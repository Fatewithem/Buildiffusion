import datetime
import math
import os

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

import sys
import time
from pathlib import Path
from accelerate import Accelerator
from typing import Any, Iterable, List, Optional

import torch
import wandb
import hydra
from torch.utils.tensorboard import SummaryWriter

import training_utils
import open3d as o3d
import numpy as np
from dataset import get_dataset
from model import get_plane_model
from config.structured import ProjectConfig


# Configuration for the model and training
from accelerate import Accelerator
import torch.nn.functional as F


# Configuration for the model and training
@hydra.main(config_path='/home/code/Buildiffusion/config', config_name='config', version_base='1.1')
def main(cfg: ProjectConfig):
    # 确认是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 在main函数中初始化 accelerator
    accelerator = Accelerator(mixed_precision='fp16')  # Enable mixed precision

    # Logging setup
    writer = SummaryWriter(log_dir=Path('tensorboard_logs'))

    if cfg.logging.wandb:
        wandb.init(project=cfg.logging.wandb_project, name=cfg.run.name, job_type=cfg.run.mode,
                   config=OmegaConf.to_container(cfg))
        wandb.run.log_code(root=hydra.utils.get_original_cwd(),
                           include_fn=lambda p: any(
                               p.endswith(ext) for ext in ('.py', '.json', '.yaml', '.md', '.txt.', '.gin')))

    # Set random seed
    training_utils.set_seed(cfg.run.seed)

    # Model initialization
    model = get_plane_model(cfg)
    model = model.to(device)  # 将模型移到 GPU 或 CPU

    print(f'Total Parameters: {sum(p.numel() for p in model.parameters()):_d}')
    print(f'Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):_d}')

    # Optimizer and scheduler
    optimizer = training_utils.get_optimizer(cfg, model, accelerator)
    scheduler = training_utils.get_scheduler(cfg, optimizer)

    # Resume from checkpoint and create the initial training state
    train_state: training_utils.TrainState = training_utils.resume_from_checkpoint(cfg, model, optimizer, scheduler)

    # Datasets
    dataloader_train, dataloader_val, dataloader_vis = get_dataset(cfg)

    # Prepare model, optimizer, and dataloaders (ensure they're on the correct device)
    model, optimizer, scheduler, dataloader_train, dataloader_val, dataloader_vis = accelerator.prepare(
        model, optimizer, scheduler, dataloader_train, dataloader_val, dataloader_vis
    )

    # Sample from the model
    if cfg.run.mode == 'sample':
        sample(
            cfg=cfg,
            model=model,
            dataloader=dataloader_val,
            # dataloader=dataloader_train,
            # accelerator=accelerator,
        )
        if cfg.logging.wandb and accelerator.is_main_process:
            wandb.finish()
        time.sleep(5)
        return

    # 计算每个 epoch 应该包含多少个 step
    steps_per_epoch = len(dataloader_train)

    # Train loop
    print(f'***** Starting training at {datetime.datetime.now()} *****')
    while True:
        log_header = f'Epoch: [{train_state.epoch}]'
        metric_logger = training_utils.MetricLogger(delimiter="  ")

        metric_logger.add_meter('step', training_utils.SmoothedValue(window_size=1, fmt='{value:.0f}'))
        metric_logger.add_meter('lr', training_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        progress_bar = metric_logger.log_every(dataloader_train, cfg.run.print_step_freq, header=log_header)

        for i, batch in enumerate(progress_bar):
            if (cfg.run.limit_train_batches is not None) and (i >= cfg.run.limit_train_batches): break
            model.train()

            # Gradient accumulation
            with accelerator.accumulate(model):

                # Forward pass
                pointcloud = batch['pointclouds']  # 将点云数据移到 GPU
                masks = batch['masks_list']
                camera = batch['camera_list']
                images = batch['images_list']
                planes = batch['plane_list']
                colors = batch['plane_color_list']
                plane_infos = batch['plane_info_list']

                # Forward pass
                with torch.cuda.amp.autocast():  # Enables mixed precision for forward pass
                    # loss, total_loss, global_loss, emd_loss, color_loss = \
                    #     model(pointcloud, masks=masks, camera=camera, images=images, planes=planes, colors=colors, plane_infos=plane_infos)

                    loss, total_loss, global_loss, emd_loss = \
                        model(pointcloud, masks=masks, camera=camera, images=images, planes=planes, colors=colors,
                              plane_infos=plane_infos)

                    # Check if the loss is NaN
                    if torch.isnan(loss).any() or torch.isinf(loss).any():
                        print(f"NaN or Inf detected in loss: {loss}")
                        sys.exit(1)

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        # 梯度裁剪
                        if cfg.optimizer.clip_grad_norm is not None:
                            accelerator.clip_grad_norm_(model.parameters(), cfg.optimizer.clip_grad_norm)
                        grad_norm_clipped = training_utils.compute_grad_norm(model.parameters())

                        optimizer.step()
                        optimizer.zero_grad()

                    if accelerator.sync_gradients and accelerator.is_main_process:
                        writer.add_scalar('Loss/train', loss.item(), train_state.step)
                        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], train_state.step)

                    if accelerator.sync_gradients:
                        scheduler.step()
                        train_state.step += 1

                        # Exit if loss was NaN
                    loss_value = loss.item()
                    total_loss = total_loss.item()
                    global_loss = global_loss.item()
                    # color_loss = color_loss.item()

                    if not math.isfinite(loss_value):
                        print("Loss is {}, stopping training".format(loss_value))
                        writer.close()  # 关闭 TensorboardX 日志记录器
                        sys.exit(1)

                if accelerator.sync_gradients:
                    # Logging
                    log_dict = {
                        'lr': optimizer.param_groups[0]["lr"],
                        'step': train_state.step,
                        'train_loss': loss_value,
                        'cd_loss': total_loss,
                        # 'color_loss': color_loss,
                        'global_loss': global_loss,
                        'emd_loss': emd_loss,
                        'grad_norm_clipped': grad_norm_clipped,
                    }
                    metric_logger.update(**log_dict)

                    # Save a checkpoint
                    if accelerator.is_main_process and (train_state.step % cfg.run.checkpoint_freq == 0):
                        checkpoint_dict = {
                            'model': accelerator.unwrap_model(model).state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'epoch': train_state.epoch,
                            'step': train_state.step,
                            'best_val': train_state.best_val,
                            'cfg': cfg
                        }

                        # 根据 step 生成唯一的文件名
                        checkpoint_path = f'checkpoint-step-{train_state.step}.pth'

                        # 保存检查点
                        accelerator.save(checkpoint_dict, checkpoint_path)
                        print(f'Saved checkpoint to {Path(checkpoint_path).resolve()}')

                    if train_state.step >= cfg.run.max_steps:
                        print(f'Ending training at: {datetime.datetime.now()}')
                        print(f'Final train state: {train_state}')

                        wandb.finish()
                        time.sleep(5)
                        return

            if train_state.step % steps_per_epoch == 0:
                train_state.epoch += 1

            # Gather stats from all processes
            metric_logger.synchronize_between_processes(device=accelerator.device)
            if accelerator.is_main_process:
                print(f'{log_header}  Average stats --', metric_logger)


@torch.no_grad()
def sample(
        *,
        cfg: ProjectConfig,
        model: torch.nn.Module,
        dataloader: Iterable,
        output_dir: str = 'sample',
):
    from pytorch3d.io import IO
    from pytorch3d.structures import Pointclouds
    from tqdm import tqdm

    print("Sample start!!!!!!!!!!!!!!")

    # Eval mode
    model.eval()

    # Output dir
    output_dir: Path = Path(output_dir)

    # PyTorch3D IO
    io = IO()

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 10:  # 只处理 10 到 20 之间的 batch
            break

        print(f"Processing batch {batch_idx + 1}...")

        filename = f'sample_{batch_idx}.ply'
        filestr = str(output_dir / 'building' / filename)

        pointcloud = batch['pointclouds']
        camera = batch['camera_list']
        images = batch['images_list']
        planes = batch['plane_list']
        colors = batch['plane_color_list']

        result, tensor, plane = model.forward_sample(pointcloud, camera, images, planes, colors)

        sequence_name = f"sample_{batch_idx}"
        sequence_category = 'buildings'
        (output_dir / 'gt' / sequence_category).mkdir(exist_ok=True, parents=True)
        (output_dir / 'pred' / sequence_category).mkdir(exist_ok=True, parents=True)
        (output_dir / 'images' / sequence_category).mkdir(exist_ok=True, parents=True)
        (output_dir / 'metadata' / sequence_category).mkdir(exist_ok=True, parents=True)
        (output_dir / 'evolutions' / sequence_category).mkdir(exist_ok=True, parents=True)

        # 创建 Open3D 的 PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(result.points_packed().cpu().numpy())
        # pcd.colors = o3d.utility.Vector3dVector(result.features_packed().cpu().numpy().astype(np.float32))

        # 保存为 PLY 格式（不同 batch 存不同文件）
        ply_path = f"/home/code/Buildiffusion/result/sample_{batch_idx}.ply"
        o3d.io.write_point_cloud(ply_path, pcd)
        print(f"✅ Batch {batch_idx} 点云已保存: {ply_path}")

        # 创建 Open3D 的 PointCloud
        pcgt = o3d.geometry.PointCloud()
        pcgt.points = o3d.utility.Vector3dVector(plane.points_packed().cpu().numpy())
        pcgt.colors = o3d.utility.Vector3dVector(plane.features_packed().cpu().numpy().astype(np.float32))

        save_path_gt = f"/home/code/Buildiffusion/result/sample_gt_{batch_idx}.ply"
        # io.save_pointcloud(data=plane, path=save_path_gt)
        o3d.io.write_point_cloud(save_path_gt, pcgt)

        save_path_tensor = f"/home/code/Buildiffusion/result/sample_{batch_idx}.pt"
        torch.save(tensor, save_path_tensor)

    print('Saved all samples to: ')
    print(output_dir.absolute())


if __name__ == "__main__":
    main()