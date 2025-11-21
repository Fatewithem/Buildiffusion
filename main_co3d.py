import datetime
import math
import os
import json

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
from model import get_co3d_model
from config.structured import ProjectConfig

# Configuration for the model and training
from accelerate import Accelerator
import torch.nn.functional as F
import torchvision.utils as vutils

from accelerate.utils import DistributedType


# Configuration for the model and training
@hydra.main(config_path='/home/code/Buildiffusion/config', config_name='config', version_base='1.1')
def main(cfg: ProjectConfig):
    # 确认是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    from accelerate.utils import DistributedDataParallelKwargs

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], mixed_precision='fp16')

    # 在main函数中初始化 accelerator
    # accelerator = Accelerator(find_unused_parameters=True, mixed_precision='no')
    # accelerator = Accelerator(mixed_precision='no')  # Enable mixed precision

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
    model = get_co3d_model(cfg)
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
            dataloader=dataloader_train,
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
                pointcloud = batch.sequence_point_cloud_pre # 将点云数据移到 GPU
                if pointcloud is None:
                    raise ValueError(f"Error: batch.sequence_point_cloud_fps is None. 请检查是否缺少 fps 点云文件。batch.keys(): {batch.keys()}")
                masks = batch.fg_probability
                camera = batch.camera
                images = batch.image_rgb
                planes = batch.sequence_point_cloud_fps
                if planes is None:
                    raise ValueError(f"Error: batch.sequence_point_cloud_fps is None. 请检查是否缺少 fps 点云文件。batch.keys(): {batch.keys()}")

                loss_output = model(pointcloud, masks=masks, camera=camera, images=images, planes=planes)

                if loss_output is None:
                    print("⚠️ Skipping batch due to invalid loss.")
                    if hasattr(batch, "image_path"):
                        print(f"Problematic image path(s): {batch.image_path}")
                    if hasattr(batch, "sequence_name"):
                        print(f"Sequence name: {batch.sequence_name}")
                    continue

                loss, cd_loss, emd_loss, normal_loss = loss_output

                # Check if the loss is None, NaN, or Inf
                if loss is None or torch.isnan(loss).any() or torch.isinf(loss).any():
                    print("⚠️ Skipping batch due to NaN, Inf, or None loss.")
                    if hasattr(batch, "image_path"):
                        print(f"Problematic image path(s): {batch.image_path}")
                    if hasattr(batch, "sequence_name"):
                        print(f"Sequence name: {batch.sequence_name}")
                    continue
                # Ensure loss is a scalar before backward
                if loss.dim() > 0:
                    loss = loss.mean()

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
                cd_loss = cd_loss.item()
                # global_loss = global_loss.item()
                emd_loss = emd_loss.item()
                # center_loss = center_loss.item()
                normal_loss = normal_loss.item()
                # extent_loss = extent_loss.item()

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
                        'cd_loss': cd_loss,
                        # 'global_loss': global_loss,
                        'emd_loss': emd_loss,
                        # 'center_loss': center_loss,
                        'normal_loss': normal_loss,
                        # 'extent_loss': extent_loss,
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
    import torchvision.utils as vutils

    print("Sample start!!!!!!!!!!!!!!")

    model.eval()

    from pathlib import Path
    output_dir = Path("/home/code/Buildiffusion")
    (output_dir / "result").mkdir(exist_ok=True, parents=True)

    import torchvision.transforms.functional as TF
    import PIL.Image

    for batch_idx, batch in enumerate(dataloader):
        print(f"Processing batch {batch_idx}...")

        print(batch.image_path)
        if batch_idx >= 10:
            break

        # Forward pass
        pointcloud = batch.sequence_point_cloud_pre  # 将点云数据移到 GPU
        if pointcloud is None:
            raise ValueError(
                f"Error: batch.sequence_point_cloud_fps is None. 请检查是否缺少 fps 点云文件。batch.keys(): {batch.keys()}")
        masks = batch.fg_probability
        camera = batch.camera
        images = batch.image_rgb
        planes = batch.sequence_point_cloud_fps
        if planes is None:
            raise ValueError(
                f"Error: batch.sequence_point_cloud_fps is None. 请检查是否缺少 fps 点云文件。batch.keys(): {batch.keys()}")

        result, pred_points, planes, camera_used, images_used = model.forward_sample(
            pointcloud, camera, images, planes, masks
        )

        # -------------------------------
        # 1. 保存 Pred Points （整体点云）
        # -------------------------------
        pred_points_np = pred_points[0].detach().cpu().numpy()  # [Q*P, 3]
        pcd_pred = o3d.geometry.PointCloud()
        pcd_pred.points = o3d.utility.Vector3dVector(pred_points_np)

        save_pred = output_dir / "result" / f"sample_{batch_idx}_pred.ply"
        o3d.io.write_point_cloud(str(save_pred), pcd_pred)
        print(f"✔ Saved pred pointcloud → {save_pred}")

        # -------------------------------
        # 2. 保存 GT planes （Pointclouds）
        # -------------------------------
        gt_np = planes.points_packed().detach().cpu().numpy()
        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(gt_np)

        if planes.features_packed() is not None and planes.features_packed().shape[1] >= 3:
            colors = planes.features_packed()[:, :3].cpu().numpy()
            colors = np.clip(colors, 0, 1)
            pcd_gt.colors = o3d.utility.Vector3dVector(colors)

        save_gt = output_dir / "result" / f"sample_{batch_idx}_gt.ply"
        o3d.io.write_point_cloud(str(save_gt), pcd_gt)
        print(f"✔ Saved gt → {save_gt}")


if __name__ == "__main__":
    main()