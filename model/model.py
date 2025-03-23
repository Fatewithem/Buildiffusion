import inspect
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
import numpy as np
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union, Any
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_pndm import PNDMScheduler
from pytorch3d.implicitron.dataset.data_loader_map_provider import FrameData
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Pointclouds
from pytorch3d.loss import chamfer_distance

# 保留注释，移除顶层导入，避免循环依赖
from .model_utils import get_num_points, get_custom_betas, render_point_cloud
from .point_cloud_model import PointCloudModel
from .projection_model import PointCloudProjectionModel
# from cost_volume.cost_volume import MVSFormerWithDino
from cost_volume.utils import load_config


class ConditionalPointCloudDiffusionModel(PointCloudProjectionModel):
    def __init__(
        self,
        beta_start: float,
        beta_end: float,
        beta_schedule: str,
        point_cloud_model: str,
        point_cloud_model_embed_dim: int,
        **kwargs,  # projection arguments
    ):
        super().__init__(**kwargs)

        # Checks
        if not self.predict_shape:
            raise NotImplementedError('Must predict shape if performing diffusion.')

        num_train_timesteps = 1000

        # Create diffusion model schedulers which define the sampling timesteps
        scheduler_kwargs = {
            'num_train_timesteps': num_train_timesteps
        }
        if beta_schedule == 'custom':
            scheduler_kwargs.update(dict(trained_betas=get_custom_betas(beta_start=beta_start, beta_end=beta_end)))
        else:
            scheduler_kwargs.update(dict(beta_start=beta_start, beta_end=beta_end, beta_schedule=beta_schedule, num_train_timesteps=num_train_timesteps))
        self.schedulers_map = {
            'ddpm': DDPMScheduler(**scheduler_kwargs, clip_sample=False),
            'ddim': DDIMScheduler(**scheduler_kwargs, clip_sample=False),
            'pndm': PNDMScheduler(**scheduler_kwargs),
        }
        self.scheduler = self.schedulers_map['ddpm']  # this can be changed for inference

        # Create point cloud model for processing point cloud at each diffusion step
        self.point_cloud_model = PointCloudModel(
            model_type=point_cloud_model,
            embed_dim=point_cloud_model_embed_dim,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
        )

        # # 加载配置和初始化模型
        # config_path = "/home/code/Dino/config/config.json"
        # config = load_config(config_path)
        # transformer_config = config['transformer_config']
        # model_path = "/home/code/Buildiffusion/cost_volume/dinov2_base"
        #
        # self.mvs_cost_volume_model = MVSFormerWithDino(
        #     dino_model_path=model_path,
        #     transformer_config=transformer_config,
        # )
        #
        # # 如果你只想使用预训练特征，不需要更新它的参数，就这样冻结
        # self.mvs_cost_volume_model.eval()  # 切换到 eval 模式（关闭 BatchNorm、Dropout 等）
        # for param in self.mvs_cost_volume_model.parameters():
        #     param.requires_grad_(False)  # 关闭梯度，使其在反向传播中不被更新

        # self.mask_weight_fusion = MaskWeightFusion(
        #     num_masks=5,  # 根据你的具体需求调整
        #     mask_0_bias=1.0,
        # )
        # self.mask_weight_fusion = self.mask_weight_fusion.to(self.device)  # 确保在正确的设备上

    def forward_train(
            self,
            pc: Pointclouds,
            masks: Optional[Tensor],
            camera: Optional[Dict],
            images: Optional[Tensor],
            planes: Optional[Dict],
    ):
        # Normalize colors and convert to tensor
        x_0 = self.point_cloud_to_tensor(pc, normalize=True, scale=False)  # (-1, 1)

        B, N, D = x_0.shape  # Batch; Num; Dimen

        device = self.device

        # Sample random noise
        noise = torch.randn_like(x_0)

        # Sample random timesteps for each point_cloud
        timestep = torch.randint(0, self.scheduler.config.num_train_timesteps, (B,),
                                 device=self.device, dtype=torch.long)

        # Add noise to points
        x_t = self.scheduler.add_noise(x_0, noise, timestep)

        # render_point_cloud(x_0, "x_0")
        
        # render_point_cloud(x_t, "x_t")

        # Bae Conditioning
        # x_t_input = self.get_input_with_bae(x_t, camera=camera, mask=masks, t=timestep)  # B N 6+1

        x_t_input = self.get_input_with_conditioning(x_t, camera=camera, images=images, mask=masks, t=timestep)

        # Cost Volume Conditioning
        # x_t_input = self.get_input_with_cost_volume(x_t, camera=camera, image=images, model=self.mvs_cost_volume_model, mask=masks, t=timestep, device=device)  # B N 6+32

        # x_t_input = x_t

        # Forward
        noise_pred = self.point_cloud_model(x_t_input, timestep)  # [B, N, 3+3+1]

        # Check
        if not noise_pred.shape == noise.shape:
            raise ValueError(f'{noise_pred.shape=} and {noise.shape=}')

        # DDPM Loss
        loss_ddpm = F.mse_loss(noise_pred, noise)

        # loss_normals = F.mse_loss(noise_pred[..., 3:], noise[..., 3:])  # 法向量部分
        # cosine_similarity = F.cosine_similarity(noise_pred[..., 3:], noise[..., 3:], dim=-1)
        # loss_normals = 1 - cosine_similarity.mean()  # 余弦相似性损失

        # Chamfer Distance Loss
        # loss_cd, _ = chamfer_distance(noise_pred, noise)

        # 总损失
        # loss = 0.8 * loss_ddpm + 0.2 * loss_normals

        loss = loss_ddpm

        # Whether to return intermediate steps
        if return_intermediate_steps:
            return loss, (x_0, x_t, noise, noise_pred)

        # return loss, loss_ddpm, loss_normals
        return loss

    @torch.no_grad()
    def forward_sample(
            self,
            num_points: int,
            camera: Optional[CamerasBase],
            images: Optional[Tensor],
            masks: Optional[Tensor],
            # Optional overrides
            scheduler: Optional[str] = 'ddpm',
            # Inference parameters
            num_inference_steps: Optional[int] = 1000,
            eta: Optional[float] = 0.0,  # for DDIM
            # Whether to return all the intermediate steps in generation
            return_sample_every_n_steps: int = -1,
            # Whether to disable tqdm
            disable_tqdm: bool = False,
    ):
        # Get scheduler from mapping, or use self.scheduler if None
        scheduler = self.scheduler if scheduler is None else self.schedulers_map[scheduler]

        # Get the size of the noise
        N = num_points
        # B = 1 if image_rgb is None else image_rgb.shape[0]
        B = 1

        D = 3 + (self.color_channels if self.predict_color else 0)
        # print(f"D: {D}")

        # device = self.device if image_rgb is None else image_rgb.device
        device = self.device

        # Sample noise
        x_t = torch.randn(B, N, D, device=device)

        # Set timesteps
        accepts_offset = "offset" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys())  # 获取 scheduler.set_timesteps 函数的签名
        extra_set_kwargs = {"offset": 1} if accepts_offset else {}
        scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        scheduler.timesteps = scheduler.timesteps.to(device)

        # Prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
        extra_step_kwargs = {"eta": eta} if accepts_eta else {}

        # Loop over timesteps
        all_outputs = []
        return_all_outputs = (return_sample_every_n_steps > 0)
        progress_bar = tqdm(scheduler.timesteps.to(device), desc=f'Sampling ({x_t.shape})', disable=disable_tqdm)

        for i, t in enumerate(progress_bar):

            # Conditioning
            x_t_input = self.get_input_with_conditioning(x_t, camera=camera, images=images, mask=masks, t=t)

            # 全部添加特征
            # x_t_input = [x_t.to(device)]
            # x_t_input.append(cv_features)
            # x_t_input = torch.cat(x_t_input, dim=2)

            # Bae Conditioning
            # x_t_input = self.get_input_with_cost_volume(x_t, camera=camera, image=images, model=self.mvs_cost_volume_model, mask=masks,
            #                                             t=t, device=device)  # B N 6+32

            # x_t_input = self.get_input_with_bae(x_t, camera=camera, mask=masks, t=t)

            # x_t_input = x_t

            # Forward
            noise_pred = self.point_cloud_model(x_t_input, t.reshape(1).expand(B))
            # noise_pred_normals = noise_pred[..., 3:] / (noise_pred[..., 3:].norm(dim=-1, keepdim=True) + 1e-8)
            # noise_pred = torch.cat([noise_pred[..., :3], noise_pred_normals], dim=-1)

            # device = x_t.device  # 假设 x_t 的设备是目标设备
            # scheduler.timesteps = scheduler.timesteps.to(device)

            # Step
            x_t = scheduler.step(noise_pred, t, x_t, **extra_step_kwargs).prev_sample

            # 打印 x_t 的统计信息
            # print(
            #     f"x_t Min: {x_t.min().item():.6f}, Max: {x_t.max().item():.6f}, Mean: {x_t.mean().item():.6f}, Std: {x_t.std().item():.6f}")

            # 对法向量进行归一化处理
            # if self.predict_normals:
            #     normals_pred = x_t[..., 3:]
            #     normals_pred = F.normalize(normals_pred, dim=-1)
            #     x_t = torch.cat([x_t[..., :3], normals_pred], dim=-1)

            # render_point_cloud(noise_pred, "noise")

            # render_point_cloud(x_t, "pred")

            # Append to output list if desired
            if (return_all_outputs and (i % return_sample_every_n_steps == 0 or i == len(scheduler.timesteps) - 1)):
                all_outputs.append(x_t)

        # Convert output back into a point cloud, undoing normalization and scaling
        output = self.tensor_to_point_cloud(x_t , denormalize=True, unscale=False)

        if return_all_outputs:
            all_outputs = torch.stack(all_outputs, dim=1)  # (B, sample_steps, N, D)
            all_outputs = [self.tensor_to_point_cloud(o, denormalize=True, unscale=True) for o in all_outputs]

        return (output, all_outputs) if return_all_outputs else output

    def forward(self, batch: dict, mode: str = 'train', **kwargs):
        """A wrapper around the forward method for training and inference"""

        # if isinstance(batch, dict):  # fixes a bug with multiprocessing where batch becomes a dict
        #     batch = FrameData(**batch)  # it really makes no sense, I do not understand it

        if mode == 'train':
            return self.forward_train(
                pc=batch['pointclouds'],
                masks=batch['masks_list'],
                camera=batch['camera_list'],
                images=batch['images_list'],
                # mask=batch.fg_probability,
                **kwargs)
        elif mode == 'sample':
            num_points = kwargs.pop('num_points', get_num_points(batch['pointclouds']))
            return self.forward_sample(
                num_points=30000,
                masks=batch['masks_list'],
                camera=batch['camera_list'],
                images=batch['images_list'],
                # mask=batch.fg_probability,
                **kwargs)
        else:
            raise NotImplementedError()