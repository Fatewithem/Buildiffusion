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

# 保留注释，移除顶层导入，避免循环依赖
from .model_utils import get_num_points, get_custom_betas, render_point_cloud
from .point_cloud_model import PointCloudModel
from .projection_model import PointCloudProjectionModel


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

    def forward_train(
            self,
            pc: Pointclouds,
            masks: Optional[Tensor],
            camera: Optional[Dict],
            # image_rgb: Optional[Tensor],
            return_intermediate_steps: bool = False
    ):
        # Normalize colors and convert to tensor
        x_0 = self.point_cloud_to_tensor(pc, normalize=True, scale=True)  # (-1, 1)
        # render_point_cloud(x_0, "origin")
        B, N, D = x_0.shape  # Batch; Num; Dimen

        # Sample random noise
        noise = torch.randn_like(x_0)

        # Sample random timesteps for each point_cloud
        timestep = torch.randint(0, self.scheduler.num_train_timesteps, (B,),
                                 device=self.device, dtype=torch.long)

        # Add noise to points
        x_t = self.scheduler.add_noise(x_0, noise, timestep)

        # Conditioning
        # x_t_input = self.get_input_with_conditioning(x_t, camera=camera,
        #                                              image_rgb=image_rgb, mask=mask, t=timestep)

        # Without Conditioning
        # x_t_input = x_t

        # Bae Conditioning
        x_t_input = self.get_input_with_bae(x_t, camera=camera, mask=masks, t=timestep)



        # Forward
        noise_pred = self.point_cloud_model(x_t_input, timestep)

        # 打印数值范围：最小值、最大值和平均值
        # print(f"Min value: {noise_pred.min().item()}")
        # print(f"Max value: {noise_pred.max().item()}")
        # print(f"Mean value: {noise_pred.mean().item()}")

        # Check
        if not noise_pred.shape == noise.shape:
            raise ValueError(f'{noise_pred.shape=} and {noise.shape=}')

        # def reconstruct_x0(x_t, noise_pred, t, scheduler):
        #     """
        #     使用模型预测的噪声来计算去噪后的数据 x_reconstructed。
        #     """
        #     # 获取设备
        #     device = x_t.device
        #
        #     # 将 timestep, alpha_t, beta_t 移动到同一设备
        #     alpha_t = scheduler.alphas_cumprod[t].to(device).view(-1, 1, 1)  # 累积 alpha
        #     beta_t = scheduler.betas[t].to(device).view(-1, 1, 1)  # 当前步的 beta
        #
        #     # 反向公式计算去噪后的 x_{t-1}
        #     x_reconstructed = (1 / torch.sqrt(alpha_t)) * (
        #             x_t - beta_t / torch.sqrt(1 - alpha_t) * noise_pred
        #     )
        #
        #     return x_reconstructed

        # 计算去噪后的数据 x_reconstructed
        # x_reconstructed = reconstruct_x0(x_t, noise_pred, timestep, self.scheduler)
        # render_point_cloud(x_reconstructed, "reconstructed")

        # DDPM Loss
        loss_ddpm = F.mse_loss(noise_pred, noise)

        # BAE Loss
        # from .mask_loss import get_mask_loss  # 在这里导入 get_mask_loss，避免循环依赖
        # loss_bae = get_mask_loss(x_reconstructed, masks, B)
        #
        # loss = 0.5 * loss_ddpm + 0.5 * loss_bae

        loss = loss_ddpm

        # Whether to return intermediate steps
        if return_intermediate_steps:
            return loss, (x_0, x_t, noise, noise_pred)

        return loss

    @torch.no_grad()
    def forward_sample(
            self,
            num_points: int,
            camera: Optional[CamerasBase],
            # image_rgb: Optional[Tensor],
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
        # device = self.device if image_rgb is None else image_rgb.device
        device = self.device

        # Sample noise
        x_t = torch.randn(B, N, D, device=device)

        # Set timesteps
        accepts_offset = "offset" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys())  # 获取 scheduler.set_timesteps 函数的签名
        extra_set_kwargs = {"offset": 1} if accepts_offset else {}
        scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

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
            # x_t_input = self.get_input_with_conditioning(x_t, camera=camera,
            #                                              image_rgb=image_rgb, mask=mask, t=t)

            # Bae Conditioning
            x_t_input = self.get_input_with_bae(x_t, camera=camera, mask=masks, t=t)

            # Without Conditioning
            # x_t_input = x_t

            # Forward
            noise_pred = self.point_cloud_model(x_t_input, t.reshape(1).expand(B))

            # Step
            x_t = scheduler.step(noise_pred, t, x_t, **extra_step_kwargs).prev_sample

            render_point_cloud(x_t, "pred")

            # Append to output list if desired
            if (return_all_outputs and (i % return_sample_every_n_steps == 0 or i == len(scheduler.timesteps) - 1)):
                all_outputs.append(x_t)

        # Convert output back into a point cloud, undoing normalization and scaling
        output = self.tensor_to_point_cloud(x_t, denormalize=True, unscale=True)
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
                # image_rgb=batch.image_rgb,
                # mask=batch.fg_probability,
                **kwargs)
        elif mode == 'sample':
            num_points = kwargs.pop('num_points', get_num_points(batch['pointclouds']))
            return self.forward_sample(
                num_points=100000,
                masks=batch['masks_list'],
                camera=batch['camera_list'],
                # image_rgb=batch.image_rgb,
                # mask=batch.fg_probability,
                **kwargs)
        else:
            raise NotImplementedError()