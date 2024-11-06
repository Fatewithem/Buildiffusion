from typing import Optional, Union

import torch
from diffusers.schedulers import DDIMScheduler, DDPMScheduler, PNDMScheduler
from diffusers.schedulers.scheduling_lms_discrete import LMSDiscreteScheduler
from diffusers import ModelMixin
from pytorch3d.implicitron.dataset.data_loader_map_provider import FrameData
from pytorch3d.renderer import PointsRasterizationSettings, PointsRasterizer
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Pointclouds
from torch import Tensor

from .feature_model import FeatureModel
from .model_utils import compute_distance_transform

SchedulerClass = Union[DDPMScheduler, DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler]


class PointCloudProjectionModel(ModelMixin):
    def __init__(
            self,
            image_height: int,
            image_width: int,
            use_top: bool,
            use_depth: bool,
            image_feature_model: str,
            use_local_colors: bool = False,
            use_local_features: bool = False,
            use_global_features: bool = False,
            use_mask: bool = False,
            use_distance_transform: bool = False,
            predict_shape: bool = True,
            predict_color: bool = False,
            process_color: bool = False,
            image_color_channels: int = 3,  # for the input image, not the points
            color_channels: int = 3,  # for the points, not the input image
            colors_mean: float = 0.5,
            colors_std: float = 0.5,
            scale_factor: float = 20.0,
            # Rasterization settings
            raster_point_radius: float = 0.005,  # point size
            raster_points_per_pixel: int = 10,  # a single point per pixel, for now
            bin_size: int = 0,
    ):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.use_top = use_top
        self.use_depth = use_depth
        self.scale_factor = scale_factor
        self.use_local_colors = use_local_colors
        self.use_local_features = use_local_features
        self.use_global_features = use_global_features
        self.use_mask = use_mask
        self.use_distance_transform = use_distance_transform
        self.predict_shape = predict_shape
        self.predict_color = predict_color
        self.process_color = process_color
        self.image_color_channels = image_color_channels
        self.color_channels = color_channels
        self.colors_mean = colors_mean
        self.colors_std = colors_std

        # Types of conditioning that are used
        self.use_local_conditioning = self.use_local_colors or self.use_local_features or self.use_mask
        self.use_global_conditioning = self.use_global_features

        # Create feature model
        # self.feature_model = FeatureModel(image_height, image_width, image_feature_model)

        # Input size
        # self.in_channels = 3  # 3 for 3D point positions
        self.in_channels = 13  # use_distance_transform

        # if self.use_local_colors:
        #     self.in_channels += self.image_color_channels
        # if self.use_local_features:
        #     self.in_channels += self.feature_model.feature_dim
        # if self.use_global_features:
        #     self.in_channels += self.feature_model.feature_dim
        # if self.use_mask:
        #     self.in_channels += 2 if self.use_distance_transform else 1
        # if self.process_color:  # whether color should be an input
        #     self.in_channels += self.color_channels

        # Output size
        self.out_channels = 0
        if self.predict_shape:
            self.out_channels += 3
        if self.predict_color:
            self.out_channels += self.color_channels

        # Save rasterization settings
        self.raster_settings = PointsRasterizationSettings(
            image_size=(1080, 1920),
            radius=raster_point_radius,
            points_per_pixel=raster_points_per_pixel,
            bin_size=bin_size,
        )

    def denormalize(self, x: Tensor, /, clamp: bool = True):
        x = x * self.colors_std + self.colors_mean
        return torch.clamp(x, 0, 1) if clamp else x

    def normalize(self, x: Tensor, /):
        x = (x - self.colors_mean) / self.colors_std
        return x

    def get_global_conditioning(self, image_rgb: Tensor):
        global_conditioning = []
        if self.use_global_features:
            global_conditioning.append(self.feature_model(image_rgb,
                                                          return_cls_token_only=True))  # (B, D)
        global_conditioning = torch.cat(global_conditioning, dim=1)  # (B, D_cond)
        return global_conditioning

    def get_local_conditioning(self, image_rgb: Tensor, mask: Tensor):
        local_conditioning = []
        if self.use_local_colors:
            local_conditioning.append(self.normalize(image_rgb))
        if self.use_local_features:
            local_conditioning.append(self.feature_model(image_rgb))
        if self.use_mask:
            local_conditioning.append(mask.float())
        if self.use_distance_transform:
            if not self.use_mask:
                raise ValueError('No mask for distance transform?')
            if mask.is_floating_point():
                mask = mask > 0.5
            local_conditioning.append(compute_distance_transform(mask))
        local_conditioning = torch.cat(local_conditioning, dim=1)  # (B, D_cond, H, W)
        return local_conditioning

    @torch.autocast('cuda', dtype=torch.float32)
    def surface_projection(self, points: Tensor, camera: CamerasBase, local_features: Tensor):
        B, C, H, W, device = *local_features.shape, local_features.device
        R = self.raster_settings.points_per_pixel
        N = points.shape[1]

        # Scale camera by scaling T. ASSUMES CAMERA IS LOOKING AT ORIGIN!
        camera = camera.clone()
        camera.T = camera.T / self.scale_factor

        # Create rasterizer
        rasterizer = PointsRasterizer(cameras=camera, raster_settings=self.raster_settings)

        # Associate points with features via rasterization
        fragments = rasterizer(Pointclouds(points))  # (B, H, W, R)
        fragments_idx: Tensor = fragments.idx.long()

        visible_pixels = (fragments_idx > -1)  # (B, H, W, R)

        points_to_visible_pixels = fragments_idx[visible_pixels]

        # 确保 local_features 维度顺序一致
        local_features = local_features.permute(0, 2, 3, 1).unsqueeze(-2).expand(-1, -1, -1, R, -1)  # (B, H, W, R, C)

        # 初始化 local_features_proj 并检查形状
        local_features_proj = torch.zeros(B * N, C, device=device)  # (B * N, C)

        # 尝试赋值，并捕获索引错误
        try:
            local_features_proj[points_to_visible_pixels] = local_features[visible_pixels]
        except IndexError as e:
            print(f"IndexError: {e}")
            print(f"points_to_visible_pixels shape: {points_to_visible_pixels.shape}")
            print(f"visible_pixels shape: {visible_pixels.shape}")

        # 重新调整输出形状
        local_features_proj = local_features_proj.reshape(B, N, C)

        return local_features_proj

    def point_cloud_to_tensor(self, pc: Pointclouds, /, normalize: bool = False, scale: bool = False):
        """Converts a point cloud to a tensor, with color if and only if self.predict_color"""
        points = pc.points_padded() / (self.scale_factor if scale else 1)

        if self.predict_color and pc.features_padded() is not None:
            colors = self.normalize(pc.features_padded()) if normalize else pc.features_padded()
            return torch.cat((points, colors), dim=2)
        else:
            return points

    def tensor_to_point_cloud(self, x: Tensor, /, denormalize: bool = False, unscale: bool = False):
        points = x[:, :, :3] * (self.scale_factor if unscale else 1)
        if self.predict_color:
            colors = self.denormalize(x[:, :, 3:]) if denormalize else x[:, :, 3:]
            return Pointclouds(points=points, features=colors)
        else:
            assert x.shape[2] == 3
            return Pointclouds(points=points)

    def get_input_with_conditioning(
            self,
            x_t: Tensor,
            camera: Optional[CamerasBase],
            image_rgb: Optional[Tensor],
            mask: Optional[Tensor],
            t: Optional[Tensor],
    ):
        """ Extracts local features from the input image and projects them onto the points
            in the point cloud to obtain the input to the model. Then extracts global
            features, replicates them across points, and concats them to the input."""
        B, N = x_t.shape[:2]

        # Initial input is the point locations (and colors if and only if predicting color)
        x_t_input = [x_t]

        # Local conditioning
        if self.use_local_conditioning:

            # Get local features and check that they are the same size as the input image
            local_features = self.get_local_conditioning(image_rgb=image_rgb, mask=mask)
            if local_features.shape[-2:] != image_rgb.shape[-2:]:
                raise ValueError(f'{local_features.shape=} and {image_rgb.shape=}')

            # Project local features. Here that we only need the point locations, not colors
            local_features_proj = self.surface_projection(points=x_t[:, :, :3],
                                                          camera=s,
                                                          local_features=local_features)  # (B, N, D_local)

            x_t_input.append(local_features_proj)

        # Global conditioning
        if self.use_global_conditioning:
            # Get and repeat global features
            global_features = self.get_global_conditioning(image_rgb=image_rgb)  # (B, D_global)
            global_features = global_features.unsqueeze(1).expand(-1, N, -1)  # (B, D_global, N)

            x_t_input.append(global_features)

        # Concatenate together all the pointwise features
        x_t_input = torch.cat(x_t_input, dim=2)  # (B, N, D)

        return x_t_input

    def get_input_with_bae(
            self,
            x_t: Tensor,
            camera: Optional[CamerasBase],
            mask: Optional[Tensor],  # turple B
            t: Optional[Tensor],
    ):
        # 确定输入张量所在的设备
        device = x_t.device

        B, N = x_t.shape[:2]

        # 初始化各个mask的张量 (B, 1, H, W)
        mask_0_tensor = torch.zeros((B, 1, 1080, 1920), device=device)
        mask_1_tensor = torch.zeros((B, 1, 1080, 1920), device=device)
        mask_2_tensor = torch.zeros((B, 1, 1080, 1920), device=device)
        mask_3_tensor = torch.zeros((B, 1, 1080, 1920), device=device)
        mask_4_tensor = torch.zeros((B, 1, 1080, 1920), device=device)

        # 将 mask 数据逐个添加到 mask_0_tensor 和 mask_1_tensor 中
        for i, m in enumerate(mask):
            mask_0_tensor[i] = m[0].to(device)
            mask_1_tensor[i] = m[1].to(device)
            mask_2_tensor[i] = m[2].to(device)
            mask_3_tensor[i] = m[3].to(device)
            mask_4_tensor[i] = m[4].to(device)

        mask_list = [mask_0_tensor, mask_1_tensor, mask_2_tensor, mask_3_tensor, mask_4_tensor]

        # 初始化输入列表，并确保 x_t 在同一设备上
        x_t_input = [x_t.to(device)]

        # 遍历相机和 mask_list，确保所有张量在同一设备上
        for c, m in zip(camera, mask_list):  # B 1 H W
            camera = c

            # 将 mask 和相关计算结果移动到相同设备
            cdt = compute_distance_transform(m).to(device)

            bae_cdt_proj = self.surface_projection(
                points=x_t[:, :, :3],  # B, N, 1
                camera=camera,
                local_features=cdt
            ).to(device)

            mask_proj = self.surface_projection(
                points=x_t[:, :, :3],  # B, N, 1
                camera=camera,
                local_features=m
            ).to(device)

            # 将计算的特征添加到输入列表
            x_t_input.append(bae_cdt_proj)
            x_t_input.append(mask_proj)

        # 确保所有张量在同一设备上后拼接
        x_t_input = torch.cat(x_t_input, dim=2)  # (B, N, D)

        return x_t_input

    def forward(self, batch: FrameData, mode: str = 'train', **kwargs):
        """ The forward method may be defined differently for different models. """
        raise NotImplementedError()























