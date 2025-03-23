from typing import Optional, Union

import torch
import os

from diffusers.schedulers import DDIMScheduler, DDPMScheduler, PNDMScheduler
from diffusers.schedulers.scheduling_lms_discrete import LMSDiscreteScheduler
from diffusers import ModelMixin
from pytorch3d.implicitron.dataset.data_loader_map_provider import FrameData
from pytorch3d.renderer import PointsRasterizationSettings, PointsRasterizer
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Pointclouds
from torch import Tensor
import matplotlib.pyplot as plt
import torch.nn as nn

from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns

from .feature_model import FeatureModel
from .model_utils import compute_distance_transform, render_point_cloud
# from cost_volume.cost_volume import build_cost_volume, MVSFormerWithDino, DinoFeatureExtractor
from cost_volume.cost_volume import DinoFeatureExtractor
from sklearn.decomposition import PCA
from cost_volume.utils import load_config

SchedulerClass = Union[DDPMScheduler, DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler]

from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
)
from pytorch3d.structures import Pointclouds


class PointCloudProjectionModel(ModelMixin):
    def __init__(
            self,
            image_height: int = 224,
            image_width: int = 224,
            use_top: bool = True,
            use_depth: bool = True,
            image_feature_model: str = "2222222",
            use_local_colors: bool = True,
            use_local_features: bool = True,
            use_global_features: bool = True,
            use_mask: bool = True,
            use_distance_transform: bool = False,
            predict_shape: bool = True,
            predict_normal: bool = False,
            predict_color: bool = True,
            process_color: bool = False,
            image_color_channels: int = 3,  # for the input image, not the points
            color_channels: int = 3,  # for the points, not the input image
            colors_mean: float = 0.5,
            colors_std: float = 0.5,
            scale_factor: float = 10.0,
            # Rasterization settings
            raster_point_radius: float = 0.008,  # point size
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
        self.predict_normal = predict_normal
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
        self.dino_extractor = DinoFeatureExtractor(model_path="/home/code/Buildiffusion/cost_volume/dinov2_base")

        # 加载配置和初始化模型
        # config_path = "/home/code/Dino/config/config.json"
        # config = load_config(config_path)
        # transformer_config = config['transformer_config']
        # model_path = "/home/code/Buildiffusion/cost_volume/dinov2_base"

        # self.mvs_cost_volume_model = MVSFormerWithDino(
        #     dino_model_path=model_path,
        #     transformer_config=transformer_config,
        # )

        # Input size
        # self.in_channels = 3  # 3 for 3D point positions
        self.in_channels = 779  # 779

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
        if self.predict_normal:
            self.out_channels += 3
        if self.predict_color:
            self.out_channels += self.color_channels

        # Save rasterization settings
        self.raster_settings = PointsRasterizationSettings(
            image_size=(546, 966),
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
            cls_token_feature = self.dino_extractor(image_rgb, return_type="cls_token")  # (B, D)
            print(f"cls: {cls_token_feature.shape}")
            global_conditioning.append(cls_token_feature)

        global_conditioning = torch.cat(global_conditioning, dim=1)  # (B, D_cond)
        return global_conditioning

    def get_local_conditioning(self, image_rgb: Tensor, mask: Tensor):
        local_conditioning = []
        if self.use_local_colors:
            # print(f"self.normalize(image_rgb): {self.normalize(image_rgb).shape}")
            local_conditioning.append(self.normalize(image_rgb))
        if self.use_local_features:
            local_features = self.dino_extractor(image_rgb, return_type="features")  # (B, D, H_tokens, W_tokens)
            # print(f"local_features: {local_features.shape}")
            local_conditioning.append(local_features)
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
    def surface_projection(
            self, points: Tensor, camera: CamerasBase, local_features: Tensor
    ):
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
        # print(f"local feature: {local_features.shape}")

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

        if self.predict_normal and pc.normals_padded() is not None:
            normals = pc.normals_padded()

            # 检查 normals 是否归一化
            norm_lengths = normals.norm(dim=-1)  # 计算每个法向量的 L2 范数
            normals = normals / (norm_lengths.unsqueeze(-1) + 1e-8)  # 避免除以零

            return torch.cat((points, normals), dim=2)
        else:
            return points

    def tensor_to_point_cloud(self, x: Tensor, /, denormalize: bool = False, unscale: bool = False):
        points = x[:, :, :3] * (self.scale_factor if unscale else 1)

        self.predict_color = True

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
            images: Optional[Tensor],
            mask: Optional[Tensor],
            t: Optional[Tensor],
    ):
        """ Extracts local features from the input image and projects them onto the points
            in the point cloud to obtain the input to the model. Then extracts global
            features, replicates them across points, and concats them to the input."""
        B, N = x_t.shape[:2]

        # Initial input is the point locations (and colors if and only if predicting color)
        x_t_input = [x_t]

        image_tensor = torch.stack(images, dim=0)
        mask_tensor = torch.stack(mask, dim=0)

        image_rgb = image_tensor[:, 1:2, :, :].squeeze(1)
        mask = mask_tensor[:, 1:2, :, :].squeeze(1)
        camera = camera[1]

        # def visualize_and_save_images(image_rgb, mask, save_dir):
        #     """
        #     可视化并保存图像和对应的 mask。
        #     :param image_rgb: 输入的 RGB 图像张量，形状为 [B, H, W]。
        #     :param mask: 输入的 Mask 张量，形状为 [B, H, W]。
        #     :param save_dir: 保存图像的目录路径。
        #     """
        #     # 确保保存目录存在
        #     os.makedirs(save_dir, exist_ok=True)
        #
        #     # 将图像和 mask 保存
        #     to_pil = ToPILImage()
        #     for i in range(image_rgb.shape[0]):
        #         # 获取单张图像和 mask
        #         img = to_pil(image_rgb[i])
        #         mask_img = to_pil(mask[i])
        #
        #         # 创建一个新图形
        #         fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        #         axes[0].imshow(img)
        #         axes[0].set_title("Image")
        #         axes[0].axis("off")
        #
        #         axes[1].imshow(mask_img, cmap="gray")
        #         axes[1].set_title("Mask")
        #         axes[1].axis("off")
        #
        #         # 保存可视化图像
        #         save_path = os.path.join(save_dir, f"image_mask_{i}.png")
        #         plt.savefig(save_path, bbox_inches="tight")
        #         plt.close(fig)
        #
        #         print(f"Saved visualization to {save_path}")

        # visualize_and_save_images(image_rgb, mask, "/home/code/Buildiffusion")

        # Local conditioning
        if self.use_local_conditioning:

            # Get local features and check that they are the same size as the input image
            local_features = self.get_local_conditioning(image_rgb=image_rgb, mask=mask)
            if local_features.shape[-2:] != image_rgb.shape[-2:]:
                raise ValueError(f'{local_features.shape=} and {image_rgb.shape=}')

            # Project local features. Here that we only need the point locations, not colors
            local_features_proj = self.surface_projection(points=x_t[:, :, :3],
                                                          camera=camera,
                                                          local_features=local_features)  # (B, N, D_local)

            x_t_input.append(local_features_proj)

        # Global conditioning
        if self.use_global_conditioning:
            # Get and repeat global features
            global_features = self.get_global_conditioning(image_rgb=image_rgb)  # (B, D_global)
            global_features = global_features.unsqueeze(1).expand(-1, N, -1)  # (B, D_global, N)

            print(f"global_features: {global_features.shape}")

            x_t_input.append(global_features)

        # Concatenate together all the pointwise features
        x_t_input = torch.cat(x_t_input, dim=2)  # (B, N, D)

        return x_t_input

    # def get_input_with_cost_volume(
    #         self,
    #         x_t: torch.Tensor,
    #         camera: Optional[CamerasBase],
    #         image: Optional[torch.Tensor],
    #         mask: Optional[torch.Tensor],  # tuple B
    #         t: Optional[torch.Tensor],
    #         device: torch.device,  # 添加 device 参数
    # ):
    #     # 确定输入张量所在的设备
    #     device = x_t.device
    #     B, N = x_t.shape[:2]
    #
    #     # 堆叠成张量
    #     image_tensor = torch.stack(image, dim=0)
    #     mask_tensor = torch.stack(mask, dim=0)
    #
    #     model = self.mvs_cost_volume_model
    #
    #     outputs = build_cost_volume(B, image_tensor, mask_tensor, model, device=device)
    #     prob = outputs['prob_volume']
    #
    #     # 打印张量的最大值和最小值
    #     # print(f"Max value: {prob.max().item()}")
    #     # print(f"Min value: {prob.min().item()}")
    #     # print(f"Mean value: {prob.mean().item()}")
    #
    #     x_t_input = [x_t.to(device)]
    #
    #     camera = camera[1]
    #     cost_volume_proj = self.surface_projection(points=x_t[:, :, :3], camera=camera, local_features=prob).to(device)
    #     # print(f"Max value: {cost_volume_proj.max().item()}")
    #     # print(f"Min value: {cost_volume_proj.min().item()}")
    #     # print(f"Mean value: {cost_volume_proj.mean().item()}")
    #
    #     # # x.shape = [1, 50000, 32]
    #     # x_2d = cost_volume_proj[0].cpu().numpy()  # [50000, 32], 先转到 numpy
    #     # pca = PCA(n_components=2)
    #     # points_2d = pca.fit_transform(x_2d)  # [50000, 2]
    #     #
    #     # plt.figure(figsize=(6, 6))
    #     # plt.scatter(points_2d[:, 0], points_2d[:, 1], s=1)  # s=1: 点大小
    #     # plt.title("PCA to 2D")
    #     #
    #     # # 将图片保存到本地，文件名可自定义
    #     # plt.savefig("/home/code/Buildiffusion/pca_2d_scatter.png", dpi=300, bbox_inches="tight")
    #
    #     x_t_input.append(cost_volume_proj)
    #     x_t_input = torch.cat(x_t_input, dim=2)
    #
    #     return x_t_input

    # def cost_volume(
    #         self,
    #         x_t: torch.Tensor,
    #         camera: Optional[CamerasBase],
    #         image: Optional[torch.Tensor],
    #         model: MVSFormerWithDino,
    # ):
    #     # 确定输入张量所在的设备
    #     device = x_t.device
    #     B, N = x_t.shape[:2]
    #
    #     # 堆叠成张量
    #     image_tensor = torch.stack(image, dim=0)
    #
    #     outputs = build_cost_volume(B, image_tensor, device=device)
    #
    #     prob = outputs['prob_volume']
    #
    #     camera = camera[1]
    #     cost_volume_proj = self.surface_projection(points=x_t[:, :, :3], camera=camera, local_features=prob).to(device)
    #
    #     return cost_volume_proj

    # def get_input_with_bae(
    #         self,
    #         x_t: torch.Tensor,
    #         camera: Optional[CamerasBase],
    #         # image: Optional[torch.Tensor],  # tuple B
    #         mask: Optional[torch.Tensor],  # tuple B
    #         t: Optional[torch.Tensor],
    # ):
    #     device = x_t.device
    #     B, N = x_t.shape[:2]
    #
    #     # 初始化 mask 张量
    #     num_masks = 5  # 假设有 5 个 mask
    #     mask_list = []
    #     for i in range(num_masks):
    #         mask_tensor = torch.zeros((B, 1, 546, 966), device=device)
    #         for b, m in enumerate(mask):
    #             mask_tensor[b] = m[i].to(device)
    #         mask_list.append(mask_tensor)
    #
    #     # 特征投影
    #     x_t_input = [x_t.to(device)]
    #     x_t_feature = []
    #     x_t_bae = []
    #
    #     for c, m in zip(camera, mask_list):
    #         cdt = compute_distance_transform(m).to(device)
    #
    #         mask_proj = self.surface_projection(points=x_t[:, :, :3], camera=c, local_features=m).to(device)
    #         cdt_proj = self.surface_projection(points=x_t[:, :, :3], camera=c, local_features=cdt).to(device)
    #
    #         x_t_feature.append(mask_proj)
    #         x_t_bae.append(cdt_proj)
    #
    #     # 堆叠特征
    #     mask_proj_features = torch.stack(x_t_feature, dim=2).squeeze(-1)  # (B, N, num_masks, feature_dim)
    #     cdt_proj_features = torch.stack(x_t_bae, dim=2).squeeze(-1)  # (B, N, num_masks, feature_dim)
    #
    #     fused_features = self.mask_weight_fusion(mask_proj_features)  # (B, N, 2 * feature_dim)
    #     fused_features = fused_features.unsqueeze(-1)
    #
    #     # 拼接最终输入
    #     x_t_input.append(fused_features)
    #     x_t_input.append(mask_proj_features[:, :, 0:1])
    #     x_t_input.append(cdt_proj_features[:, :, 0:1])
    #
    #     x_t_input = torch.cat(x_t_input, dim=2)  # (B, N, feature_dim + 2 * feature_dim)
    #
    #     return x_t_input

    def forward(self, batch: FrameData, mode: str = 'train', **kwargs):
        """ The forward method may be defined differently for different models. """
        raise NotImplementedError()