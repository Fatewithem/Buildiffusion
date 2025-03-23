import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from omegaconf import DictConfig
from tqdm import tqdm
from einops import repeat
import open3d as o3d
import torchvision.transforms as T
import os

from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union, Any
from cost_volume.cost_volume import DinoFeatureExtractor
from model.transformer.pointnet import PointNet2FeatureExtractorWithFP

from pytorch3d.renderer import PointsRasterizationSettings, PointsRenderer, AlphaCompositor
from pytorch3d.renderer.points import PointsRasterizer

from model.projection_model import PointCloudProjectionModel
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Pointclouds


class ImageSupervisedPointCloudModel(PointCloudProjectionModel):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

        self.pointcloud_extractor = PointNet2FeatureExtractorWithFP()

        self.raster_settings = PointsRasterizationSettings(
            image_size=(546, 966),  # 或来自 config
            radius=0.01,
            points_per_pixel=20
        )
        self.compositor = AlphaCompositor(background_color=(1.0, 1.0, 1.0))

        self.initial_mlp = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # xyz + rgb
        )
        self.refine_mlp = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )

    def forward(self, pc: Pointclouds,
                masks: Optional[Tensor],
                camera: Optional[Dict],
                images: Optional[Tensor],
               ):
        x = self.point_cloud_to_tensor(pc, normalize=True, scale=True)
        B, N, D = x.shape

        # 进行特征提取
        pointcloud_features = self.pointcloud_extractor(x[:, :, :3])  # (batch_size, pointcloud_feature_dim)
        pointcloud_features = pointcloud_features.permute(0, 2, 1)  # [0, 10]

        image_tensor = torch.stack(images, dim=0)

        # 提取每个 batch 中 index=1 的图像
        image_1_tensor = image_tensor[:, 1]  # shape: (B, 3, H, W)
        image_1_tensor = image_1_tensor.permute(0, 2, 3, 1)  # to (B, H, W, 3)

        # 构建 mask：原始 mask 为黑色前景（0），白色背景（1）
        if masks is not None:
            mask_tensor = torch.stack(masks, dim=0)[:, 1]  # (B, 1, H, W) from batch index 1
            mask_tensor = mask_tensor.squeeze(1).unsqueeze(-1)  # (B, H, W, 1)
            mask_tensor = (mask_tensor < 0.5).float()  # 黑色前景为1，白色背景为0

        loss_total = 0.0

        delta = self.initial_mlp(pointcloud_features).float()
        delta_xyz, rgb1 = delta[..., :3], torch.sigmoid(delta[..., 3:])

        x_0 = self.point_cloud_to_tensor(pc, normalize=True, scale=False)

        xyz1 = x_0[:, :, :3] + delta_xyz
        pc1 = Pointclouds(
            points=[xyz1[i].float() for i in range(B)],
            features=[rgb1[i].float() for i in range(B)]
        )
        img1 = self.render_pointcloud_features(pc1, camera[1]).clamp(0, 1)

        if masks is not None:
            loss_total += F.mse_loss(img1 * mask_tensor, image_1_tensor * mask_tensor)
        else:
            loss_total += F.mse_loss(img1, image_1_tensor)

        # Convert tensor to PIL image and save
        to_pil = T.ToPILImage()
        img_to_save = img1[0].permute(2, 0, 1).cpu()

        output_path = os.path.join("/home/code/Blender/color", "rendered_output.png")
        to_pil(img_to_save).save(output_path)

        gt_to_save = image_1_tensor[0].permute(2, 0, 1).cpu()
        gt_output_path = os.path.join("/home/code/Blender/color", "rendered_gt.png")
        to_pil(gt_to_save).save(gt_output_path)

        # print(f"Rendered image saved to {output_path}")

        return loss_total

    def forward_sample(self, pc: Pointclouds,
                       masks: Optional[Tensor],
                       camera: Optional[Dict],
                       images: Optional[Tensor],
                      ):
        x = self.point_cloud_to_tensor(pc, normalize=True, scale=True)
        B, N, D = x.shape

        # 提取点云特征
        pointcloud_features = self.pointcloud_extractor(x[:, :, :3])
        pointcloud_features = pointcloud_features.permute(0, 2, 1)

        delta = self.initial_mlp(pointcloud_features).float()
        delta_xyz, rgb1 = delta[..., :3], torch.sigmoid(delta[..., 3:])

        x_0 = self.point_cloud_to_tensor(pc, normalize=True, scale=False)
        xyz1 = x_0[:, :, :3] + delta_xyz

        # 构建新的点云
        pc1 = Pointclouds(
            points=[xyz1[i].float() for i in range(B)],
            features=[rgb1[i].float() for i in range(B)]
        )

        # 渲染并返回图像与点云
        img = self.render_pointcloud_features(pc1, camera[1]).clamp(0, 1)
        return pc1, img, pc

    def render_pointcloud_features(self, pointcloud: Pointclouds, cameras: CamerasBase):
        if pointcloud.features_padded().shape[-1] == 0:
            N, device = pointcloud.num_points_per_cloud()[0].item(), pointcloud.device
            gray = torch.ones((1, N, 3), device=device) * 0.5
            pointcloud = Pointclouds(points=pointcloud.points_padded(), features=gray)

        # ensure float32 dtype
        pointcloud = Pointclouds(
            points=[p.to(torch.float32) for p in pointcloud.points_list()],
            features=[f.to(torch.float32) for f in pointcloud.features_list()]
        )

        cameras = cameras.clone()

        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=self.raster_settings)
        renderer = PointsRenderer(rasterizer=rasterizer, compositor=self.compositor)
        rendered = renderer(pointcloud)

        return rendered