import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from omegaconf import DictConfig
from tqdm import tqdm
from einops import repeat
import open3d as o3d

from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union, Any
from cost_volume.cost_volume import DinoFeatureExtractor
from .pointnet import PointNet2FeatureExtractorWithFP
from .transformer import Transformer
from .output import OutputLayer

from model.projection_model import PointCloudProjectionModel
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Pointclouds
from .utils import positionalencoding1d, convert_to_tensor
from .loss import compute_loss

from pytorch3d.loss import chamfer_distance


# 综合模型框架
class Co3dQueryModel(PointCloudProjectionModel):
    def __init__(
            self,
            config: DictConfig,
            **kwargs
    ):
        # 调用父类构造函数
        super(Co3dQueryModel, self).__init__(**kwargs)

        # 特征提取网络
        self.pointcloud_extractor = PointNet2FeatureExtractorWithFP()
        self.dino_extractor = DinoFeatureExtractor(model_path="/home/code/Buildiffusion/cost_volume/dinov2_base")
        self.img_proj = nn.Linear(768, 256)

        # 查询
        self.num_queries = 10
        self.d_model = 256

        # 读取transformer参数
        self.cfg_transformer = config.network.transformer
        self.cfg_transformer.d_model = 256

        # 初始化
        self.learned_queries = nn.Parameter(torch.rand((self.num_queries, self.d_model)))
        self.transformer = Transformer(**self.cfg_transformer)
        self.output_layer = OutputLayer(self.d_model, num_queries=self.num_queries, points_per_query=1000) # 500
        self.camera_embed = nn.Linear(12, self.d_model)

        self.pc_mlp = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

    def forward(
            self,
            pc: Pointclouds,
            masks: Optional[Tensor],
            camera: Optional[Dict],
            images: Optional[Tensor],  # [B, C, H, W]
            planes: Optional[Dict],
    ):
        # ------------ PRE-TRANSFORMER PART ------------
        # 转换点云
        x = self.point_cloud_to_tensor(pc, normalize=True, scale=False)  # 用于特征提取
        B, N, D = x.shape  # Batch; Num; Dimen

        if masks is not None:
            image_1_tensor = images * masks  # masks shape: (B, 1, H, W)，自动广播
        else:
            image_1_tensor = images

        # 进行特征提取
        # 点云特征
        pointcloud_features = self.pointcloud_extractor(x[:, :, :3].contiguous())  # (B, N, 254)
        pointcloud_features = pointcloud_features.permute(0, 2, 1)
        pc_feat = self.pc_mlp(pointcloud_features)  # [B, N, 768]

        # 图像特征
        image_features = self.dino_extractor(image_1_tensor)  # [B, Features, H_tokens, W_tokens]

        B, N, C = pc_feat.shape
        _, C_img, H, W = image_features.shape
        img_feat = image_features.flatten(2).permute(0, 2, 1)  # (B, H*W, 768)
        img_feat = self.img_proj(img_feat)
        # Downsample img_feat to 2048 tokens
        # img_feat = F.adaptive_avg_pool1d(img_feat.transpose(1, 2), output_size=2048).transpose(1, 2)

        # 相机位姿特征
        R = camera.R  # (B, 3, 3)
        T = camera.T  # (B, 3)

        pose = torch.cat([R.view(B, -1), T], dim=-1)  # (B, 12)
        pose_embedding = self.camera_embed(pose)  # (B, 13)

        # Prepare query with pose embedding added
        # query = pc_feat + pose_embedding.unsqueeze(1)  # broadcast addition

        # 准备decoder第一层的输入
        tgt = self.prepare_learned_queries(self.learned_queries, B, pc_feat)
        decoder_attn_mask = self.get_decoder_mask()

        # ------------ TRANSFORMER PART ------------

        prediction = self.transformer(trg=tgt, pc_feat=pc_feat, img_feat=img_feat, mod=pose_embedding, trg_mask=decoder_attn_mask)

        points, _ = self.output_layer(prediction)
        points = points.contiguous()

        planes = self.point_cloud_to_tensor(planes, normalize=True, scale=False)  # (-1, 1)

        # Normalize GT planes
        planes = planes

        # Normalize predicte
        # d points and centers to same scale
        points = points
        planes = planes[..., :3].contiguous()

        final_loss, cd_loss, emd_loss, repulsion_loss = compute_loss(points, planes)

        # 渲染输出点云为图像用于计算2D投影loss
        # rendered_image = self.render_pointcloud(output, camera, images)  # 自定义函数，需要你实现
        # masks_rgb = masks.expand(-1, 3, -1, -1)  # [1, 3, 800, 800]
        # projected_loss = F.mse_loss(rendered_image, masks_rgb)  # 示例使用MSE loss，可换为其他

        # final_loss += projected_loss * 0.1 # 合并loss

        return final_loss, cd_loss, emd_loss, repulsion_loss  # (points, centers)

    @torch.no_grad()
    def forward_sample(
            self,
            pc: Pointclouds,
            camera: Optional[Dict],
            images: Optional[Tensor],
            planes: Optional[Tensor],
            masks: Optional[Tensor],
    ):
        # 和forward保持一致
        x = self.point_cloud_to_tensor(pc, normalize=True, scale=False)  # (-1, 1)
        B, N, D = x.shape

        image_1_tensor = images * masks

        # 特征提取
        pointcloud_features = self.pointcloud_extractor(x[:, :, :3].contiguous())
        pointcloud_features = pointcloud_features.permute(0, 2, 1)
        pc_feat = self.pc_mlp(pointcloud_features)

        image_features = self.dino_extractor(image_1_tensor)
        img_feat = image_features.flatten(2).permute(0, 2, 1)
        img_feat = self.img_proj(img_feat)
        img_feat = F.adaptive_avg_pool1d(img_feat.transpose(1, 2), output_size=2048).transpose(1, 2)

        R = camera.R  # 注意，不是camera[1]，是整个batch
        T = camera.T
        pose = torch.cat([R.view(B, -1), T], dim=-1)
        pose_embedding = self.camera_embed(pose)

        # decoder输入
        tgt = self.prepare_learned_queries(self.learned_queries, B, pc_feat)
        decoder_attn_mask = self.get_decoder_mask()

        prediction = self.transformer(trg=tgt, pc_feat=pc_feat, img_feat=img_feat, mod=pose_embedding,
                                      trg_mask=decoder_attn_mask)

        points, _ = self.output_layer(prediction)
        points = points.contiguous()

        result = self.tensor_to_point_cloud(points, denormalize=False, unscale=False)
        return result, points, planes, camera, images.squeeze(0)

    def prepare_learned_queries(self, learned_queries: torch.Tensor, batch_size: int, pc_feat: torch.Tensor) -> torch.Tensor:
        # Add positional encoding
        # pos_enc.shape => same as `learned_queries` | [num_queries (12), d_model (768)]
        pos_enc = positionalencoding1d(self.d_model, self.num_queries).to(self.device)
        learned_queries = learned_queries + pos_enc  # [nq, d_model]
        learned_queries = repeat(learned_queries, 'nq dmodel -> b nq dmodel', b=batch_size)  # [B, nq, d_model]

        # 点云全局摘要特征，作为每个 query 的几何感知输入
        pc_summary = pc_feat.mean(dim=1, keepdim=True).expand(-1, self.num_queries, -1)  # [B, nq, d_model]

        fused_query = learned_queries + pc_summary
        return fused_query

    def get_decoder_mask(self):
        """
            Generate decoder-side attention mask
        :return: Attention mask
        """
        # Create boolean identity matrix
        tgt_mask = torch.ones((self.num_queries, self.num_queries), dtype=torch.bool, device=self.device)
        # Select diagonal entries
        tgt_mask_diag = torch.diagonal(tgt_mask)
        # Replace diagonal entries with False
        tgt_mask_diag[:] = False
        # Replace diagonals with -inf and everything else with 0
        tgt_mask = tgt_mask.float()
        tgt_mask = tgt_mask.masked_fill(tgt_mask == 0, float('-inf'))
        tgt_mask = tgt_mask.masked_fill(tgt_mask == 1, float(0.0))
        return tgt_mask

    def render_pointcloud(self, pointcloud_tensor: Tensor, camera: Dict, images: Tensor) -> Tensor:
        """
        将输出的点云进行渲染为图像，用于2D投影loss计算。
        这里只是一个模板，需要根据你的渲染逻辑完成。
        """
        # 示例：使用pytorch3d进行可微渲染
        from pytorch3d.renderer import (
            FoVPerspectiveCameras, PointsRasterizationSettings,
            PointsRenderer, AlphaCompositor, PointsRasterizer,
            NormWeightedCompositor
        )
        from pytorch3d.structures import Pointclouds
        import torchvision
        import os

        cameras = camera.to(pointcloud_tensor.device)
        point_clouds = Pointclouds(points=pointcloud_tensor, features=torch.ones_like(pointcloud_tensor))

        raster_settings = PointsRasterizationSettings(
            image_size=images.shape[-1],
            radius=0.02,
            points_per_pixel=10
        )

        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        renderer = PointsRenderer(rasterizer=rasterizer, compositor=AlphaCompositor())

        rendered = renderer(point_clouds)

        # img_to_save = rendered[0].clamp(0.0, 1.0).cpu()
        # save_path = "/home/code/Buildiffusion/color/image.png"
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # torchvision.utils.save_image(img_to_save.permute(2, 0, 1), save_path)

        return rendered.permute(0, 3, 1, 2)
