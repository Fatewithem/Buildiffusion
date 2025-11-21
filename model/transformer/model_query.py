import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from omegaconf import DictConfig
from tqdm import tqdm
from einops import repeat
import open3d as o3d

from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union, Any
from cost_volume.cost_volume import DinoFeatureExtractorUrbanBIS
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
class MultiModalQueryModel(PointCloudProjectionModel):
    def __init__(
            self,
            config: DictConfig,
            **kwargs
    ):
        # 调用父类构造函数
        super(MultiModalQueryModel, self).__init__(**kwargs)

        # 特征提取网络
        self.pointcloud_extractor = PointNet2FeatureExtractorWithFP()
        self.dino_extractor = DinoFeatureExtractorUrbanBIS(model_path="/home/code/Buildiffusion/cost_volume/dinov2_base")

        # 查询
        self.num_queries = 10
        self.d_model = 768

        # 读取transformer参数
        self.cfg_transformer = config.network.transformer

        # 初始化
        self.learned_queries = nn.Parameter(torch.rand((self.num_queries, self.d_model)))
        self.transformer = Transformer(**self.cfg_transformer)
        self.output_layer = OutputLayer(self.d_model, num_queries=self.num_queries, points_per_query=1000)
        self.camera_embed = nn.Linear(12, self.d_model)

        self.pc_mlp = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 768)
        )

    def forward(
            self,
            pc: Pointclouds,
            masks: Optional[Tensor],
            camera: Optional[Dict],
            images: Optional[Tensor],
            planes: Optional[Dict],
    ):
        # ------------ PRE-TRANSFORMER PART ------------

        # 转换点云
        x = self.point_cloud_to_tensor(pc, normalize=True, scale=True)  # (-1, 1)
        B, N, D = x.shape  # Batch; Num; Dimen

        image_tensor = torch.stack(images, dim=0)
        image_1_tensor = image_tensor[:, 1, :, :, :]  # shape: [B, H, W, C]
        image_1_tensor = image_1_tensor.unsqueeze(1)  # shape: [B, 1, H, W, C]
        image_1_tensor = image_1_tensor.permute(0, 1, 4, 2, 3).squeeze(1)  # convert to [B, C, H, W]

        # 进行特征提取
        # 点云特征
        pointcloud_features = self.pointcloud_extractor(x[:, :, :3])  # (B, N, 254)
        pointcloud_features = pointcloud_features.permute(0, 2, 1)
        pc_feat = self.pc_mlp(pointcloud_features)  # [B, N, 768]

        # 图像特征
        image_features = self.dino_extractor(image_1_tensor)  # [B, Features, H_tokens, W_tokens]
        # project_features = self.surface_projection(points=x[:, :, :3],
        #                                            camera=camera[1],
        #                                            local_features=image_features)

        B, N, C = pc_feat.shape
        _, C_img, H, W = image_features.shape
        img_feat = image_features.flatten(2).permute(0, 2, 1)  # (B, H*W, 768)
        img_feat = F.adaptive_avg_pool1d(img_feat.transpose(1, 2), output_size=2048).transpose(1, 2)

        # 相机位姿特征
        R = camera[1].R  # (B, 3, 3)
        T = camera[1].T  # (B, 3)

        R = R.expand(B, -1, -1)  # [B, 3, 3]
        T = T.expand(B, -1)  # [B, 3]

        pose = torch.cat([R.view(B, -1), T], dim=-1)  # (B, 12)
        pose_embedding = self.camera_embed(pose)  # (B, 13)

        # 准备decoder第一层的输入
        tgt = self.prepare_learned_queries(self.learned_queries, B, pc_feat)
        decoder_attn_mask = self.get_decoder_mask()

        # ------------ TRANSFORMER PART ------------
        prediction = self.transformer(trg=tgt, pc_feat=pc_feat, img_feat=img_feat, mod=pose_embedding, trg_mask=decoder_attn_mask)

        # print(
        #     f"Prediction Min: {image_features.min().item()}, Max: {image_features.max().item()}, Mean: {image_features.mean().item()}")

        # 输出xyz
        output = self.output_layer(prediction)

        planes = torch.stack(planes, dim=0)  # [-1, 1]

        # print("output shape:", output.shape)
        # print("planes shape:", planes.shape)

        final_loss, cd_loss, emd_loss, repulsion_loss = compute_loss(output, planes)

        return final_loss, cd_loss, emd_loss, repulsion_loss

    @torch.no_grad()
    def forward_sample(
            self,
            pc: Pointclouds,
            camera: Optional[Dict],
            images: Optional[Tensor],
            planes: Optional[Tensor],
                       ):
        # 转换点云
        x = self.point_cloud_to_tensor(pc, normalize=True, scale=True)  # (-1, 1)
        B, N, D = x.shape  # Batch; Num; Dimen

        image_tensor = torch.stack(images, dim=0)
        image_1_tensor = image_tensor[:, 1, :, :, :]  # shape: [B, H, W, C]
        image_1_tensor = image_1_tensor.unsqueeze(1)  # shape: [B, 1, H, W, C]
        image_1_tensor = image_1_tensor.permute(0, 1, 4, 2, 3).squeeze(1)  # convert to [B, C, H, W]

        # 进行特征提取
        pointcloud_features = self.pointcloud_extractor(x[:, :, :3])  # (batch_size, pointcloud_feature_dim)
        pointcloud_features = pointcloud_features.permute(0, 2, 1)  # [0, 10]
        pc_feat = self.pc_mlp(pointcloud_features)  # [B, N, 768]

        image_features = self.dino_extractor(image_1_tensor)
        img_feat = image_features.flatten(2).permute(0, 2, 1)
        img_feat = F.adaptive_avg_pool1d(img_feat.transpose(1, 2), output_size=2048).transpose(1, 2)

        # 准备decoder第一层的输入
        tgt = self.prepare_learned_queries(self.learned_queries, B, pc_feat)

        R = camera[1].R  # (B, 3, 3)
        T = camera[1].T  # (B, 3)
        pose = torch.cat([R.view(B, -1), T], dim=-1)  # (B, 12)
        pose_embedding = self.camera_embed(pose)  # (B, 13)
        # combined_features = torch.cat([cond_feat, pose_embedding.unsqueeze(1).expand(-1, cond_feat.shape[1], -1)], dim=2)

        decoder_attn_mask = self.get_decoder_mask()

        # ------------ TRANSFORMER PART ------------

        prediction = self.transformer(trg=tgt, pc_feat=pc_feat, img_feat=img_feat, mod=pose_embedding, trg_mask=decoder_attn_mask)


        output = self.output_layer(prediction)  # [B, 9000, 6]

        # print(
        #     f"Prediction Min: {output[0][0].min().item()}, Max: {output[0][0].max().item()}, Mean: {output[0][0].mean().item()}")

        # output = convert_to_tensor(output)

        result = self.tensor_to_point_cloud(output, denormalize=False, unscale=False)

        planes = torch.stack(planes, dim=0)

        planes_colors = self.tensor_to_point_cloud(planes, denormalize=False, unscale=False)

        return result, planes_colors

    def prepare_learned_queries(self, learned_queries: torch.Tensor, batch_size: int,
                                pc_feat: torch.Tensor) -> torch.Tensor:
        """
            Prepare the learned decomposition factor queries for the decoder
            Adds sine-cos positional encoding and replicates single-set of queries among the batch dimension
            Fuses coarse point cloud features as global summary to each query
        :param learned_queries: Single set of learned queries, shape: [num_queries (12), d_model (768)]
        :param batch_size: Size of the batch
        :param pc_feat: Point cloud features, shape: [B, N, d_model]
        :return: Learned decomposition factor queries fused with pc global summary
        """

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
