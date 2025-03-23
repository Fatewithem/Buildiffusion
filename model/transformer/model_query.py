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
from .output import OutputLayer, OutputLayerColor

from model.projection_model import PointCloudProjectionModel
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Pointclouds
from .utils import positionalencoding1d, convert_to_tensor
from .loss import compute_loss, compute_color_loss

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
        self.dino_extractor = DinoFeatureExtractor(model_path="/home/code/Buildiffusion/cost_volume/dinov2_base")

        # 查询
        self.num_queries = 10
        self.d_model = 768 + 64

        # 读取transformer参数
        self.cfg_transformer = config.network.transformer

        # 初始化
        self.learned_queries = nn.Parameter(torch.rand((self.num_queries, self.d_model)))
        self.transformer = Transformer(**self.cfg_transformer)
        self.output_layer = OutputLayer(self.d_model, num_queries=self.num_queries, max_points=2000)

        # self.output_layer = OutputLayerColor(self.d_model, num_queries=self.num_queries, max_points=2000)

        # **定义每个查询的目标点数（固定模式）**
        self.target_points_per_query = torch.tensor(
            [2000, 2000, 1000, 1000, 1000, 500, 500, 500, 250, 250], dtype=torch.long
        )

    def forward(
            self,
            pc: Pointclouds,
            masks: Optional[Tensor],
            camera: Optional[Dict],
            images: Optional[Tensor],
            planes: Optional[Dict],
            colors: Optional[Tensor],
            plane_infos: Optional[Tensor],
    ):
        # ------------ PRE-TRANSFORMER PART ------------

        # 转换点云
        x = self.point_cloud_to_tensor(pc, normalize=True, scale=True)  # (-1, 1)
        B, N, D = x.shape  # Batch; Num; Dimen

        image_tensor = torch.stack(images, dim=0)
        image_1_tensor = image_tensor[0, 1:2, :, :, :]

        # 进行特征提取
        pointcloud_features = self.pointcloud_extractor(x[:, :, :3])  # (batch_size, pointcloud_feature_dim)
        pointcloud_features = pointcloud_features.permute(0, 2, 1)  # [0, 10]

        image_features = self.dino_extractor(image_1_tensor)  # (batch_size, image_feature_dim)
        project_features = self.surface_projection(points=x[:, :, :3],
                                                   camera=camera[1],
                                                   local_features=image_features)

        combined_features = torch.cat([pointcloud_features, project_features],
                                      dim=2)  # (batch_size, N，pointcloud_feature_dim + image_feature_dim)

        # 准备decoder第一层的输入
        tgt = self.prepare_learned_queries(self.learned_queries, B)
        decoder_attn_mask = self.get_decoder_mask()

        # ------------ TRANSFORMER PART ------------

        input = combined_features.permute((1, 0, 2))  # [-22, 20]

        prediction = self.transformer(input, tgt, decoder_attn_mask)
        prediction = prediction.permute((1, 0, 2))  # [B, N_query, C] [-3, 3]

        # print(
        #     f"Prediction Min: {image_features.min().item()}, Max: {image_features.max().item()}, Mean: {image_features.mean().item()}")

        # **创建目标点数列表**
        target_num_points_list = self.target_points_per_query.unsqueeze(0).repeat(B, 1)

        # 只输出xyz
        # output = self.output_layer(prediction, target_num_points_list)

        # 输出xyz + color
        output = self.output_layer(prediction, target_num_points_list)

        planes = torch.stack(planes, dim=0) / 10.0  # [-1, 1]
        # colors = torch.stack(colors, dim=0)  # [0,1]

        # plane_infos = torch.stack(plane_infos, dim=0)

        loss, total_loss, global_loss, emd_loss = compute_loss(output, planes, self.target_points_per_query)
        # loss, total_loss, global_loss, emd_loss, color_loss = compute_color_loss(output, planes, colors, self.target_points_per_query)

        # return loss, total_loss, global_loss, emd_loss, color_loss
        return loss, total_loss, global_loss, emd_loss

    @torch.no_grad()
    def forward_sample(
            self,
            pc: Pointclouds,
            camera: Optional[Dict],
            images: Optional[Tensor],
            planes: Optional[Tensor],
            colors: Optional[Tensor],
                       ):
        # 转换点云
        x = self.point_cloud_to_tensor(pc, normalize=True, scale=True)  # (-1, 1)
        B, N, D = x.shape  # Batch; Num; Dimen

        image_tensor = torch.stack(images, dim=0)
        image_1_tensor = image_tensor[0, 1:2, :, :, :]

        # 进行特征提取
        pointcloud_features = self.pointcloud_extractor(x[:, :, :3])  # (batch_size, pointcloud_feature_dim)
        pointcloud_features = pointcloud_features.permute(0, 2, 1)  # [0, 10]

        image_features = self.dino_extractor(image_1_tensor)  # (batch_size, image_feature_dim)
        project_features = self.surface_projection(points=x[:, :, :3],
                                                   camera=camera[1],
                                                   local_features=image_features)

        combined_features = torch.cat([pointcloud_features, project_features],
                                      dim=2)  # (batch_size, N，pointcloud_feature_dim + image_feature_dim)

        # 准备decoder第一层的输入
        tgt = self.prepare_learned_queries(self.learned_queries, B)
        decoder_attn_mask = self.get_decoder_mask()

        # ------------ TRANSFORMER PART ------------

        input = combined_features.permute((1, 0, 2))  # [-22, 20]

        prediction = self.transformer(input, tgt, decoder_attn_mask)
        prediction = prediction.permute((1, 0, 2))  # [B, N_query, C] [-3, 3]

        # **创建目标点数列表**
        target_num_points_list = self.target_points_per_query.unsqueeze(0).repeat(B, 1)

        output = self.output_layer(prediction, target_num_points_list)  # [B, 9000, 6]

        # print(
        #     f"Prediction Min: {output[0][0].min().item()}, Max: {output[0][0].max().item()}, Mean: {output[0][0].mean().item()}")

        output = convert_to_tensor(output)

        result = self.tensor_to_point_cloud(output, denormalize=False, unscale=True)

        planes = torch.stack(planes, dim=0)
        colors = torch.stack(colors, dim=0)

        planes_colors = torch.cat((planes, colors), dim=-1)

        # print(output.shape)
        #
        # print(
        #     f"Prediction Min: {output.min().item()}, Max: {output.max().item()}, Mean: {output.mean().item()}")

        planes_colors = self.tensor_to_point_cloud(planes_colors, denormalize=False, unscale=False)

        return result, output, planes_colors

    def prepare_learned_queries(self, learned_queries: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
            Prepare the learned decomposition factor queries for the decoder
            Adds sine-cos positional encoding and replicates single-set of queries among the batch dimension
        :param learned_queries: Single set of learned queries, shape: [num_queries (12), d_model (768)]
        :param batch_size: Size of the batch
        :return: Learned decomposition factor queries
        """

        # Add positional encoding
        # pos_enc.shape => same as `learned_queries` | [num_queries (12), d_model (768)]
        pos_enc = positionalencoding1d(self.d_model, self.num_queries).to(self.device)
        learned_queries = learned_queries + pos_enc

        # Expand (replicate) among the batch dimension
        learned_queries = repeat(learned_queries, 'nq dmodel -> nq b dmodel', b=batch_size)
        # learned_queries.shape => [num_queries (12), B_SIZE, d_model (768)]

        return learned_queries

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
