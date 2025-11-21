import torch
import pytorch_lightning as pl
import torch.nn as nn
from pytorch3d.ops import sample_farthest_points


class OutputLayer(pl.LightningModule):
    """
    Output Layer of the LegoFormer.
    Maps Transformer's output vector to decomposition factors and generates 3D point clouds.
    """

    def __init__(self, dim_model, num_queries=15, points_per_query=100):
        super().__init__()

        self.points_per_query = points_per_query
        self.num_queries = num_queries
        self.total_points = num_queries * points_per_query

        self.mlp = nn.Sequential(
            nn.Linear(dim_model, 512),  # Add 3D Positional Encoding
            nn.LeakyReLU(0.1),
            nn.Linear(512, points_per_query * 3)
        )

        # self.param_head = nn.Sequential(
        #     nn.Linear(dim_model, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 7)  # center(3) + normal(3) + extent(1)
        # )

        # self.class_head = nn.Linear(dim_model, 3)  # 假设3类几何结构

    def forward(self, x):
        """
        x: [B, num_queries, dim_model]
        return: [B, num_queries * points_per_query, 3], [B, num_queries, 3]
        """
        B, nq, dmodel = x.shape  # [B, num_queries, dim_model]
        x = x.view(B * nq, dmodel)  # [B * nq, dim_model]
        points = self.mlp(x)  # [B * nq, points_per_query * 3]
        points = points.view(B, nq * self.points_per_query, 3)  # [B, total_points, 3]
        return points, None


    def forward_split_by_query(self, x):
        """
        x: [B, num_queries, dim_model]
        return: [B, num_queries, points_per_query, 3], [B, num_queries, 3] - for per-query visualization
        """
        B, nq, dmodel = x.shape
        x = x.view(B * nq, dmodel)
        points = self.mlp(x)
        points = points.view(B, nq, self.points_per_query, 3)
        return points, None


# 返回包含颜色信息
# class OutputLayerColor(pl.LightningModule):
#     """
#     Output Layer of the LegoFormer.
#     Maps Transformer's output vector to decomposition factors and generates 3D point clouds.
#     现在改为输出 6 维：前 3 维为坐标 (x, y, z)，后 3 维为颜色 (r, g, b)。
#     """
#
#     def __init__(self, dim_model, num_queries=10, max_points=9000):
#         super().__init__()
#
#         self.max_points = max_points  # 设置最大点数作为所有点云的上限
#
#         # 注意，这里将输出维度从 3 * max_points 改为 6 * max_points
#         self.mlp = nn.Sequential(
#             nn.Linear(dim_model, 512),
#             nn.LeakyReLU(0.1),
#             nn.Linear(512, max_points * 6)  # 每个点 6 维：xyz + rgb
#         )
#
#     def forward(self, x):
#         """
#         :param x: 输入张量，形状为 [B, C]
#         :return: Tensor, 形状为 (B, N, 6)，前 3 维为坐标，后 3 维为颜色
#         """
#         points = self.mlp(x)  # [B, max_points * 6]
#         points = points.view(-1, self.max_points, 6)  # [B, N, 6]
#         return points