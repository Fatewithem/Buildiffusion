# import torch
# import pytorch_lightning as pl
# import torch.nn as nn
# from pytorch3d.ops import sample_farthest_points
#
# class OutputLayer(pl.LightningModule):
#     """
#     Output Layer of the LegoFormer.
#     Maps Transformer's output vector to decomposition factors and generates 3D point clouds.
#     """
#
#     def __init__(self, dim_model, num_queries=10, max_points=2000):
#         super().__init__()
#
#         self.num_queries = num_queries
#         self.max_points = max_points  # 设置最大点数作为所有点云的上限
#
#         # 定义MLP，将每个查询映射为 N * 3 坐标（3D点）
#         self.mlp = nn.Sequential(
#             nn.Linear(dim_model, 512),
#             nn.LeakyReLU(0.1),  # 允许负数通过
#             nn.Linear(512, max_points * 3)
#         )
#
#     def forward(self, x, target_num_points_list):
#         """
#         通过MLP生成点云，并根据每个查询的目标点数调整点云的数量
#         :param x: 输入张量，形状为 [B, num_queries, C]
#         :param target_num_points_list: 每个查询的目标点数列表，形状为 [B, num_queries]
#         :return: 生成的3D点云，形状为 [B, num_queries, N, 3]
#         """
#         batch_size, num_queries, _ = x.shape
#
#         # 通过MLP生成每个查询的点云坐标
#         points = self.mlp(x)  # 形状: [B, num_queries, max_points * 3]
#
#         # 将输出重塑为 [B, num_queries, max_points, 3]
#         points = points.view(batch_size, num_queries, self.max_points, 3)
#
#         # 进行最远点采样 (FPS)
#         sampled_points = []
#         for b in range(batch_size):
#             batch_selected_points = []
#             for q in range(num_queries):
#                 num_points = target_num_points_list[b, q].item()  # 获取当前查询的目标点数
#
#                 if num_points > 0:
#                     # 进行最远点采样
#                     sampled_pts, _ = sample_farthest_points(points[b, q].unsqueeze(0), K=num_points)
#                     batch_selected_points.append(sampled_pts.squeeze(0))  # 移除batch维度
#                 else:
#                     # 如果目标点数为 0，返回空点云
#                     batch_selected_points.append(torch.zeros(0, 3, device=x.device))
#
#             # 删除空点云并合并
#             batch_selected_points = [p for p in batch_selected_points if p.numel() > 0]  # 过滤掉空点云
#
#             sampled_points.append(torch.cat(batch_selected_points, dim=0))  # 按不同大小拼接
#
#         return torch.stack(sampled_points, dim=0)  # 返回一个包含所有批次点云的张量”


import torch
import pytorch_lightning as pl
import torch.nn as nn
from pytorch3d.ops import sample_farthest_points


class OutputLayer(pl.LightningModule):
    """
    Output Layer of the LegoFormer.
    Maps Transformer's output vector to decomposition factors and generates 3D point clouds.
    """

    def __init__(self, dim_model, num_queries=10, max_points=2000):
        super().__init__()

        self.num_queries = num_queries
        self.max_points = max_points  # 设置最大点数作为所有点云的上限

        # 定义MLP，将每个查询映射为 N * 3 坐标（3D点）
        self.mlp = nn.Sequential(
            nn.Linear(dim_model, 512),
            nn.LeakyReLU(0.1),  # 允许负数通过
            nn.Linear(512, max_points * 3)
        )

    def forward(self, x, target_num_points_list):
        """
        通过MLP生成点云，并根据每个查询的目标点数调整点云的数量
        :param x: 输入张量，形状为 [B, num_queries, C]
        :param target_num_points_list: 每个查询的目标点数列表，形状为 [B, num_queries]
        :return: **List of Tensors**, 其中每个 `output[b][q]` 形状为 `(N, 3)`
        """
        batch_size, num_queries, _ = x.shape

        # 通过MLP生成每个查询的点云坐标
        points = self.mlp(x)  # 形状: [B, num_queries, max_points * 3]
        points = points.view(batch_size, num_queries, self.max_points, 3)  # 变形 [B, num_queries, max_points, 3]

        # 采样后存储点云
        output = []  # 存储 batch 内的所有点云
        for b in range(batch_size):
            batch_selected_points = []  # 存储当前 batch 内所有 query 生成的点云

            for q in range(num_queries):
                num_points = target_num_points_list[b, q].item()  # 获取当前查询的目标点数

                if num_points > 0:
                    # 进行最远点采样 (FPS)
                    sampled_pts, _ = sample_farthest_points(points[b, q].unsqueeze(0), K=num_points)
                    batch_selected_points.append(sampled_pts.squeeze(0))  # 形状: (num_points, 3)
                else:
                    # 如果目标点数为 0，返回空点云
                    batch_selected_points.append(torch.zeros((0, 3), device=x.device))

            output.append(batch_selected_points)  # 形状: List[B] -> List[num_queries] -> Tensor[N, 3]

        return output  # **返回 List of List of Tensors**