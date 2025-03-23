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
            nn.Linear(dim_model, 512),  # Add 3D Positional Encoding
            nn.LeakyReLU(0.1),
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


# 返回包含颜色信息
class OutputLayerColor(pl.LightningModule):
    """
    Output Layer of the LegoFormer.
    Maps Transformer's output vector to decomposition factors and generates 3D point clouds.
    现在改为输出 6 维：前 3 维为坐标 (x, y, z)，后 3 维为颜色 (r, g, b)。
    """

    def __init__(self, dim_model, num_queries=10, max_points=2000):
        super().__init__()

        self.num_queries = num_queries
        self.max_points = max_points  # 设置最大点数作为所有点云的上限

        # 注意，这里将输出维度从 3 * max_points 改为 6 * max_points
        self.mlp = nn.Sequential(
            nn.Linear(dim_model, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, max_points * 6)  # 每个点 6 维：xyz + rgb
        )

    def forward(self, x, target_num_points_list):
        """
        :param x: 输入张量，形状为 [B, num_queries, C]
        :param target_num_points_list: 每个查询的目标点数列表，形状为 [B, num_queries]
        :return: List of Tensors, 每个元素形状为 (N, 6)，前 3 维为坐标，后 3 维为颜色
        """
        batch_size, num_queries, _ = x.shape

        # 通过 MLP 生成 (max_points * 6) 的输出
        points = self.mlp(x)  # 形状: [B, num_queries, max_points * 6]
        points = points.view(batch_size, num_queries, self.max_points, 6)
        # points[..., :3] -> xyz，points[..., 3:] -> rgb

        output = []
        for b in range(batch_size):
            batch_selected_points = []

            for q in range(num_queries):
                num_points = target_num_points_list[b, q].item()

                if num_points > 0:
                    # 1) 只用坐标部分做最远点采样 (FPS)
                    xyz = points[b, q, :, :3].unsqueeze(0)  # (1, max_points, 3)
                    sampled_xyz, sampled_indices = sample_farthest_points(xyz, K=num_points)
                    # sampled_xyz: (1, K, 3)， sampled_indices: (1, K)

                    # 2) 根据 sampled_indices 获取对应的颜色
                    rgb = points[b, q, :, 3:]  # (max_points, 3)
                    sampled_rgb = rgb[sampled_indices[0]]  # (K, 3)

                    # 3) 拼接 xyz 和 rgb => (K, 6)
                    sampled_xyz = sampled_xyz.squeeze(0)  # (K, 3)
                    sampled_pts_6d = torch.cat([sampled_xyz, sampled_rgb], dim=-1)

                    batch_selected_points.append(sampled_pts_6d)
                else:
                    batch_selected_points.append(torch.zeros((0, 6), device=x.device))

            output.append(batch_selected_points)

        return output