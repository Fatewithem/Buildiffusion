import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
import numpy as np

from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG


def index_points(points, idx):
    """
    根据索引从 points 中获取对应点信息
    """
    device = points.device
    B = points.shape[0]

    if idx.dim() == 1:
        return torch.index_select(points, 1, idx)

    if len(idx.shape) == 2:
        S = idx.shape[1]
        batch_idx = torch.arange(B, dtype=torch.long, device=device).view(-1, 1).repeat(1, S)
        new_points = points[batch_idx, idx, :]
    else:
        B, S, K = idx.shape
        batch_idx = torch.arange(B, dtype=torch.long, device=device).view(-1, 1, 1).repeat(1, S, K)
        new_points = points[batch_idx, idx, :]
    return new_points


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    在给定的球半径内，查找邻域点(近邻搜索)，并限制最大邻域点数量
    """
    device = xyz.device
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]

    dist = torch.cdist(new_xyz, xyz, p=2)  # (B, S, N)
    group_idx = torch.argsort(dist, dim=-1)[:, :, :nsample]  # 取最近的 nsample 个点

    # 找到超出半径的点
    mask = dist.gather(-1, group_idx) > radius
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)  # 取第一个有效索引
    group_idx[mask] = group_first[mask]  # 用第一个有效索引填充

    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    对点云进行采样并将邻域分组。
    """
    B, N, C_xyz = xyz.shape
    S = npoint

    # 1. 采样中心点
    new_xyz, fps_idx = sample_farthest_points(xyz, K=npoint)

    # 2. 查找邻域点
    group_idx = query_ball_point(radius, nsample, xyz, new_xyz)

    # 3. 提取邻域点坐标 & 特征
    grouped_xyz = index_points(xyz, group_idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)

    if points is not None:
        grouped_points = index_points(points, group_idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points


class PointNet2FeatureExtractorWithFP(nn.Module):
    def __init__(self, out_channels=256):
        super(PointNet2FeatureExtractorWithFP, self).__init__()
        self.sa1 = PointnetSAModuleMSG(
            npoint=4096,
            radii=[0.2],
            nsamples=[32],
            mlps=[[0, 64, 64, 128]],
            use_xyz=True
        )
        self.sa2 = PointnetSAModuleMSG(
            npoint=1024,
            radii=[0.4],
            nsamples=[64],
            mlps=[[128, 128, 128, 256]],
            use_xyz=True
        )
        self.sa3 = PointnetSAModuleMSG(
            npoint=512,
            radii=[0.6],
            nsamples=[128],
            mlps=[[256, 256, 256, 512]],
            use_xyz=True
        )

        self.fp3 = PointnetFPModule(mlp=[512 + 256, 512, 256])
        self.fp2 = PointnetFPModule(mlp=[256 + 128, 256, 128])
        # self.fp1 = PointnetFPModule(mlp=[128, 128, out_channels])

    def forward(self, xyz):
        with torch.cuda.amp.autocast(enabled=False):
            l0_xyz = xyz
            l0_points = None

            l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
            l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
            l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

            l2_points_up = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
            l1_points_up = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points_up)
            # l0_points_up = self.fp1(l0_xyz, l1_xyz, None, l1_points_up)
            # 包裹在禁用 autocast 的上下文中以避免 fp16 类型冲突

            # l0_points_up = F.adaptive_avg_pool1d(l0_points_up, output_size=2048)

        return l1_points_up

class PointNet2FeatureExtractorWithFP_ShapeNet(nn.Module):
    def __init__(self, out_channels=256):
        super(PointNet2FeatureExtractorWithFP_ShapeNet, self).__init__()
        self.sa1 = PointnetSAModuleMSG(
            npoint=512,
            radii=[0.2],
            nsamples=[32],
            mlps=[[0, 64, 64, 128]],
            use_xyz=True
        )
        self.sa2 = PointnetSAModuleMSG(
            npoint=128,
            radii=[0.4],
            nsamples=[64],
            mlps=[[128, 128, 128, 256]],
            use_xyz=True
        )
        self.sa3 = PointnetSAModuleMSG(
            npoint=32,
            radii=[0.6],
            nsamples=[128],
            mlps=[[256, 256, 256, 512]],
            use_xyz=True
        )

        self.fp3 = PointnetFPModule(mlp=[512 + 256, 512, 256])
        self.fp2 = PointnetFPModule(mlp=[256 + 128, 256, 128])
        self.fp1 = PointnetFPModule(mlp=[128, 128, 256])

    def forward(self, xyz):
        with torch.cuda.amp.autocast(enabled=False):
            l0_xyz = xyz
            l0_points = None

            l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
            l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
            l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

            l2_points_up = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
            l1_points_up = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points_up)
            l0_points_up = self.fp1(l0_xyz, l1_xyz, None, l1_points_up)

        return l0_points_up
