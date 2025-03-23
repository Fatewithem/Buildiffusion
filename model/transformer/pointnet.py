import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
import numpy as np

from pytorch3d.ops import sample_farthest_points


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


def three_nn_interpolate(xyz_src, xyz_dst, feats_dst):
    """
    对 xyz_src 中的每个点，到 xyz_dst 找最近的3个点，用距离加权插值 feats_dst。
    参数:
        xyz_src:  (B, nSrc, 3) 目标分辨率坐标（要插值到的分辨率）
        xyz_dst:  (B, nDst, 3) 已知特征所在的低分辨率坐标
        feats_dst:(B, cDst, nDst) 对应 xyz_dst 的特征
    返回:
        feats_src: (B, cDst, nSrc)  插值后在 xyz_src 上的特征
    """
    device = xyz_src.device
    B, nSrc, _ = xyz_src.shape
    _, cDst, nDst = feats_dst.shape

    feats_src_list = []
    for b in range(B):
        # (nSrc,1,3) vs (1,nDst,3) => (nSrc, nDst)
        dist = torch.sum((xyz_src[b].unsqueeze(1) - xyz_dst[b].unsqueeze(0))**2, dim=-1)
        # 找最近的3个点
        dist_sorted, idx_sorted = torch.sort(dist, dim=-1)
        idx_knn = idx_sorted[:, :3]   # (nSrc, 3)
        dist_knn = dist_sorted[:, :3] # (nSrc, 3)

        # 距离越大，权重越小；加 1e-8 防止除0
        weight = 1.0 / (dist_knn + 1e-8)  # (nSrc, 3)
        weight = weight / torch.sum(weight, dim=1, keepdim=True)  # 归一化

        # feats_dst[b]: (cDst, nDst)
        feats_3 = feats_dst[b].index_select(1, idx_knn.reshape(-1))  # (cDst, nSrc*3)
        feats_3 = feats_3.reshape(cDst, nSrc, 3)                     # (cDst, nSrc, 3)

        # 使用 unsqueeze(0) 将 weight 形状从 (nSrc, 3) 变为 (1, nSrc, 3)
        # 两者对应相乘后在最后一维求和，得到 (cDst, nSrc)
        feats_b_src = torch.sum(feats_3 * weight.unsqueeze(0), dim=-1)
        feats_src_list.append(feats_b_src.unsqueeze(0))

    # 拼接得到 (B, cDst, nSrc)
    feats_src = torch.cat(feats_src_list, dim=0)
    return feats_src


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channels, mlp):
        """
        npoint: 采样中心点数
        radius: 邻域球半径
        nsample: 邻域点数限制
        in_channels: 输入特征通道数（xyz坐标不算在 in_channels 中，此处仅特征维数）
        mlp: list，表示 MLP 每层输出通道数，如 [64, 64, 128]
        """
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = in_channels + 3  # +3 因为要拼接局部坐标 (dx, dy, dz)
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points=None):
        """
        xyz: (B, N, 3) 原始点坐标
        points: (B, N, C) 每个点的特征，可选
        输出:
            new_xyz: (B, npoint, 3) 下采样后的中心点坐标
            new_points: (B, npoint, mlp最后一层输出通道)
        """
        # 1. 下采样并分组
        new_xyz, new_points = sample_and_group(
            self.npoint, self.radius, self.nsample, xyz, points
        )
        # new_points: (B, npoint, nsample, 3 + C_in)

        # 2. 对每个分组的点（特征向量）应用 MLP (PointNet 结构)
        # shape转换: (B, npoint, nsample, C) -> (B, C, npoint, nsample) 方便卷积
        new_points = new_points.permute(0, 3, 1, 2)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        # 3. 通过最大池化将 (B, mlp[-1], npoint, nsample) 缩减到 (B, mlp[-1], npoint)
        new_points = torch.max(new_points, -1)[0]  # (B, mlp[-1], npoint)

        # 4. 交换维度到 (B, npoint, mlp[-1])
        new_points = new_points.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channels, mlp_channels):
        """
        参数:
            in_channels:   输入通道数 = 上一分辨率的特征 + 当前分辨率的特征
            mlp_channels:  list[int], 表示 MLP 的各层输出通道，如 [256, 128]
        """
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_ch = in_channels
        for out_ch in mlp_channels:
            self.mlp_convs.append(nn.Conv1d(last_ch, out_ch, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_ch))
            last_ch = out_ch

    def forward(self, xyz1, xyz2, feat1, feat2):
        """
        输入:
            xyz1:  (B, n1, 3)  高分辨率点坐标
            xyz2:  (B, n2, 3)  低分辨率点坐标
            feat1: (B, c1, n1) 在 xyz1 上的特征(可能来自上个 FP 或最初)
            feat2: (B, c2, n2) 在 xyz2 上的特征(由SA得到)
        输出:
            new_feat: (B, mlp_channels[-1], n1)
        """
        # 如果 n2=1, 说明只有一个点(类似全局特征)，直接 broadcast
        if xyz2.shape[1] == 1:
            interpolated_feat = feat2.repeat(1, 1, xyz1.shape[1])  # (B, c2, n1)
        else:
            # 做插值
            interpolated_feat = three_nn_interpolate(xyz_src=xyz1, xyz_dst=xyz2, feats_dst=feat2)
            # (B, c2, n1)

        # 与本分辨率原有特征拼接
        if feat1 is not None:
            new_feat = torch.cat([feat1, interpolated_feat], dim=1)  # (B, c1+c2, n1)
        else:
            new_feat = interpolated_feat

        # 通过 1D Conv (MLP)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_feat = F.relu(bn(conv(new_feat)))  # (B, out_ch, n1)

        return new_feat


class PointNet2FeatureExtractorWithFP(nn.Module):
    def __init__(self):
        super(PointNet2FeatureExtractorWithFP, self).__init__()

        # ---------- 3层 SA ----------
        self.sa1 = PointNetSetAbstraction(
            npoint=4096, radius=0.2, nsample=32,  # 改为 4096 个点
            in_channels=0,  # 初始没有额外特征
            mlp=[64, 64, 128]  # 输出 128 通道
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=1024, radius=0.4, nsample=64,
            in_channels=128,  # 上层输出128
            mlp=[128, 128, 256]  # 输出 256 通道
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=512, radius=0.6, nsample=128,
            in_channels=256,  # 上层输出256
            mlp=[256, 256, 512]  # 输出 512 通道
        )

        # ---------- 3层 FP ----------
        self.fp3 = PointNetFeaturePropagation(in_channels=256 + 512, mlp_channels=[512, 256])
        self.fp2 = PointNetFeaturePropagation(in_channels=128 + 256, mlp_channels=[256, 128])
        # self.fp1 = PointNetFeaturePropagation(in_channels=128 + 0, mlp_channels=[128, 128])
        self.fp1 = PointNetFeaturePropagation(in_channels=128 + 0, mlp_channels=[128, 64])

    def forward(self, xyz):
        """
        参数:
            xyz: (B, N, 3) 点云坐标
        返回:
            points_up: (B, 128, N) 最终上采样到原分辨率的特征
        """
        points = None  # 假设初始没有额外的特征

        # ---------- 下采样(SA) ----------
        l1_xyz, l1_points = self.sa1(xyz, points)  # (B, 4096, 3), (B, 4096, 128)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # (B, 1024, 3), (B, 1024, 256)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # (B, 512, 3), (B, 512, 512)

        # 在使用 FP 前，要让特征维度变为 (B, C, n)
        l1_points = l1_points.permute(0, 2, 1)  # (B, 128, 4096)
        l2_points = l2_points.permute(0, 2, 1)  # (B, 256, 1024)
        l3_points = l3_points.permute(0, 2, 1)  # (B, 512, 512)

        # ---------- 上采样(FP) ----------
        # fp3: 从第三层(512个点)插值回第二层(1024个点)
        l2_points_up = self.fp3(xyz1=l2_xyz, xyz2=l3_xyz, feat1=l2_points, feat2=l3_points)  # (B, 256, 1024)

        # fp2: 从第二层(1024个点)插值回第一层(2048个点)
        l1_points_up = self.fp2(xyz1=l1_xyz, xyz2=l2_xyz, feat1=l1_points, feat2=l2_points_up)  # (B, 128, 2048)

        # fp1: 从第一层(2048个点)插值回原始分辨率(N个点)
        points_up = self.fp1(xyz1=xyz, xyz2=l1_xyz, feat1=None, feat2=l1_points_up)  # (B, 128, N) 64

        return points_up


def load_ply_to_torch_tensor(ply_path):
    """
    使用 open3d 读取 .ply 文件，并返回 shape = (1, N, 3) 的 float Tensor
    """
    # 1. 用 Open3D 读取 .ply 文件
    pcd = o3d.io.read_point_cloud(ply_path)
    # pcd 是一个 open3d.geometry.PointCloud 对象

    # 2. 提取点坐标 (N, 3)
    points_np = np.asarray(pcd.points, dtype=np.float32)
    if points_np.shape[1] != 3:
        raise ValueError("点云坐标不是 (N, 3)，请检查输入文件。")

    # 3. 转换为 PyTorch Tensor，并添加 batch 维度 -> (1, N, 3)
    points_torch = torch.from_numpy(points_np).unsqueeze(0)  # (1, N, 3)
    return points_torch


if __name__ == "__main__":
    # 替换为你实际的 ply 文件路径
    ply_file_path = "/home/code/Blender/untitled.ply"

    # 1. 加载点云并转换为 Tensor
    pointcloud_tensor = load_ply_to_torch_tensor(ply_file_path)  # (1, N, 3)

    new_xyz, fps_idx = sample_farthest_points(pointcloud_tensor, K=20000)
    pointcloud_tensor = index_points(pointcloud_tensor, fps_idx)  # 根据索引获得采样后的点云，形状 (B, 20000, 3)

    print("读取到的点云数据维度：", pointcloud_tensor.shape)

    # 构建带FP的网络
    model_fp = PointNet2FeatureExtractorWithFP()
    model_fp.eval()

    with torch.no_grad():
        per_point_feats = model_fp(pointcloud_tensor)  # (1, 128, N)

    print("带FP的网络输出维度：", per_point_feats.shape)
    # 现在每个原始点都有一个128维的特征

    # 2. 构建网络
    # model = PointNet2FeatureExtractor()
    # model.eval()  # 推理模式
    #
    # # 如果有 GPU 并希望使用的话
    # # model = model.cuda()
    # # pointcloud_tensor = pointcloud_tensor.cuda()
    #
    # # 3. 前向传播获取特征
    # with torch.no_grad():
    #     features = model(pointcloud_tensor)  # (1, 1024) - 具体视你的网络输出
    #
    # print("PointNet++ 提取到的特征维度：", features.shape)
    # print("特征向量：", features)

