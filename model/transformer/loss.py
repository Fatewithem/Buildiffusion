from torch.cuda.amp import autocast
import torch
from pytorch3d.loss import chamfer_distance
import torch.nn as nn
import torch.nn.functional as F
import ot  # Optimal Transport
import open3d as o3d
import numpy as np
from geomloss import SamplesLoss

def compute_chamfer_loss(pred_points, gt_points):
    """
    计算 Chamfer 距离和全局 Chamfer 距离。
    :param pred_points: (B, N, 3) 预测点云
    :param gt_points: (B, M, 3) GT 点云
    :return: chamfer_loss, chamfer_global_loss
    """
    with autocast(enabled=False):
        chamfer_loss, _ = chamfer_distance(pred_points.float(), gt_points.float())
    return chamfer_loss


def compute_emd_loss(pred_points, gt_points):
    """
    使用 Sinkhorn 距离（来自 geomloss）计算近似 EMD。
    :param pred_points: (B, N, 3) 预测点云
    :param gt_points: (B, M, 3) GT 点云
    :return: Sinkhorn EMD Loss
    """
    loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.01, backend="tensorized")  # blur: 0.01
    with autocast(enabled=False):
        pred_points = pred_points.float()
        gt_points = gt_points.float()

        # 自动替换 NaN 和 Inf 为有限值，避免训练中断
        pred_points = torch.nan_to_num(pred_points, nan=0.0, posinf=1e4, neginf=-1e4)
        gt_points = torch.nan_to_num(gt_points, nan=0.0, posinf=1e4, neginf=-1e4)

        return loss_fn(pred_points, gt_points)


def compute_repulsion_loss(pred_points, k=5, h=0.03):
    """
    计算预测点云的 repulsion loss，鼓励点之间的均匀分布。
    :param pred_points: Tensor (B, N, 3)
    :param k: 最近邻个数
    :param h: 控制高斯距离衰减的参数
    :return: scalar loss
    """
    from pytorch3d.ops import knn_points
    with autocast(enabled=False):
        pred_points = pred_points.float()
        B, N, _ = pred_points.shape
        dists, _, _ = knn_points(pred_points, pred_points, K=k+1, return_nn=False)
        dists = dists[:, :, 1:]  # 排除自身
        repulsion = torch.exp(-dists / (h ** 2)).mean()
    return repulsion


def compute_loss(pred_points, planes):
    """
    pred_points: [B, N, 3]   — 模型预测点云
    planes:      [B, M, 3]   — GT 点云
    """
    with autocast(enabled=False):
        pred_points = pred_points.float()
        planes      = planes.float()

        # ====== Query Diversity Loss ======
        # pred_points: [B, N, 3]
        B, N, _ = pred_points.shape
        Q = 10  # your query count
        P = N // Q

        pred_q = pred_points.view(B, Q, P, 3)   # [B, Q, P, 3]

        # Query global feature = mean pooling of each query output
        q_feat = pred_q.mean(dim=2)             # [B, Q, 3]

        # Compute pairwise L2 distances
        dist = torch.cdist(q_feat, q_feat)      # [B, Q, Q]

        # Mask out diagonal
        mask = 1 - torch.eye(Q, device=dist.device)
        dist = dist * mask

        # Diversity loss: exp(-distance)
        diversity_loss = torch.exp(-dist).mean()

        # Sanitize invalid values
        pred_points = torch.nan_to_num(pred_points, nan=0.0, posinf=1e4, neginf=-1e4)
        planes      = torch.nan_to_num(planes, nan=0.0, posinf=1e4, neginf=-1e4)

        # -------- 1) Chamfer Loss --------
        cd_loss = compute_chamfer_loss(pred_points, planes)

        # -------- 2) EMD Loss (Sinkhorn) --------
        emd_loss = compute_emd_loss(pred_points, planes)

        # -------- 3) Repulsion loss --------
        repulsion_loss = compute_repulsion_loss(pred_points)

        # -------- Final Loss --------
        lambda_cd = 1.0
        lambda_emd = 0.15
        lambda_rep = 0.1
        lambda_div = 0.05  # 推荐 0.02 - 0.1，可根据 collapse 程度调整

        final_loss = (
                lambda_cd * cd_loss +
                lambda_emd * emd_loss +
                lambda_rep * repulsion_loss +
                lambda_div * diversity_loss
        )

        final_loss = final_loss.mean()

    return final_loss, cd_loss.mean(), emd_loss.mean(), diversity_loss


def compute_loss_shapenet(output, planes):
    with autocast(enabled=False):
        output = output.float()
        planes = planes.float()

        # Clamp invalid values
        output = torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)
        planes = torch.nan_to_num(planes, nan=0.0, posinf=1e4, neginf=-1e4)

        # l1_loss = F.l1_loss(output, planes, reduction='mean')

        cd_loss, _ = chamfer_distance(output, planes)

        # 计算 EMD loss
        loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.01, backend="tensorized")
        emd_loss = loss_fn(output, planes)

        total_loss = cd_loss + 0.2 * emd_loss

    return total_loss, cd_loss, emd_loss