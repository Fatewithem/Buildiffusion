import torch
from pytorch3d.loss import chamfer_distance
import torch.nn as nn
import torch.nn.functional as F
import ot  # Optimal Transport


def manual_emd_loss(pred_points, gt_points):
    """
    手动计算 Earth Mover’s Distance (EMD) Loss。
    :param pred_points: (B, N, 3) 预测点云
    :param gt_points: (B, M, 3) GT 点云
    :return: EMD Loss
    """
    B, N, _ = pred_points.shape
    _, M, _ = gt_points.shape
    total_loss = 0.0

    for b in range(B):
        C = torch.cdist(pred_points[b], gt_points[b])  # 计算欧几里得距离 (N, M)
        C_np = C.detach().cpu().numpy()
        P = ot.emd([], [], C_np)  # 计算最优传输
        P = torch.tensor(P, device=pred_points.device, dtype=torch.float32)
        total_loss += (P * C).sum()  # 计算 EMD 损失

    return total_loss / B


def split_planes(planes, target_points_per_query):
    """
    按 `target_points_per_query` 将 `planes[B, N, 3]` 切分为 List[B] -> List[num_queries] -> Tensor[N_q, 3]
    :param planes: [B, N, 3] 的 GT 点云
    :param target_points_per_query: 每个 query 对应的点数
    :return: 切分后的 List[B] -> List[num_queries] -> Tensor[N_q, 3]
    """
    B, N, _ = planes.shape  # batch_size, 总点数, 3
    num_queries = len(target_points_per_query)

    split_planes_list = []  # 存储切分后的点云
    for b in range(B):
        start_idx = 0
        batch_planes = []  # 当前 batch 的 GT planes

        for q_idx in range(num_queries):
            num_pts = target_points_per_query[q_idx]  # 当前 query 需要的点数
            end_idx = start_idx + num_pts
            batch_planes.append(planes[b, start_idx:end_idx, :])  # 取出对应的点
            start_idx = end_idx

        split_planes_list.append(batch_planes)  # 存储 batch 级 GT planes

    return split_planes_list  # 返回 List[B] -> List[num_queries] -> Tensor[N_q, 3]


# def compute_point_cloud_normals(points):
#     """
#     计算预测点云的法向量，基于最近邻计算。
#     :param points: Tensor [N, 3] 预测点云
#     :return: Tensor [N, 3] 法向量
#     """
#     from torch_cluster import knn
#     k = 3  # 最近邻
#     edge_index = knn(points, points, k=k)
#
#     v1 = points[edge_index[1][0]] - points[edge_index[1][1]]
#     v2 = points[edge_index[1][0]] - points[edge_index[1][2]]
#     normals = torch.cross(v1, v2)
#     return F.normalize(normals, dim=-1)
#
#
# def normal_consistency_loss(output, plane_infos):
#     """
#     计算预测点云的法向量和 GT 平面法向量的一致性损失。
#     :param output: List[B] -> List[10] -> Tensor[N_q, 3]，预测点云
#     :param plane_infos: Tensor[B, 10, 4]，GT 平面方程 (A, B, C, D)
#     :return: Scalar loss
#     """
#     total_loss = 0.0
#     B = len(output)
#
#     for b_idx in range(B):
#         sample_loss = 0.0
#         output_b = output[b_idx]
#         plane_info_b = plane_infos[b_idx]
#
#         for q_idx in range(plane_info_b.shape[0]):
#             pred_q = output_b[q_idx]
#             plane_eq = plane_info_b[q_idx]
#
#             if pred_q.shape[0] == 0:
#                 continue
#
#             # GT 平面的法向量
#             gt_normal = torch.tensor([plane_eq[0], plane_eq[1], plane_eq[2]], device=pred_q.device)
#             gt_normal = F.normalize(gt_normal, dim=0)
#
#             # 计算预测点云的法向量
#             pred_normals = compute_point_cloud_normals(pred_q)
#
#             # 计算余弦相似度 loss
#             normal_loss = 1 - F.cosine_similarity(pred_normals, gt_normal.expand_as(pred_normals), dim=-1).mean()
#             sample_loss += normal_loss
#
#         total_loss += sample_loss / plane_info_b.shape[0]
#
#     return total_loss / B


def compute_chamfer_loss(output, planes, target_points_per_query):
    """
    计算 Chamfer Distance Loss，并加入法向量损失 (Normal Consistency Loss)
    """
    total_loss = 0.0
    global_loss = 0.0
    B = len(output)  # Batch 大小

    split_gt_planes = split_planes(planes, target_points_per_query)

    for b_idx in range(B):  # 遍历 batch
        sample_loss = 0.0
        output_b = output[b_idx]  # 当前 batch 的预测点云
        planes_b = split_gt_planes[b_idx]  # 当前 batch 的 GT

        # 拼接所有 query 的点云用于全局 loss 计算
        pred_all = torch.cat(output_b, dim=0)  # 合并所有预测点
        pred_all = pred_all[:, :3]
        gt_all = torch.cat(planes_b, dim=0)    # 合并所有 GT 点

        for q_idx in range(len(output_b)):  # 遍历 queries
            pred_q = output_b[q_idx][:, :3]  # 预测点云 (N_q, 3)
            gt_q = planes_b[q_idx]    # GT 点云 (N_q, 3)

            if pred_q.shape[0] == 0 or gt_q.shape[0] == 0:
                continue

            # 计算 Chamfer Distance（局部 loss）
            cd, _ = chamfer_distance(pred_q.unsqueeze(0), gt_q.unsqueeze(0))
            sample_loss += cd

        # 计算全局 Chamfer Distance（整体 loss）
        global_cd, _ = chamfer_distance(pred_all.unsqueeze(0), gt_all.unsqueeze(0))

        total_loss += sample_loss / len(output_b)  # 对 queries 求平均
        global_loss += global_cd  # 累计全局 loss

    total_loss = total_loss / B  # 对 batch 求平均
    global_loss = global_loss / B  # 对 batch 求平均

    # 结合局部 loss、全局 loss 和法向量 loss
    # chamfer_loss = total_loss + 0.15 * global_loss # 0.05 权重可调整

    return total_loss, global_loss


def compute_emd_loss(output, planes, target_points_per_query):
    """
    计算 Earth Mover's Distance (EMD) Loss。
    """
    total_loss = 0.0
    B = len(output)  # Batch 大小

    split_gt_planes = split_planes(planes, target_points_per_query)

    for b_idx in range(B):  # 遍历 batch
        sample_loss = 0.0
        output_b = output[b_idx]  # 当前 batch 的预测点云
        planes_b = split_gt_planes[b_idx]  # 当前 batch 的 GT

        for q_idx in range(len(output_b)):  # 遍历 queries
            pred_q = output_b[q_idx][:, :3] # 预测点云 (N_q, 3)
            gt_q = planes_b[q_idx]    # GT 点云 (N_q, 3)

            if pred_q.shape[0] == 0 or gt_q.shape[0] == 0:
                continue

            # 计算 Earth Mover’s Distance (EMD)
            emd = manual_emd_loss(pred_q.unsqueeze(0), gt_q.unsqueeze(0))
            sample_loss += emd

        total_loss += sample_loss / len(output_b)  # 对 queries 求平均

    total_loss = total_loss / B  # 对 batch 求平均
    return total_loss


def compute_global_emd_loss(output, planes, target_points_per_query):
    """
    计算全局拼接后的 EMD loss。
    """
    total_loss = 0.0
    B = len(output)
    split_gt_planes = split_planes(planes, target_points_per_query)

    for b_idx in range(B):
        output_b = output[b_idx]
        planes_b = split_gt_planes[b_idx]

        pred_all = torch.cat([q[:, :3] for q in output_b], dim=0)
        gt_all = torch.cat(planes_b, dim=0)

        min_len = min(pred_all.shape[0], gt_all.shape[0])
        emd = manual_emd_loss(pred_all[:min_len].unsqueeze(0), gt_all[:min_len].unsqueeze(0))
        total_loss += emd

    return total_loss / B


def compute_loss(output, planes, target_points_per_query):
    """
    计算综合 Loss：Chamfer Loss + 点到平面 Loss + 法向量 Loss
    """

    # CD loss
    cd_loss, cd_global_loss = compute_chamfer_loss(output, planes, target_points_per_query)

    # EMD loss
    emd_loss = compute_emd_loss(output, planes, target_points_per_query)

    # normal_loss = normal_consistency_loss(output, plane_infos)  # 法向量一致性

    # 组合 Loss
    # final_loss = total_loss + 0.5 * global_loss + 1 * plane_dist_loss + 1 * normal_loss

    final_loss = cd_loss + 0.2 * cd_global_loss + 0.1 * emd_loss

    return final_loss, cd_loss, cd_global_loss, emd_loss


# --------------------------------------------------color---------------------------------------------------------------
def compute_color(output, colors, target_points_per_query):
    """
    计算颜色损失，包括 MSE Loss 和 Chamfer Distance (CD) Loss。
    :param output: 预测点云，形状 [B, N, 6] (xyz + rgb)
    :param colors: GT 颜色，形状 [B, N, 3]
    :param target_points_per_query: 每个 query 目标点数
    :param lambda_rgb: MSE 颜色损失的权重
    :param lambda_cd: CD 颜色损失的权重
    :return: 颜色损失
    """
    total_loss = 0.0
    B = len(output)  # Batch 大小

    split_gt_colors = split_planes(colors, target_points_per_query)

    for b_idx in range(B):
        sample_loss = 0.0
        output_b = output[b_idx]
        colors_b = split_gt_colors[b_idx]

        for q_idx in range(len(output_b)):
            pred_q = output_b[q_idx]
            gt_color_q = colors_b[q_idx]

            if pred_q.shape[0] == 0 or gt_color_q.shape[0] == 0:
                continue

            pred_rgb = pred_q[:, 3:]  # 取出预测点云中的颜色部分

            # 计算颜色 MSE Loss
            mse_rgb = F.mse_loss(pred_rgb, gt_color_q)

            # 计算 Chamfer Distance for colors
            cd_rgb, _ = chamfer_distance(pred_rgb.unsqueeze(0), gt_color_q.unsqueeze(0))

            # 计算最终 Loss
            sample_loss += 0.4 * mse_rgb + 0.6 * cd_rgb

        total_loss += sample_loss / len(output_b)

    total_loss = total_loss / B
    return total_loss


def compute_color_loss(output, planes, colors, target_points_per_query):
    total_loss, global_loss = compute_chamfer_loss(output, planes, target_points_per_query)  # 你的 Chamfer 计算
    emd_loss = compute_emd_loss(output, planes, target_points_per_query)
    color_loss = compute_color(output, colors, target_points_per_query)

    final_loss = total_loss + 0.15 * global_loss + 0.1 * emd_loss + 0.1 * color_loss

    return final_loss, total_loss, global_loss, emd_loss, color_loss