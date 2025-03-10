import torch
from pytorch3d.loss import chamfer_distance
import torch.nn as nn
import torch.nn.functional as F
import ot  # Optimal Transport


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


def compute_loss(output, planes, target_points_per_query):
    """
    计算 Chamfer Distance Loss
    :param output: 预测的 List[B] -> List[num_queries] -> Tensor[N_q, 3]
    :param planes: GT 的 Tensor[B, N, 3]，需要先划分
    :param target_points_per_query: 每个 Query 的目标点数
    :return: loss 值
    """
    total_loss = 0.0
    B = len(output)  # Batch 大小

    # ✅ 先将 `planes` 切分成 `List[B] -> List[num_queries] -> Tensor[N_q, 3]`
    split_gt_planes = split_planes(planes, target_points_per_query)

    for b_idx in range(B):  # 遍历 batch
        sample_loss = 0.0
        output_b = output[b_idx]  # 当前 batch 的预测点云
        planes_b = split_gt_planes[b_idx]  # 当前 batch 的 GT

        for q_idx in range(len(output_b)):  # 遍历 queries
            pred_q = output_b[q_idx]  # 预测点云 (N_q, 3)
            gt_q = planes_b[q_idx]  # GT 点云 (N_q, 3)

            # 确保两个点云都有点，否则跳过
            if pred_q.shape[0] == 0 or gt_q.shape[0] == 0:
                continue

            # 计算 Chamfer Distance
            cd, _ = chamfer_distance(
                pred_q.unsqueeze(0),  # [1, N_q, 3]
                gt_q.unsqueeze(0)  # [1, N_q, 3]
            )
            sample_loss += cd

        total_loss += sample_loss / len(output_b)  # 对 queries 求平均

    total_loss = total_loss / B  # 对 batch 求平均
    return total_loss

#
# def compute_loss(output, planes, target_points_per_query):
#     """
#     计算 Chamfer Distance Loss
#     :param output: 预测的 List[B] -> List[num_queries] -> Tensor[N_q, 3]
#     :param planes: GT 的 Tensor[B, N, 3]，需要先划分
#     :param target_points_per_query: 每个 Query 的目标点数
#     :return: loss 值
#     """
#     total_loss = 0.0
#     B = len(output)  # Batch 大小
#
#     # ✅ 先将 `planes` 切分成 `List[B] -> List[num_queries] -> Tensor[N_q, 3]`
#     split_gt_planes = split_planes(planes, target_points_per_query)
#
#     for b_idx in range(B):  # 遍历 batch
#         sample_loss = 0.0
#         output_b = output[b_idx]  # 当前 batch 的预测点云
#         planes_b = split_gt_planes[b_idx]  # 当前 batch 的 GT
#
#         for q_idx in range(len(output_b)):  # 遍历 queries
#             pred_q = output_b[q_idx]  # 预测点云 (N_q, 3)
#             gt_q = planes_b[q_idx]  # GT 点云 (N_q, 3)
#
#             # 确保两个点云都有点，否则跳过
#             if pred_q.shape[0] == 0 or gt_q.shape[0] == 0:
#                 continue
#
#             # 计算 Chamfer Distance
#             cd = F.mse_loss(
#                 pred_q,  # [1, N_q, 3]
#                 gt_q  # [1, N_q, 3]
#             )
#             sample_loss += cd
#
#         total_loss += sample_loss / len(output_b)  # 对 queries 求平均
#
#     total_loss = total_loss / B  # 对 batch 求平均
#     return total_loss
