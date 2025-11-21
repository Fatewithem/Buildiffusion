import torch
import numpy as np
from pytorch3d.ops import knn_points
import open3d as o3d
from pytorch3d.ops import sample_farthest_points

TARGET_POINTS = 8192

def read_ply_to_tensor_repeat(path, target_points=TARGET_POINTS, center=False):
    """读取PLY文件并通过重复采样或裁剪补齐到 target_points 点"""
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points).astype(np.float32)
    pts = torch.from_numpy(pts).unsqueeze(0)  # [1, N, 3]

    N = pts.shape[1]
    if N < target_points:
        repeat_factor = int(np.ceil(target_points / N))
        pts_repeated = pts.repeat(1, repeat_factor, 1)[:, :target_points, :]
        pts = pts_repeated
    elif N > target_points:
        pts = pts[:, :target_points, :]

    if center:
        centroid = pts.mean(dim=1, keepdim=True)
        pts = pts - centroid

    return pts

def read_ply_to_tensor_fps(path, target_points=8192, center=False):
    """
    读取 PLY 文件并通过 FPS 插值补齐到 target_points 个点
    """
    # 读取点云
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=np.float32)  # [N, 3]
    pts = torch.from_numpy(pts).unsqueeze(0)  # [1, N, 3]

    N = pts.shape[1]

    if N >= target_points:
        # 使用 FPS 下采样
        pts_sampled, _ = sample_farthest_points(pts, K=target_points)
        pts = pts_sampled
    else:
        # FPS 采样原始点作为基点
        fps_idx = torch.randperm(N)[:N]  # 先打乱顺序
        base_pts = pts[:, fps_idx, :]  # [1, N, 3]

        # 生成插值权重（随机插值到目标点数）
        idx = torch.randint(0, N, (1, target_points - N))
        alpha = torch.rand(1, target_points - N, 1)
        interp_pts = alpha * pts[:, idx.squeeze(0), :] + (1 - alpha) * base_pts[:, idx.squeeze(0), :]

        pts = torch.cat([pts, interp_pts], dim=1)

    if center:
        centroid = pts.mean(dim=1, keepdim=True)
        pts = pts - centroid

    return pts  # [1, target_points, 3]

def compute_fscore(pred, gt, threshold=0.02):
    """使用 PyTorch3D 计算 F-score、Precision、Recall"""
    dists1 = knn_points(pred, gt, K=1).dists.sqrt()  # pred -> gt
    dists2 = knn_points(gt, pred, K=1).dists.sqrt()  # gt -> pred

    precision = (dists1 < threshold).float().mean().item()
    recall = (dists2 < threshold).float().mean().item()
    fscore = 2 * precision * recall / (precision + recall + 1e-8)

    pred_center = pred.mean(dim=1).squeeze().tolist()
    gt_center = gt.mean(dim=1).squeeze().tolist()
    pred_range = (pred.max(dim=1).values - pred.min(dim=1).values).squeeze().tolist()
    gt_range = (gt.max(dim=1).values - gt.min(dim=1).values).squeeze().tolist()

    return fscore, precision, recall, pred_center, gt_center, pred_range, gt_range

if __name__ == "__main__":
    pred_path = "//home/code/Buildiffusion/result_shape/sample_0.ply"
    gt_path = "/home/code/Buildiffusion/result_shape/sample_0_gt.ply"

    pred = read_ply_to_tensor_repeat(pred_path, center=False)
    gt = read_ply_to_tensor_repeat(gt_path, center=False)

    fscore, prec, rec, pred_center, gt_center, pred_range, gt_range = compute_fscore(pred, gt)
    print(f"F-score: {fscore:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
    print(f"Pred Center: {pred_center}")
    print(f"GT Center:   {gt_center}")
    print(f"Pred Range:  {pred_range}")
    print(f"GT Range:    {gt_range}")