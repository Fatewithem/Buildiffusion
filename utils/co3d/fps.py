import os
import open3d as o3d
import numpy as np
import torch
from pytorch3d.ops import sample_farthest_points
from torch_cluster import knn
from typing import Tuple

def farthest_point_sample(points: torch.Tensor, num_points: int) -> Tuple[torch.Tensor, torch.Tensor]:
    points = points.unsqueeze(0)  # [1, N, 3]
    sampled_points, sampled_idx = sample_farthest_points(points, K=num_points)
    return sampled_points.squeeze(0), sampled_idx.squeeze(0)  # [num_points, 3], [num_points]

def fps_downsample_pointcloud(input_file: str, output_file: str, target_num_points: int = 20000):
    # 读取点云
    pcd = o3d.io.read_point_cloud(input_file)
    points_np = np.asarray(pcd.points)
    colors_np = np.asarray(pcd.colors) if pcd.has_colors() else None

    if points_np.shape[0] <= target_num_points:
        print(f"Skipping {input_file}: only {points_np.shape[0]} points, no need to downsample.")
        return

    # 转成Tensor
    points = torch.tensor(points_np, dtype=torch.float32).cuda()
    if colors_np is not None:
        colors = torch.tensor(colors_np, dtype=torch.float32).cuda()
    else:
        colors = None

    # FPS下采样
    sampled_points, sampled_idx = farthest_point_sample(points, num_points=target_num_points)

    # 提取对应颜色
    if colors is not None:
        sampled_colors = colors[sampled_idx]

    # 保存下采样后的点云
    downsampled_pcd = o3d.geometry.PointCloud()
    downsampled_pcd.points = o3d.utility.Vector3dVector(sampled_points.cpu().numpy())
    if colors is not None:
        downsampled_pcd.colors = o3d.utility.Vector3dVector(sampled_colors.cpu().numpy())

    o3d.io.write_point_cloud(output_file, downsampled_pcd)
    print(f"Saved downsampled point cloud to {output_file}")

def process_ply_with_knn_blur(input_ply_path, output_ply_path, k=10, noise_std=0.02):
    """
    读取 .ply 文件，基于 KNN 计算局部扰动进行模糊化，并保存处理后的 .ply 文件。

    :param input_ply_path: str, 输入的 .ply 文件路径
    :param output_ply_path: str, 处理后保存的 .ply 文件路径
    :param k: int，近邻点的数量，值越大模糊越明显
    :param noise_std: float，高斯噪声标准差，控制模糊程度
    """
    pcd = o3d.io.read_point_cloud(input_ply_path)
    points = torch.tensor(np.asarray(pcd.points), dtype=torch.float32)

    # 计算 KNN
    edge_index = knn(points, points, k=k)

    # 计算邻居点的均值
    neighbors = points[edge_index[1]].reshape(points.shape[0], k, 3)
    avg_neighbors = neighbors.mean(dim=1)  # 计算邻居均值

    # 添加噪声
    noise = torch.randn_like(points) * noise_std
    blurred_points = avg_neighbors + noise

    # 保存点云
    pcd.points = o3d.utility.Vector3dVector(blurred_points.numpy())
    o3d.io.write_point_cloud(output_ply_path, pcd)
    print(f"模糊化点云已保存至: {output_ply_path}")

def batch_process_hydrant(root_folder: str):
    """
    遍历 hydrant 类别下每个 sequence，如果有 pointcloud.ply，就进行FPS采样到2w点
    """
    for seq_name in os.listdir(root_folder):
        seq_path = os.path.join(root_folder, seq_name)
        if not os.path.isdir(seq_path):
            continue

        input_ply = os.path.join(seq_path, "pointcloud_fps.ply")
        output_ply = os.path.join(seq_path, "pointcloud_sampled.ply")  # 保存到新文件，防止覆盖

        if os.path.exists(input_ply):
            # if os.path.exists(output_ply):
            #     print(f"Skipping {output_ply}: already exists.")
            #     continue
            print(f"Processing {input_ply} ...")
            fps_downsample_pointcloud(input_ply, output_ply, target_num_points=10000)
        else:
            print(f"No pointcloud.ply found in {seq_path}, skipping.")

def batch_blur_fps_pointclouds(root_folder: str, k: int = 10, noise_std: float = 0.02):
    """
    遍历 hydrant 类别下每个 sequence，对 pointcloud_fps.ply 进行模糊处理，保存为 pointcloud_fps_blur.ply
    :param root_folder: str, hydrant 类别根目录
    :param k: int, 近邻点数量
    :param noise_std: float, 高斯噪声标准差
    """
    for seq_name in os.listdir(root_folder):
        seq_path = os.path.join(root_folder, seq_name)
        if not os.path.isdir(seq_path):
            continue

        input_ply = os.path.join(seq_path, "pointcloud_fps.ply")
        output_ply = os.path.join(seq_path, "pointcloud_blur.ply")

        if os.path.exists(input_ply):
            # if os.path.exists(output_ply):
            #     print(f"Skipping {output_ply}: already exists.")
            #     continue
            print(f"Blurring {input_ply} ...")
            process_ply_with_knn_blur(input_ply, output_ply, k=k, noise_std=noise_std)
        else:
            print(f"No pointcloud_fps.ply found in {seq_path}, skipping.")

if __name__ == "__main__":
    hydrant_root = "/home/datasets/Co3dv2/toytruck"  # 替换成你的hydrant类别根目录
    # batch_process_hydrant(hydrant_root)
    batch_blur_fps_pointclouds(hydrant_root)