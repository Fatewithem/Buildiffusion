import torch
import numpy as np
import json
import open3d as o3d


# 输出点云的总点数
def print_total_point_count(points):
    total_points = len(points)  # 获取点云中的总点数
    print(f"Total points in the point cloud: {total_points}")


# 读取点云的 .pt 文件
def read_tensor_point_cloud(file_path):
    # 加载 Tensor 文件
    data = torch.load(file_path)

    points = data['points']  # 获取点云坐标

    # 将数据转换为 NumPy 数组
    points = points.numpy()

    print(f"点云中的点数: {len(points)}")
    return points


def save_tensor_as_ply(tensor_points, output_ply_path):
    """
    将 PyTorch Tensor 格式的点云保存为 PLY 文件
    :param tensor_points: torch.Tensor, 形状为 [N, 3]
    :param output_ply_path: str, PLY 文件的保存路径
    """
    if isinstance(tensor_points, torch.Tensor):
        tensor_points = tensor_points.cpu().numpy()  # 转换为 NumPy 数组

    # 创建 Open3D 点云对象
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(tensor_points)  # 设置点坐标

    # 保存为 PLY 文件
    o3d.io.write_point_cloud(output_ply_path, point_cloud)
    print(f"✅ 点云已成功保存到 {output_ply_path}")


# 主程序
def main():
    # 设置文件路径
    json_file_path = "/home/datasets/UrbanBIS/Qingdao/0020-0026/building/building10/untitled_plane_info.json"
    point_cloud_file = "/home/datasets/UrbanBIS/Qingdao/0020-0026/building/building10/untitled_plane.pt"  # 假设点云文件是 .pt 格式

    # 读取点云数据（点坐标、颜色和标签）
    points = read_tensor_point_cloud(point_cloud_file)

    save_tensor_as_ply(points, "/home/code/Buildiffusion/output.ply")


if __name__ == "__main__":
    main()