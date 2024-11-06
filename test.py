import open3d as o3d
import numpy as np


def get_ply_axis_range(ply_file_path):
    # 读取 PLY 文件
    pcd = o3d.io.read_point_cloud(ply_file_path)

    # 获取点云的所有点
    points = np.asarray(pcd.points)

    # 分别计算 X、Y、Z 轴的最小值和最大值
    x_min, y_min, z_min = points.min(axis=0)
    x_max, y_max, z_max = points.max(axis=0)

    # 输出每个轴的范围
    print(f"X axis range: [{x_min}, {x_max}]")
    print(f"Y axis range: [{y_min}, {y_max}]")
    print(f"Z axis range: [{z_min}, {z_max}]")

    return (x_min, x_max), (y_min, y_max), (z_min, z_max)


def get_ply_bounding_box_center(ply_file_path):
    # 读取 PLY 文件
    pcd = o3d.io.read_point_cloud(ply_file_path)

    # 获取点云的所有点
    points = np.asarray(pcd.points)

    # 计算 X、Y、Z 轴的最小值和最大值
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)

    # 计算包围盒中心
    center = (min_bound + max_bound) / 2

    # 输出包围盒的中心位置
    print(f"Bounding box center: {center}")

    return center


# 使用示例
ply_file_path = '/home/code/Buildiffusion/outputs/debug/2024-10-15--14-59-59/sample/pred/buildings/buildings_test.ply'
get_ply_axis_range(ply_file_path)
get_ply_bounding_box_center(ply_file_path)