import open3d as o3d
import numpy as np
import os
import re


def fit_and_project_to_plane(point_cloud):
    """
    拟合点云到单个平面，并将所有点投影到该平面上。

    参数：
    - point_cloud (o3d.geometry.PointCloud): Open3D 点云对象

    返回：
    - projected_cloud (o3d.geometry.PointCloud): 所有点投影到同一平面的点云
    - plane_equation (tuple): 拟合平面的方程 (A, B, C, D)
    """
    if len(point_cloud.points) == 0:
        raise ValueError("点云为空，无法进行平面拟合！")

    # 1️⃣ RANSAC 拟合平面
    plane_model, inliers = point_cloud.segment_plane(
        distance_threshold=0.01,
        ransac_n=3,
        num_iterations=1000
    )

    # 拟合的平面方程 Ax + By + Cz + D = 0
    A, B, C, D = plane_model

    # 2️⃣ 计算所有点到平面的投影
    points = np.asarray(point_cloud.points)
    normal = np.array([A, B, C])
    dists = (points @ normal + D) / np.linalg.norm(normal)  # 点到平面的距离
    projected_points = points - np.outer(dists, normal)  # 计算投影点

    # 3️⃣ 生成新的点云
    projected_cloud = o3d.geometry.PointCloud()
    projected_cloud.points = o3d.utility.Vector3dVector(projected_points)

    # 4️⃣ 处理颜色（如果原始点云有颜色）
    if point_cloud.has_colors():
        projected_cloud.colors = point_cloud.colors

    return projected_cloud, (A, B, C, D)


def process_split_ply_files(input_folder):
    """
    读取文件夹中以 'split_' 开头且编号在 0 到 9 之间的 .ply 文件，
    进行平面拟合并投影所有点，然后保存结果。

    参数：
    - input_folder (str): 包含 .ply 文件的文件夹路径
    """
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"文件夹不存在: {input_folder}")

    # 筛选出匹配 "split_0.ply" - "split_9.ply" 的文件
    pattern = re.compile(r"split_[0-9]\.ply$")  # 正则匹配 split_0.ply - split_9.ply
    ply_files = [f for f in os.listdir(input_folder) if pattern.match(f)]

    if not ply_files:
        print("⚠️  未找到符合条件的 .ply 文件（split_0.ply - split_9.ply）！")
        return

    print(f"📂 发现 {len(ply_files)} 个符合条件的 .ply 文件，开始处理...")

    for ply_file in ply_files:
        input_ply_path = os.path.join(input_folder, ply_file)

        # 读取点云
        point_cloud = o3d.io.read_point_cloud(input_ply_path)

        # 进行平面拟合和投影
        projected_cloud, plane_eq = fit_and_project_to_plane(point_cloud)

        # 构造输出文件名（添加 `_projected` 后缀）
        output_ply_path = os.path.join(input_folder, f"{os.path.splitext(ply_file)[0]}_projected.ply")

        # 保存投影后的点云
        o3d.io.write_point_cloud(output_ply_path, projected_cloud)

        print(f"✅ {ply_file} 处理完成，结果已保存为: {output_ply_path}")

    print("🎉 所有符合条件的文件处理完成！")


# 示例：处理文件夹
if __name__ == "__main__":
    folder_path = "/path/to/your/ply/folder"  # 请替换为你的 .ply 文件夹路径
    process_split_ply_files(folder_path)