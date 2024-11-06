import open3d as o3d
import numpy as np
from scipy.spatial import ConvexHull
import random
import json
import os


def compute_centroid(points):
    """
    计算给定点的几何重心（质心）。

    Parameters:
        points: numpy 数组，形状为 (N, 3)，表示平面上的点。

    Returns:
        centroid: 平面的几何重心，形状为 (3,)。
    """
    return np.mean(points, axis=0)


def are_planes_close(centroid1, centroid2, normal, distance_threshold=3):
    """
    判断两个平面的重心是否沿着法向量方向接近。

    Parameters:
        centroid1: 第一个平面的重心。
        centroid2: 第二个平面的重心。
        normal: 用于投影的平面的法向量。
        distance_threshold: 判定两个平面接近的距离阈值，默认为8。

    Returns:
        bool: True 如果平面重心在法向量方向上的距离小于阈值，False 否则。
    """
    # 将两个重心分别投影到法向量方向上，计算投影距离
    projected_distance = np.abs(np.dot((centroid2 - centroid1), normal))
    # print(projected_distance)

    # 取投影距离的绝对值，判断是否小于阈值
    return abs(projected_distance) < distance_threshold


def are_normals_similar(normal1, normal2, threshold=np.pi / 12):
    """
    判断两个法向量是否相似，使用点积计算夹角。
    Parameters:
        normal1: 第一个平面的法向量。
        normal2: 第二个平面的法向量。
        threshold: 判定法向量相似的阈值，默认为10度（弧度制）。
    Returns:
        bool: True 如果法向量相似，False 否则。
    """
    angle = np.arccos(np.clip(np.dot(normal1, normal2), -1.0, 1.0))
    return angle < threshold


def down_sampling(cloud):
    # 体素下采样，设置体素大小，例如0.05
    voxel_size = 0.1  # 体素的大小可以根据需要调整
    cloud_downsampled = cloud.voxel_down_sample(voxel_size=voxel_size)

    return cloud_downsampled


# 平面拟合与法向量存储
def fit_plane(cloud):
    print(f"PointCloud before filtering has: {len(cloud.points)} data points.")

    # 估计法向量
    cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

    # 初始化平面分割对象
    planes = []
    plane_normals = []  # 用于存储平面的平均法向量
    plane_centroids = []  # 用于存储平面的重心
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]  # 预定义颜色
    nr_points = len(cloud.points)

    # 循环分割平面，直到剩余点云小于总数的15%
    plane_id = 0
    while len(cloud.points) > 0.1 * nr_points:
        plane_model, inliers = cloud.segment_plane(distance_threshold=0.01,  # 调整阈值
                                                            ransac_n=3,
                                                            num_iterations=1000)

        if len(inliers) == 0:
            print("Could not estimate a planar model for the given dataset.")
            break

        # 提取属于平面的点
        inlier_cloud = cloud.select_by_index(inliers)
        normal = plane_model[:3]
        centroid = compute_centroid(np.asarray(inlier_cloud.points))

        similar_normals = [normal]
        merged = False

        for idx, existing_normal in enumerate(plane_normals):
            if are_normals_similar(normal, existing_normal) and are_planes_close(centroid, plane_centroids[idx], normal):
                inlier_cloud.paint_uniform_color(colors[idx % len(colors)])  # 使用相同的颜色
                planes[idx] += inlier_cloud  # 合并到已有的平面
                similar_normals.append(plane_normals[idx])
                merged = True
                break

        if not merged:
            # 新增一个平面
            color_idx = plane_id % len(colors)  # 循环使用预定义颜色
            inlier_cloud.paint_uniform_color(colors[color_idx])
            planes.append(inlier_cloud)
            average_normal = np.mean(similar_normals, axis=0)
            plane_normals.append(average_normal)
            plane_centroids.append(centroid)
            plane_id += 1

        # 移除平面内的点，继续分割剩余点云
        cloud = cloud.select_by_index(inliers, invert=True)

    return planes, plane_normals


def move_points_to_plane(plane_points, plane_normals, num_points, threshold_up=0.1):
    """
    将 plane 中的所有点移动到与法向量垂直的平面上，并将它们沿法向量方向取平均高度。
    仅当平面点数占比大于阈值时才进行操作。

    Parameters:
        plane_points: open3d.geometry.PointCloud 对象，包含所有点的坐标。
        plane_normals: 平面的法向量（已归一化）。
        total_points: 点云的总点数，用于计算平面占比。
        threshold: 占比阈值，默认为 0.1（10%）。

    Returns:
        moved_points: numpy 数组，调整后的点。如果占比小于阈值，返回原始点。
    """
    # 判断平面占比是否大于阈值
    if len(plane_points.points) / num_points < threshold_up:
        # print("Plane points are less than 10%, skipping move operation.")
        # 返回原始点，不进行移动操作
        return np.asarray(plane_points.points)

    # 将法向量归一化
    normal = plane_normals / np.linalg.norm(plane_normals)

    # 将 plane_points 转换为 NumPy 数组
    points_np = np.asarray(plane_points.points)

    # 计算每个点在法向量方向上的分量（高度）
    heights = np.dot(points_np, normal)

    # 计算沿法向量方向上的平均高度
    avg_height = np.mean(heights)

    # 计算每个点在法向量方向的分量（投影到平面上的点）
    projected_points = points_np - np.outer(heights, normal)

    # 将所有点的高度设置为平均高度
    moved_points = projected_points + avg_height * normal

    return moved_points


# 生成随机颜色
def random_color():
    return [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]


# 将所有平面组合成一个点云并为每个平面着色
def combine_planes_with_colors(filled_planes):
    combined_cloud = o3d.geometry.PointCloud()

    for filled_plane in filled_planes:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(filled_plane)

        # 给每个平面着色，使用随机颜色或预定义颜色
        color = random_color()
        cloud.paint_uniform_color(color)

        # 合并点云
        combined_cloud += cloud

    return combined_cloud


def process_point_cloud(file_path, threshold_zero=0.1, threshold_one=0.1, threshold_neg_one=0.1):
    """
    处理点云数据，对平面进行分割和处理，将结果保存到同一文件夹，文件名后加 _plane。

    Parameters:
        file_path: 点云文件的路径 (.ply 格式)。
        threshold_zero: 接近 0 的阈值。
        threshold_one: 接近 1 的阈值。
        threshold_neg_one: 接近 -1 的阈值。

    Returns:
        保存处理后的点云文件，文件名后加 _plane。
    """
    # 读取点云数据 (.ply 格式)
    cloud = o3d.io.read_point_cloud(file_path)
    ds_pc = down_sampling(cloud)
    ds_num = len(ds_pc.points)

    # 平面拟合与法向量计算
    planes, planes_normal = fit_plane(ds_pc)

    # 替换法向量值
    for i in range(len(planes_normal)):
        for j in range(len(planes_normal[i])):
            if abs(planes_normal[i][j]) < threshold_zero:
                planes_normal[i][j] = 0
            elif abs(planes_normal[i][j] - 1) < threshold_one:
                planes_normal[i][j] = 1
            elif abs(planes_normal[i][j] + 1) < threshold_neg_one:
                planes_normal[i][j] = -1

    # 对每个平面进行均匀填充，仅对大于 1% 的平面进行处理
    filled_planes = [
        move_points_to_plane(plane, normal, num_points=ds_num)
        for plane, normal in zip(planes, planes_normal)
        if len(plane.points) / ds_num >= 0.01
    ]

    print("Len: ", len(filled_planes))

    # 将所有填充后的平面组合并仅保留外层点
    final_cloud = combine_planes_with_colors(filled_planes)

    # 进行统计滤波以去除噪声
    cl, ind = final_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    final_cloud_filtered = final_cloud.select_by_index(ind)

    # 生成输出文件路径
    dir_path, file_name = os.path.split(file_path)
    file_base, file_ext = os.path.splitext(file_name)
    output_file = os.path.join(dir_path, f"{file_base}_plane{file_ext}")

    # 保存处理后的结果
    o3d.io.write_point_cloud(output_file, final_cloud_filtered)
    print(f"Combined planes saved to {output_file}")


if __name__ == '__main__':
    # 输入和输出文件夹路径
    # input_folder = "/home/datasets/UrbanBIS/Qingdao/"
    #
    # json_file_path = "/home/code/Buildiffusion/data/Qingdao/filter.json"
    #
    # with open(json_file_path, 'r') as f:
    #     data = json.load(f)
    #
    # for folder, files in data.items():
    #     file_name = 'untitled.ply'
    #     obj_file_path = os.path.join("/home/datasets/UrbanBIS/Qingdao", folder, file_name)
    #     print(obj_file_path)
    #     process_point_cloud(obj_file_path)

    # 输入文件路径，用户可以在此指定单个点云文件路径
    file_path = "/home/code/Buildiffusion/outputs/debug/2024-10-26--12-28-30/sample/pred/buildings/pred.ply"

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
    else:
        print(f"Processing file: {file_path}")
        process_point_cloud(file_path)
















