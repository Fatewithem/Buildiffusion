import open3d as o3d
import numpy as np
import random
import torch
import json
import os

# ============== 新增：在平面上插值/补充点的函数 ==============

def interpolate_points_in_plane(plane_points, target_num_points):
    """
    若 plane_points 点数不足 target_num_points 时，
    根据该平面的法向量及质心，在平面内生成新的点进行补充。

    plane_points: [N, 3] Tensor
    target_num_points: int
    """
    num_points = plane_points.shape[0]
    if num_points == 0:
        # 如果平面上原本就没有点，则可直接返回 0 点云（也可以默认在原点附近补）
        return torch.zeros(target_num_points, 3).to(plane_points.device)

    # 1) 计算质心
    centroid = torch.mean(plane_points, dim=0)  # shape: (3,)

    # 2) 估计平面法向量 (可直接从分割时的 plane_model 拿，也可从当前点云用SVD算)
    #   这里用SVD方式：
    shifted = plane_points - centroid
    u, s, vh = torch.svd(shifted)  # PyTorch <= 1.9 可以用 torch.svd；之后是 torch.linalg.svd
    normal = vh[:, -1]  # 最后一列对应最小奇异值
    normal = normal / torch.norm(normal)  # 单位化

    # 3) 构造平面内的两个正交基 e1, e2
    #   随意找一个与 normal 不平行的向量做叉积
    #   这里选一个简单的参考向量 (0,0,1) 或 (1,0,0)，根据情况防止叉积为 0 向量
    ref = torch.tensor([0.0, 0.0, 1.0], device=plane_points.device)
    if torch.abs(torch.dot(ref, normal)) > 0.99:  # 若太接近，则换一个
        ref = torch.tensor([1.0, 0.0, 0.0], device=plane_points.device)

    e1 = torch.cross(normal, ref)
    e1 = e1 / torch.norm(e1)
    e2 = torch.cross(normal, e1)
    e2 = e2 / torch.norm(e2)

    # 4) 将平面点投影到 (u, v) 2D 坐标
    uv = torch.stack([
        torch.sum(shifted * e1, dim=1),
        torch.sum(shifted * e2, dim=1)
    ], dim=1)  # shape: (N, 2)

    # 5) 获取 uv 的最小外接矩形范围
    min_u, _ = torch.min(uv[:, 0], dim=0)
    max_u, _ = torch.max(uv[:, 0], dim=0)
    min_v, _ = torch.min(uv[:, 1], dim=0)
    max_v, _ = torch.max(uv[:, 1], dim=0)

    # 需要补充的点数
    needed_points = target_num_points - num_points
    # 若 needed_points < 0，则说明原点数超过目标点数，此时不在这里裁剪，而由外部自行处理

    # 6) 在 [min_u, max_u] x [min_v, max_v] 中做均匀随机采样
    u_rand = torch.empty(needed_points, device=plane_points.device).uniform_(float(min_u), float(max_u))
    v_rand = torch.empty(needed_points, device=plane_points.device).uniform_(float(min_v), float(max_v))

    # 7) 将 (u_rand, v_rand) 映射回 3D 空间
    rand_shifted_3d = u_rand.unsqueeze(1) * e1 + v_rand.unsqueeze(1) * e2  # shape: (needed_points, 3)
    rand_points_3d = centroid.unsqueeze(0) + rand_shifted_3d

    # 8) 合并原平面点 + 随机补充点
    combined = torch.cat([plane_points, rand_points_3d], dim=0)

    # 9) 如果合并后超过 target_num_points，可再次下采样（此处可用 FPS 也可用随机采样）
    if combined.shape[0] > target_num_points:
        combined = farthest_point_sampling(combined, target_num_points)

    return combined


def farthest_point_sampling(points, num_samples):
    """
    从点云中采样指定数量的点，使用 Farthest Point Sampling (FPS)。
    points: [N, 3] - 原始点云的坐标 (Tensor)
    num_samples: int - 要采样的点数
    """
    device = points.device
    N = points.shape[0]
    centroids = torch.zeros(num_samples, dtype=torch.long, device=device)
    distance = torch.ones(N, device=device) * 1e10
    farthest_pts = torch.randint(0, N, (1,), device=device)  # 随机选择第一个点

    for i in range(num_samples):
        centroids[i] = farthest_pts[0].item()
        dist = torch.sum((points - points[farthest_pts[0]])**2, dim=1)
        distance = torch.min(distance, dist)
        farthest_pts = torch.argmax(distance, dim=0, keepdim=True)
    return points[centroids]


# ============== 将原代码的 pad_points_to_target 改为插值逻辑 ==============

def pad_points_to_target(plane_points, target_num_points):
    """
    当 plane_points.shape[0] < target_num_points 时，
    在平面内进行差值/采样来“填充”。
    """
    num_points = plane_points.shape[0]
    if num_points == 0:
        # 如果平面本身没有点，可以直接返回一个 0 点云
        return torch.zeros(target_num_points, 3).to(plane_points.device)
    elif num_points < target_num_points:
        # 改为根据平面内插值生成新的点来补
        return interpolate_points_in_plane(plane_points, target_num_points)
    else:
        # 不做额外处理，直接返回原有点
        return plane_points


# ======================== 下面是你的其它函数，改动较少 ========================

def compute_centroid(points):
    return np.mean(points, axis=0)

def are_planes_close(centroid1, centroid2, normal, distance_threshold=3):
    projected_distance = np.abs(np.dot((centroid2 - centroid1), normal))
    return abs(projected_distance) < distance_threshold

def are_normals_similar(normal1, normal2, threshold=np.pi / 12):
    angle = np.arccos(np.clip(np.dot(normal1, normal2), -1.0, 1.0))
    return angle < threshold

def downsample_points(plane_points, target_num_points):
    """
    对平面进行下采样或插值补点，返回指定数量的点。
    """
    if plane_points.shape[0] == 0:
        # 如果平面没有点，返回一个全为零的点云
        return torch.zeros(target_num_points, 3).to(plane_points.device)
    elif plane_points.shape[0] > target_num_points:
        # 如果平面点数超过目标数量，则进行下采样
        return farthest_point_sampling(plane_points, target_num_points)
    else:
        # 如果点数不足目标数量，则进行插值/补点
        return pad_points_to_target(plane_points, target_num_points)


def fit_plane(cloud, max_planes=10, target_num_points_list=None):
    print(f"PointCloud before filtering has: {len(cloud.points)} data points.")
    cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

    planes, plane_normals, plane_centroids, plane_inliers, plane_labels, plane_equations = [], [], [], [], [], []
    colors = [
        [1, 0, 0], [0, 1, 0],
        [0, 0, 1], [1, 1, 0],
        [1, 0, 1], [0, 1, 1]
    ]
    nr_points = len(cloud.points)
    plane_id = 0

    while len(cloud.points) > 0.1 * nr_points:
        plane_model, inliers = cloud.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
        if len(inliers) == 0:
            print("Could not estimate a planar model for the given dataset.")
            break

        inlier_cloud = cloud.select_by_index(inliers)
        normal = plane_model[:3]
        centroid = compute_centroid(np.asarray(inlier_cloud.points))
        similar_normals = [normal]
        merged = False

        for idx, existing_normal in enumerate(plane_normals):
            if are_normals_similar(normal, existing_normal) and are_planes_close(centroid, plane_centroids[idx], normal):
                inlier_cloud.paint_uniform_color(colors[idx % len(colors)])
                planes[idx] += inlier_cloud
                similar_normals.append(plane_normals[idx])
                plane_inliers[idx] += len(inliers)
                merged = True
                break

        if not merged:
            color_idx = plane_id % len(colors)
            inlier_cloud.paint_uniform_color(colors[color_idx])
            planes.append(inlier_cloud)
            plane_inliers.append(len(inliers))
            average_normal = np.mean(similar_normals, axis=0)
            plane_normals.append(average_normal)
            plane_centroids.append(centroid)
            plane_labels.append(plane_id)  # 给平面分配标签

            # 计算平面方程的偏移量 D
            A, B, C = average_normal
            D = -np.dot(average_normal, centroid)
            plane_equations.append((A, B, C, D))

            plane_id += 1

        cloud = cloud.select_by_index(inliers, invert=True)

    # 对平面根据点数进行排序，选择前 max_planes 个点数最大的平面
    sorted_planes = sorted(zip(planes, plane_inliers, plane_normals, plane_centroids, plane_labels, plane_equations),
                           key=lambda x: x[1], reverse=True)
    planes, plane_inliers, plane_normals, plane_centroids, plane_labels, plane_equations = zip(*sorted_planes[:max_planes])

    # 重新给 plane_labels 从 0 开始排序
    plane_labels = list(range(len(plane_labels)))

    # 对每个平面进行点云数量调整（包括插值补点）
    adjusted_planes = []
    for idx, plane in enumerate(planes):
        plane_points = np.asarray(plane.points)
        target_num_points = target_num_points_list[idx] if (target_num_points_list and idx < len(target_num_points_list)) else 100
        adjusted_plane = downsample_points(torch.tensor(plane_points, dtype=torch.float32), target_num_points)
        new_o3d_plane = o3d.geometry.PointCloud()
        new_o3d_plane.points = o3d.utility.Vector3dVector(adjusted_plane.cpu().numpy())
        adjusted_planes.append(new_o3d_plane)

    return list(adjusted_planes), list(plane_normals), list(plane_labels), list(plane_equations)

def move_points_to_plane(plane_points, plane_normals, num_points, threshold_up=0.1):
    if len(plane_points.points) / num_points < threshold_up:
        return np.asarray(plane_points.points)

    normal = plane_normals / np.linalg.norm(plane_normals)
    points_np = np.asarray(plane_points.points)
    heights = np.dot(points_np, normal)
    avg_height = np.mean(heights)
    projected_points = points_np - np.outer(heights, normal)
    return projected_points + avg_height * normal

def random_color():
    return [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]

def combine_planes_with_labels(filled_planes, plane_labels):
    all_points = []  # 存储所有点
    all_labels = []  # 存储标签

    for filled_plane, label in zip(filled_planes, plane_labels):
        if isinstance(filled_plane, o3d.geometry.PointCloud):
            points = np.asarray(filled_plane.points)
        else:
            points = filled_plane
        all_points.append(points)
        all_labels.extend([label] * len(points))

    combined_points = np.vstack(all_points)
    combined_labels = np.array(all_labels)

    combined_cloud = o3d.geometry.PointCloud()
    combined_cloud.points = o3d.utility.Vector3dVector(combined_points)
    return combined_cloud, combined_labels

def save_tensor_with_labels(final_cloud, final_labels, output_tensor_file):
    points = np.asarray(final_cloud.points)
    colors = np.asarray(final_cloud.colors)
    points_tensor = torch.tensor(points, dtype=torch.float32)
    colors_tensor = torch.tensor(colors, dtype=torch.float32)
    labels_tensor = torch.tensor(final_labels, dtype=torch.int32)

    torch.save({
        'points': points_tensor,
        'colors': colors_tensor,
        'labels': labels_tensor
    }, output_tensor_file)

    print(f"✅ 点云数据已保存为 Tensor 格式：{output_tensor_file}！")

def save_plane_info_to_json(plane_labels, plane_equations, output_json_path):
    sorted_indices = np.argsort(plane_labels)
    plane_labels_sorted = [plane_labels[i] for i in sorted_indices]
    plane_equations_sorted = [plane_equations[i] for i in sorted_indices]

    data = []
    for label, equation in zip(plane_labels_sorted, plane_equations_sorted):
        data.append({
            "label": label,
            "plane_equation": {
                "A": equation[0],
                "B": equation[1],
                "C": equation[2],
                "D": equation[3]
            }
        })

    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Plane information saved to {output_json_path}")

def process_point_cloud(file_path, threshold_zero=0.1, threshold_one=0.1, threshold_neg_one=0.1):
    cloud = o3d.io.read_point_cloud(file_path)
    ds_num = len(cloud.points)

    target_num_points_list = [2000, 2000, 1000, 1000, 1000, 500, 500, 500, 250, 250]
    planes, planes_normal, plane_labels, plane_equations = fit_plane(
        cloud, max_planes=10, target_num_points_list=target_num_points_list
    )

    # 根据给定阈值，将平面法向量近似到 0、1 或 -1
    for i in range(len(planes_normal)):
        for j in range(len(planes_normal[i])):
            if abs(planes_normal[i][j]) < threshold_zero:
                planes_normal[i][j] = 0
            elif abs(planes_normal[i][j] - 1) < threshold_one:
                planes_normal[i][j] = 1
            elif abs(planes_normal[i][j] + 1) < threshold_neg_one:
                planes_normal[i][j] = -1

    filled_planes, filled_normals = [], []
    for plane, normal in zip(planes, planes_normal):
        if len(plane.points) / ds_num >= 0.01:
            # 将该平面点云移动到某个高度
            moved = move_points_to_plane(plane, normal, num_points=ds_num)
            filled_planes.append(moved)
            filled_normals.append(normal)

    final_cloud, final_labels = combine_planes_with_labels(filled_planes, plane_labels)

    final_points = np.asarray(final_cloud.points)
    final_points_tensor = torch.tensor(final_points, dtype=torch.float32)

    # 如果想在最终点云上再做一次统一填充到 9000，可根据需求选择
    target_num_points = 9000
    if final_points_tensor.shape[0] < target_num_points:
        # 这里也可以用插值逻辑，示例中依旧沿用 pad 函数
        final_points_tensor = pad_points_to_target(final_points_tensor, target_num_points)

    print("Final output shape:", final_points_tensor.shape)

    dir_path, file_name = os.path.split(file_path)
    file_base, file_ext = os.path.splitext(file_name)
    output_file = os.path.join(dir_path, f"untitled_plane.pt")
    output_json_file = os.path.join(dir_path, f"untitled_plane_info.json")

    # 仅保存点和标签
    torch.save({
        'points': final_points_tensor,
        'labels': torch.tensor(final_labels, dtype=torch.int32)
    }, output_file)
    print(f"Combined planes saved to {output_file}")

    save_plane_info_to_json(plane_labels, plane_equations, output_json_file)


if __name__ == '__main__':
    # 示例主函数
    input_folder = "/home/datasets/UrbanBIS/Lihu/"
    json_file_path = "/home/datasets/UrbanBIS/Lihu/filter.json"

    with open(json_file_path, 'r') as f:
        data = json.load(f)

    for folder, files in data.items():
        file_name = 'untitled_fps.ply'
        obj_file_path = os.path.join("/home/datasets/UrbanBIS/Lihu", folder, file_name)
        print("Processing:", obj_file_path)
        process_point_cloud(obj_file_path)