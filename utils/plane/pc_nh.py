import open3d as o3d
import numpy as np
import random
import torch
import json
import os

from pytorch3d.ops import sample_farthest_points

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

    # 2) 估计平面法向量 (可从 plane_model 拿，也可从当前点云用SVD算)
    shifted = plane_points - centroid
    u, s, vh = torch.svd(shifted)  # 对老版本PyTorch可用torch.svd
    normal = vh[:, -1]  # 最后一列对应最小奇异值
    normal = normal / torch.norm(normal)  # 单位化

    # 3) 构造平面内的两个正交基 e1, e2
    ref = torch.tensor([0.0, 0.0, 1.0], device=plane_points.device)
    if torch.abs(torch.dot(ref, normal)) > 0.99:
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

    needed_points = target_num_points - num_points
    u_rand = torch.empty(needed_points, device=plane_points.device).uniform_(float(min_u), float(max_u))
    v_rand = torch.empty(needed_points, device=plane_points.device).uniform_(float(min_v), float(max_v))

    # 6) 将 (u_rand, v_rand) 映射回 3D 空间
    rand_shifted_3d = u_rand.unsqueeze(1) * e1 + v_rand.unsqueeze(1) * e2
    rand_points_3d = centroid.unsqueeze(0) + rand_shifted_3d

    # 7) 合并原平面点 + 随机补充点
    combined = torch.cat([plane_points, rand_points_3d], dim=0)

    # 8) 如果合并后超过 target_num_points，可再次下采样（此处可用 FPS 也可用随机采样）
    if combined.shape[0] > target_num_points:
        combined = farthest_point_sampling(combined, target_num_points)

    return combined


def farthest_point_sampling(points, num_samples):
    """
    使用 PyTorch3D 的 sample_farthest_points 实现的远点采样 (FPS)。

    参数:
    ----
    points: [N, 3] - 原始点云的坐标 (torch.Tensor)
    num_samples: int - 要采样的点数

    返回:
    ----
    torch.Tensor, 形状为 [num_samples, 3]
    """
    points_batch = points.unsqueeze(0)  # (1, N, 3)
    sampled_points, sampled_indices = sample_farthest_points(points_batch, K=num_samples)
    sampled_points = sampled_points.squeeze(0)
    return sampled_points


def pad_points_to_target(plane_points, target_num_points):
    """
    当 plane_points.shape[0] < target_num_points 时，
    在平面内进行差值/采样来“填充”。
    """
    num_points = plane_points.shape[0]
    if num_points == 0:
        return torch.zeros(target_num_points, 3).to(plane_points.device)
    elif num_points < target_num_points:
        return interpolate_points_in_plane(plane_points, target_num_points)
    else:
        return plane_points


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
        return torch.zeros(target_num_points, 3).to(plane_points.device)
    elif plane_points.shape[0] > target_num_points:
        return farthest_point_sampling(plane_points, target_num_points)
    else:
        return pad_points_to_target(plane_points, target_num_points)


def merge_point_clouds(pc1, pc2):
    """
    手动合并两个 PointCloud 的点和颜色。
    - 若两者都有颜色，则将颜色拼接；若只有坐标，就只合并坐标。
    """
    pts1 = np.asarray(pc1.points)
    pts2 = np.asarray(pc2.points)
    merged_points = np.vstack((pts1, pts2))

    has_color_1 = pc1.has_colors()
    has_color_2 = pc2.has_colors()
    merged_colors = None

    if has_color_1 and has_color_2:
        col1 = np.asarray(pc1.colors)
        col2 = np.asarray(pc2.colors)
        merged_colors = np.vstack((col1, col2))

    new_pc = o3d.geometry.PointCloud()
    new_pc.points = o3d.utility.Vector3dVector(merged_points)
    if merged_colors is not None:
        new_pc.colors = o3d.utility.Vector3dVector(merged_colors)

    return new_pc


import numpy as np
import open3d as o3d
import torch

def fit_plane(cloud, max_planes=10, target_num_points_list=None):
    cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

    planes = []
    plane_normals = []
    plane_centroids = []
    plane_inliers = []
    plane_equations = []

    nr_points = len(cloud.points)
    cloud_colors = np.asarray(cloud.colors)

    while len(cloud.points) > 0.1 * nr_points:
        plane_model, inliers = cloud.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
        if len(inliers) == 0:
            print("Could not estimate a planar model for the given dataset.")
            break

        inlier_cloud = cloud.select_by_index(inliers)
        inlier_colors = cloud_colors[inliers]
        inlier_cloud.colors = o3d.utility.Vector3dVector(inlier_colors)

        normal = plane_model[:3]
        centroid = np.mean(np.asarray(inlier_cloud.points), axis=0)
        merged = False

        for idx, existing_normal in enumerate(plane_normals):
            if are_normals_similar(normal, existing_normal) and are_planes_close(centroid, plane_centroids[idx], normal):
                # 合并点云（`merge_point_clouds()` 直接管理颜色）
                planes[idx] = merge_point_clouds(planes[idx], inlier_cloud)
                plane_inliers[idx] += len(inliers)
                merged = True
                break

        if not merged:
            planes.append(inlier_cloud)
            plane_inliers.append(len(inliers))
            plane_normals.append(normal)
            plane_centroids.append(centroid)
            plane_equations.append((*normal, -np.dot(normal, centroid)))

        cloud = cloud.select_by_index(inliers, invert=True)
        cloud_colors = np.delete(cloud_colors, inliers, axis=0)

    # 依据平面点数排序，并只保留前 max_planes 个
    sorted_planes = sorted(
        zip(planes, plane_inliers, plane_normals, plane_centroids, plane_equations),
        key=lambda x: x[1],
        reverse=True
    )[:max_planes]
    planes, plane_inliers, plane_normals, plane_centroids, plane_equations = zip(*sorted_planes)

    # 对每个平面进行调整（包括插值/下采样）
    adjusted_planes = []
    adjusted_plane_colors = []
    for idx, plane in enumerate(planes):
        plane_points = np.asarray(plane.points)
        plane_col = np.asarray(plane.colors)

        target_num_points = (
            target_num_points_list[idx]
            if (target_num_points_list and idx < len(target_num_points_list))
            else 100
        )

        # 获取下采样点的索引
        adjusted_plane_tensor, sampled_indices = downsample_points_with_indices(
            torch.tensor(plane_points, dtype=torch.float32),
            target_num_points
        )
        adjusted_plane_np = adjusted_plane_tensor.cpu().numpy()

        # 创建新的 o3d 点云
        new_o3d_plane = o3d.geometry.PointCloud()
        new_o3d_plane.points = o3d.utility.Vector3dVector(adjusted_plane_np)

        # 确保颜色匹配
        if len(plane_col) == len(plane_points):
            if len(adjusted_plane_np) > len(plane_points):
                avg_color = np.mean(plane_col, axis=0)
                new_colors = np.vstack((plane_col[sampled_indices], np.tile(avg_color, (len(adjusted_plane_np) - len(sampled_indices), 1))))
            else:
                new_colors = plane_col[sampled_indices]
            new_o3d_plane.colors = o3d.utility.Vector3dVector(new_colors)

        adjusted_planes.append(new_o3d_plane)
        adjusted_plane_colors.append(new_o3d_plane.colors)

    return list(adjusted_planes), list(plane_normals), list(plane_equations), list(adjusted_plane_colors)


def move_points_to_plane(plane_points, plane_normals, num_points, threshold_up=0.1):
    """
    移动点云到平面，同时保持颜色信息。
    - 如果点数占比 < threshold_up，则不移动，直接返回原始点和颜色
    - 否则，点云会被投影到平面
    """

    if len(plane_points.points) / num_points < threshold_up:
        if plane_points.has_colors():
            return np.asarray(plane_points.points), np.asarray(plane_points.colors)
        else:
            return np.asarray(plane_points.points)

    normal = plane_normals / np.linalg.norm(plane_normals)
    points_np = np.asarray(plane_points.points)
    heights = np.dot(points_np, normal)
    avg_height = np.mean(heights)

    # 计算投影点
    projected_points = points_np - np.outer(heights, normal)
    final_points = projected_points + avg_height * normal

    # 处理颜色
    if plane_points.has_colors():
        colors = np.asarray(plane_points.colors)
        return final_points, colors  # 返回点和颜色
    else:
        return final_points  # 仅返回点


def combine_planes_with_labels(filled_planes, plane_labels):
    """
    将多个平面的点云合并为一个 PointCloud，并返回对应的标签。
    这里也一并合并颜色（若有）。
    """
    all_points = []
    all_labels = []
    all_colors = []

    for filled_plane, label in zip(filled_planes, plane_labels):
        # 若它是 PointCloud
        if isinstance(filled_plane, o3d.geometry.PointCloud):
            points = np.asarray(filled_plane.points)
            if filled_plane.has_colors():
                colors = np.asarray(filled_plane.colors)
            else:
                colors = np.zeros((len(points), 3), dtype=np.float32)
        else:
            # 如果不是 o3d 点云，而是 array
            points = filled_plane
            colors = np.zeros((len(points), 3), dtype=np.float32)

        all_points.append(points)
        all_labels.extend([label] * len(points))
        all_colors.append(colors)

    combined_points = np.vstack(all_points)
    combined_labels = np.array(all_labels)
    combined_colors = np.vstack(all_colors)

    combined_cloud = o3d.geometry.PointCloud()
    combined_cloud.points = o3d.utility.Vector3dVector(combined_points)
    combined_cloud.colors = o3d.utility.Vector3dVector(combined_colors)

    return combined_cloud, combined_labels


def save_tensor_with_labels(final_cloud, final_labels, output_tensor_file):
    """
    将最终的点云(含颜色)和标签一起保存为 .pt 文件
    """
    points = np.asarray(final_cloud.points)
    colors = np.asarray(final_cloud.colors)  # 如果没有颜色，这里将是一个空数组
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
    """
    保存平面方程和对应标签到 JSON。
    """
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


def pad_final_cloud_with_color(final_cloud, target_num_points):
    """
    若最终合并后的点数不足 target_num_points，则补点并使用“全局平均颜色”进行填充。
    """
    points = np.asarray(final_cloud.points, dtype=np.float32)
    colors = np.asarray(final_cloud.colors, dtype=np.float32)

    points_tensor = torch.tensor(points, dtype=torch.float32)
    N = points_tensor.shape[0]
    if N >= target_num_points:
        # 已经够了，直接下采样
        sampled_pts = farthest_point_sampling(points_tensor, target_num_points)
        # 下采样索引若要映射颜色，需要修改 farthest_point_sampling 返回索引。此处简单处理：
        # 暂时不做颜色严格匹配，只示范同等长度
        # (若想严格FPS对应，可自行修改 farthest_point_sampling 函数以返回indices)
        new_cloud = o3d.geometry.PointCloud()
        new_cloud.points = o3d.utility.Vector3dVector(sampled_pts.cpu().numpy())
        # 为简单起见，这里先不匹配颜色，全部赋平均色
        avg_color = np.mean(colors, axis=0)
        new_colors = np.tile(avg_color, (target_num_points, 1))
        new_cloud.colors = o3d.utility.Vector3dVector(new_colors)
        return new_cloud
    else:
        # 需要补点
        needed_points = target_num_points - N
        padded_points = pad_points_to_target(points_tensor, target_num_points).cpu().numpy()
        # 原始部分颜色照旧
        new_colors = []
        for c in colors:
            new_colors.append(c)
        # 对补充的点使用平均颜色
        avg_color = np.mean(colors, axis=0) if len(colors) > 0 else np.array([0, 0, 0])
        for _ in range(needed_points):
            new_colors.append(avg_color)

        new_cloud = o3d.geometry.PointCloud()
        new_cloud.points = o3d.utility.Vector3dVector(padded_points)
        new_cloud.colors = o3d.utility.Vector3dVector(np.array(new_colors))
        return new_cloud


def process_point_cloud(file_path, threshold_zero=0.1, threshold_one=0.1, threshold_neg_one=0.1):
    """
    主处理流程：读取点云 -> 分割平面 -> 下采样/补点 -> 合并并移动 -> 最终输出
    """
    cloud = o3d.io.read_point_cloud(file_path)
    ds_num = len(cloud.points)

    # 先设置每个平面大致想要的点数
    target_num_points_list = [2000, 2000, 1000, 1000, 1000, 500, 500, 500, 250, 250]

    # 注意 fit_plane 实际返回5个值，这里要全部接收
    planes, planes_normal, plane_labels, plane_equations, plane_colors = fit_plane(
        cloud, max_planes=10, target_num_points_list=target_num_points_list
    )

    # 将法向量近似到 0、1 或 -1
    for i in range(len(planes_normal)):
        for j in range(len(planes_normal[i])):
            if abs(planes_normal[i][j]) < threshold_zero:
                planes_normal[i][j] = 0
            elif abs(planes_normal[i][j] - 1) < threshold_one:
                planes_normal[i][j] = 1
            elif abs(planes_normal[i][j] + 1) < threshold_neg_one:
                planes_normal[i][j] = -1

    # 对每个平面做 move_points_to_plane
    filled_planes, filled_normals = [], []
    for plane, normal in zip(planes, planes_normal):
        # 处理颜色：move_points_to_plane 可能返回 (moved_points, moved_colors)
        if plane.has_colors():
            moved_points, moved_colors = move_points_to_plane(plane, normal, num_points=ds_num)
        else:
            moved_points = move_points_to_plane(plane, normal, num_points=ds_num)
            moved_colors = None  # 无颜色情况

        # 如果 plane 是 open3d 点云，并且有颜色，需要重新构造
        if isinstance(plane, o3d.geometry.PointCloud()):
            new_pc = o3d.geometry.PointCloud()
            new_pc.points = o3d.utility.Vector3dVector(moved_points)

            if moved_colors is not None:
                if len(moved_colors) == len(moved_points):
                    new_pc.colors = o3d.utility.Vector3dVector(moved_colors)
                else:
                    # 颜色数量不匹配 -> 用最近邻插值或平均色填充
                    avg_color = np.mean(moved_colors, axis=0)
                    filled_colors = np.tile(avg_color, (len(moved_points), 1))
                    new_pc.colors = o3d.utility.Vector3dVector(filled_colors)

            filled_planes.append(new_pc)
        else:
            filled_planes.append(moved_points)

        filled_normals.append(normal)

    # 如果想对最终合并后的点云整体也做一次补点/下采样, 例如补到9000
    target_num_points_final = 9000
    final_cloud = pad_final_cloud_with_color(final_cloud, target_num_points_final)

    # 保存到指定文件
    dir_path, file_name = os.path.split(file_path)
    file_base, file_ext = os.path.splitext(file_name)
    output_file = os.path.join(dir_path, f"untitled_plane.pt")
    output_json_file = os.path.join(dir_path, f"untitled_plane_info.json")

    # 将最终点云（含颜色）与标签保存
    save_tensor_with_labels(final_cloud, final_labels, output_file)
    # 将平面方程信息保存到json
    save_plane_info_to_json(plane_labels, plane_equations, output_json_file)
    print("Final processing done!")


if __name__ == '__main__':
    # 测试示例
    test_path = "/home/datasets/UrbanBIS/Qingdao/3-0205-0228-8/building/building1/untitled_fps.ply"
    process_point_cloud(test_path)

    # 其他批处理逻辑可按需添加，如：
    # input_folder = "/home/datasets/UrbanBIS/Qingdao/"
    # json_file_path = "/home/datasets/UrbanBIS/Qingdao/filter.json"
    # with open(json_file_path, 'r') as f:
    #     data = json.load(f)
    # for folder, files in data.items():
    #     file_name = 'untitled_fps.ply'
    #     obj_file_path = os.path.join(input_folder, folder, file_name)
    #     print("Processing:", obj_file_path)
    #     process_point_cloud(obj_file_path)