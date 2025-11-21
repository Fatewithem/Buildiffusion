import open3d as o3d
import numpy as np

def transfer_color_by_nearest(gt_pcd: o3d.geometry.PointCloud, source_pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    # 构建 KDTree
    gt_kd_tree = o3d.geometry.KDTreeFlann(gt_pcd)

    # 获取 source 点的位置
    source_points = np.asarray(source_pcd.points)
    gt_points = np.asarray(gt_pcd.points)
    gt_colors = np.asarray(gt_pcd.colors)

    assigned_colors = []

    for point in source_points:
        # 查找最近邻
        _, idx, _ = gt_kd_tree.search_knn_vector_3d(point, 1)
        nearest_color = gt_colors[idx[0]]
        assigned_colors.append(nearest_color)

    # 创建一个新点云或直接赋予颜色
    source_pcd.colors = o3d.utility.Vector3dVector(np.array(assigned_colors))
    return source_pcd

import glob
import os

if __name__ == "__main__":
    # # 批量读取 result 目录下所有 GT 点云文件
    # gt_files = glob.glob("/home/code/Buildiffusion/result/*_gt.ply")
    # for gt_path in gt_files:
    #     # 推断对应的 source 文件
    #     source_path = gt_path.replace("_gt.ply", ".ply")
    #     if not os.path.exists(source_path):
    #         print(f"对应源点云未找到: {source_path}")
    #         continue
    #
    #     # 读取点云
    #     gt_pcd = o3d.io.read_point_cloud(gt_path)
    #     source_pcd = o3d.io.read_point_cloud(source_path)
    #
    #     # 检查 gt 是否包含颜色
    #     if not gt_pcd.has_colors():
    #         print(f"{gt_path} 不包含颜色，跳过")
    #         continue
    #
    #     # 转移颜色
    #     colored_source = transfer_color_by_nearest(gt_pcd, source_pcd)
    #
    #     # 构造保存路径，添加 _color 后缀
    #     save_path = source_path.replace(".ply", "_color.ply")
    #     o3d.io.write_point_cloud(save_path, colored_source)
    #     print(f"颜色转移完成，结果已保存为 {save_path}")

    # 👉 修改为你自己的路径
    gt_path = "/home/code/Buildiffusion/building4/untitled_fps.ply"  # 带颜色的 Ground Truth 点云
    source_path = "/home/code/Buildiffusion/result_plane/sample_7.ply"  # 需要转移颜色的点云

    # 检查文件是否存在
    if not os.path.exists(gt_path) or not os.path.exists(source_path):
        print("路径无效，请检查输入路径")
        exit()

    gt_pcd = o3d.io.read_point_cloud(gt_path)
    source_pcd = o3d.io.read_point_cloud(source_path)

    if not gt_pcd.has_colors():
        print(f"{gt_path} 不包含颜色，无法转移")
        exit()

    # 转移颜色并保存
    colored_pcd = transfer_color_by_nearest(gt_pcd, source_pcd)
    save_path = source_path.replace(".ply", "_color.ply")
    o3d.io.write_point_cloud(save_path, colored_pcd)

    print(f"颜色转移完成，结果保存为: {save_path}")
