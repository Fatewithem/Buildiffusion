from torch_cluster import knn
import open3d as o3d
import torch
import numpy as np
import os
import json


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


if __name__ == "__main__":
    # 输入和输出文件夹路径
    json_file_path = "/home/datasets/UrbanBIS/Qingdao/filter.json"

    print(f"Loading JSON data from {json_file_path}")
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    for folder, files in data.items():
        for file_name in files.keys():
            if file_name.endswith(".obj"):
                obj_file_path = os.path.join("/home/datasets/UrbanBIS/Qingdao", folder, file_name)
                ply_file_name = os.path.splitext(file_name)[0] + "_fps.ply"
                ply_file_fps_name = os.path.splitext(file_name)[0] + "_blur.ply"

                root = os.path.join("/home/datasets/UrbanBIS/Qingdao", folder)
                print(f"Root: {root}")
                ply_file_path = os.path.join(root, ply_file_name)
                ply_file_fps_path = os.path.join(root, ply_file_fps_name)

                # 检查.ply文件是否存在
                if not os.path.exists(ply_file_path):
                    print(f"{ply_file_path} does not exist! Skipping this file.")
                    continue  # 如果文件不存在，跳过

                print(f"Processing {ply_file_path}")

                # downsample_ply_with_fps(ply_file_path, ply_file_fps_path, 10000)

                # 示例调用
                process_ply_with_knn_blur(ply_file_path, ply_file_fps_path, k=10, noise_std=0.5)


