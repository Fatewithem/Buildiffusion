import numpy as np
import open3d as o3d
import json
import os


def farthest_point_sampling(points, num_samples):
    """
    Perform Farthest Point Sampling (FPS) on a set of points.

    :param points: Input point cloud as a numpy array of shape (N, 3).
    :param num_samples: Number of points to sample.
    :return: Sampled points as a numpy array of shape (num_samples, 3), and sampled indices.
    """
    N = points.shape[0]
    sampled_points = np.zeros((num_samples, 3))
    sampled_indices = []

    # Initialize: randomly select the first point
    farthest_index = np.random.randint(0, N)
    sampled_points[0] = points[farthest_index]
    sampled_indices.append(farthest_index)

    # Initialize distances
    distances = np.full(N, np.inf)

    for i in range(1, num_samples):
        # Update distances: min distance to any sampled point
        dist_to_new_point = np.linalg.norm(points - points[farthest_index], axis=1)
        distances = np.minimum(distances, dist_to_new_point)

        # Find the farthest point
        farthest_index = np.argmax(distances)
        sampled_points[i] = points[farthest_index]
        sampled_indices.append(farthest_index)

    return sampled_points, np.array(sampled_indices)


def downsample_ply_with_fps(input_ply_path, output_ply_path, num_samples):
    """
    使用最远点采样（FPS）来降采样 PLY 文件，并尽可能保留颜色信息。

    :param input_ply_path: 输入 PLY 文件路径。
    :param output_ply_path: 输出降采样 PLY 文件路径。
    :param num_samples: 要采样的点数。
    """
    try:
        # 加载 PLY 文件
        pcd = o3d.io.read_point_cloud(input_ply_path)
        points = np.asarray(pcd.points)

        if len(points) == 0:
            print("PLY 文件中没有点数据。")
            return None

        print(f"原始点的数量: {len(points)}")

        # 如果点云包含颜色信息，则提取颜色
        colors = None
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
            print("检测到颜色信息，采样后将保留颜色。")

        # 执行 FPS，得到采样后的点和采样时选中的索引
        sampled_points, sampled_indices = farthest_point_sampling(points, num_samples)

        # 创建一个新的点云对象，并使用采样后的点
        downsampled_pcd = o3d.geometry.PointCloud()
        downsampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)

        # 如果原点云有颜色信息，则根据索引取出对应的颜色
        if colors is not None:
            sampled_colors = colors[sampled_indices]
            downsampled_pcd.colors = o3d.utility.Vector3dVector(sampled_colors)

        # 保存降采样后的点云文件
        o3d.io.write_point_cloud(output_ply_path, downsampled_pcd)
        print(f"FPS 之后的点数: {num_samples}")
        print(f"降采样后的 PLY 文件已保存至: {output_ply_path}")

        return downsampled_pcd

    except Exception as e:
        print(f"处理 PLY 文件时出错: {e}")
        return None


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
                ply_file_name = os.path.splitext(file_name)[0] + ".ply"
                ply_file_fps_name = os.path.splitext(file_name)[0] + "_fps.ply"

                root = os.path.join("/home/datasets/UrbanBIS/Qingdao", folder)
                print(f"Root: {root}")
                ply_file_path = os.path.join(root, ply_file_name)
                ply_file_fps_path = os.path.join(root, ply_file_fps_name)

                # 检查.ply文件是否存在
                if not os.path.exists(ply_file_path):
                    print(f"{ply_file_path} does not exist! Skipping this file.")
                    continue  # 如果文件不存在，跳过

                print(f"Processing {ply_file_path}")

                # 调用转换函数
                downsample_ply_with_fps(ply_file_path, ply_file_fps_path, 10000)


# Example usage
# input_ply = "/home/code/Blender/untitled.ply"  # Replace with your input PLY file path
# output_ply = "/home/code/Blender/untitled_fps.ply"  # Replace with the desired output PLY file path
# num_samples = 2048  # Adjust the number of points to sample
# downsample_ply_with_fps(input_ply, output_ply, num_samples)