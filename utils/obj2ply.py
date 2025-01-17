import os
import trimesh
import json
import open3d as o3d
import numpy as np


def calculate_bounding_box_center(mesh):
    """
    计算网格的包围盒中心
    :param mesh: Trimesh 对象
    :return: 包围盒中心坐标
    """
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2.0  # 取包围盒的中点
    return center


def center_and_scale_object(mesh, target_scale):
    """
    将3D对象根据包围盒中心进行居中并缩放到目标尺寸。
    :param mesh: Trimesh 对象
    :param target_scale: 目标缩放尺寸，所有对象的最大维度将被缩放到此大小
    :return: 居中并缩放后的 Trimesh 对象
    """
    # 计算包围盒中心
    bounding_box_center = calculate_bounding_box_center(mesh)

    # 将对象移到原点（基于包围盒中心）
    translation_vector = -bounding_box_center
    mesh.apply_translation(translation_vector)

    # 获取对象的包围盒尺寸
    dimensions = mesh.extents  # 获取 X、Y、Z 方向的包围盒尺寸
    max_dimension = max(dimensions)

    # 计算缩放因子
    scale_factor = target_scale / max_dimension

    # 对网格的顶点应用缩放
    mesh.apply_scale(scale_factor)

    return mesh


def convert_obj_to_point_cloud(obj_file_path, ply_file_path, target_scale, num_points=50000):
    """将OBJ文件转换为点云，并保存为PLY文件。"""
    # 加载OBJ文件
    mesh = trimesh.load(obj_file_path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        print("Failed to load the OBJ file. Please check the file path and format.")
        return

    # 居中和缩放3D对象
    mesh = center_and_scale_object(mesh, target_scale)

    # 创建Open3D三角网格对象
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces)

    # 检查UV数据和材质图像
    if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None and \
            hasattr(mesh.visual.material, 'image') and mesh.visual.material.image is not None:
        # 应用纹理
        vertex_colors = trimesh.visual.color.uv_to_color(
            mesh.visual.uv, mesh.visual.material.image
        )[:, :3] / 255.0  # 转换为0-1范围的RGB
        mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        print("Texture applied successfully.")
    else:
        print("No texture or vertex colors found. Applying default color.")
        # 默认颜色设置为白色
        mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(
            np.ones((len(mesh.vertices), 3))
        )

    # 均匀采样点云
    point_cloud = mesh_o3d.sample_points_poisson_disk(num_points)

    # 保存为PLY文件
    o3d.io.write_point_cloud(ply_file_path, point_cloud)
    print(f"Successfully converted {obj_file_path} to {ply_file_path}")


if __name__ == "__main__":
    # 输入和输出文件夹路径
    input_folder = "/home/datasets/UrbanBIS/Qingdao/0005-0019"

    json_file_path = "/home/code/Buildiffusion/data/Lihu/filter.json"
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    for folder, files in data.items():
        for file_name in files.keys():
            if file_name.endswith(".obj"):
                obj_file_path = os.path.join("/home/datasets/UrbanBIS/Lihu", folder, file_name)
                ply_file_name = os.path.splitext(file_name)[0] + ".ply"

                root = os.path.join("/home/datasets/UrbanBIS/Lihu", folder)
                print(f"Root: {root}")
                ply_file_path = os.path.join(root, ply_file_name)

                # 检查.ply文件是否存在
                if os.path.exists(ply_file_path):
                    print(f"{ply_file_path} already exists. Skipping conversion.")
                    continue  # 如果存在，跳过转换

                # 调用转换函数
                convert_obj_to_point_cloud(obj_file_path, ply_file_path, 20.0)
    # convert_obj_to_point_cloud("/home/code/Blender/data/untitled.obj",
    #                            "/home/code/Blender/data/processed_object_0.ply",
    #                            10.0)