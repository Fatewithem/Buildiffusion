import open3d as o3d
import trimesh
import numpy as np


def reconstruct_mesh_from_ply(ply_path, output_path):
    # 读取 PLY 文件
    pcd = o3d.io.read_point_cloud(ply_path)

    # 估计法向量
    pcd.estimate_normals()

    # 进行泊松重建
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)

    # 保存为 OBJ 文件
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"Mesh saved to {output_path}")


if __name__ == "__main__":
    ply_file = "/home/code/Buildiffusion/result/sample_gt_0.ply"  # 输入的 PLY 文件路径
    output_file = "/home/code/Buildiffusion/result/sample_gt_0.obj"  # 输出的网格文件（可改为 .stl）
    reconstruct_mesh_from_ply(ply_file, output_file)
