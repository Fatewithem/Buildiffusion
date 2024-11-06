import open3d as o3d
import numpy as np

# 1. 加载带有材质的 obj 文件
mesh = o3d.io.read_triangle_mesh("/home/datasets/UrbanBIS/Qingdao/0005-0019/building/building1/untitled.obj")
mesh.compute_vertex_normals()

# 2. 进行表面采样，生成点云
pcd = mesh.sample_points_uniformly(number_of_points=100000)

# 3. 如果需要颜色信息，请确保 obj 文件中包含颜色信息
if mesh.has_vertex_colors():
    colors = np.asarray(mesh.vertex_colors)
    points = np.asarray(pcd.points)
    pcd.colors = o3d.utility.Vector3dVector(colors[:len(points)])

# 5. 保存生成的点云为 PLY 格式
o3d.io.write_point_cloud("output.ply", pcd)