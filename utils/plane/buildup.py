import numpy as np
import itertools
import open3d as o3d
from scipy.spatial import ConvexHull

# 你的 10 个平面方程 (ax + by + cz + d = 0)
planes = [
    [1, 0, 0, -10],   # 例如：x = 10
    [-1, 0, 0, 0],    # x = 0
    [0, 1, 0, -10],   # y = 10
    [0, -1, 0, 0],    # y = 0
    [0, 0, 1, -10],   # z = 10
    [0.5, 0.5, -1, 5], # 斜面示例
    [-0.5, 0.5, -1, 8], # 斜面示例
    [0.5, -0.5, -1, 7], # 斜面示例
    [-0.5, -0.5, -1, 6], # 斜面示例
    [0, 0, 1, 0]       # z = 0 (底面)
]

# 存储所有交点
vertices = []

# 遍历所有 3 个平面组合，求交点
for p1, p2, p3 in itertools.combinations(planes, 3):
    A = np.array([p1[:3], p2[:3], p3[:3]])  # 取法向量
    b = np.array([-p1[3], -p2[3], -p3[3]])  # 取 d 值

    # 检查行列式，避免奇异矩阵
    if np.linalg.det(A) != 0:
        # 计算交点
        point = np.linalg.solve(A, b)

        # 只保留 z >= 0 的点（位于最底层平面上方）
        if point[2] >= 0:
            vertices.append(point)

# 转换为 NumPy 数组
vertices = np.array(vertices)

# 计算 3D 包围盒（凸包）
hull = ConvexHull(vertices)

# 获取包围盒顶点
bounding_box_vertices = vertices[hull.vertices]

# 1. 生成 Open3D 点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(bounding_box_vertices)

# 2. 计算 Alpha Shape 表面重建 (适用于凹凸多面体)
alpha = 0.2  # Alpha 值，控制细节保留度
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)

# 3. 生成法向量
mesh.compute_vertex_normals()

# 4. 保存为 PLY 文件
output_ply_file = "/home/code/Buildiffusion/sample.ply"
o3d.io.write_triangle_mesh(output_ply_file, mesh)

print(f"表面重建后的 PLY 文件已保存到: {output_ply_file}")

# 5. 可视化
o3d.visualization.draw_geometries([mesh])