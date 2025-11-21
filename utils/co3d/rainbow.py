import open3d as o3d
import numpy as np
import colorsys

# === 加载点云 ===
pcd = o3d.io.read_point_cloud("/home/code/Buildiffusion/result/sample_0.ply")  # 替换路径

# 获取点的坐标
points = np.asarray(pcd.points)

# 颜色锚点（立方体 8 个角的颜色）
anchors = np.array([
    [1.0, 0.0, 0.0],  # Red
    [0.0, 1.0, 0.0],  # Green
    [0.0, 0.0, 1.0],  # Blue
    [1.0, 1.0, 0.0],  # Yellow
    [0.0, 1.0, 1.0],  # Cyan
    [1.0, 0.0, 1.0],  # Magenta
    [1.0, 1.0, 1.0],  # White
    [1.0, 0.5, 0.0],  # Orange
])

# 归一化坐标到 [0, 1]
pc_min = points.min(axis=0)
pc_max = points.max(axis=0)
pc_range = pc_max - pc_min + 1e-8
normed = (points - pc_min) / pc_range

# 计算相对于8个锚点的距离的反比加权
from itertools import product
corner_coords = np.array(list(product([0, 1], repeat=3)))  # 8 corners
scale = 1.0  # 可调节的比例系数，越小则颜色插值越趋于均匀
weights = []
for p in normed:
    dists = np.linalg.norm(corner_coords - p, axis=1) / scale
    w = 1.0 / (dists + 1e-6)
    w /= w.sum()
    weights.append(w)
weights = np.array(weights)
colors = weights @ anchors
colors = np.clip(colors * 1.0, 0, 1)
colors = np.power(colors, 1)

pcd.colors = o3d.utility.Vector3dVector(colors)

# === 保存彩色点云 ===
o3d.io.write_point_cloud("/home/code/Buildiffusion/result/sample_0_colored.ply", pcd)