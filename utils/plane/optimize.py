import torch
import numpy as np
import os
import open3d as o3d

# 配置路径
input_pt_file = "/home/code/Buildiffusion/result/sample_0.pt"  # 原始点云路径
output_dir = "/home/code/Buildiffusion/test/"  # 存放拆分后的 PLY 目录
merged_ply_file = "/home/code/Buildiffusion/test/merged_output.ply"  # 合并后的 PLY 文件
merged_pt_file = "/home/code/Buildiffusion/test/merged_output.pt"  # 合并后的 PyTorch Tensor 文件

os.makedirs(output_dir, exist_ok=True)

# 拆分点云的大小
split_sizes = [2000, 2000, 1000, 1000, 1000, 500, 500, 500, 250, 250]

# 预设颜色列表（10个不同颜色）
colors = [
    [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1],
    [0, 1, 1], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5], [0.3, 0.7, 0.2]
]

# 1. 加载 .pt 文件
pointcloud = torch.load(input_pt_file)  # 可能是 (1, 9000, 3)
pointcloud = pointcloud.squeeze(0)  # 确保形状为 [9000, 3]

pointcloud = pointcloud.cpu().numpy()  # 转换为 NumPy 数组
assert pointcloud.shape == (9000, 3), f"点云数据形状错误: {pointcloud.shape}"

# 2. 拆分点云并保存
start_idx = 0
all_sub_clouds = []  # 存储所有子点云
all_colors = []  # 存储所有颜色

for i, size in enumerate(split_sizes):
    sub_cloud = pointcloud[start_idx:start_idx + size]  # 取子点云
    color = np.tile(colors[i], (sub_cloud.shape[0], 1))  # 赋予颜色
    colored_sub_cloud = np.hstack((sub_cloud, color))  # 变为 [N, 6] 格式
    start_idx += size

    # 3. 创建 Open3D 点云对象并保存为 PLY
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(colored_sub_cloud[:, :3])  # 取 XYZ 坐标
    pcd.colors = o3d.utility.Vector3dVector(colored_sub_cloud[:, 3:])  # 取颜色信息
    output_ply_file = os.path.join(output_dir, f"split_{i}.ply")
    o3d.io.write_point_cloud(output_ply_file, pcd)

    all_sub_clouds.append(sub_cloud)  # 存储当前子点云
    all_colors.append(color)  # 存储当前子点云颜色
    print(f"子点云 {i} 已保存为: {output_ply_file}")

# 4. 合并所有子点云（包括颜色）
merged_points = np.vstack(all_sub_clouds)  # 合并所有子点云
merged_colors = np.vstack(all_colors)  # 合并所有颜色

# 5. 保存合并后的点云（带颜色）
merged_pcd = o3d.geometry.PointCloud()
merged_pcd.points = o3d.utility.Vector3dVector(merged_points)
merged_pcd.colors = o3d.utility.Vector3dVector(merged_colors)  # 赋予颜色
o3d.io.write_point_cloud(merged_ply_file, merged_pcd)
print(f"✅ 合并后的点云已保存为: {merged_ply_file}")

# 6. 保存合并后的点云为 PyTorch Tensor
merged_tensor = torch.tensor(merged_points, dtype=torch.float32)
merged_colors_tensor = torch.tensor(merged_colors, dtype=torch.float32)
torch.save({'points': merged_tensor, 'colors': merged_colors_tensor}, merged_pt_file)
print(f"✅ 合并后的点云已保存为 PyTorch Tensor: {merged_pt_file}")