import inspect
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
import open3d as o3d
# from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
# from diffusers.schedulers.scheduling_ddim import DDIMScheduler
# from diffusers.schedulers.scheduling_pndm import PNDMScheduler
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Pointclouds


def load_ply_as_pointclouds(ply_path):
    # 使用 Open3D 读取 PLY 文件
    pcd = o3d.io.read_point_cloud(ply_path)

    # 提取点坐标 (Nx3)
    points = torch.tensor(pcd.points, dtype=torch.float32)

    # 提取颜色信息 (Nx3) 如果没有颜色信息，默认设为白色
    if pcd.has_colors():
        colors = torch.tensor(pcd.colors, dtype=torch.float32)
    else:
        colors = torch.ones_like(points)  # 全部设为白色 (1, 1, 1)

    # 创建 PyTorch3D 的 Pointclouds 对象
    point_cloud = Pointclouds(points=[points], features=[colors])

    return point_cloud


def save_pointclouds_as_ply(point_cloud: Pointclouds, file_path: str):
    """
    将 PyTorch3D 的 Pointclouds 对象保存为 .ply 文件。

    Args:
        point_cloud: PyTorch3D 的 Pointclouds 对象。
        file_path: 保存的 .ply 文件路径。
    """
    # 提取点云的点 (Nx3)
    points = point_cloud.points_padded()[0].cpu().numpy()  # 提取第一个点云并转为 NumPy

    # 提取颜色信息 (Nx3)，若无颜色则默认设为白色
    if point_cloud.features_padded() is not None:
        colors = point_cloud.features_padded()[0].cpu().numpy()
    else:
        colors = torch.ones_like(point_cloud.points_padded()[0]).cpu().numpy()  # 白色

    # 创建 Open3D 的 PointCloud 对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)  # 设置点坐标
    pcd.colors = o3d.utility.Vector3dVector(colors)  # 设置点颜色

    # 保存为 .ply 文件
    o3d.io.write_point_cloud(file_path, pcd)
    print(f"Point cloud saved to {file_path}")


def get_bounding_box_and_size(point_cloud: Pointclouds):
    """
    计算点云的包围盒和尺寸范围。

    Args:
        point_cloud: PyTorch3D 的 Pointclouds 对象。

    Returns:
        min_coords: 包围盒的最小坐标 (x_min, y_min, z_min)。
        max_coords: 包围盒的最大坐标 (x_max, y_max, z_max)。
        size: 包围盒的尺寸范围 (width, height, depth)。
    """
    # 提取点云的点坐标 (Nx3)
    points = point_cloud.points_padded()[0]  # 获取第一个点云

    # 计算最小和最大坐标
    min_coords = points.min(dim=0)[0]  # (3,)
    max_coords = points.max(dim=0)[0]  # (3,)

    # 计算包围盒的尺寸范围
    size = max_coords - min_coords

    return min_coords, max_coords, size


def generate_bounded_noise(B, N, D, bbox_size=40.0):
    """
    生成在指定包围盒范围内的噪声。

    Args:
        B: Batch size
        N: Number of points
        D: Dimensions (should be 3 for point cloud)
        bbox_size: 包围盒的尺寸范围 (默认 40x40x40)

    Returns:
        noise: 在指定包围盒范围内的噪声张量 (B, N, D)。
    """
    # 生成标准正态分布的噪声
    noise = torch.randn(B, N, D)

    # 缩放噪声到 [-20, 20] 区间
    noise = noise / noise.abs().max() * (bbox_size / 2.0)

    return noise


def point_cloud_to_tensor(
    pc: Pointclouds,
    scale_factor: float = 1.0,
    predict_color: bool = False,
    normalize: bool = False,
    scale: bool = False
):
    """将点云转换为张量，根据参数是否包含颜色信息和是否进行标准化与缩放"""
    points = pc.points_padded() * (scale_factor if scale else 1.0)

    if predict_color and pc.features_padded() is not None:
        colors = normalize_tensor(pc.features_padded()) if normalize else pc.features_padded()
        return torch.cat((points, colors), dim=2)
    else:
        return points


def tensor_to_point_cloud(
    x: Tensor,
    scale_factor: float = 1.0,
    predict_color: bool = False,
    denormalize: bool = False,
    unscale: bool = False
):
    """将张量转换回点云，根据参数是否包含颜色信息和是否进行反标准化与反缩放"""
    points = x[:, :, :3] / (scale_factor if unscale else 1.0)

    if predict_color:
        colors = denormalize_tensor(x[:, :, 3:]) if denormalize else x[:, :, 3:]
        return Pointclouds(points=points, features=colors)
    else:
        assert x.shape[2] == 3, "输入张量的最后一维必须是3（只包含位置）"
        return Pointclouds(points=points)


def denormalize(
    x: Tensor,
    colors_mean: Tensor,
    colors_std: Tensor,
    clamp: bool = True
) -> Tensor:
    """反标准化张量，并可选地将其值限制在 [0, 1] 区间内"""
    x = x * colors_std + colors_mean
    return torch.clamp(x, 0, 1) if clamp else x


def normalize(x: Tensor, colors_mean: Tensor, colors_std: Tensor) -> Tensor:
    """标准化张量，使用给定的均值和标准差"""
    return (x - colors_mean) / colors_std


def main():
    # 示例：从指定路径加载 PLY 文件
    ply_file_path = "/home/datasets/UrbanBIS/Qingdao/0020-0026/building/building6/untitled_plane.ply"  # 替换为你的文件路径
    point_cloud = load_ply_as_pointclouds(ply_file_path)

    # Normalize colors and convert to tensor
    x_0 = point_cloud_to_tensor(point_cloud, normalize=True, scale=False)

    B, N, D = x_0.shape  # Batch; Num; Dimen

    noise = generate_bounded_noise(B, N, D)

    pointcloud_out = tensor_to_point_cloud(noise)

    # 获取包围盒和尺寸范围
    min_coords, max_coords, size = get_bounding_box_and_size(pointcloud_out)

    print(f"Bounding Box Min Coordinates: {min_coords}")
    print(f"Bounding Box Max Coordinates: {max_coords}")
    print(f"Bounding Box Size: {size}")

    # 保存为 example.ply
    save_pointclouds_as_ply(pointcloud_out, "/home/code/Buildiffusion/example.ply")


if __name__ == '__main__':
    main()
























