import cv2
import numpy as np
import torch
import torch.nn as nn
import os
from torch import Tensor
from pytorch3d.structures import Pointclouds
from torch.cuda.amp import autocast  # 导入自动混合精度的上下文管理器
import matplotlib.pyplot as plt

from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
)
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.transforms import euler_angles_to_matrix


# 当 requires_grad 设置为 True 时，这些参数会在反向传播过程中计算梯度；如果设置为 False，则不会计算梯度
# 用于frozen
def set_requires_grad(module: nn.Module, requires_grad: bool):
    for p in module.parameters():
        p.requires_grad_(requires_grad)


# 计算距离变换
def compute_distance_transform(mask: torch.Tensor):
    # print(f"Input mask shape: {mask.shape}")

    # 确保 mask 形状为 (B, H, W)

    mask = mask.squeeze(dim=1)  # 移除所有大小为 1 的维度
    mask = mask.squeeze(dim=1)

    # print(f"After squeeze, mask shape: {mask.shape}")

    # 检查是否为 (B, 1080, 1920)
    # assert mask.shape[-2:] == (1080, 1920), f"Unexpected mask shape: {mask.shape[-2:]}"

    # 遍历每个批次的掩膜并计算距离变换
    batch_size = mask.shape[0]
    distance_transform_list = []

    for i in range(batch_size):
        # 将掩膜转换为 NumPy 格式，并计算距离变换
        m = mask[i].detach().cpu().numpy().astype(np.uint8)
        dist_transform = cv2.distanceTransform(
            1 - m, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_3
        )
        # 正则化到 [0, 1] 范围，并转换为张量
        dist_transform /= dist_transform.max()  # 归一化到 [0, 1]
        dist_transform_tensor = torch.from_numpy(dist_transform).float()
        distance_transform_list.append(dist_transform_tensor)

    # 将所有距离变换张量堆叠，并添加通道维度
    distance_transform = torch.stack(distance_transform_list).unsqueeze(1).to(mask.device)

    return distance_transform


def default(x, d):
    return d if x is None else x


def get_num_points(x: Pointclouds, /):
    return x.points_padded().shape[1]


def tensor_to_point_cloud(x: Tensor, predict_color: bool = True, denormalize: bool = False, unscale: bool = False):
    points = x[:, :, :3]
    if predict_color:
        colors = denormalize(x[:, :, 3:]) if denormalize else x[:, :, 3:]
        return Pointclouds(points=points, features=colors)
    else:
        assert x.shape[2] == 3
        return Pointclouds(points=points)


# 设置 diffusion 的 beta 参数
def get_custom_betas(beta_start: float, beta_end: float, warmup_frac: float = 0.3, num_train_timesteps: int = 1000):
    """Custom beta schedule"""
    betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
    warmup_frac = 0.3
    warmup_time = int(num_train_timesteps * warmup_frac)
    warmup_steps = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    warmup_time = min(warmup_time, num_train_timesteps)
    betas[:warmup_time] = warmup_steps[:warmup_time]
    return betas


# 可视化点云效果
def render_point_cloud(point_cloud: Tensor, idx: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pointclouds = tensor_to_point_cloud(point_cloud)

    # if pointclouds.features_padded() is None:
    #     # 添加全白的 RGB 特征
    #     features = torch.ones_like(pointclouds.points_padded(), device=device)  # 形状为 [B, N, 3]
    #     pointclouds = Pointclouds(points=pointclouds.points_padded(), features=features)

    features = torch.ones_like(pointclouds.points_padded(), device=device)  # 形状为 [B, N, 3]
    pointclouds = Pointclouds(points=pointclouds.points_padded(), features=features)

    camera_params = [
        {"name": "_4", "location": [-7.8681e-07,  2.9535e+00,  4.5796e+01], "rotation": [165.0, 0.0, 180.0]},
    ]

    # 渲染参数
    focal_length = 3.85  # 替换成你自己的焦距
    image_size = (1080, 1920)

    # 配置渲染器
    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=0.005,
        points_per_pixel=10,
        bin_size=0  # 设置为0以自动选择最佳大小
    )
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(
            cameras=None,  # 我们将在循环中动态设置 cameras
            raster_settings=raster_settings),
        compositor=AlphaCompositor()
    )

    # 创建输出目录
    output_dir = "/home/code/Blender/output"
    os.makedirs(output_dir, exist_ok=True)

    for cam in camera_params:
        # 相机位置与旋转
        location = torch.tensor(cam["location"], dtype=torch.float32, device=device).unsqueeze(0)
        rotation = torch.tensor(cam["rotation"], dtype=torch.float32, device=device).unsqueeze(0)
        rotation = rotation * (torch.pi / 180.0)
        R = euler_angles_to_matrix(rotation, "XYZ").float()  # 转换为 float32

        T = location / 10.0

        cameras = PerspectiveCameras(
            focal_length=((focal_length, focal_length),),
            principal_point=((0.0, 0.0),),
            image_size=(image_size,),
            in_ndc=True,
            R=R, T=T, device=device
        )

        with autocast(enabled=False):
            # 渲染点云图像，确保 pointcloud 和 cameras 是 float32 类型
            rendered_image = renderer(point_clouds=pointclouds, cameras=cameras)
            rendered_image = torch.clamp(rendered_image, 0, 1)  # 将像素值限制在 0 到 1 之间

        # 保存生成的图像
        if cam['name'] == '_4':
            image_path = os.path.join(output_dir, f"{idx}.png")
            # 确保图像的形状正确
            if rendered_image[0, ..., :3].ndimension() != 3 or rendered_image[0, ..., :3].shape[-1] != 3:
                print(f"Rendered image shape: {rendered_image[0, ..., :3].shape}")
                raise ValueError("Rendered image must have shape [height, width, 3] for RGB format.")

            # 保存图像
            plt.imsave(image_path, rendered_image[0, ..., :3].detach().cpu().numpy())  # 先 detach 再转为 numpy






