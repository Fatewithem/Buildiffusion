import torch
import open3d as o3d
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
)
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer.cameras import FoVPerspectiveCameras
from pytorch3d.transforms import euler_angles_to_matrix
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast  # 导入自动混合精度的上下文管理器
from .projection_model import PointCloudProjectionModel

import os
import numpy as np


def iou_loss(pred, target, smooth=1.):
    """
    计算 IoU 损失（交并比损失）。

    参数:
    pred (torch.Tensor): 预测的 mask，值为 0 或 1，shape 为 (N, H, W) 或 (N, C, H, W)
    target (torch.Tensor): 真实的 mask，值为 0 或 1，shape 为 (N, H, W) 或 (N, C, H, W)
    smooth (float): 平滑参数，防止除零，默认为 1.

    返回:
    torch.Tensor: IoU Loss
    """

    # print(f"pred: {pred.shape}")
    # print(f"target: {target.shape}")

    threshold = 0.5

    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()

    # 计算交集和并集
    intersection = torch.sum(pred_bin * target_bin)
    union = torch.sum(pred_bin + target_bin) - intersection

    # 避免除零错误
    iou = intersection / (union + 1e-6)

    # IoU 损失
    iou_loss = 1 - iou

    return iou_loss


def save_iou_masks(rendered_image, mask, cam_name, output_dir, step):
    """
    将渲染图像和 mask 保存到指定路径。
    """
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 文件路径
    rendered_image_path = os.path.join(output_dir, f"{cam_name}_rendered_{step}.png")
    mask_path = os.path.join(output_dir, f"{cam_name}_mask_{step}.png")

    # 确保 mask 的形状为 [1080, 1920]，然后扩展为 RGB 格式
    if mask.dim() == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)  # 从 [1, 1080, 1920] 变为 [1080, 1920]

    # 将 mask 转换为 RGB 格式 (单通道复制 3 次)
    mask_rgb = mask.unsqueeze(-1).repeat(1, 1, 3)  # [1080, 1920] -> [1080, 1920, 3]
    mask_rgb = mask_rgb.detach().cpu().numpy()

    # 保存渲染图像和 mask
    plt.imsave(rendered_image_path, rendered_image.detach().cpu().numpy(), cmap='gray')
    plt.imsave(mask_path, mask_rgb)

    print(f"Saved rendered image to: {rendered_image_path}")
    print(f"Saved mask to: {mask_path}")


def get_mask_loss(tensors, masks, batch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 实例化 PointCloudProjectionModel 类
    model = PointCloudProjectionModel(
        image_height=1080,
        image_width=1920,
        use_top=True,
        use_depth=False,
        image_feature_model='resnet50',
        predict_shape=True,
        predict_color=False
    )

    camera_params = [
        # {"name": "_1", "location": [-100, 0, 0], "rotation": [90.0, -76.0, 90.0]},  # 相机 1
        # {"name": "_2", "location": [100, 0, 0], "rotation": [90.0, 76.0, -90.0]},  # 相机 2
        # {"name": "_3", "location": [0, -23, -100], "rotation": [14.0, 0.0, 0.0]},  # 相机 3
        # {"name": "_4", "location": [0, 23, 100], "rotation": [166.0, 0.0, 180.0]},  # 相机 4
        {"name": "_0", "location": [0, 150, 0], "rotation": [90.0, 0.0, 180.0]}  # 相机 0
    ]

    # 渲染参数
    fov = 23.14
    image_size = (1080, 1920)

    # 渲染设置
    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=0.005,
        points_per_pixel=10,
        bin_size=0
    )

    # 渲染器初始化
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(
            cameras=None,  # 稍后动态指定摄像机
            raster_settings=raster_settings
        ),
        compositor=AlphaCompositor()
    )

    # 创建输出目录
    output_dir = "/home/code/Blender/output"
    os.makedirs(output_dir, exist_ok=True)

    # 初始化总的 IoU 损失
    total_loss = 0.0

    pointclouds = model.tensor_to_point_cloud(tensors)

    for step, (pc, mask) in enumerate(zip(pointclouds, masks)):
        # 如果点云没有特征，则为其添加默认特征，例如全白（RGB: [1, 1, 1]）
        if pc.features_padded() is None:
            features = torch.ones_like(pc.points_padded(), device=device)  # 形状为 [B, N, 3]
            pc = Pointclouds(points=pc.points_padded(), features=features)

        rendered_images = []  # 用于存储所有相机的渲染结果

        for cam in camera_params:
            # 相机位置与旋转
            location = torch.tensor(cam["location"], dtype=torch.float32, device=device).unsqueeze(0)
            rotation = torch.tensor(cam["rotation"], dtype=torch.float32, device=device).unsqueeze(0)
            rotation = rotation * (torch.pi / 180.0)
            R = euler_angles_to_matrix(rotation, "XYZ").float()
            T = -torch.matmul(R, location.unsqueeze(2)).squeeze(-1)

            # 定义相机
            cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=fov).float()

            # 渲染图像
            with autocast(enabled=False):
                rendered_image = renderer(point_clouds=pc, cameras=cameras)
                rendered_image = torch.clamp(rendered_image, 0, 1)

            # 转为灰度图
            rendered_gray = (
                    0.299 * rendered_image[..., 0] +
                    0.587 * rendered_image[..., 1] +
                    0.114 * rendered_image[..., 2]
            )

            # 生成 mask
            threshold = 0.5
            rendered_mask = (rendered_gray > threshold).float()

            # 保存渲染图像和 mask
            save_iou_masks(rendered_mask[0], mask[0], cam['name'], "/home/code/Blender/image", step)

            # 将渲染的 mask 添加到列表中
            rendered_images.append(rendered_mask)

        # 堆叠所有渲染图像为张量
        rendered_images_tensor = torch.stack(rendered_images, dim=0)

        # 计算 IoU 损失
        sub_loss = iou_loss(rendered_images_tensor, mask, smooth=1.0)
        # print(f"sub_loss: {sub_loss}")
        total_loss += sub_loss

    return total_loss / batch










