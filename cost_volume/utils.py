import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import math
import json

import torchvision.transforms as transforms


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def prepare_camera_projections(camera_params, K, device="cpu"):
    """
    将相机参数和内参矩阵转换为齐次投影矩阵。
    :param camera_params: 相机参数列表，每个包含 "R" (旋转矩阵) 和 "T" (平移向量)。
    :param K: 内参矩阵，形状为 [3, 3] 的张量。
    :param device: 设备信息，例如 "cpu" 或 "cuda:0"。
    :return: 齐次投影矩阵列表，每个为 [4, 4] 的张量。
    """
    proj_matrices = []
    K = K.to(device)  # 将 K 移动到目标设备上
    for cam in camera_params:
        R = torch.tensor(cam["R"], dtype=torch.float32, device=device)  # [3, 3]
        T = torch.tensor(cam["T"], dtype=torch.float32, device=device).view(3, 1)  # [3, 1]

        # 构造 [3, 4] 的外参矩阵 [R | T]
        extrinsics = torch.cat([R, T], dim=1)  # [3, 4]

        # 计算投影矩阵 K * [R | T]
        projection_3x4 = torch.matmul(K, extrinsics)  # [3, 4]

        # 转换为齐次投影矩阵 [4, 4]
        proj_matrix = torch.eye(4, dtype=torch.float32, device=device)  # [4, 4]
        proj_matrix[:3, :4] = projection_3x4

        proj_matrices.append(proj_matrix)

    return proj_matrices


def compute_intrinsic_matrix(fx, fy, cx, cy, scale=1.0, device="cpu"):
    """
    构建统一的相机内参矩阵 K
    :param fx: 焦距 fx
    :param fy: 焦距 fy
    :param cx: 主点 cx
    :param cy: 主点 cy
    :param device: PyTorch 设备
    :return: 内参矩阵 K [3, 3]
    """
    K = torch.eye(3, device=device, dtype=torch.float32)
    K[0, 0] = fx * scale  # 焦距 fx
    K[1, 1] = fy * scale  # 焦距 fy
    K[0, 2] = cx * scale  # 主点 cx
    K[1, 2] = cy * scale  # 主点 cy
    return K


def load_image_as_tensor(image_path, target_size=None):
    """
    加载图像并将其转换为Tensor，同时将尺寸调整为可以被16*16整除的大小。
    """
    # 加载图像并转换为RGB格式
    image = Image.open(image_path).convert("RGB")

    # 将图像转换为Tensor
    transform_list = [transforms.ToTensor()]
    if target_size:
        transform_list.insert(0, transforms.Resize(target_size))
    transform = transforms.Compose(transform_list)
    image_tensor = transform(image).unsqueeze(0)  # [1, C, H, W]

    # 获取当前尺寸
    _, _, H, W = image_tensor.shape

    # 计算新的尺寸，使其可以被14整除
    new_H = math.ceil(H / 14) * 14
    new_W = math.ceil(W / 14) * 14

    # 使用插值调整尺寸
    image_tensor_resized = F.interpolate(image_tensor, size=(new_H, new_W), mode='bicubic', align_corners=False)

    return image_tensor_resized


def homo_warping_3D_with_mask(src_fea, src_proj, ref_proj, depth_values):
    """
    进行3D同质变换并生成投影掩码。
    :param src_fea: 源图像特征，形状为 [B, C, H, W]
    :param src_proj: 源视图的投影矩阵，形状为 [B, 4, 4]
    :param ref_proj: 参考视图的投影矩阵，形状为 [B, 4, 4]
    :param depth_values: 深度假设值，形状为 [B, Ndepth] 或 [B, Ndepth, H, W]
    :return: 重投影后的特征 [B, C, Ndepth, H, W] 和投影掩码 [B, Ndepth, H, W]
    """
    # 确保所有输入张量在相同设备
    device = src_fea.device
    src_proj = src_proj.to(device)
    ref_proj = ref_proj.to(device)
    depth_values = depth_values.to(device)

    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B, 3, 3]
        trans = proj[:, :3, 3:4]  # [B, 3, 1]

        # 构造网格坐标
        y, x = torch.meshgrid(
            torch.arange(0, height, dtype=torch.float32, device=device),
            torch.arange(0, width, dtype=torch.float32, device=device),
            indexing='ij'
        )
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = xyz.unsqueeze(0).repeat(batch, 1, 1)  # [B, 3, H*W]

        # 旋转并进行深度投影
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth, -1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / (proj_xyz[:, 2:3, :, :] + 1e-6)  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    # 计算投影掩码
    X_mask = ((proj_x_normalized > 1) + (proj_x_normalized < -1)).detach()
    Y_mask = ((proj_y_normalized > 1) + (proj_y_normalized < -1)).detach()
    proj_mask = ((X_mask + Y_mask) > 0).view(batch, num_depth, height, width)
    z = proj_xyz[:, 2:3, :, :].view(batch, num_depth, height, width)
    proj_mask = (proj_mask + (z <= 0)) > 0

    # 使用 F.grid_sample 进行特征重投影
    warped_src_fea = F.grid_sample(
        src_fea,
        grid.view(batch, num_depth * height, width, 2),
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea, proj_mask