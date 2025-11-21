import os
import torch
import open3d as o3d
import numpy as np
import cv2
import json
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.transforms import euler_angles_to_matrix

# --- PyTorch3D渲染函数 ---
from pytorch3d.renderer import (
    FoVPerspectiveCameras, PointsRasterizationSettings, PointsRenderer,
    PointsRasterizer, AlphaCompositor, NormWeightedCompositor,
    look_at_view_transform
)
from pytorch3d.structures import Pointclouds
from torchvision.utils import save_image

def render_with_pytorch3d(points, colors, R, T, focal_length, principal_point, image_size, device, output_path):
    # 构造相机
    cameras = PerspectiveCameras(
        R=R.to(device),
        T=T.to(device),
        focal_length=focal_length.to(device),
        principal_point=principal_point.to(device),
        image_size=image_size,
        in_ndc=False,
        device=device
    )

    # 构造点云对象
    point_cloud = Pointclouds(points=[points], features=[colors])

    # 渲染器配置
    raster_settings = PointsRasterizationSettings(
        image_size=image_size[0],  # 假设传入为 [(H, W)]
        radius=0.003,
        points_per_pixel=10
    )
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
        compositor=AlphaCompositor()
    )

    # 渲染图像
    rendered = renderer(point_cloud)
    rendered = rendered[0].permute(2, 0, 1)  # (3, H, W)

    save_image(rendered.clamp(0,1), output_path)
    print(f"Saved rendered image to {output_path}")

def create_camera(R, T, focal_length, principal_point, image_size=None, device="cpu"):
    print(image_size)
    cameras = PerspectiveCameras(
        R=R.to(device),
        T=T.to(device),
        focal_length=focal_length.to(device),
        principal_point=principal_point.to(device),
        image_size=(1066, 1896),
        in_ndc=False,
        device=device
    ).float()
    return cameras


def project_points(points, cameras):
    points = points.unsqueeze(0)  # (1, N, 3)
    xy_depth = cameras.get_full_projection_transform().transform_points(points)[0]  # (N, 3)
    xy = xy_depth[:, :2]
    z = xy_depth[:, 2]
    return xy, z


def sample_colors_from_image(image, uv, W, H):
    """
    image: (3, H, W)
    uv: (N, 2) in pixel coordinates
    """
    # 归一化到[-1,1]，grid_sample要求这样
    uv_norm = uv.clone()
    uv_norm[:, 0] = (uv_norm[:, 0] / (W - 1)) * 2 - 1  # 注意W-1
    uv_norm[:, 1] = (uv_norm[:, 1] / (H - 1)) * 2 - 1  # 注意H-1
    uv_norm = uv_norm.view(1, -1, 1, 2)  # (1, N, 1, 2)

    # grid_sample expects (B, C, H, W), so unsqueeze(0)
    sampled = torch.nn.functional.grid_sample(image.unsqueeze(0), uv_norm, mode='bilinear', align_corners=True)
    sampled = sampled.squeeze(0).squeeze(2).permute(1, 0)  # (N, 3)
    return sampled


def colorize_point_cloud(pcd_path, image_path, output_dir, camera_params_list, sample_idx):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 读取点云
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)
    points = torch.from_numpy(points).float().to(device)  # (N, 3)

    # 2. 读取图像
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        raise ValueError(f"Failed to load image at {image_path}")
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image_cv).float().permute(2, 0, 1) / 255.0  # (3, H, W)
    image = image.to(device)
    _, H, W = image.shape

    os.makedirs(output_dir, exist_ok=True)

    for idx, cam_param in enumerate(camera_params_list):
        R = torch.tensor(cam_param["R"], dtype=torch.float32).unsqueeze(0)
        # 在原始R上增加旋转调整
        from pytorch3d.transforms import euler_angles_to_matrix
        # 使用角度输入，例如 [90, 0, 90]，并转换为弧度
        angles_deg = [180, 0, 0]  # XYZ 顺序
        angle_adjust = torch.tensor(angles_deg, dtype=torch.float32) * np.pi / 180.0
        # 应用旋转增量（注意乘法顺序）
        R_delta = euler_angles_to_matrix(angle_adjust, convention="XYZ").to(R.device)
        R = torch.bmm(R_delta.unsqueeze(0), R)
        T = torch.tensor(cam_param["T"], dtype=torch.float32).unsqueeze(0)
        # 将 focal_length 和 principal_point 从归一化单位转换为像素单位
        fx_ndc, fy_ndc = cam_param["focal_length"]
        ppx_ndc, ppy_ndc = cam_param["principal_point"]

        fx = fx_ndc * W / 2
        fy = fy_ndc * H / 2
        ppx = W / 2
        ppy = H / 2

        # print(ppx_ndc)
        # print(ppy_ndc)

        focal_length = torch.tensor([[fx, fy]], dtype=torch.float32)
        principal_point = torch.tensor([[ppx, ppy]], dtype=torch.float32)

        cameras = create_camera(R, T, focal_length, principal_point, image_size=((H, W),), device=device)

        # 4. 投影
        uv, depth = project_points(points, cameras)
        print(f"Camera {idx} UV min/max:", uv.min().item(), uv.max().item())
        print(f"Camera {idx} Depth min/max:", depth.min().item(), depth.max().item())

        # 5. mask筛选
        mask = (uv[:, 0] >= 0) & (uv[:, 0] <= (W-1)) & (uv[:, 1] >= 0) & (uv[:, 1] <= (H-1)) & (depth > 0)
        print(f"Camera {idx} valid points after masking:", mask.sum().item(), "/", mask.shape[0])

        if mask.sum() == 0:
            print(f"Camera {idx}: No valid points inside view frustum, skipping.")
            continue

        uv_valid = uv[mask]
        points_valid = points[mask]

        # 8. 计算图像中心对应的3D点
        cx = (W - 1) / 2
        cy = (H - 1) / 2
        center_uv = torch.tensor([[cx, cy]], device=device)

        # 归一化坐标到 [-1,1]
        center_uv_norm = center_uv.clone()
        center_uv_norm[:, 0] = (center_uv_norm[:, 0] / (W - 1)) * 2 - 1
        center_uv_norm[:, 1] = (center_uv_norm[:, 1] / (H - 1)) * 2 - 1
        center_uv_norm = center_uv_norm.view(1, 1, 1, 2)

        # 反向投影，从图像中心采样对应的深度
        # 这里简单地用所有valid点里面，找到最靠近(cx, cy)的那个点
        dists = (uv_valid - center_uv).norm(dim=1)
        idx_closest = torch.argmin(dists)
        center_3d_point = points_valid[idx_closest].unsqueeze(0)  # (1, 3)

        # 6. 采样颜色
        colors = sample_colors_from_image(image, uv_valid, W, H)

        # 9. 标记中心点周围小球为红色
        center = center_3d_point  # (1, 3)
        radius = 0.3  # 半径0.5米可调整
        distances = torch.norm(points_valid - center, dim=1)
        neighbor_mask = distances < radius
        colors[neighbor_mask] = torch.tensor([1.0, 0.0, 0.0], device=device)  # 红色

        # 7. 保存点云
        pcd_valid = o3d.geometry.PointCloud()
        pcd_valid.points = o3d.utility.Vector3dVector(points_valid.cpu().numpy())
        pcd_valid.colors = o3d.utility.Vector3dVector(colors.clamp(0,1).cpu().numpy())
        output_ply_path = os.path.join(output_dir, f"sample_{sample_idx}_colored_output_{idx}.ply")
        o3d.io.write_point_cloud(output_ply_path, pcd_valid)
        print(f"Saved colorized point cloud to {output_ply_path}")

        # 10. 使用 PyTorch3D 渲染图像
        render_output_path = os.path.join(output_dir, f"sample_{sample_idx}_rendered_{idx}.png")
        render_with_pytorch3d(points_valid, colors, R, T, focal_length, principal_point, ((H, W),), device, render_output_path)


if __name__ == "__main__":
    # 配置基本路径
    pcd_path = "/home/code/Buildiffusion/result/sample_0_gt.ply"
    # pcd_path = "/home/code/Buildiffusion/building5/untitled_fps.ply"
    image_path = "/home/code/Buildiffusion/result/frame000021.jpg"
    output_dir = "/home/code/Buildiffusion/color/"

    camera_params_list = [{
        "R": [
            [0.7089390158653259, 0.48484131693840027, 0.5121858716011047],
            [-0.5104979872703552, -0.14830778539180756, 0.8469926714897156],
            [0.48661819100379944, -0.8619360327720642, 0.1423693597316742]
        ],
        "T": [-1.3809274435043335, 1.459787368774414, 8.683882713317871],
        "focal_length": [3.0737552642822266, 3.0737552642822266],
        "principal_point": [0.0, 0.0]
    }]
    colorize_point_cloud(pcd_path, image_path, output_dir, camera_params_list, sample_idx='0')