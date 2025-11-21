import os
import torch
import open3d as o3d
import numpy as np
import cv2
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.transforms import euler_angles_to_matrix


def create_camera(location, rotation, focal_length, device="cpu"):
    location = location.unsqueeze(0).to(device)
    rotation = rotation.unsqueeze(0).to(device)
    rotation_rad = rotation * (torch.pi / 180.0)
    R = euler_angles_to_matrix(rotation_rad, "XYZ").float()
    T = location
    cameras = PerspectiveCameras(
        R=R,
        T=T,
        focal_length=focal_length,
        principal_point=((483,273),),
        image_size=((966,546),),
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
        location = torch.tensor(cam_param["location"], dtype=torch.float32)
        rotation = torch.tensor(cam_param["rotation"], dtype=torch.float32)
        focal_length = (cam_param["focal_length"],)

        cameras = create_camera(location, rotation, focal_length, device=device)

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
        # center = center_3d_point  # (1, 3)
        # radius = 0.5  # 半径0.5米可调整
        # distances = torch.norm(points_valid - center, dim=1)
        # neighbor_mask = distances < radius
        # colors[neighbor_mask] = torch.tensor([1.0, 0.0, 0.0], device=device)  # 红色

        # 7. 保存点云
        pcd_valid = o3d.geometry.PointCloud()
        pcd_valid.points = o3d.utility.Vector3dVector(points_valid.cpu().numpy())
        pcd_valid.colors = o3d.utility.Vector3dVector(colors.clamp(0,1).cpu().numpy())
        output_ply_path = os.path.join(output_dir, f"sample_{sample_idx}_colored_output_{idx}.ply")
        o3d.io.write_point_cloud(output_ply_path, pcd_valid)
        print(f"Saved colorized point cloud to {output_ply_path}")


if __name__ == "__main__":
    # 配置基本路径
    base_pcd_path = "/home/code/Buildiffusion/result/sample_{}_gt.ply"
    base_image_path = "/home/code/Buildiffusion/result/sample_{}_image.png"
    output_dir = "/home/code/Buildiffusion/color/"

    camera_params_list = [
        {"location": [0, 0, 4.5796e+01], "rotation": [0.0, -75.0, 180.0], "focal_length": (2102.0, 2102.0)},
        {"location": [0, 1, 4.5796e+01], "rotation": [0.0, -75.0, 180.0], "focal_length": (2102.0, 2102.0)},
        {"location": [0, 2, 4.5796e+01], "rotation": [0.0, -75.0, 180.0], "focal_length": (2102.0, 2102.0)},
        {"location": [0, -1.22, 4.5796e+01], "rotation": [0.0, -75.0, 180.0], "focal_length": (2102.0, 2102.0)},
    ]

    # for i in range(10):  # 处理sample_0 到 sample_9
    #     pcd_path = base_pcd_path.format(i)
    #     image_path = base_image_path.format(i)
    #     print(f"Processing sample {i}...")
    #     colorize_point_cloud(pcd_path, image_path, output_dir, camera_params_list, sample_idx=i)

    # 单独测试 sample_7
    sample_idx = 7
    pcd_path = f"/home/code/Buildiffusion/result/sample_{sample_idx}_gt.ply"
    image_path = f"/home/code/Buildiffusion/result/sample_{sample_idx}_image.png"

    print(f"Processing sample {sample_idx}...")
    colorize_point_cloud(pcd_path, image_path, output_dir, camera_params_list, sample_idx=sample_idx)