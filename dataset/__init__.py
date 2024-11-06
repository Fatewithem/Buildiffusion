import sys
import os

sys.path.append("..")

from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union, Any
import hydra
import torch
import pandas as pd
from dataclasses import dataclass, field
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from omegaconf import DictConfig
from config.structured import ProjectConfig, BuildingsConfig, DataloaderConfig
from pytorch3d.structures import Pointclouds
from PIL import Image
from torchvision import transforms
import open3d as o3d
import numpy as np
import json
from pytorch3d.renderer.cameras import FoVPerspectiveCameras
from pytorch3d.transforms import euler_angles_to_matrix

from PIL import Image
import matplotlib.pyplot as plt


@dataclass
class BuildingsDataset(Dataset):
    dataset_cfg: BuildingsConfig
    split: str
    root_dir: str = field(init=False)
    scene: List[str] = field(default_factory=list)
    file_paths: List[str] = field(init=False)
    fov: float = 23.14  # 视场角
    scale_factor: float = 1.0  # 缩放因子
    device: torch.device = field(default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def __post_init__(self):
        self.root_dir = self.dataset_cfg.root
        self.scene = self.dataset_cfg.scene
        self.file_paths = self._load_file_paths()

    def _load_file_paths(self) -> List[str]:
        file_folder_path = []
        for s in self.scene:
            scene_dir = os.path.join(self.root_dir, s)
            json_path = os.path.join(scene_dir, "filter.json")
            with open(json_path, 'r') as f:
                data = json.load(f)
            # print(f"Scene_dir: {type(scene_dir)}")
            for folder in data.items():
                folder_path = os.path.join(scene_dir, folder[0])
                file_folder_path.append(folder_path)

        # print(f"Nums: {len(file_folder_path)}")
        return file_folder_path

    def create_camera(self, location: List[float], rotation: List[float]) -> FoVPerspectiveCameras:
        """
        根据位置、旋转和 fov 创建相机对象。
        Args:
            location (List[float]): 相机的三维位置 (x, y, z)。
            rotation (List[float]): 相机的旋转角度 [yaw, pitch, roll]。

        Returns:
            FoVPerspectiveCameras: 创建好的相机对象。
        """
        # 转换位置和旋转为张量
        location_tensor = torch.tensor(location, dtype=torch.float32).unsqueeze(0)
        rotation_tensor = torch.tensor(rotation, dtype=torch.float32).unsqueeze(0)

        # 将角度从度数转换为弧度，并计算旋转矩阵 R
        rotation_rad = rotation_tensor * (torch.pi / 180.0)
        R = euler_angles_to_matrix(rotation_rad, "XYZ").float()

        # 计算平移向量 T
        T = -torch.matmul(R, location_tensor.unsqueeze(2)).squeeze(-1)

        # 缩放平移向量 T
        T = T * self.scale_factor

        # 创建相机对象
        cameras = FoVPerspectiveCameras(R=R, T=T, fov=self.fov).float()
        return cameras

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Any:
        file_path = self.file_paths[idx]
        # print(f"Loading point cloud from: {file_path}")  # 打印正在加载的文件路径

        temp_path = 'blender'
        masks_plane_path = 'masks_plane'
        masks_path = os.path.join(file_path, masks_plane_path)

        pointcloud_path = os.path.join(file_path, "untitled_plane.ply")

        transform = transforms.ToTensor()
        # masks_tensor = torch.zeros((5, 1, 1080, 1920))
        masks_tensor = torch.zeros((5, 1, 1080, 1920))

        # 遍历 mask 文件并处理
        for i in range(5):
            mask_name = f"_{i}.png"
            mask_path = os.path.join(masks_path, mask_name)

            # 检查文件是否存在
            if not os.path.exists(mask_path):
                print(f"Error: Mask file not found at {mask_path}")
                continue

            # 读取并转换为灰度图像
            mask = Image.open(mask_path).convert("L")  # 灰度图像
            mask_tensor = transform(mask)  # 转换为 Tensor

            # 将 mask 插入到 batch 维度中
            masks_tensor[i] = mask_tensor

            # # 保存可视化结果到文件
            # save_path = os.path.join("/home/code/Blender/test_output", f"mask_{i}.png")
            # fig = plt.figure(figsize=(10, 5))  # 创建一个图像对象
            # plt.imshow(mask_tensor.squeeze(0).cpu().numpy(), cmap='gray')  # 绘制灰度图
            # plt.title(f"Mask {i}")
            # plt.axis('off')  # 关闭坐标轴
            # plt.savefig(save_path, bbox_inches='tight', pad_inches=0)  # 保存为 PNG 文件
            # plt.close(fig)  # 关闭图像，释放内存
            #
            # print(f"Saved mask visualization to: {save_path}")

        # 使用 Open3D 加载点云文件
        point_cloud = o3d.io.read_point_cloud(pointcloud_path)

        # 提取点云的坐标数据，转换为 numpy 数组
        points = np.asarray(point_cloud.points)

        # 检查是否有点云坐标
        if points.shape[0] == 0:
            print(f"Warning: No points found in point cloud at {file_path}")
            return None, None

        # 如果有颜色信息，可以提取颜色数据
        if point_cloud.has_colors():
            colors = np.asarray(point_cloud.colors)
            colors_tensor = torch.tensor(colors, dtype=torch.float32)
        else:
            colors_tensor = None  # 如果没有颜色信息，可以设为 None

        # 将 numpy 数组转换为 PyTorch 张量
        points_tensor = torch.tensor(points, dtype=torch.float32)

        camera_params = [
            {"name": "_0", "location": [0, 150, 0], "rotation": [90.0, 0.0, 180.0]},
            {"name": "_1", "location": [-100, 0, 0], "rotation": [90.0, -76.0, 90.0]},
            {"name": "_2", "location": [100, 0, 0], "rotation": [90.0, 76.0, -90.0]},
            {"name": "_3", "location": [0, -23, -100], "rotation": [14.0, 0.0, 0.0]},
            {"name": "_4", "location": [0, 23, 100], "rotation": [166.0, 0.0, 180.0]},
        ]

        # 为每个相机参数创建相机对象
        camera_dict = [self.create_camera(cam["location"], cam["rotation"]) for cam in camera_params]

        return points_tensor, colors_tensor, masks_tensor, camera_dict


def custom_collate_fn(batch):
    # 分离出坐标和颜色
    points_list, colors_list, masks_list, camera_list = zip(*batch)
    # points_list, colors_list = zip(*batch)

    # 检查是否所有点云都有颜色
    if all(c is not None for c in colors_list):
        pointcloud_batch = Pointclouds(points=list(points_list), features=list(colors_list))
    else:
        pointcloud_batch = Pointclouds(points=list(points_list))

    # 将 Pointclouds 封装到字典里
    batch_dict = {
        'pointclouds': pointcloud_batch,
        'points_list': points_list,  # 也可以返回单独的点和颜色列表
        'colors_list': colors_list,
        'masks_list': masks_list,
        'camera_list': camera_list[0],
    }

    return batch_dict


def get_dataset(cfg: ProjectConfig):
    if cfg.dataset.type == 'buildings':
        dataset_cfg: BuildingsConfig = cfg.dataset
        dataloader_cfg: DataloaderConfig = cfg.dataloader

        datasets = {
            "train": BuildingsDataset(dataset_cfg, split="train"),
            "val": BuildingsDataset(dataset_cfg, split="val"),
            "test": BuildingsDataset(dataset_cfg, split="test"),
        }

        dataloaders = {
            "train": DataLoader(
                datasets["train"],
                batch_size=dataloader_cfg.batch_size,
                num_workers=dataloader_cfg.num_workers,
                shuffle=True,
                drop_last=False,
                collate_fn=custom_collate_fn,
            ),
            "val": DataLoader(
                datasets["val"],
                batch_size=dataloader_cfg.batch_size,
                sampler=SequentialSampler(datasets["val"]),
                num_workers=dataloader_cfg.num_workers,
                drop_last=False,
                collate_fn=custom_collate_fn,
            ),
            "test": DataLoader(
                datasets["test"],
                batch_size=dataloader_cfg.batch_size,
                sampler=SequentialSampler(datasets["test"]),
                num_workers=dataloader_cfg.num_workers,
                drop_last=False,
                collate_fn=custom_collate_fn,
            ),
        }

        dataloader_train = dataloaders['train']
        dataloader_val = dataloader_vis = dataloaders[dataset_cfg.eval_split]

    else:
        raise NotImplementedError(cfg.dataset.type)

    return dataloader_train, dataloader_val, dataloader_vis


# @hydra.main(config_path='/home/code/Buildiffusion/config', config_name='config', version_base='1.1')
# def main(cfg: ProjectConfig):
#     # 调用 get_dataset 函数
#     dataloader_train, dataloader_val, dataloader_vis = get_dataset(cfg)
#
#     # 检查 dataloader 是否正常工作
#     try:
#         for batch in dataloader_train:
#             # print(f"Batchsize length: {batch['masks_list'][42]}")
#             print("Train batch loaded successfully")
#             break
#
#         for batch in dataloader_val:
#             print("Validation batch loaded successfully")
#             break
#
#         for batch in dataloader_vis:
#             print("Visualization batch loaded successfully")
#             break
#
#     except Exception as e:
#         print(f"Error loading dataset: {e}")
#
#     for i, batch in enumerate(dataloader_train):
#         print(f"Batch {i} loaded successfully")
#         if i >= len(dataloader_train) - 1:
#             break
#
#
# if __name__ == "__main__":
#     main()