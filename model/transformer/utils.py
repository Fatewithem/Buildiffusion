import os

import numpy as np
import open3d
import mcubes
import math
import torch
from omegaconf import OmegaConf, DictConfig

from pytorch3d.ops import knn_points, sample_farthest_points

def voxel_grid_to_mesh(vox_grid: np.array) -> open3d.geometry.TriangleMesh:
    """
        taken from: https://github.com/lmb-freiburg/what3d

        Converts a voxel grid represented as a numpy array into a mesh.
    """
    sp = vox_grid.shape
    if len(sp) != 3 or sp[0] != sp[1] or \
            sp[1] != sp[2] or sp[0] == 0:
        raise ValueError("Only non-empty cubic 3D grids are supported.")
    padded_grid = np.pad(vox_grid, ((1, 1), (1, 1), (1, 1)), 'constant')
    m_vert, m_tri = mcubes.marching_cubes(padded_grid, 0)
    m_vert = m_vert / (padded_grid.shape[0] - 1)
    out_mesh = open3d.geometry.TriangleMesh()
    out_mesh.vertices = open3d.utility.Vector3dVector(m_vert)
    out_mesh.triangles = open3d.utility.Vector3iVector(m_tri)
    return out_mesh


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


def init_weights(m):
    """
        Utility method to initialize the backbone network weights

        Source: https://github.com/hzxie/Pix2Vox/blob/master/utils/network_utils.py
    :param m:
    :return:
    """
    if type(m) == torch.nn.Conv2d or type(m) == torch.nn.Conv3d or type(m) == torch.nn.ConvTranspose3d:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.BatchNorm3d:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        torch.nn.init.constant_(m.bias, 0)


def save_config(cfg: DictConfig, dir: str) -> None:
    """
        Save configuration to a file
    :param cfg: Configuration dict or DictConfig
    :param dir: Path to the save folder
    :return: None
    """
    if not os.path.exists(dir):
        os.makedirs(dir)
    path = os.path.join(dir, 'config_model.yaml')
    OmegaConf.save(cfg, path)


def load_config(cfg_path: str) -> DictConfig:
    """
        Load configuration file. `base_config.yaml` is taken as a base for every config.
    :param cfg_path: Path to the configuration file
    :return: Loaded configuration
    """
    base_cfg = OmegaConf.load('legoformer/config/base_config.yaml')
    curr_cfg = OmegaConf.load(cfg_path)
    return OmegaConf.merge(base_cfg, curr_cfg)


def convert_to_tensor(output):
    """
    直接将 `output` 转换为 `Tensor[B, N, 3]`（无 Padding）
    :param output: `OutputLayer` 生成的 List[B] -> List[num_queries] -> Tensor[N, 3]
    :return: `Tensor[B, N_total, 3]`
    """
    batch_size = len(output)  # 计算 batch 维度
    assert batch_size == 1, "当前函数仅支持 batch_size=1 的情况"

    all_points = []
    for query_points in output[0]:  # 遍历 queries
        if query_points.numel() > 0:
            all_points.append(query_points)

    # 直接拼接所有 query 的点云，保持 [B, N_total, 3] 格式
    return torch.cat(all_points, dim=0).unsqueeze(0)  # 添加 batch 维度，变成 [1, N_total, 3]


def chamfer_f1_score(pred, gt, threshold=0.01, target_points=8192, mode="fps"):
    """
    Compute Chamfer-based F-score between two point clouds.
    If point count < target_points, use FPS + interpolation instead of simple repeat.

    :param pred: Tensor [B, N, 3] - predicted point cloud
    :param gt: Tensor [B, M, 3] - ground truth point cloud
    :param threshold: float - distance threshold for F-score
    :param target_points: int - desired number of points per cloud
    :param mode: str - interpolation mode, either "fps" or "repeat"
    :return: precision, recall, fscore (all float)
    """

    def fps_interpolate(x, target):
        B, N, C = x.shape
        if N >= target:
            x_fps, _ = sample_farthest_points(x, K=target)
            return x_fps

        # 插值补点
        idx1 = torch.randint(0, N, (B, target - N), device=x.device)
        idx2 = torch.randint(0, N, (B, target - N), device=x.device)
        rand_alpha = torch.rand(B, target - N, 1, device=x.device)

        # 从 x 中选出 idx1 和 idx2 的点，注意 gather 用法确保维度一致
        pt1 = torch.gather(x, dim=1, index=idx1.unsqueeze(-1).expand(-1, -1, C))  # [B, target-N, 3]
        pt2 = torch.gather(x, dim=1, index=idx2.unsqueeze(-1).expand(-1, -1, C))  # [B, target-N, 3]

        x_interp = rand_alpha * pt1 + (1 - rand_alpha) * pt2  # [B, target-N, 3]

        return torch.cat([x, x_interp], dim=1)  # [B, target, 3]

    def repeat_interpolate(x, target):
        B, N, C = x.shape
        if N >= target:
            return x[:, :target, :]
        repeat_factor = (target + N - 1) // N
        x_repeat = x.repeat(1, repeat_factor, 1)[:, :target, :]
        return x_repeat

    if mode == "fps":
        pred = fps_interpolate(pred, target_points)
        gt = fps_interpolate(gt, target_points)
    elif mode == "repeat":
        pred = repeat_interpolate(pred, target_points)
        gt = repeat_interpolate(gt, target_points)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    dist1 = knn_points(pred, gt, K=1).dists.squeeze(-1)  # [B, N]
    dist2 = knn_points(gt, pred, K=1).dists.squeeze(-1)  # [B, M]

    precision = torch.mean((dist1 < threshold ** 2).float(), dim=1)
    recall = torch.mean((dist2 < threshold ** 2).float(), dim=1)
    fscore = 2 * precision * recall / (precision + recall + 1e-8)
    fscore[torch.isnan(fscore)] = 0

    return precision.mean().item(), recall.mean().item(), fscore.mean().item()
