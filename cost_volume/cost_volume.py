import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import math
import json
import random
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

import torchvision.transforms as transforms
import plotly.graph_objects as go


from .utils import compute_intrinsic_matrix, prepare_camera_projections, homo_warping_3D_with_mask, load_image_as_tensor, load_config
from .module import *


# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


class DinoFeatureExtractor(nn.Module):
    def __init__(self, model_path):
        """
        初始化特征提取器。
        :param model_path: DINO 模型路径
        """
        super(DinoFeatureExtractor, self).__init__()
        self.processor = AutoImageProcessor.from_pretrained(
            model_path,
            size={"height": 546, "width": 966},
            do_resize=False,
            do_rescale=False
        )
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32)
        self.model.eval()  # 设置模型为评估模式
        for param in self.model.parameters():
            param.requires_grad_(False)  # 关闭梯度，使其在反向传播中不被更新

    def forward(self, images, return_type="features"):
        """
        对输入图片直接提取特征。
        :param images: 输入图片张量，形状为 [B, C, H, W]
        :return: 提取的特征张量 [B, Features, H_tokens, W_tokens]
        """
        device = next(self.model.parameters()).device  # 获取模型所在设备

        # 确保输入数据类型与模型一致
        images = images.to(dtype=torch.float32, device=device)
        images = torch.clamp(images, 0, 1)

        # 转换为适合模型输入的格式
        images = images.permute(0, 2, 3, 1).cpu().numpy()  # 转为 numpy 格式
        inputs = self.processor(images=list(images), return_tensors="pt")
        inputs = {k: v.to(dtype=torch.float32, device=device) for k, v in inputs.items()}  # 转换为模型的设备和类型

        # 特征提取
        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = self.model(**inputs)

        # # 提取特征
        # features = outputs.last_hidden_state[:, 1:, :]  # [B, Tokens - CLS, Features]
        #
        # # 重新调整特征维度
        # B, Tokens, Features = features.shape
        # H_tokens, W_tokens = 546 // self.model.config.patch_size, 966 // self.model.config.patch_size
        # features = features.view(B, H_tokens, W_tokens, Features)
        # features = features.permute(0, 3, 1, 2)  # [B, Features, H_tokens, W_tokens]
        #
        # return features

        if return_type == "features":
            features = outputs.last_hidden_state[:, 1:, :]  # [B, Tokens - CLS, Features]
            B, Tokens, Features = features.shape
            H_tokens, W_tokens = 546 // self.model.config.patch_size, 966 // self.model.config.patch_size
            features = features.view(B, H_tokens, W_tokens, Features)
            features = features.permute(0, 3, 1, 2)  # [B, Features, H_tokens, W_tokens]
            features = F.interpolate(
                features,
                size=(546, 966),  # 目标尺寸
                mode='bicubic',
                align_corners=False
            )
            return features

        elif return_type == "cls_token":
            return outputs.last_hidden_state[:, 0, :]  # 返回 CLS token 特征 [B, Features]

        else:
            raise ValueError(f"Unsupported return_type: {return_type}")

#
# def channel_range_normalization(input_tensor):
#     B, C, H, W = input_tensor.shape
#     normalized_channels = []
#
#     for c in range(C):  # 遍历每个通道
#         channel = input_tensor[:, c, :, :]  # 取出第 c 个通道 [B, H, W]
#         channel_min = channel.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
#         channel_max = channel.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
#         normalized_channel = (channel - channel_min) / (channel_max - channel_min + 1e-8)  # 归一化到 [0, 1]
#
#         # 如果需要归一化到 [-1, 1]，取消下面注释
#         # normalized_channel = 2 * normalized_channel - 1
#
#         normalized_channels.append(normalized_channel)
#
#     # 拼接回原来的形状 [B, 32, H, W]
#     normalized_tensor = torch.stack(normalized_channels, dim=1)
#     return normalized_tensor


# def multi_scale_cost_volume(self, ref_features, src_features_list, depth_values, ref_proj, src_projs, device):
#     """
#     构建多尺度代价体积
#     """
#
#     src_proj_list = torch.unbind(src_projs, dim=1)
#
#     volume_sum = 0.0
#     vis_sum = 0.0
#
#     vis = self.vis
#     vis = vis.to(device=device)
#
#     with torch.cuda.amp.autocast():
#         for src_feat, src_proj in zip(src_features_list, src_proj_list):
#             # 调用函数
#             warped_volume, _ = homo_warping_3D_with_mask(src_feat, ref_proj, src_proj, depth_values)
#
#             B, C, D, H, W = warped_volume.shape
#
#             # ref_volume 的 dtype 和形状
#             ref_volume = ref_features.view(B, C, 1, H, W)
#
#             # 计算 in_prod_vol
#             in_prod_vol = ref_volume * warped_volume  # [B,C(G),D,H,W]
#
#             # 计算 sim_vol
#             sim_vol = in_prod_vol.sum(dim=1)  # [B,D,H,W]
#
#             # 计算 sim_vol_norm
#             sim_vol_norm = F.softmax(sim_vol.detach(), dim=1)
#
#             # 计算 entropy
#             entropy = (-sim_vol_norm * torch.log(sim_vol_norm + 1e-7)).sum(dim=1, keepdim=True)
#
#             # 计算 vis_weight
#             vis_weight = vis(entropy)
#
#             # 更新 volume_sum 和 vis_sum
#             volume_sum = volume_sum + in_prod_vol * vis_weight.unsqueeze(1)
#             vis_sum = vis_sum + vis_weight
#
#         # aggregate multiple feature volumes by variance
#         volume_mean = volume_sum / (vis_sum.unsqueeze(1) + 1e-6)  # volume_sum / (num_views - 1)
#
#     # config_path = "/home/code/Dino/config/config.json"  # 配置文件路径
#     # config = load_config(config_path)
#     # transformer_config = config['transformer_config']  # 提取 transformer 配置
#     #
#     # in_channels = 768
#     # transformer_config[0]['base_channel'] = in_channels
#     # cost_reg = PureTransformerCostReg(in_channels, **transformer_config[0])
#     # cost_reg = cost_reg.to(device=device)
#
#     position3d = None
#     cost = self.cost_reg(volume_mean, position3d)
#     prob_volume_pre = cost.squeeze(1)
#
#     prob_volume = F.softmax(prob_volume_pre, dim=1)
#
#     # 各个通道归一化
#     # prob_volume = channel_range_normalization(prob_volume)
#
#     # 插值prob_volume_pre
#     prob_volume_resized = F.interpolate(
#         prob_volume,
#         size=(546, 966),  # 目标尺寸
#         mode='bicubic',
#         align_corners=False
#     )
#
#     outputs = {'prob_volume': prob_volume_resized}
#
#     return outputs
#
#
# class MVSFormerWithDino(nn.Module):
#     def __init__(self, dino_model_path, transformer_config):
#         super(MVSFormerWithDino, self).__init__()
#         self.vis = nn.Sequential(
#             ConvBnReLU(1, 16),
#             ConvBnReLU(16, 16),
#             ConvBnReLU(16, 8),
#             nn.Conv2d(8, 1, 1),
#             nn.Sigmoid()
#         )
#         self.in_channels = 768
#         self.feature_extractor = DinoFeatureExtractor(dino_model_path)
#         self.cost_reg = PureTransformerCostReg(self.in_channels, **transformer_config[0])
#
#     def forward(self, ref_images, src_images_list, depth_values, ref_proj, src_projs, device):
#         ref_features = self.feature_extractor(ref_images)
#         src_features_list = [self.feature_extractor(src_image) for src_image in src_images_list]
#
#         outputs = multi_scale_cost_volume(self, ref_features, src_features_list, depth_values, ref_proj, src_projs, device)
#         return outputs
#
#
# def build_cost_volume(B, rgb_features, masks, model, device="cuda"):
#     """
#     处理多视图立体匹配的主函数。
#
#     :param rgb_features: RGB图像特征，形状为 [B, C, H, W]
#     :param B: 批量大小
#     :param dino_model_path: DINO模型路径
#     :param device: 使用的设备 ("cuda" 或 "cpu")
#     :return: 计算结果，包括深度图和其他中间结果
#     """
#     set_seed(42)
#
#     # 图像尺寸和深度范围
#     H, W = 546, 966  # 插值后图像尺寸
#     D = 64  # 深度假设数量
#     depth_min, depth_max = 0.1, 20.0  # 深度范围
#
#     dino_model_path = "/home/code/Buildiffusion/cost_volume/dinov2_base"
#
#     # 提取参考图像和源图像列表
#     ref_images = rgb_features[:, 0, :, :].to(device)  # [B, H, W]
#     print(ref_images.shape)
#     src_images_list = [rgb_features[:, i, :, :].to(device) for i in range(1, 5)]  # 4 个 [B, H, W]
#
#     ref_mask = masks[:, 0, :, :].unsqueeze(1).to(device)  # [B,1,H,W]
#     src_masks_list = [
#         masks[:, i, :, :].unsqueeze(1).to(device)
#         for i in range(1, 5)
#     ]
#
#     ref_images = ref_images * (1 - ref_mask).to(dtype=torch.float32, device=device)
#     src_images_list = [img * (1 - m) for img, m in zip(src_images_list, src_masks_list)]
#
#     # 相机参数定义
#     camera_params = [
#         {"name": "_1", "T": [-7.8681e-07, 2.9535e+00, 4.5796e+01], "R": [[4.3711388e-08, -3.8213709e-15, 1.0000000e+00],
#                                                                          [-2.5881904e-01, 9.6592581e-01, 1.1313344e-08],
#                                                                          [-9.6592581e-01, -2.5881904e-01, 4.2221959e-08]]},  # 相机 1
#         {"name": "_0", "T": [-4.8083e-06, -2.4041e-06, 5.5000e+01], "R": [[-1.000000e+00, 8.742278e-08, 0.000000e+00],
#                                                                           [3.821371e-15, 4.371139e-08, -1.000000e+00],
#                                                                           [-8.742278e-08, -1.000000e+00, -4.371139e-08]]},  # 相机 0
#         {"name": "_2", "T": [-7.8681e-07, 2.9535e+00, 4.5796e+01],  "R": [[4.3711388e-08, -3.8213709e-15, -1.0000000e+00],
#                                                                           [2.5881922e-01, 9.6592581e-01, 1.1313344e-08],
#                                                                           [9.6592581e-01, -2.5881922e-01, 4.2221959e-08]]},  # 相机 2
#         {"name": "_3", "T": [-7.8681e-07, 2.9535e+00, 4.5796e+01],  "R": [[1.00000000e+00, -8.74227766e-08, -8.74227766e-08],
#                                                                           [1.07070605e-07, 9.65925813e-01, 2.58819133e-01],
#                                                                           [6.18172322e-08, -2.58819133e-01, 9.65925813e-01]]},  # 相机 3
#         {"name": "_4", "T": [-7.8681e-07, 2.9535e+00, 4.5796e+01], "R": [[-1.0000000e+00, 8.7422777e-08, 0.0000000e+00],
#                                                                          [8.4443919e-08, 9.6592581e-01, -2.5881913e-01],
#                                                                          [-2.2626688e-08, -2.5881913e-01, -9.6592581e-01]]},  # 相机 4
#     ]
#
#     # 内参矩阵
#     fx, fy = 75.075, 75.075  # 焦距
#     cx, cy = 34.5, 19.5  # 主点
#     scale = 1
#     K = compute_intrinsic_matrix(fx, fy, cx, cy, scale, device=device)
#
#     # 计算投影矩阵
#     proj_matrices = prepare_camera_projections(camera_params, K, device=device)
#     ref_proj = proj_matrices[0].unsqueeze(0).expand(B, -1, -1)  # [B, 4, 4]
#     src_projs = torch.stack(proj_matrices[1:], dim=0).unsqueeze(0).expand(B, -1, -1, -1)  # [B, N_src, 4, 4]
#
#     # 深度假设
#     depth_values = torch.linspace(depth_min, depth_max, D).view(1, D).expand(B, -1).to(device)  # [B, D]
#
#     # config_path = "/home/code/Dino/config/config.json"
#     # config = load_config(config_path)  # 确保此函数正确运行
#     # transformer_config = config['transformer_config']  # 确保 transformer_config 正确加载
#
#     # 定义模型
#     # model = MVSFormerWithDino(dino_model_path, transformer_config).to(device)
#
#     # 推理
#     with torch.no_grad():
#         outputs = model(ref_images, src_images_list, depth_values, ref_proj, src_projs, device)
#
#     return outputs






