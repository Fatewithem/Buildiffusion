import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from omegaconf import DictConfig
from tqdm import tqdm
from einops import repeat
import open3d as o3d

from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union, Any
from cost_volume.cost_volume import DinoFeatureExtractorShapeNet
from .pointnet import PointNet2FeatureExtractorWithFP_ShapeNet
from .transformer import Transformer
from .output import OutputLayer

from model.projection_model import PointCloudProjectionModel
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Pointclouds
from .utils import positionalencoding1d, convert_to_tensor, chamfer_f1_score
from .loss import compute_loss, compute_loss_shapenet

from pytorch3d.loss import chamfer_distance


# 综合模型框架
class R2N2QueryModel(PointCloudProjectionModel):
    def __init__(
            self,
            config: DictConfig,
            **kwargs
    ):
        # 调用父类构造函数
        super(R2N2QueryModel, self).__init__(**kwargs)

        # 特征提取网络
        self.pointcloud_extractor = PointNet2FeatureExtractorWithFP_ShapeNet()
        self.dino_extractor = DinoFeatureExtractorShapeNet(model_path="/home/code/Buildiffusion/cost_volume/dinov2_base")

        # 查询
        self.num_queries = 8
        self.d_model = 768

        # 读取transformer参数
        self.cfg_transformer = config.network.transformer

        # 初始化
        self.learned_queries = nn.Parameter(torch.rand((self.num_queries, self.d_model)))
        self.transformer = Transformer(**self.cfg_transformer)
        self.output_layer = OutputLayer(self.d_model, num_queries=self.num_queries, points_per_query=128) # 500
        self.camera_embed = nn.Linear(12, self.d_model)

        self.pc_mlp = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 768)
        )

    def _per_view_forward(self, pc_feat, img, cam):
        # img: [C, H, W]
        # cam: camera object or (R, T) tuple
        img_feat = self.dino_extractor(img.unsqueeze(0))  # shape: [1, C, H, W]
        B = 1

        R, T = cam.R, cam.T
        R = R.squeeze(0) if R.dim() == 3 else R
        T = T.squeeze(0) if T.dim() == 2 else T
        pose = torch.cat([R.reshape(1, -1), T.reshape(1, -1)], dim=-1)  # shape: [1, 12]
        pose_embedding = self.camera_embed(pose)  # shape: [1, d_model]
        tgt = self.prepare_learned_queries(self.learned_queries, B, pc_feat)
        decoder_attn_mask = self.get_decoder_mask()
        pred = self.transformer(trg=tgt, pc_feat=pc_feat, img_feat=img_feat.flatten(2).permute(0, 2, 1),
                                mod=pose_embedding, trg_mask=decoder_attn_mask)
        output = self.output_layer(pred)  # shape: [1, nq, 3]
        return output

    def forward(
            self,
            pc: Pointclouds,
            camera: Optional[Dict],
            images: Optional[Tensor],  # [B, V, C, H, W]
            blurs: Pointclouds,
    ):
        x = self.point_cloud_to_tensor(blurs, normalize=True, scale=False)
        B, N, D = x.shape

        images = torch.stack(images, dim=0)  # [B, V, C, H, W]
        V = images.shape[1]
        images = images.view(B * V, *images.shape[2:])  # [B*V, C, H, W]

        # Flatten cameras
        flat_cameras = [cam for batch in camera for cam in batch]

        # 特征提取
        pointcloud_features = self.pointcloud_extractor(x[:, :, :3].contiguous())  # (B, N, 254)
        pointcloud_features = pointcloud_features.permute(0, 2, 1)
        pc_feat = self.pc_mlp(pointcloud_features)  # [B, N, 768]
        pc_feat = pc_feat.repeat_interleave(V, dim=0)  # [B*V, N, 768]

        outputs = self._batch_view_forward(pc_feat, images, flat_cameras)

        planes = self.point_cloud_to_tensor(pc, normalize=True, scale=False)[..., :3].contiguous()
        final_loss = 0
        cd_loss = 0
        emd_loss = 0
        idx = 0
        for i in range(B):
            for j in range(V):
                total, cd, emd = compute_loss_shapenet(outputs[idx:idx+1], planes[i:i+1])
                final_loss += total
                cd_loss += cd
                emd_loss += emd
                idx += 1
        final_loss /= V
        return final_loss, cd_loss, emd_loss

    def _batch_view_forward(self, pc_feat, images, cameras):
        B = pc_feat.shape[0]
        img_feats = self.dino_extractor(images)  # shape: [B, C, H, W]
        poses = []
        for cam in cameras:
            R, T = cam.R, cam.T
            R = R.squeeze(0) if R.dim() == 3 else R
            T = T.squeeze(0) if T.dim() == 2 else T
            pose = torch.cat([R.reshape(1, -1), T.reshape(1, -1)], dim=-1)  # shape: [1, 12]
            poses.append(pose)
        poses = torch.cat(poses, dim=0)  # [B, 12]
        pose_embeddings = self.camera_embed(poses)  # [B, d_model]

        tgt = self.prepare_learned_queries(self.learned_queries, B, pc_feat)
        decoder_attn_mask = self.get_decoder_mask()

        pred = self.transformer(
            trg=tgt,
            pc_feat=pc_feat,
            img_feat=img_feats.flatten(2).permute(0, 2, 1),
            mod=pose_embeddings,
            trg_mask=decoder_attn_mask,
        )
        output = self.output_layer(pred)  # [B, nq, 3]
        return output


    @torch.no_grad()
    def forward_sample(
            self,
            pc: Pointclouds,
            # masks: Optional[Tensor],
            camera: Optional[Dict],
            images: Optional[Tensor],  # [B, C, H, W]
            blurs: Optional[Dict],
    ):
        # 和forward保持一致
        x = self.point_cloud_to_tensor(blurs, normalize=True, scale=False)  # (-1, 1)
        B, N, D = x.shape

        images = torch.stack(images, dim=0)

        # 特征提取
        pointcloud_features = self.pointcloud_extractor(x[:, :, :3].contiguous())
        pointcloud_features = pointcloud_features.permute(0, 2, 1)
        pc_feat = self.pc_mlp(pointcloud_features)

        outputs = []
        for i in range(B):
            for j in range(images.shape[1]):
                img = images[i, j]
                cam = camera[i][j]
                pc_feat_i = pc_feat[i:i+1]
                output = self._per_view_forward(pc_feat_i, img, cam)
                outputs.append(output)

        outputs = torch.cat(outputs, dim=0)  # [B*num_views, nq, 3]
        pc_tensor = self.point_cloud_to_tensor(pc, normalize=True, scale=False)  # (-1, 1)

        result = None
        total_prec = 0
        total_rec = 0
        total_f1 = 0
        idx = 0
        for i in range(B):
            view_outputs = []
            view_precs = []
            view_recs = []
            view_f1s = []
            for j in range(images.shape[1]):
                out = outputs[idx:idx+1]
                prec, rec, f1 = chamfer_f1_score(pc_tensor[i:i+1], out, threshold=0.01)
                view_outputs.append(out)
                view_precs.append(prec)
                view_recs.append(rec)
                view_f1s.append(f1)
                idx += 1
            avg_output = torch.mean(torch.cat(view_outputs, dim=0), dim=0, keepdim=True)
            avg_prec = sum(view_precs) / len(view_precs)
            avg_rec = sum(view_recs) / len(view_recs)
            avg_f1 = sum(view_f1s) / len(view_f1s)

            total_prec += avg_prec
            total_rec += avg_rec
            total_f1 += avg_f1

            if result is None:
                result = avg_output
            else:
                result = torch.cat([result, avg_output], dim=0)

        best_prec = total_prec / B
        best_rec = total_rec / B
        best_f1 = total_f1 / B

        result = self.tensor_to_point_cloud(result, denormalize=False, unscale=False);

        return result, pc, best_prec, best_rec, best_f1

    def prepare_learned_queries(self, learned_queries: torch.Tensor, batch_size: int, pc_feat: torch.Tensor) -> torch.Tensor:
        """
            Prepare the learned decomposition factor queries for the decoder
            Adds sine-cos positional encoding and replicates single-set of queries among the batch dimension
            Fuses coarse point cloud features as global summary to each query
        :param learned_queries: Single set of learned queries, shape: [num_queries (12), d_model (768)]
        :param batch_size: Size of the batch
        :param pc_feat: Point cloud features, shape: [B, N, d_model]
        :return: Learned decomposition factor queries fused with pc global summary
        """

        # Add positional encoding
        # pos_enc.shape => same as `learned_queries` | [num_queries (12), d_model (768)]
        pos_enc = positionalencoding1d(self.d_model, self.num_queries).to(self.device)
        learned_queries = learned_queries + pos_enc  # [nq, d_model]
        learned_queries = repeat(learned_queries, 'nq dmodel -> b nq dmodel', b=batch_size)  # [B, nq, d_model]

        # 点云全局摘要特征，作为每个 query 的几何感知输入
        pc_summary = pc_feat.mean(dim=1, keepdim=True).expand(-1, self.num_queries, -1)  # [B, nq, d_model]

        fused_query = learned_queries + pc_summary
        return fused_query

    def get_decoder_mask(self):
        """
            Generate decoder-side attention mask
        :return: Attention mask
        """
        # Create boolean identity matrix
        tgt_mask = torch.ones((self.num_queries, self.num_queries), dtype=torch.bool, device=self.device)
        # Select diagonal entries
        tgt_mask_diag = torch.diagonal(tgt_mask)
        # Replace diagonal entries with False
        tgt_mask_diag[:] = False
        # Replace diagonals with -inf and everything else with 0
        tgt_mask = tgt_mask.float()
        tgt_mask = tgt_mask.masked_fill(tgt_mask == 0, float('-inf'))
        tgt_mask = tgt_mask.masked_fill(tgt_mask == 1, float(0.0))
        return tgt_mask

    def render_pointcloud(self, pointcloud_tensor: Tensor, camera: Dict, images: Tensor) -> Tensor:
        """
        将输出的点云进行渲染为图像，用于2D投影loss计算。
        这里只是一个模板，需要根据你的渲染逻辑完成。
        """
        # 示例：使用pytorch3d进行可微渲染
        from pytorch3d.renderer import (
            FoVPerspectiveCameras, PointsRasterizationSettings,
            PointsRenderer, AlphaCompositor, PointsRasterizer,
            NormWeightedCompositor
        )
        from pytorch3d.structures import Pointclouds
        import torchvision
        import os

        cameras = camera.to(pointcloud_tensor.device)
        point_clouds = Pointclouds(points=pointcloud_tensor, features=torch.ones_like(pointcloud_tensor))

        raster_settings = PointsRasterizationSettings(
            image_size=images.shape[-1],
            radius=0.02,
            points_per_pixel=10
        )

        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        renderer = PointsRenderer(rasterizer=rasterizer, compositor=AlphaCompositor())

        rendered = renderer(point_clouds)

        # img_to_save = rendered[0].clamp(0.0, 1.0).cpu()
        # save_path = "/home/code/Buildiffusion/color/image.png"
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # torchvision.utils.save_image(img_to_save.permute(2, 0, 1), save_path)

        return rendered.permute(0, 3, 1, 2)
