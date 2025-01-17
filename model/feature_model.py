import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers import ModelMixin
from timm.models.vision_transformer import VisionTransformer, resize_pos_embed
from torch import Tensor
from torchvision.transforms import functional as TVF
import inspect

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

MODEL_URLS = {
    'vit_base_patch16_224_mae': '/home/code/Buildiffusion/vit_model/mae_pretrain_vit_base.pth',
    # 'vit_small_patch16_224_msn': '/home/code/Buildiffusion/vit_model/vits16_800ep.pth.tar',
    # 'vit_large_patch7_224_msn': '/home/code/Buildiffusion/vit_model/vitl7_200ep.pth.tar',
}

NORMALIZATION = {
    'vit_base_patch16_224_mae': (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    # 'vit_small_patch16_224_msn': (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    # 'vit_large_patch7_224_msn': (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
}

MODEL_KWARGS = {
    'vit_base_patch16_224_mae': dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
    ),
    # 'vit_small_patch16_224_msn': dict(
    #     patch_size=16, embed_dim=384, depth=12, num_heads=6,
    # ),
    # 'vit_large_patch7_224_msn': dict(
    #     patch_size=7, embed_dim=1024, depth=24, num_heads=16,
    # )
}


class FeatureModel(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
            self,
            image_height: int = 546,  # 修改为支持指定高度
            image_width: int = 966,  # 修改为支持指定宽度
            model_name: str = 'vit_small_patch16_224_mae',
            global_pool: str = '',  # '' or 'token'
    ) -> None:
        super().__init__()
        self.model_name = model_name

        # Identity
        if self.model_name == 'identity':
            return

        # Create model
        self.model = VisionTransformer(
            img_size=(image_height, image_width), num_classes=0, global_pool=global_pool,
            **MODEL_KWARGS[model_name])

        # Model properties
        self.feature_dim = self.model.embed_dim
        self.mean, self.std = NORMALIZATION[model_name]

        # # Modify MSN model with output head from training
        # if model_name.endswith('msn'):
        #     use_bn = True
        #     emb_dim = (192 if 'tiny' in model_name else 384 if 'small' in model_name else
        #         768 if 'base' in model_name else 1024 if 'large' in model_name else 1280)
        #     hidden_dim = 2048
        #     output_dim = 256
        #     self.model.fc = None
        #     fc = OrderedDict([])
        #     fc['fc1'] = torch.nn.Linear(emb_dim, hidden_dim)
        #     if use_bn:
        #         fc['bn1'] = torch.nn.BatchNorm1d(hidden_dim)
        #     fc['gelu1'] = torch.nn.GELU()
        #     fc['fc2'] = torch.nn.Linear(hidden_dim, hidden_dim)
        #     if use_bn:
        #         fc['bn2'] = torch.nn.BatchNorm1d(hidden_dim)
        #     fc['gelu2'] = torch.nn.GELU()
        #     fc['fc3'] = torch.nn.Linear(hidden_dim, output_dim)
        #     self.model.fc = torch.nn.Sequential(fc)

        # Load pretrained checkpoint
        # checkpoint = torch.hub.load_state_dict_from_url(MODEL_URLS[model_name])
        checkpoint = torch.load("/home/code/Buildiffusion/vit_model/mae_pretrain_vit_base.pth", map_location='cpu')

        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'target_encoder' in checkpoint:
            state_dict = checkpoint['target_encoder']
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            # NOTE: Comment the line below if using the projection head, uncomment if not using it
            # See https://github.com/facebookresearch/msn/blob/81cb855006f41cd993fbaad4b6a6efbb486488e6/src/msn_train.py#L490-L502
            # for more info about the projection head
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
        else:
            raise NotImplementedError()
        # state_dict['pos_embed'] = resize_pos_embed(state_dict['pos_embed'], self.model.pos_embed)

        # print(inspect.signature(resize_pos_embed))

        # 调整 pos_embed 的大小
        if 'pos_embed' in state_dict:
            # 从预训练权重中加载的 `pos_embed`
            pos_embed_checkpoint = state_dict['pos_embed']

            # 获取当前模型的 `pos_embed`，以确定目标形状
            pos_embed_model = self.model.pos_embed

            # 确定预训练和当前模型的网格大小
            num_tokens_checkpoint = pos_embed_checkpoint.shape[1]
            num_tokens_model = pos_embed_model.shape[1]

            # 处理位置嵌入大小不一致
            if 'pos_embed' in state_dict:
                # 获取预训练的 pos_embed
                pos_embed_checkpoint = state_dict['pos_embed']

                # 拆分 CLS token 和 Patch token
                cls_token = pos_embed_checkpoint[:, :1]  # 取出 CLS token (1, 1, 768)
                patch_embed_checkpoint = pos_embed_checkpoint[:, 1:]  # 去掉 CLS token (1, num_patches, 768)

                # 获取当前模型的 Patch token 数量
                num_patches_checkpoint = patch_embed_checkpoint.shape[1]
                num_patches_model = self.model.pos_embed.shape[1] - 1  # 减去 CLS token

                # 调整 Patch token 的形状
                if num_patches_checkpoint != num_patches_model:
                    gs_checkpoint = int(math.sqrt(num_patches_checkpoint))  # 原始网格大小
                    gs_model = int(math.sqrt(num_patches_model))  # 当前模型网格大小

                    # 插值调整 Patch token
                    patch_embed_checkpoint = patch_embed_checkpoint.reshape(1, gs_checkpoint, gs_checkpoint, -1)
                    patch_embed_checkpoint = F.interpolate(
                        patch_embed_checkpoint, size=(gs_model, gs_model), mode='bicubic', align_corners=False
                    )
                    patch_embed_checkpoint = patch_embed_checkpoint.reshape(1, num_patches_model, -1)  # 恢复形状

                # 重新拼接 CLS token 和调整后的 Patch token
                pos_embed_checkpoint = torch.cat([cls_token, patch_embed_checkpoint], dim=1)

                # 更新 state_dict
                state_dict['pos_embed'] = pos_embed_checkpoint

        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        # # Modify MSN model with output head from training
        # if model_name.endswith('msn'):
        #     self.fc = self.model.fc
        #     del self.model.fc
        # else:
        #     self.fc = nn.Identity()

        # NOTE: I've disabled the whole projection head stuff for simplicity for now
        self.fc = nn.Identity()

    def denormalize(self, img: Tensor):
        img = TVF.normalize(img, mean=[-m / s for m, s in zip(self.mean, self.std)], std=[1 / s for s in self.std])
        return torch.clip(img, 0, 1)

    def normalize(self, img: Tensor):
        return TVF.normalize(img, mean=self.mean, std=self.std)

    def forward(
            self,
            x: Tensor,
            return_type: str = 'features',
            return_upscaled_features: bool = True,
            return_projection_head_output: bool = False,
    ):
        """Normalizes the input `x` and runs it through `model` to obtain features"""
        assert return_type in {'cls_token', 'features', 'all'}

        # Identity
        if self.model_name == 'identity':
            return x

        # Normalize and forward
        B, C, H, W = x.shape

        print(f"Shape: {x.shape}")

        x = self.normalize(x)
        feats = self.model(x)

        print(f"Feats: {feats.shape}")

        # Reshape to image-like size
        if return_type in {'features', 'all'}:
            B, T, D = feats.shape
            assert math.sqrt(T - 1).is_integer()
            HW_down = int(math.sqrt(T - 1))  # subtract one for CLS token
            output_feats: Tensor = feats[:, 1:, :].reshape(B, HW_down, HW_down, D).permute(0, 3, 1,
                                                                                           2)  # (B, D, H_down, W_down)
            if return_upscaled_features:
                output_feats = F.interpolate(output_feats, size=(H, W), mode='bilinear',
                                             align_corners=False)  # (B, D, H_orig, W_orig)

        # Head for MSN
        output_cls = feats[:, 0]
        if return_projection_head_output and return_type in {'cls_token', 'all'}:
            output_cls = self.fc(output_cls)

        # Return
        if return_type == 'cls_token':
            return output_cls
        elif return_type == 'features':
            return output_feats
        else:
            return output_cls, output_feats
