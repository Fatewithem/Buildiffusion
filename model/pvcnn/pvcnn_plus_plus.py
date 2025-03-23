import torch
import torch.nn as nn

from model.pvcnn.pvcnn import PVCNN2
from model.pvcnn.pvcnn_utils import create_mlp_components
from model.simple.simple_model import SimplePointModel


class PVCNN2PlusPlus(nn.Module):
    def __init__(
        self,
        *,
        embed_dim,  # 嵌入维度，表示特征的维度
        num_classes,  # 分类类别的数量，用于最终输出的分类任务
        extra_feature_channels,  # 额外的特征通道数量，用于输入数据的特征维度
    ):
        super().__init__()
        
        # Create models
        self.simple_point_model = SimplePointModel(num_classes=embed_dim, embed_dim=embed_dim, 
            extra_feature_channels=extra_feature_channels, num_layers=3)
        self.pvcnn = PVCNN2(num_classes=embed_dim, embed_dim=embed_dim, 
            extra_feature_channels=(embed_dim - 3))
        
        # Tie timestep embeddings
        self.pvcnn.embedf = self.simple_point_model.timestep_projection  # 确保两个模型使用相同的时间步嵌入方式

        # # Remove output projections 
        # self.pvcnn.classifier = nn.Identity()
        # self.simple_point_model.output_projection = nn.Identity()

        # Create new output projection
        layers, _ = create_mlp_components(
            in_channels=embed_dim, out_channels=[128, self.pvcnn.dropout, num_classes],
            classifier=True, dim=2, width_multiplier=self.pvcnn.width_multiplier)
        self.output_projection = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor, t: torch.Tensor):
        x = self.simple_point_model(inputs, t)  # (B, D_emb, N)
        x = x + self.pvcnn(x, t)  # (B, D_emb, N)
        x = self.output_projection(x)  # (B, D_out, N)
        return x
