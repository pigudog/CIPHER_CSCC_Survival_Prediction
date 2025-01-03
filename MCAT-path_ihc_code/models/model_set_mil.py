from collections import OrderedDict
from os.path import join
import pdb

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import *



###################### path only#######################################################
class MIL_Attention_FC_surv_Net(nn.Module):
    def __init__(self, input_dim=512, size_arg="small", dropout=0.25, n_classes=4):
        r"""
        Attention MIL Implementation

        Args:
            omic_input_dim (int): Dimension size of genomic features.
            fusion (str): Fusion method (Choices: concat, bilinear, or None)
            size_arg (str): Size of NN architecture (Choices: small or large)
            dropout (float): Dropout rate
            n_classes (int): Output shape of NN
        """
        super(MIL_Attention_FC_surv_Net, self).__init__()

        self.size_dict_path = {"small": [input_dim, 256, 256], "big": [1024, 512, 384]}

        ### Deep Sets Architecture Construction
        size = self.size_dict_path[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.rho = nn.Sequential(*[nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)])

        self.classifier = nn.Linear(size[2], n_classes)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to('cuda:0')

        self.rho = self.rho.to(device)
        self.classifier = self.classifier.to(device)

    def forward(self, x):
        A, h_path = self.attention_net(x)
        A = torch.transpose(A, 1, 0)
        A_raw = A
        A = F.softmax(A, dim=1)
        h_path = torch.mm(A, h_path)
        h_path = self.rho(h_path).squeeze()

        h = h_path  # [256] vector

        logits = self.classifier(h).unsqueeze(0)  # logits needs to be a [1 x 4] vector
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        return hazards, S, Y_hat, None, None


class IHC_Projector(nn.Module):
    """Projector layer with two fully connected layers."""
    def __init__(self, input_dim: int, hidden_dim: int):
        super(IHC_Projector, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x


class WSI_Projector(nn.Module):
    """Projector layer with two fully connected layers."""
    def __init__(self, input_dim: int, hidden_dim: int):
        super(WSI_Projector, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x


class MultiFeatureFusionNetworkAdvanced(nn.Module):
    def __init__(self, feature_dim=512, num_features=12):
        super(MultiFeatureFusionNetworkAdvanced, self).__init__()
        self.num_features = num_features
        self.feature_dim = feature_dim

        # Transformer Encoder 用于更好地融合预训练特征
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # 门控机制，用于学习每个特征的贡献
        self.gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )

        # 用于融合预训练特征和feat_wsi的自注意力模块
        self.attention_layer_final = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=4, batch_first=True)

    def forward(self, pretrained_features, feat_wsi):
        """
        Args:
            pretrained_features: list of 12 tensors, each with shape (n, feature_dim)
            feat_wsi: Tensor of shape (n, feature_dim)

        Returns:
            fused_features: Tensor of shape (n, feature_dim)
        """
        # 将预训练特征堆叠起来，形状为 (n, num_features, feature_dim)
        stacked_features = torch.stack(pretrained_features, dim=1)

        # 使用 Transformer Encoder 层对预训练特征进行融合
        fused_pretrained_features = self.transformer_encoder(stacked_features)

        # 使用门控机制对每个特征的贡献进行学习
        gate_weights = self.gate(fused_pretrained_features)  # 形状为 (n, num_features, 1)
        gated_features = fused_pretrained_features * gate_weights  # 形状为 (n, num_features, feature_dim)

        # 对融合后的特征取平均，得到形状为 (n, feature_dim) 的张量
        fused_pretrained_features = torch.mean(gated_features, dim=1)

        # 将融合后的预训练特征与 feat_wsi 堆叠起来，形成形状为 (n, 2, feature_dim)
        combined_features = torch.stack([fused_pretrained_features, feat_wsi], dim=1)

        # 使用多头注意力层融合预训练特征和 feat_wsi
        attn_output_final, _ = self.attention_layer_final(combined_features, combined_features, combined_features)

        # 对融合后的特征取平均，得到形状为 (n, feature_dim) 的张量
        fused_features = torch.mean(attn_output_final, dim=1)

        return fused_features

# Example usage:
# Assuming mri_features is a Tensor of shape (batch_size, 256)
# and wsi_features is a Tensor of shape (batch_size, 512)
# mri_features, wsi_features = ...
# model = Multimodal_surNet()
# hazards, S, Y_hat, _, _ = model(mri_features, wsi_features)
