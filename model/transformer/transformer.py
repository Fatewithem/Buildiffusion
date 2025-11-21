"""
    Transformer class definition

    The implementation mainly follows the implementation found in the PyTorch
        with added support of pre-residual connection normalization.

    Resources used to develop this script:
        - https://github.com/jwang0306/transformer-pytorch
"""

import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(hidden_size, filter_size)
        self.fc2 = nn.Linear(hidden_size, filter_size)  # Gated Linear Unit
        self.fc3 = nn.Linear(filter_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src = self.fc1(src) * torch.sigmoid(self.fc2(src))  # GLU 机制
        src = self.dropout(self.fc3(src))
        return src


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.scale * norm


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, n_head, pre_lnorm, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(hidden_size, n_head, dropout=dropout, batch_first=True)
        self.self_attn_norm = RMSNorm(hidden_size)

        self.cross_attn_img = nn.MultiheadAttention(hidden_size, n_head, dropout=dropout, batch_first=True)
        self.cross_attn_img_norm = RMSNorm(hidden_size)

        self.cross_attn_pc = nn.MultiheadAttention(hidden_size, n_head, dropout=dropout, batch_first=True)
        self.cross_attn_pc_norm = RMSNorm(hidden_size)

        # feed forward network part
        self.pff = PositionwiseFeedForward(hidden_size, filter_size, dropout)
        self.pff_norm = RMSNorm(hidden_size)

        self.pre_lnorm = pre_lnorm

        self.modulation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size * 2)
        )

    def forward(self, trg, img_feat, pc_feat, trg_mask, mod):
        if self.pre_lnorm:
            ris = self.self_attn_norm(trg)
            gamma, beta = self.modulation(mod).chunk(2, dim=-1)
            gamma, beta = gamma.unsqueeze(1), beta.unsqueeze(1)
            trg = trg * (1 + gamma) + beta
            trg = trg + self.self_attn(ris, ris, ris, attn_mask=trg_mask)[0]

            ris = self.cross_attn_img_norm(trg)
            img_ctx = self.cross_attn_img(ris, img_feat, img_feat)[0]

            ris = self.cross_attn_pc_norm(trg)
            pc_ctx = self.cross_attn_pc(ris, pc_feat, pc_feat)[0]

            trg = trg + img_ctx + pc_ctx

            ris = self.pff_norm(trg)
            trg = trg + self.pff(ris)
        else:
            trg = self.self_attn_norm(trg + self.self_attn(trg, trg, trg, attn_mask=trg_mask)[0])

            trg = self.cross_attn_img_norm(trg + self.cross_attn_img(trg, img_feat, img_feat)[0])
            trg = self.cross_attn_pc_norm(trg + self.cross_attn_pc(trg, pc_feat, pc_feat)[0])

            trg = self.pff_norm(trg + self.pff(trg))

        return trg


class Decoder(nn.Module):
    def __init__(self, hidden_size, filter_size, n_head, dropout, n_layers, pre_lnorm):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.embed_scale = hidden_size ** 0.5
        self.layers = nn.ModuleList(
            [DecoderLayer(hidden_size, filter_size, n_head, pre_lnorm, dropout) for _ in range(n_layers)])
        self.pre_lnorm = pre_lnorm
        self.last_norm = RMSNorm(hidden_size)

    def forward(self, trg, img_feat, pc_feat, trg_mask=None, mod=None):
        for layer in self.layers:
            trg = layer(trg, img_feat, pc_feat, trg_mask, mod)

        if self.pre_lnorm:
            trg = self.last_norm(trg)

        return trg


class Transformer(nn.Module):
    def __init__(self, d_model, dim_feedforward, nhead, dropout, num_decoder_layers, pre_lnorm=True):
        super(Transformer, self).__init__()
        self.decoder = Decoder(d_model, dim_feedforward, nhead, dropout, num_decoder_layers, pre_lnorm)

    def forward(self, trg, pc_feat, img_feat, mod=None, trg_mask=None):
        dec_out = self.decoder(trg, img_feat, pc_feat, trg_mask, mod)
        return dec_out

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
