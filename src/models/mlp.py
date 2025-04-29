import torch
from torch import nn
from einops import rearrange
from typing import Literal
from .dct import get_dct_matrix

class LayerNormSpatial(nn.Module):
    def __init__(self, dim: int, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, dim, 1]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, dim, 1]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y

class LayerNormTemporal(nn.Module):
    def __init__(self, dim: int, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, 1, dim]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y

class FCSpatial(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        x = rearrange(x, "b d n -> b n d")
        x = self.fc(x)
        x = rearrange(x, "b n d -> b d n")
        return x

class FCTemoral(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.fc(x)
        return x

class MLPBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        seq: int,
        layer_norm: Literal["spatial", "temporal", "all", False] = "spatial",
        use_spatial_fc: bool = False
    ) -> None:
        super().__init__()

        self.fc = FCSpatial(dim) if use_spatial_fc else FCTemoral(seq)

        if layer_norm == False:
            self.norm = nn.Identity()
        elif layer_norm == "spatial":
            self.norm = LayerNormSpatial(dim)
        elif layer_norm == "temporal":
            self.norm = LayerNormTemporal(seq)
        elif layer_norm == "all":
            self.norm = nn.LayerNorm([dim, seq])
        else:
            raise NotImplementedError

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.fc.weight, gain=1e-8)
        nn.init.constant_(self.fc.fc.bias, 0)

    def forward(self, x):
        x_ = self.fc(x)
        x_ = self.norm(x_)
        x = x + x_
        return x

class MLP(nn.Module):
    def __init__(self, dim, seq, layer_norm, use_spatial_fc, num_layers):
        super().__init__()
        self.mlps = nn.Sequential(*[
            MLPBlock(dim, seq, layer_norm, use_spatial_fc)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        x = self.mlps(x)
        return x

class MotionMLP(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.configs = configs

        self.dct_m, self.idct_m = get_dct_matrix(configs.input_length)

        self.mlp = MLP(
            dim=configs.mlp_dim,
            seq=configs.input_length,
            layer_norm=configs.mlp_layer_norm,
            use_spatial_fc=configs.mlp_spatial_fc,
            num_layers=configs.mlp_num_layers
        )

        self.fc_in = nn.Linear(configs.motion_dim, configs.mlp_dim)
        self.fc_out = nn.Linear(configs.mlp_dim, configs.motion_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc_out.weight, gain=1e-8)
        nn.init.constant_(self.fc_out.bias, 0)

    def forward(self, x):
        self.dct_m, self.idct_m = self.dct_m.to(x.device), self.idct_m.to(x.device)

        x_ = x.clone()
        x_ = torch.matmul(self.dct_m, x_)

        motion_feats = self.fc_in(x_)
        motion_feats = rearrange(motion_feats, "b n d -> b d n")

        motion_feats = self.mlp(motion_feats)

        motion_feats = rearrange(motion_feats, "b d n -> b n d")
        motion_feats = self.fc_out(motion_feats)

        motion_feats = torch.matmul(self.idct_m, motion_feats)

        offset = x[:, -1:]
        motion_feats = motion_feats[:, -self.configs.target_length_train:] + offset

        return motion_feats
