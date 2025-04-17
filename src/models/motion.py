import torch
from torch import nn
from einops.layers.torch import Rearrange
from .mlp import MLP
from .dct import get_dct_matrix

class MotionPrediction(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.configs = configs

        self.arr0 = Rearrange("b n d -> b d n")
        self.arr1 = Rearrange("b d n -> b n d")

        self.dct_m, self.idct_m = get_dct_matrix(configs.amass_input_length)

        self.mlp = MLP(
            dim=configs.mlp_dim,
            seq=configs.amass_input_length,
            use_norm=configs.mlp_with_normalization,
            use_spatial_fc=configs.mlp_spatial_fc_only,
            num_layers=configs.mlp_num_layers,
            layer_norm_axis=configs.mlp_norm_axis
        )

        self.fc_in = nn.Linear(configs.motion_dim, configs.motion_dim)
        self.fc_out = nn.Linear(configs.motion_dim, configs.motion_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc_out.weight, gain=1e-8)
        nn.init.constant_(self.fc_out.bias, 0)

    def forward(self, x):
        x_ = x.clone()
        x_ = torch.matmul(self.dct_m, x_)

        motion_feats = self.fc_in(x_)
        motion_feats = self.arr0(motion_feats)

        motion_feats = self.mlp(motion_feats)

        motion_feats = self.arr1(motion_feats)
        motion_feats = self.fc_out(motion_feats)

        motion_feats = torch.matmul(self.idct_m, motion_feats)

        offset = x[:, -1:]
        motion_feats = motion_feats[:, :self.configs.amass_target_length] + offset

        return motion_feats
