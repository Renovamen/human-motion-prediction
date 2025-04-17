import torch
from torch import nn
from einops.layers.torch import Rearrange

class LayNormSpatial(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
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

class LayNormTemporal(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
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
    def __init__(self, dim):
        super().__init__()

        self.fc = nn.Linear(dim, dim)
        self.arr0 = Rearrange("b n d -> b d n")
        self.arr1 = Rearrange("b d n -> b n d")

    def forward(self, x):
        x = self.arr0(x)
        x = self.fc(x)
        x = self.arr1(x)
        return x

class FCTemoral(nn.Module):
    def __init__(self, dim):
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
        use_norm: bool = True,
        use_spatial_fc: bool = False,
        layer_norm_axis: str = "spatial"
    ) -> None:
        super().__init__()

        if not use_spatial_fc:
            self.fc = FCTemoral(seq)
        else:
            self.fc = FCSpatial(dim)

        if use_norm:
            if layer_norm_axis == "spatial":
                self.norm = LayNormSpatial(dim)
            elif layer_norm_axis == "temporal":
                self.norm = LayNormTemporal(seq)
            elif layer_norm_axis == "all":
                self.norm = nn.LayerNorm([dim, seq])
            else:
                raise NotImplementedError
        else:
            self.norm = nn.Identity()

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
    def __init__(self, dim, seq, use_norm, use_spatial_fc, num_layers, layer_norm_axis):
        super().__init__()
        self.mlps = nn.Sequential(*[
            MLPBlock(dim, seq, use_norm, use_spatial_fc, layer_norm_axis)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        x = self.mlps(x)
        return x
