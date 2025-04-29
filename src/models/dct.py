import torch

def get_dct_matrix(N: int):
    i = torch.arange(N).float()
    k = torch.arange(N).float().unsqueeze(1)

    w = torch.sqrt(torch.ones_like(k) * (2.0 / N))
    w[0] = (1.0 / N) ** 0.5

    dct_m = w * torch.cos(torch.pi * (i + 0.5) * k / N)
    idct_m = torch.linalg.inv(dct_m)

    return dct_m.unsqueeze(0), idct_m.unsqueeze(0)
