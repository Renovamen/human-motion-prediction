import numpy as np
import torch

def get_dct_matrix(N):
    dct_m = np.eye(N)

    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)

            if k == 0:
                w = np.sqrt(1 / N)

            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)

    idct_m = np.linalg.inv(dct_m)

    dct_m = torch.tensor(dct_m).float().unsqueeze(0).to("cuda")
    idct_m = torch.tensor(idct_m).float().unsqueeze(0).to("cuda")

    return dct_m, idct_m
