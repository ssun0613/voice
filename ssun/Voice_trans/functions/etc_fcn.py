import os,sys
sys.path.append("..")
import numpy as np

import matplotlib
import torch
matplotlib.use("Agg")
import matplotlib.pylab as plt

def quantize_f0_torch(x, num_bins=256):
    # x is logf0
    B = x.size(0) # B=10
    x = x.view(-1).clone() # x.shape : torch.Size([1920])
    uv = (x <= 0)
    x[uv] = 0
    assert (x >= 0).all() and (x <= 1).all()
    x = torch.round(x * (num_bins - 1))
    x = x + 1
    x[uv] = 0
    enc = torch.zeros((x.size(0), num_bins + 1), device=x.device)
    enc[torch.arange(x.size(0)), x.long()] = 1

    quantize_f0 = enc.view(B, -1, num_bins + 1), x.view(B, -1).long()
    # quantize_f0[0].shape : torch.Size([10, 192, 257]), quantize_f0[1].shape : torch.Size([10, 192])
    return quantize_f0

def quantize_f0_numpy(x, num_bins=256):
    # x is logf0
    assert x.ndim == 1
    x = x.astype(float).copy()
    uv = (x <= 0)
    x[uv] = 0.0
    assert (x >= 0).all() and (x <= 1).all()
    x = np.round(x * (num_bins - 1))
    x = x + 1
    x[uv] = 0.0
    enc = np.zeros((len(x), num_bins + 1), dtype=np.float32)
    enc[np.arange(len(x)), x.astype(np.int32)] = 1.0
    return enc, x.astype(np.int64)


if __name__ == '__main__':
    x = torch.rand(10, 192, 1)
    pitch_quan = quantize_f0_torch(x)