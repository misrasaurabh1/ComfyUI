import torch
from torch import nn


class PixelNorm(nn.Module):
    def __init__(self, dim=1, eps=1e-8):
        super(PixelNorm, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        # Use torch.square for efficiency and clamp denominator for numerical safety
        denom = torch.mean(x.square(), dim=self.dim, keepdim=True).clamp_min(self.eps)
        return x / denom.sqrt()
