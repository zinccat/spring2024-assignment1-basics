import torch

import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, weight=None):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        if weight is not None:
            self.weight = weight
        else:
            self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, activation):
        norm = torch.sqrt((activation**2).mean(-1, keepdim=True) + self.eps)
        return activation / norm * self.weight

if __name__ == '__main__':
    rmsnorm = RMSNorm(512)
    ac = torch.randn(512)
    print(rmsnorm(ac))