import torch
import torch.nn as nn
from cs336_basics import gelu

class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, weights = None):
        super(FFN, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        if weights is not None:
            self.w1 = nn.Parameter(weights['w1.weight'], requires_grad=True)
            self.w2 = nn.Parameter(weights['w2.weight'], requires_grad=True)
        else:
            self.w1 = torch.nn.Parameter(torch.randn((d_ff, d_model)))
            self.w2 = torch.nn.Parameter(torch.randn((d_model, d_ff)))

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        x = in_features@self.w1.T
        x = gelu(x)
        output = x@self.w2.T
        return output