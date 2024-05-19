import torch
import torch.nn as nn
from cs336_basics import gelu

class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, weights = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = nn.Linear(d_ff, d_model, False)
        self.w2 = nn.Linear(d_model, d_ff, False)
        self.w1.weight.data = weights['w1.weight']
        self.w2.weight.data = weights['w2.weight']

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        x = self.w1(in_features)
        x = gelu(x)
        output = self.w2(x)
        return output