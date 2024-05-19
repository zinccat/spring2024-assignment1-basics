import torch
import math
from typing import Iterable

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    total_norm = 0
    for param in parameters:
        if param.grad is not None:
            total_norm += torch.linalg.norm(param.grad, 'fro') ** 2
    total_norm = math.sqrt(total_norm)
    if total_norm > max_l2_norm:
        for param in parameters:
            if param.grad is not None:
                param.grad *= max_l2_norm / (total_norm + 1e-6)
    return parameters