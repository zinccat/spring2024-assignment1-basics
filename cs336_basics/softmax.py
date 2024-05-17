import torch

def softmax(v: torch.Tensor, dim: int = -1) -> torch.Tensor:
    v_max = v.max(axis=dim, keepdim=True).values
    v = v - v_max
    exp_v = torch.exp(v)
    sum_exp_v = torch.sum(exp_v, dim=dim, keepdim=True)
    return exp_v / sum_exp_v