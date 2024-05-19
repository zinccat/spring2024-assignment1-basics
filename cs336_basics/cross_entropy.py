import torch

def cross_entropy(o: torch.Tensor, xi: torch.Tensor):
    o_max = o.max(dim=-1, keepdim=True).values
    o = o - o_max
    exp_o = torch.exp(o)
    log_sum_exp_o = torch.log(torch.sum(exp_o, dim=-1, keepdim=True))
    log_probs = o - log_sum_exp_o
    log_probs_correct_class = log_probs.gather(dim=1, index=xi.unsqueeze(1)).squeeze(1)
    return -log_probs_correct_class.mean()