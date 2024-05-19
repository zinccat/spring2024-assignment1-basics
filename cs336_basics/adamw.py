import torch
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), weight_decay=1e-4, eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        beta1, beta2 = betas
        defaults = {"lr": lr, "beta1": beta1, "beta2": beta2, "lam": weight_decay, "eps": eps}
        super().__init__(params, defaults)
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            alpha = group["lr"] # Get the learning rate.
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            lam = group["lam"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 1) # Get iteration number from the state, or initial value.
                m = state.get("m", 0)
                v = state.get("v", 0)
                g = p.grad.data # Get the gradient of loss with respect to p.
                m = beta1 * m + (1-beta1) * g
                v = beta2 * v + (1-beta2) * torch.pow(g, 2)
                alpha_t = alpha * math.sqrt(1-math.pow(beta2, t)) / (1-math.pow(beta1, t))
                p.data -= alpha_t * m / (torch.sqrt(v) + eps)
                p.data -= alpha * lam * p.data
                state["t"] = t + 1 # Increment iteration number.
                state["m"] = m
                state["v"] = v
        return loss