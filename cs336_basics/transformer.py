import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict

from cs336_basics import MultiHeadSelfAttention, RMSNorm, FFN

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, attn_pdrop: float = None, residual_pdrop: float = None, weights: Dict[str, torch.FloatTensor] = None):
        super(TransformerBlock, self).__init__()
        self.mhsa = MultiHeadSelfAttention(d_model, num_heads, attn_pdrop)
        self.rmsnorm1 = RMSNorm(d_model)
        self.dropout = nn.Dropout(residual_pdrop)
        self.rmsnorm2 = RMSNorm(d_model)
        self.ffn = FFN(d_model, d_ff)
        if weights is not None:
            self.mhsa.wq.data = weights['attn.q_proj.weight'].T
            self.mhsa.wk.data = weights['attn.k_proj.weight'].T
            self.mhsa.wv.data = weights['attn.v_proj.weight'].T
            self.mhsa.wo.data = weights['attn.output_proj.weight'].T
            self.rmsnorm1.weight.data = weights['ln1.weight']
            self.rmsnorm2.weight.data = weights['ln2.weight']
            self.ffn.w1.data = weights['ffn.w1.weight']
            self.ffn.w2.data = weights['ffn.w2.weight']

    def forward(self, x):
        y = x + self.dropout(self.mhsa(self.rmsnorm1(x)))
        y = y + self.dropout(self.ffn(self.rmsnorm2(y)))
        return y