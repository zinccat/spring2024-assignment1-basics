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
        ffn_weights = {'w1.weight': weights['ffn.w1.weight'], 'w2.weight': weights['ffn.w2.weight']} if weights else None
        self.ffn = FFN(d_model, d_ff, ffn_weights)
        if weights is not None:
            self.mhsa.wq.weight.data = weights['attn.q_proj.weight']
            self.mhsa.wk.weight.data = weights['attn.k_proj.weight']
            self.mhsa.wv.weight.data = weights['attn.v_proj.weight']
            self.mhsa.wo.weight.data = weights['attn.output_proj.weight']
            self.rmsnorm1.weight.data = weights['ln1.weight']
            self.rmsnorm2.weight.data = weights['ln2.weight']

    def forward(self, x):
        y = x + self.dropout(self.mhsa(self.rmsnorm1(x)))
        y = y + self.dropout(self.ffn(self.rmsnorm2(y)))
        return y