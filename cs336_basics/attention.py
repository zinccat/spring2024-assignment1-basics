import torch
from torch import nn
import torch.nn.functional as F
import math
from cs336_basics import softmax
from typing import Dict

def scaled_dot_product_attention(q, k, v, mask=None, pdrop=None):
    d_k = q.shape[-1]
    scores = q@torch.swapaxes(k, -2, -1)
    scores = scores / math.sqrt(d_k)
    if mask is not None:
        # print(scores.shape, mask.shape)
        scores = scores.masked_fill_(mask, float('-inf'))
    attn_weights = softmax(scores, dim=-1)
    if pdrop is not None:
        attn_weights = F.dropout(attn_weights, p=pdrop, training=True)
    output = attn_weights @ v
    return output

class MultiHead(nn.Module):
    def __init__(self, d_model: int, num_heads: int, attn_pdrop: float, weights: Dict[str, torch.FloatTensor] = None):
        super(MultiHead, self).__init__()
        self.d_model = d_model
        self.dk = self.dv = self.d_model // num_heads
        self.num_heads = num_heads
        self.attn_pdrop = attn_pdrop
        self.wq = nn.Parameter(torch.randn(d_model, d_model))
        self.wk = nn.Parameter(torch.randn(d_model, d_model))
        self.wv = nn.Parameter(torch.randn(d_model, d_model))
        self.wo = nn.Parameter(torch.randn(d_model, d_model))
        if weights is not None:
            for N in range(num_heads):
                self.wq.data[:, N*self.dk:(N+1)*self.dk] = weights[f'q_heads.{N}.weight'].T
                self.wk.data[:, N*self.dk:(N+1)*self.dk] = weights[f'k_heads.{N}.weight'].T
                self.wv.data[:, N*self.dv:(N+1)*self.dv] = weights[f'v_heads.{N}.weight'].T
            self.wo.data = weights['output_proj.weight'].T
    
    def forward(self, q, k, v):
        batch_size = q.size(0)
        seq_len = q.size(1)
        
        q = q @ self.wq
        k = k @ self.wk
        v = v @ self.wv
        
        q = q.view(batch_size, seq_len, self.num_heads, self.dk).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.dk).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.dv).transpose(1, 2)
        
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        attn = scaled_dot_product_attention(q, k, v, mask, pdrop=self.attn_pdrop)
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = attn @ self.wo
        return output


class MultiHeadSelfAttention(MultiHead):
    def forward(self, x):
        return super().forward(x, x, x)