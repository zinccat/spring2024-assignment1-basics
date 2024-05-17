import torch
import torch.nn.functional as F
import math
from cs336_basics import softmax

def scaled_dot_product_attention(q, k, v, mask=None, pdrop=None):
    d_k = q.shape[-1]
    # print(q.shape, k.T.shape)
    scores = q@torch.swapaxes(k, -2, -1)
    scores = scores / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill_(mask, -torch.inf)
    attn_weights = softmax(scores, dim=-1)
    if pdrop is not None:
        if pdrop is not None:
            attn_weights = F.dropout(attn_weights, p=pdrop, training=True)
    output = attn_weights @ v
    return output