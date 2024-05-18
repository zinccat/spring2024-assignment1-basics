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

class Transformer(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, attn_pdrop: float, residual_pdrop: float, weights: Dict[str, torch.FloatTensor] = None):
        super(Transformer, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(context_length, d_model)
        self.dropout = nn.Dropout(residual_pdrop)
        transformer_layers = []
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        if weights:
            self.token_embeddings.weight.data = weights['token_embeddings.weight']
            self.position_embeddings.weight.data = weights['position_embeddings.weight']
            for i in range(num_layers):
                transformer_weight = {
                    'attn.q_proj.weight': weights[f'layers.{i}.attn.q_proj.weight'],
                    'attn.k_proj.weight': weights[f'layers.{i}.attn.k_proj.weight'],
                    'attn.v_proj.weight': weights[f'layers.{i}.attn.v_proj.weight'],
                    'attn.output_proj.weight': weights[f'layers.{i}.attn.output_proj.weight'],
                    'ln1.weight': weights[f'layers.{i}.ln1.weight'],
                    'ln2.weight': weights[f'layers.{i}.ln2.weight'],
                    'ffn.w1.weight': weights[f'layers.{i}.ffn.w1.weight'],
                    'ffn.w2.weight':weights[f'layers.{i}.ffn.w2.weight']
                }
                transformer_layers.append(TransformerBlock(d_model, num_heads, d_ff, attn_pdrop, residual_pdrop, transformer_weight))
                self.norm.weight.data = weights['ln_final.weight']
                self.head.weight.data = weights['lm_head.weight']
        else:
            transformer_layers.append(TransformerBlock(d_model, num_heads, d_ff, attn_pdrop, residual_pdrop))

        self.transformer = nn.Sequential(*transformer_layers)
        
    def forward(self, in_indices):
        token_embedding = self.token_embeddings(in_indices)
        position_embedding = self.position_embeddings(torch.arange(in_indices.shape[1], device=in_indices.device))
        embedding = token_embedding + position_embedding
        embedding = self.dropout(embedding)
        x = self.transformer(embedding)
        x = self.norm(x)
        x = self.head(x)
        return x