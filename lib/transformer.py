import numpy as np
import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn, optim

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, autoregressive=False, norm=True):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.autoregressive = autoregressive
        self.linear1 = nn.Linear(dim, 4*dim)
        self.linear2 = nn.Linear(4*dim, dim)
        if norm:
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
        else:
            self.norm1 = None
            self.norm2 = None

    def forward(self, x):
        # Self-attention block
        x_res = x
        if self.norm1 is not None:
            x = self.norm1(x)
        if self.autoregressive:
            mask = torch.full([x.shape[1], x.shape[1]], float('-inf')).triu_(1)
        else:
            mask = None
        x = self.attn(x, x, x, attn_mask=mask)[0]
        x = x + x_res
        # Feedforward block
        x_res = x
        if self.norm2 is not None:
            x = self.norm2(x)
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.linear2(x)
        x = x + x_res
        return x

def position_codes(dim, seq_len=4096):
    period = torch.linspace(0., 10., dim).exp()[None,:]
    position = torch.arange(seq_len)[:,None]
    pc = torch.sin(float(2*np.pi) * position / period)
    pc = pc @ torch.nn.init.orthogonal_(torch.randn(dim, dim))
    return pc