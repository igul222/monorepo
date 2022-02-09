"""
Character-level text DDPM.
"""

import fire
import math
import numpy as np
import lib.lm_datasets
import lib.utils
import re
import torch
import torch.nn.functional as F
from torch import nn, optim

def position_embedding_matrix(n, dim):
    position = torch.arange(n).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
    pe = torch.zeros(n, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def main(
    T=1024,
    batch_size=64,
    seq_len=256,
    steps=100_000,
    print_freq=100,
    lr=1e-4,
    dim=1024):

    lib.utils.print_args(locals())

    train_data = lib.lm_datasets.books1()
    train_iterator = lib.lm_datasets.sequential_iterator(
        train_data, batch_size, seq_len, 0, True)

    class Block(nn.Module):
        def __init__(self, dim, dim_hid, scale, norm='group'):
            super().__init__()
            self.scale = scale
            self.conv1 = nn.Conv1d(dim, dim_hid, 5, padding='same')
            self.conv2 = nn.Conv1d(dim_hid, dim, 5, padding='same')
            assert(norm in ['group', 'none'])
            if norm == 'group':
                self.norm1 = nn.GroupNorm(8, dim)
                self.norm2 = nn.GroupNorm(8, dim_hid)
            elif norm == 'none':
                self.norm1 = (lambda x: x)
                self.norm2 = (lambda x: x)
        def forward(self, x):
            z = F.avg_pool1d(x, self.scale, self.scale)
            z = self.conv1(F.relu(self.norm1(z)))
            z = self.conv2(F.relu(self.norm2(z)))
            x = x + F.interpolate(z, scale_factor=self.scale)
            return x

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('t_embed', position_embedding_matrix(T, dim))
            self.input = nn.Conv1d(256, dim, 1, 1, padding='same')
            self.blocks = nn.Sequential(*[
                Block(dim, dim, 1, norm='none'),
                Block(dim, dim, 1),
                Block(dim, dim, 1),
                Block(dim, dim, 1),
                Block(dim, dim, 1),
                Block(dim, dim, 1),
                Block(dim, dim, 1),
                Block(dim, dim, 1)
            ])
            self.output = nn.Conv1d(dim, 256, 1, 1, 0)
            self.log_scales = nn.Parameter(torch.zeros(T))
        def forward(self, x, t):
            x_orig = x
            x = self.input(x) + self.t_embed[t][:,:,None]
            x = self.blocks(x)
            x = self.output(x)
            return (x_orig - x) * self.log_scales[t].exp()[:,None,None]
  
    model = Model().cuda()
    lib.utils.print_model(model)

    beta = torch.linspace(1e-4, 0.02, T).cuda()
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    lib.utils.print_tensor('noise scales', (1-alpha_bar).sqrt())

    def loss(X):
        # Low-discrepancy sampler for t
        n_cycles = T // X.shape[0]
        t = torch.arange(X.shape[0]).cuda() * n_cycles
        t += torch.randint(low=0, high=n_cycles, size=[1])[0].cuda()
        alpha_bar_t = alpha_bar[t][:,None,None]
        epsilon = torch.randn(X.shape, device='cuda')
        residual = epsilon - model(
            (alpha_bar_t.sqrt()*X) + ((1-alpha_bar_t).sqrt()*epsilon),
            t
        )
        return residual.pow(2).sum(dim=1)

    def forward():
        X = next(train_iterator).cuda()
        X = F.one_hot(X.long(), num_classes=256).permute(0,2,1)
        return loss(X).mean()

    opt = optim.Adam(model.parameters(), lr=lr)
    lib.utils.train_loop(forward, opt, steps, print_freq=print_freq,
        lr_warmup_steps=100, lr_cooldown_steps=steps//10)

    with torch.no_grad():
        X_samples = torch.randn([batch_size, 256, seq_len]).cuda()
        for t in range(T)[::-1]:
            X_samples = (
                (1./alpha[t].sqrt()) *
                (
                    X_samples -
                    (
                        (beta[t]/(1-alpha_bar[t]).sqrt()) * 
                        model(
                            X_samples,
                            torch.tensor([t]*batch_size).long().cuda()
                        )
                    )
                )
            )
            if t > 1:
                X_samples += beta[t].sqrt() * torch.randn_like(X_samples).cuda()
    print('Samples:')
    for x in X_samples:
        x = x.argmax(dim=0).detach().cpu().numpy().tobytes()
        x = re.sub(rb'[^\x00-\x7F]', b'X', x) # replace non-ascii bytes with 'X'
        x = x.decode('utf-8')
        print(x)
        print('---')

if __name__ == '__main__':
    fire.Fire(main)