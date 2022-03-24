"""
Character-level text DDPM.

'small' hparams: dim=256, steps=10_000
'big' hparams: dim=1024, steps=1_000_000
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
    steps=10_000,
    print_freq=1000,
    lr=1e-4,
    dim=256):

    lib.utils.print_args(locals())

    # train_data, _, _ = lib.lm_datasets.enwik8()
    train_data, _, _ = lib.lm_datasets.books1()
    train_iterator = lib.lm_datasets.sequential_iterator(
        train_data, batch_size, seq_len, 0, True)

    class Block(nn.Module):
        def __init__(self, dim, norm='group'):
            super().__init__()
            self.conv1 = nn.Conv1d(dim, dim, 5, padding='same')
            self.conv2 = nn.Conv1d(dim, dim, 5, padding='same')
            assert(norm in ['group', 'none'])
            if norm == 'group':
                self.norm1 = nn.GroupNorm(8, dim)
                self.norm2 = nn.GroupNorm(8, dim)
            elif norm == 'none':
                self.norm1 = (lambda x: x)
                self.norm2 = (lambda x: x)
        def forward(self, x):
            z = x
            z = self.conv1(F.relu(self.norm1(z)))
            z = self.conv2(F.relu(self.norm2(z)))
            x = x + z
            return x

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('t_embed', position_embedding_matrix(T, dim))
            self.input = nn.Conv1d(256, dim, 1, 1, padding='same')
            self.blocks = nn.Sequential(*[
                Block(dim, norm='none'),
                Block(dim),
                Block(dim),
                Block(dim),
                Block(dim),
                Block(dim),
                Block(dim),
                Block(dim)
            ])
            self.output_norm = nn.GroupNorm(8, dim)
            self.output = nn.Conv1d(dim, 256, 1, 1, 0)
            self.log_scales = nn.Parameter(torch.zeros(T))
        def forward(self, x, t):
            x_orig = x
            x = self.input(x) + self.t_embed[t][:,:,None]
            x = self.blocks(x)
            x = self.output(self.output_norm(x))
            return (x_orig - x) * self.log_scales[t].exp()[:,None,None]

    model = Model().cuda()
    lib.utils.print_model(model)

    beta = torch.linspace(1e-4, 0.01, T).cuda()
    # beta[0] = (0.2)**2 # noise floor
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    print('noise scales:')
    for i, x in enumerate((1-alpha_bar).sqrt().tolist()):
        if i % 10 == 0:
            lib.utils.print_row(i, x)

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
        with torch.cuda.amp.autocast():
            X_samples = torch.randn([8*batch_size, 256, seq_len]).cuda()
            for t in range(T)[::-1]:
                X_samples = (
                    (1./alpha[t].sqrt()) *
                    (
                        X_samples -
                        (
                            (beta[t]/(1-alpha_bar[t]).sqrt()) * 
                            model(
                                X_samples,
                                torch.tensor([t]*8*batch_size).long().cuda()
                            )
                        )
                    )
                )
                if t > 1:
                    epsilon = torch.randn_like(X_samples).cuda()
                    X_samples += beta[t].sqrt() * epsilon

    with torch.no_grad():
        X_samples = X_samples.argmax(dim=1)

        print('Samples:')
        for x in X_samples[:10]:
            x = x.detach().cpu().numpy().tobytes()
            # replace non-ascii bytes with 'X'
            x = re.sub(rb'[^\x00-\x7F]', b'X', x)
            x = x.decode('utf-8')
            print(x)
            print('---')

        sample_freqs = torch.zeros([256])
        for x in X_samples.view(-1):
            sample_freqs[x] += 1

        data_freqs = torch.zeros([256])
        for _ in range(8):
            X = next(train_iterator)
            for x in X.view(-1).tolist():
                data_freqs[x] += 1
        
        sample_freqs /= sample_freqs.sum()
        data_freqs /= data_freqs.sum()
        l1 = (sample_freqs - data_freqs).abs().sum().item()
        print('Unigram L1:', l1)

if __name__ == '__main__':
    fire.Fire(main)