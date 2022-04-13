"""
Baseline autoregressive LM for char_vdm.
"""

import fire
import lib.lm_datasets
import lib.utils
import lib.transformer
import numpy as np
import os
import torch
import torch.nn.functional as F
import tqdm
from torch import nn, optim

def main(**args):
    args = lib.utils.AttributeDict(args)
    args.setdefault('dataset', 'books1')
    args.setdefault('lr', 1e-3)
    args.setdefault('seq_len', 128)
    args.setdefault('n_heads', 4)
    args.setdefault('batch_size', 128)
    args.setdefault('steps', 1000_000)
    args.setdefault('dim', 512)
    args.setdefault('n_blocks', 2)
    lib.utils.print_args(args)

    if args.dataset == 'enwik8':
        train_data, _, test_data = lib.lm_datasets.enwik8()
    elif args.dataset == 'books1':
        train_data, _, test_data = lib.lm_datasets.books1()

    train_iterator = lib.lm_datasets.random_iterator(
        train_data, args.batch_size, args.seq_len+1)

    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(256, args.dim)
            self.register_buffer('pos_codes',
                lib.transformer.position_codes(args.dim, args.seq_len))
            self.blocks = nn.Sequential(*[
                lib.transformer.TransformerBlock(args.dim, args.n_heads,
                    autoregressive=True)
                for _ in range(args.n_blocks)
            ])
            self.output_norm = nn.LayerNorm(args.dim)
            self.output = nn.Linear(args.dim, 256)

        def forward(self, x):
            x = self.embedding(x) + self.pos_codes[None,:,:]
            x = self.blocks(x)
            x = self.output_norm(x)
            x = self.output(x)
            return x

    model = Transformer().cuda()
    lib.utils.print_model(model)

    nll_ema = 0.

    def forward():
        nonlocal nll_ema
        X = next(train_iterator).cuda().long()
        logits = model(X[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, 256), X[:, 1:].reshape(-1))

        nll_ema = (.9999 * nll_ema) + (.0001 * loss.item())

        return loss, torch.tensor(nll_ema)

    opt = optim.Adam(model.parameters(), lr=args.lr)
    lib.utils.train_loop(forward, opt, args.steps, print_freq=1000,
        lr_cooldown_steps=args.steps // 10, names=['nll_ema'])

if __name__ == '__main__':
    fire.Fire(main)