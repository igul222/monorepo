"""
Multimodal alignment via "deep learning magic"
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
    args.setdefault('lr', 1e-3)
    args.setdefault('seq_len', 64)
    args.setdefault('steps', 44_000)
    args.setdefault('dim', 256)
    args.setdefault('n_blocks', 6)
    args.setdefault('n_heads', 2)
    args.setdefault('vocab_size', 256)
    args.setdefault('print_freq', 100)
    args.setdefault('N', 8)
    args.setdefault('batch_size', 128 // args.N)

    lib.utils.print_args(args)

    N = args.N

    (train_data, _, test_data), _ = lib.lm_datasets.books1()

    train_iterator = lib.lm_datasets.random_iterator(
        train_data, args.batch_size, args.seq_len+1)

    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(N*args.vocab_size, args.dim)
            self.register_buffer('pos_codes',
                lib.transformer.position_codes(args.dim, args.seq_len))
            self.blocks = nn.Sequential(*[
                lib.transformer.TransformerBlock(args.dim, args.n_heads,
                    autoregressive=True)
                for _ in range(args.n_blocks)
            ])
            self.output_norm = nn.LayerNorm(args.dim)
            self.output = nn.Linear(args.dim, N*args.vocab_size)

        def forward(self, x):
            x = self.embedding(x) + self.pos_codes[None,:,:]
            x = self.blocks(x)
            # x = self.output_norm(x)
            x = self.output(x)
            return x

    model = Transformer().cuda()
    lib.utils.print_model(model)

    def forward():
        X = [next(train_iterator).cuda().long() for i in range(N)]
        logits = model(
            torch.cat([Xi + (i*256) for i, Xi in enumerate(X)], dim=0)[:, :-1]
        )
        logits_ii = [
            logits[i*args.batch_size:(i+1)*args.batch_size, :, i::N]
            for i in range(N)
        ]
        loss_ii = [
            F.cross_entropy(
                logits_ii[i].reshape(-1, args.vocab_size),
                X[i][:, 1:].reshape(-1)
            )
            for i in range(N)
        ]

        loss = torch.stack(loss_ii).mean()

        loss_12 = F.cross_entropy(
            logits[:args.batch_size, :, 1::N].reshape(-1, args.vocab_size),
            X[0][:, 1:].reshape(-1)
        )

        return loss, loss_12

    # opt = optim.Adam(model.parameters(), lr=args.lr)
    opt = optim.Adam([
        {
            'params': model.blocks.parameters(),
            'lr': args.lr
        },
        {
            'params': list(model.embedding.parameters()) + list(model.output.parameters()),
            'lr': args.lr * 10,
        }
    ])
    lib.utils.train_loop(forward, opt, args.steps, print_freq=args.print_freq,
        lr_cooldown_steps=args.steps // 10,
        names=['loss_12'])

    hook(None)

if __name__ == '__main__':
    fire.Fire(main)