"""
Baseline autoregressive LM for char_vdm.
"""

import fire
import lib.diffusion_lm_datasets
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
    args.setdefault('batch_size', 128)
    args.setdefault('dataset', 'rocstories')
    args.setdefault('lr', 1e-4)
    args.setdefault('model_size', 'small')
    args.setdefault('print_freq', 1000)
    args.setdefault('steps', 100_000)
    args.setdefault('hook_freq', 5000)

    if args.model_size == 'small':
        args.setdefault('dim', 1024)
        args.setdefault('n_blocks', 9)
        args.setdefault('n_heads', 8)
    elif args.model_size == 'medium':
        args.setdefault('dim', 1536)
        args.setdefault('n_blocks', 12)
        args.setdefault('n_heads', 12)


    lib.utils.print_args(args)

    dataset = lib.diffusion_lm_datasets.REGISTRY[args.dataset](args.batch_size)
    (train_iterator, test_iterator), (word2idx, idx2word) = dataset

    typical_seq_len = next(train_iterator)[0].shape[1]
    vocab_size = len(word2idx)
    print(f'typical_seq_len: {typical_seq_len}, vocab_size: {vocab_size}')

    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, args.dim)
            self.register_buffer('pos_codes',
                lib.transformer.position_codes(args.dim))
            self.blocks = nn.Sequential(*[
                lib.transformer.TransformerBlock(args.dim, args.n_heads,
                    autoregressive=True)
                for _ in range(args.n_blocks)
            ])
            self.output_norm = nn.LayerNorm(args.dim)
            self.output = nn.Linear(args.dim, vocab_size)

        def forward(self, x):
            x = self.embedding(x) + self.pos_codes[None,:x.shape[1],:]
            x = self.blocks(x)
            x = self.output_norm(x)
            x = self.output(x)
            return x

    model = Transformer().cuda()
    lib.utils.print_model(model)

    nll_ema = 0.

    def forward():
        nonlocal nll_ema
        X = next(train_iterator)[0].cuda().long()
        logits = model(X[:, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            X[:, 1:].reshape(-1)
        )
        # Assumes the first token is perfectly predictable.
        loss = loss * (float(X.shape[0] - 1) / float(X.shape[0]))
        nll_ema = (.9999 * nll_ema) + (.0001 * loss.item())
        return loss, torch.tensor(nll_ema)

    def hook(_):
        with torch.no_grad(), torch.cuda.amp.autocast():
            losses = []
            for _ in range(40_000 // args.batch_size):
                X = next(test_iterator)[0].cuda().long()
                logits = model(X[:, :-1])
                loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    X[:, 1:].reshape(-1)
                )
                # Assumes the first token is perfectly predictable.
                loss = loss * (float(X.shape[0] - 1) / float(X.shape[0]))
                losses.append(loss.item())
        print('Test NLL:', np.mean(losses))

    opt = optim.Adam(model.parameters(), lr=args.lr)
    lib.utils.train_loop(
        forward, opt, args.steps, print_freq=args.print_freq,
        names=['nll_ema'],
        hook=hook, hook_freq=args.hook_freq
    )

if __name__ == '__main__':
    fire.Fire(main)