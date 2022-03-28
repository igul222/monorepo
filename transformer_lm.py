"""
Transformer character-level LM. Hyperparams tuned for books1 given approx.
2 hours on a Titan V.
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
    args.setdefault('lr', 3e-4)
    args.setdefault('seq_len', 1024)
    args.setdefault('n_heads', 4)
    args.setdefault('batch_size', 16)
    args.setdefault('steps', 44_000)
    args.setdefault('dim', 1024)
    args.setdefault('n_blocks', 4)
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
                lib.transformer.position_codes(args.dim))
            self.blocks = nn.Sequential(*[
                lib.transformer.TransformerBlock(args.dim)
                for _ in range(args.n_blocks)
            ])
            self.output_norm = nn.LayerNorm(args.dim)
            self.output = nn.Linear(args.dim, 256)

        def forward(self, x):
            x = self.embedding(x) + self.pos_codes[None, :x.shape[0], :]
            x = self.blocks(x)
            x = self.output_norm(x)
            x = self.output(x)
            return x

    model = Transformer().cuda()
    lib.utils.print_model(model)

    def forward():
        X = next(train_iterator).cuda().long()
        logits = model(X[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, 256), X[:, 1:].reshape(-1))
        return loss / 0.69314718 # BPC

    opt = optim.Adam(model.parameters(), lr=args.lr)
    lib.utils.train_loop(forward, opt, args.steps, print_freq=1000,
        lr_cooldown_steps=args.steps // 10)

    # This evaluation is approximate but low-variance and only slightly biased.
    k = args.seq_len // 64 # Number of tokens to evaluate per forward pass.
    test_data = test_data.unfold(0, args.seq_len+1, k)
    losses = []
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            for i in range(0, len(test_data), args.batch_size):
                X = test_data[i:i+args.batch_size].cuda().long()
                logits = model(X[:, :-1])
                loss = F.cross_entropy(
                    logits[:,-k:].permute(0,2,1),
                    X[:, -k:]
                )
                losses.append(loss.item() / 0.69314718) # BPC
                del X, logits, loss
    test_loss = np.mean(losses)
    print(f'Test loss: {test_loss}')
    return test_loss

if __name__ == '__main__':
    fire.Fire(main)