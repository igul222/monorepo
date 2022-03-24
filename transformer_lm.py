"""
Transformer character-level LM. Hyperparams tuned for books1 given approx.
2 hours on a Titan V.
"""

import fire
import lib.lm_datasets
import lib.utils
import numpy as np
import os
import torch
import torch.utils.checkpoint
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

    class TransformerBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.attn = nn.MultiheadAttention(dim, args.n_heads,
                batch_first=True)
            attn_mask = torch.zeros([args.seq_len, args.seq_len])
            for i in range(args.seq_len):
                attn_mask[i, i+1:] = float('-inf')
            self.register_buffer('attn_mask', attn_mask)
            self.linear1 = nn.Linear(dim, 4*dim)
            self.linear2 = nn.Linear(4*dim, dim)
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)

        def forward(self, x):
            # Self-attention block
            x_res = x
            x = self.norm1(x)
            x = self.attn(x, x, x, attn_mask=self.attn_mask)[0]
            x = x + x_res
            # Feedforward block
            x_res = x
            x = self.norm2(x)
            x = self.linear1(x)
            x = F.gelu(x)
            x = self.linear2(x)
            x = x + x_res
            return x

    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(256, args.dim)
            position = torch.arange(args.seq_len).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, args.dim, 2) * float(-np.log(10000) / args.dim))
            pe = torch.zeros(1, args.seq_len, args.dim)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pos_embedding', pe)

            self.blocks = nn.Sequential(*[
                TransformerBlock(args.dim)
                for _ in range(args.n_blocks)
            ])
            self.output_norm = nn.LayerNorm(args.dim)
            self.output = nn.Linear(args.dim, 256)

        def forward(self, x, checkpoint=False):
            x = self.embedding(x)
            x = x + self.pos_embedding
            if checkpoint:
                x = torch.utils.checkpoint.checkpoint_sequential(
                    self.blocks, args.n_blocks, x)
            else:
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