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
    args.setdefault('lr', 3e-4)
    args.setdefault('seq_len', 16)
    args.setdefault('n_heads', 4)
    args.setdefault('batch_size', 1024)
    args.setdefault('steps', 2000)
    args.setdefault('dim', 1024)
    args.setdefault('n_blocks', 4)
    args.setdefault('vocab_size', 821) # 821 to match Lisa's E2E.
    args.setdefault('print_freq', 100)

    lib.utils.print_args(args)

    train_data, _, test_data = lib.lm_datasets.e2e(args.vocab_size)
    test_data = torch.cat(test_data)

    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('pos_codes',
                lib.transformer.position_codes(args.dim, args.seq_len))
            self.blocks = nn.Sequential(*[
                lib.transformer.TransformerBlock(args.dim, args.n_heads,
                    autoregressive=True, dropout=0.5)
                for _ in range(args.n_blocks)
            ])
            self.output_norm = nn.LayerNorm(args.dim)
            self.output = nn.Linear(args.dim, args.vocab_size)

        def forward(self, x):
            x = self.output.weight[x, :] * float(np.sqrt(args.dim))
            x = x + self.pos_codes[None,:,:]
            x = self.blocks(x)
            x = self.output_norm(x)
            x = self.output(x)
            return x

    model = Transformer().cuda()
    lib.utils.print_model(model)

    def forward():
        np.random.shuffle(train_data)
        X = train_data[:args.batch_size*(args.seq_len//8)]
        X = torch.cat(X)[:args.batch_size * (args.seq_len+1)].cuda().long()
        X = X.view(args.batch_size, args.seq_len+1)
        logits = model(X[:, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, args.vocab_size),
            X[:, 1:].reshape(-1)
        )
        return loss / 0.69314718 # bits per token

    def hook(_):
        model.eval()
        # This evaluation is very slightly biased but low-variance.
        k = 1 # Number of tokens to evaluate per forward pass.
        test_data_unfolded = test_data.unfold(0, args.seq_len+1, k)
        losses = []
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                iter_range = range(0, len(test_data_unfolded), args.batch_size)
                for i in tqdm.tqdm(iter_range, leave=False):
                    X = test_data_unfolded[i:i+args.batch_size].cuda().long()
                    logits = model(X[:, :-1])
                    loss = F.cross_entropy(
                        logits[:,-k:].permute(0,2,1),
                        X[:, -k:]
                    )
                    losses.append(loss.item() / 0.69314718) # bits per token
                    del X, logits, loss
        test_loss = np.mean(losses)
        print(f'Test loss: {test_loss}')
        model.train()
        return test_loss

    opt = optim.Adam(model.parameters(), lr=args.lr)
    lib.utils.train_loop(forward, opt, args.steps, print_freq=args.print_freq,
        lr_cooldown_steps=1000, hook=hook, hook_freq=500)

if __name__ == '__main__':
    fire.Fire(main)