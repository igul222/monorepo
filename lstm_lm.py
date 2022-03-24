"""
LSTM language model. The enwik8 hparams are tuned for test accuracy (1.46 bpc).
The books1 hparams are tuned for test accuracy subject to a time constraint of
~2 hours on a Titan V (1.38bpc).
"""

import fire
import lib.lm_datasets
import lib.utils
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
    args.setdefault('n_layers', 1)
    args.setdefault('seq_len', 256)
    args.setdefault('reset_prob', 0.01)
    args.setdefault('batch_size', 256)
    if args.dataset == 'enwik8':
        args.setdefault('steps', 10_000)
        args.setdefault('dim', 4096)
    elif args.dataset == 'books1':
        args.setdefault('steps', 40_000)
        args.setdefault('dim', 2048)
    lib.utils.print_args(args)

    if args.dataset == 'enwik8':
        train_data, _, test_data = lib.lm_datasets.enwik8()
    elif args.dataset == 'books1':
        train_data, _, test_data = lib.lm_datasets.books1()

    train_iterator = lib.lm_datasets.sequential_iterator(
        train_data, args.batch_size, args.seq_len+1, 1, True)

    class LSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(256, 256)
            self.lstm = nn.LSTM(
                input_size=256,
                hidden_size=args.dim,
                num_layers=args.n_layers,
                batch_first=True
            )
            self.lstm = nn.utils.weight_norm(self.lstm, 'weight_ih_l0')
            self.lstm = nn.utils.weight_norm(self.lstm, 'weight_hh_l0')
            self.readout = nn.Linear(args.dim, 256)

        def forward(self, x, state):
            x = self.embedding(x)
            all_h, new_state = self.lstm(x, state)
            new_state = (new_state[0].detach(), new_state[1].detach())
            logits = self.readout(all_h)
            return logits, new_state

    model = LSTM().cuda()
    lib.utils.print_model(model)

    state = (
        torch.zeros([args.n_layers, args.batch_size, args.dim]).half().cuda(),
        torch.zeros([args.n_layers, args.batch_size, args.dim]).half().cuda()
    )
    def forward():
        nonlocal state
        with torch.no_grad():
            mask = torch.bernoulli(torch.full([1, args.batch_size, 1],
                1-args.reset_prob, device='cuda'))
            state[0].mul_(mask)
            state[1].mul_(mask)
        X = next(train_iterator).cuda().long()
        logits, state = model(X[:, :-1], state)
        loss = F.cross_entropy(logits.view(-1, 256), X[:, 1:].reshape(-1))
        return loss / 0.69314718 # BPC

    opt = optim.Adam(model.parameters(), lr=args.lr)
    lib.utils.train_loop(forward, opt, args.steps, print_freq=1000,
        lr_cooldown_steps=args.steps//10)

    state = (
        torch.zeros([args.n_layers, 1, args.dim]).half().cuda(),
        torch.zeros([args.n_layers, 1, args.dim]).half().cuda()
    )
    test_iterator = lib.lm_datasets.sequential_iterator(
        test_data, 1, args.seq_len+1, 1, False)
    losses = []
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            for X in tqdm.tqdm(test_iterator, mininterval=10.):
                X = X.cuda().long()
                logits, state = model(X[:, :-1], state)
                loss = F.cross_entropy(logits.view(-1, 256),
                    X[:, 1:].reshape(-1))
                losses.append(loss.item() / 0.69314718) # BPC
                del X, logits, loss
    test_loss = np.mean(losses)
    print(f'Test loss: {test_loss}')

    return test_loss

if __name__ == '__main__':
    fire.Fire(main)