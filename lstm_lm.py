"""
LSTM language model.
Default hparams should get around 1.46 bpc on enwik8.
"""

import argparse
import lib.lm_datasets
import lib.utils
import numpy as np
import os
import torch
import torch.nn.functional as F
import tqdm
from torch import nn, optim

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--n_layers', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--seq_len', type=int, default=256)
parser.add_argument('--dim', type=int, default=4096)
parser.add_argument('--steps', type=int, default=20_000)
parser.add_argument('--reset_prob', type=float, default=0.01)
args = parser.parse_args()
lib.utils.print_args(args)

train_data, _, test_data = lib.lm_datasets.enwik8()

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
    global state
    with torch.no_grad():
        mask = torch.bernoulli(torch.full([1, args.batch_size, 1],
            1-args.reset_prob, device='cuda'))
        state[0].mul_(mask)
        state[1].mul_(mask)
    X = next(train_iterator).cuda().long()
    logits, state = model(X[:, :-1], state)
    loss = F.cross_entropy(logits.view(-1, 256), X[:, 1:].reshape(-1))
    return loss / 0.69314718 # BPC

def hook(step):
    state = (
        torch.zeros([args.n_layers, 1, args.dim]).half().cuda(),
        torch.zeros([args.n_layers, 1, args.dim]).half().cuda()
    )
    test_iterator = lib.enwik8.sequential_iterator(
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
    print(f'Test loss: {np.mean(losses)}')

opt = optim.Adam(model.parameters(), lr=args.lr)
lib.utils.train_loop(forward, opt, args.steps, print_freq=1000,
    hook=hook, hook_freq=5_000)