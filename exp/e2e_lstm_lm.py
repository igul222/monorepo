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
    args.setdefault('lr', 1e-3)
    args.setdefault('n_layers', 1)
    args.setdefault('seq_len', 64)
    args.setdefault('batch_size', 256)
    args.setdefault('vocab_size', 821) # 821 to match Lisa's setup.
    args.setdefault('dim', 4096)
    lib.utils.print_args(args)

    (train_data, _, test_data), idx2word = lib.lm_datasets.e2e(args.vocab_size)
    train_data_concat = torch.cat(train_data)
    test_data_concat = torch.cat(test_data)
    # synth_data_concat = torch.cat(synth_data)
    train_iterator = lib.lm_datasets.random_iterator(train_data_concat,
        args.batch_size, args.seq_len+1)
    # synth_iterator = lib.lm_datasets.sequential_iterator(synth_data_concat,
    #     args.batch_size, args.seq_len+1, 1, True)

    class LSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=args.dim,
                hidden_size=args.dim,
                num_layers=args.n_layers,
                batch_first=True
            )
            self.lstm = nn.utils.weight_norm(self.lstm, 'weight_ih_l0')
            self.lstm = nn.utils.weight_norm(self.lstm, 'weight_hh_l0')
            self.readout = nn.Linear(args.dim, args.vocab_size)

        def forward(self, x, state):
            x = self.readout.weight[x, :] * float(np.sqrt(args.dim))
            all_h, new_state = self.lstm(x, state)
            new_state = (new_state[0].detach(), new_state[1].detach())
            logits = self.readout(all_h)
            return logits, new_state

    model = LSTM().cuda()
    lib.utils.print_model(model)

    def hook(_):
        model.eval()
        state = (
            torch.zeros([args.n_layers, 1, args.dim]).half().cuda(),
            torch.zeros([args.n_layers, 1, args.dim]).half().cuda()
        )
        test_iterator = lib.lm_datasets.sequential_iterator(
            test_data_concat, 1, args.seq_len+1, 1, False)
        losses = []
        with torch.cuda.amp.autocast(), torch.no_grad():
            for X in tqdm.tqdm(test_iterator, mininterval=10., leave=False):
                X = X.cuda().long()
                logits, state = model(X[:, :-1], state)
                loss = F.cross_entropy(logits.view(-1, args.vocab_size),
                    X[:, 1:].reshape(-1))
                losses.append(loss.item() / 0.69314718) # BPC
                del X, logits, loss
        test_loss = np.mean(losses)
        model.train()
        print(f'Test loss: {test_loss}')
        return test_loss

    # state = (
    #     torch.zeros([args.n_layers, args.batch_size, args.dim]).half().cuda(),
    #     torch.zeros([args.n_layers, args.batch_size, args.dim]).half().cuda()
    # )
    # step = 0

    # def forward():
    #     nonlocal state, step
    #     step += 1
    #     X = next(synth_iterator).cuda()
    #     logits, state = model(X[:, :-1], state)
    #     loss = F.cross_entropy(
    #         logits.view(-1, args.vocab_size),
    #         X[:, 1:].reshape(-1)
    #     )
    #     return loss / 0.69314718 # BPC

    # opt = optim.Adam(model.parameters(), lr=args.lr)
    # lib.utils.train_loop(forward, opt, 2000, print_freq=100,
    #     lr_cooldown_steps=1000, hook=hook, hook_freq=500)

    state = (
        torch.zeros([args.n_layers, args.batch_size, args.dim]).half().cuda(),
        torch.zeros([args.n_layers, args.batch_size, args.dim]).half().cuda()
    )
    def forward():
        nonlocal state
        # Reshuffle data every step
        np.random.shuffle(train_data)
        train_data_concat.copy_(torch.cat(train_data))

        X = next(train_iterator).cuda()
        logits, state = model(X[:, :-1], state)
        loss = F.cross_entropy(
            logits.view(-1, args.vocab_size),
            X[:, 1:].reshape(-1)
        )
        return loss / 0.69314718 # BPC

    opt = optim.Adam(model.parameters(), lr=args.lr)
    lib.utils.train_loop(forward, opt, 2000, print_freq=100,
        lr_cooldown_steps=1000, hook=hook, hook_freq=500)


if __name__ == '__main__':
    fire.Fire(main)