"""
Toy demo of 'meta-learning' in language models.
"""

import fire
import lib.ops
import lib.utils
import torch
import torch.nn.functional as F
from torch import nn, optim

# 'how much incentive do i have to meta-learn vs revert to the prior'
# also, 'how far do i need to extrapolate'
DIVERSITY = 0.5
N_TOKENS = 10
SEQ_LEN = 64
DIM = 512
BATCH_SIZE = 256
STEPS = 10000
PRINT_FREQ = 1000

def make_probs(n):
    return F.softmax(
        torch.randn(n, N_TOKENS, device='cuda') * DIVERSITY, dim=1)

def make_data(probs):
    return torch.multinomial(probs, SEQ_LEN+1, replacement=True)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(N_TOKENS, DIM)
        self.lstm = nn.LSTM(
            input_size=DIM, hidden_size=DIM, batch_first=True)
        self.output = nn.Linear(DIM, N_TOKENS)
    def forward(self, x):
        x = self.embedding(x)
        x = self.lstm(x)[0]
        x = self.output(x)
        return x

def main():
    lib.utils.print_tensor('sample probs', make_probs(10))
    model = Model().cuda()
    def forward():
        X = make_data(make_probs(BATCH_SIZE))
        logits = model(X[:,:-1])
        loss = F.cross_entropy(
            logits.reshape(-1, N_TOKENS),
            X[:,1:].reshape(-1)
        )
        return loss
    opt = optim.Adam(model.parameters())
    lib.utils.train_loop(forward, opt, STEPS, print_freq=PRINT_FREQ)

    with torch.no_grad():
        X = torch.zeros([1, 64], dtype=torch.int64, device='cuda')
        preds = F.softmax(model(X), dim=2)[0]
        for i in range(preds.shape[0]):
            lib.utils.print_tensor(f'preds[{i}]', preds[i])

if __name__ == '__main__':
    fire.Fire(main)