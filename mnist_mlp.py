"""
MNIST MLP classifier.
"""

import fire
import lib.datasets
import lib.utils
import torch
import torch.nn.functional as F
from torch import nn, optim

def main(batch_size=1024, steps=10_000, print_freq=1000, dropout=0.5):
    lib.utils.print_args(locals())

    X_train, y_train = lib.datasets.mnist('train')
    X_test, y_test = lib.datasets.mnist('test')

    model = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(512, 10)
    ).cuda()
    lib.utils.print_model(model)

    def forward():
        X, y = lib.utils.get_batch([X_train, y_train], batch_size)
        logits = model(X)
        return F.cross_entropy(logits, y)
    opt = optim.Adam(model.parameters())
    lib.utils.train_loop(forward, opt, steps, print_freq=print_freq,
        lr_cooldown_steps=steps//10)

    with torch.no_grad():
        model.eval()
        test_acc = model(X_test).argmax(dim=1).eq(y_test).float().mean()
        print('Test acc:', test_acc.item())

if __name__ == '__main__':
    fire.Fire(main)