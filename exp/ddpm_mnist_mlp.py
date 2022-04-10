"""
MNIST DDPM but with an MLP.
"""

import fire
import math
import numpy as np
import lib.datasets
import lib.utils
import lib.transformer
import torch
import torch.nn.functional as F
from torch import nn, optim

def main(
    T=1024,
    batch_size=512,
    steps=10_000,
    print_freq=1000,
    lr=1e-3,
    dim=2048):

    lib.utils.print_args(locals())

    X_train, y_train = lib.datasets.mnist('train')
    X_test, y_test = lib.datasets.mnist('test')

    beta = torch.linspace(1e-4, 0.02, T).cuda()
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('t_codes', lib.transformer.position_codes(dim, T))
            self.h1 = nn.Linear(784, dim)
            self.h2 = nn.Linear(dim, dim)
            self.h3 = nn.Linear(dim, dim)
            self.h4 = nn.Linear(dim, 784)
            self.log_scales = nn.Parameter(torch.zeros(T))

        def forward(self, x, t):
            x_orig = x
            x = F.gelu(self.h1(x) + self.t_codes[t])
            x = F.gelu(self.h2(x))
            x = F.gelu(self.h3(x))
            x = self.h4(x)
            return (x_orig - x) * self.log_scales[t].exp()[:,None]

    model = Model().cuda()
    lib.utils.print_model(model)

    def loss(X):
        X = (2*X) - 1.
        # Low-discrepancy sampler for t
        n_cycles = T // X.shape[0]
        t = torch.arange(X.shape[0]).cuda() * n_cycles
        t += torch.randint(low=0, high=n_cycles, size=[1])[0].cuda()
        alpha_bar_t = alpha_bar[t][:,None]
        epsilon = torch.randn(X.shape, device='cuda')
        residual = epsilon - model(
            (alpha_bar_t.sqrt()*X) + ((1-alpha_bar_t).sqrt()*epsilon),
            t
        )
        return residual.pow(2).sum(dim=1)

    def forward():
        X = lib.utils.get_batch(X_train, batch_size)
        return loss(X).mean()

    opt = optim.Adam(model.parameters(), lr=lr)
    lib.utils.train_loop(forward, opt, steps, print_freq=print_freq,
        lr_warmup_steps=100, lr_cooldown_steps=steps//10)

    test_loss = lib.utils.batch_apply(loss, X_test).mean().item()
    print('Test loss:', test_loss)
    with torch.no_grad():
        X_samples = torch.randn([64, 784]).cuda()
        for t in range(T)[::-1]:
            X_samples = (
                (1./alpha[t].sqrt()) *
                (
                    X_samples -
                    (
                        (beta[t]/(1-alpha_bar[t]).sqrt()) * 
                        model(
                            X_samples,
                            torch.tensor([t]*64).long().cuda()
                        )
                    )
                )
            )
            if t > 1:
                X_samples += beta[t].sqrt() * torch.randn_like(X_samples).cuda()
        X_samples = (X_samples + 1) / 2.
    lib.utils.save_image_grid(X_samples, f'samples.png')

if __name__ == '__main__':
    fire.Fire(main)