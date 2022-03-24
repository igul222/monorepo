"""
MNIST DDPM. Trains in about 10 minutes.

Potential improvements:
- Fourier input features
- Tune the noise schedule better
"""

import fire
import math
import numpy as np
import lib.datasets
import lib.utils
import torch
import torch.nn.functional as F
from torch import nn, optim

def position_embedding_matrix(n, dim):
    position = torch.arange(n).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
    pe = torch.zeros(n, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def main(
    T=1024,
    batch_size=64,
    steps=8000,
    print_freq=100,
    lr=3e-4,
    dim=192):

    lib.utils.print_args(locals())

    X_train, y_train = lib.datasets.mnist('train')
    X_test, y_test = lib.datasets.mnist('test')

    class Block(nn.Module):
        def __init__(self, dim, dilation, norm='group'):
            super().__init__()
            self.conv1 = nn.Conv2d(dim, dim, 3,dilation=dilation,padding='same')
            self.conv2 = nn.Conv2d(dim, dim, 3,dilation=dilation,padding='same')
            assert(norm in ['group', 'none'])
            if norm == 'group':
                self.norm1 = nn.GroupNorm(8, dim)
                self.norm2 = nn.GroupNorm(8, dim)
            elif norm == 'none':
                self.norm1 = (lambda x: x)
                self.norm2 = (lambda x: x)
        def forward(self, x):
            x_res = x
            x = self.conv1(F.relu(self.norm1(x)))
            x = self.conv2(F.relu(self.norm2(x)))
            return x + x_res

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('t_embed', position_embedding_matrix(T, dim))
            self.input = nn.Conv2d(1, dim, 1, 1, padding='same')
            # We use dilated convs to keep the implementation simple, but
            # realistically a U-net might work better.
            self.block1 = Block(dim, 1, norm='none')
            self.block2 = Block(dim, 1)
            self.block3 = Block(dim, 2)
            self.block4 = Block(dim, 2)
            self.block5 = Block(dim, 2)
            self.block6 = Block(dim, 2)
            self.block7 = Block(dim, 1)
            self.block8 = Block(dim, 1)
            self.output = nn.Conv2d(dim, 1, 1, 1, 0)
        def forward(self, x, t):
            x = x.view(-1, 1, 28, 28)
            x = self.input(x) + self.t_embed[t][:,:,None,None]
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            x = self.block5(x)
            x = self.block6(x)
            x = self.block7(x)
            x = self.block8(x)
            x = self.output(x / 30.)
            return x.view(-1, 784)
  
    model = Model().cuda()
    lib.utils.print_model(model)

    beta = torch.linspace(1e-4, 0.02, T).cuda()
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

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