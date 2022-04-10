"""
Character-level continuous-time variational diffusion model.

TODO:
- Condition the model on gamma
- Implement sampling
- Scale up and run
"""

import fire
import math
import numpy as np
import lib.lm_datasets
import lib.transformer
import lib.utils
import re
import torch
import torch.nn.functional as F
import tqdm
from torch import nn, optim, autograd

def main(
    batch_size=64,
    seq_len=256,
    steps=1_000_000,
    print_freq=1000,
    lr=1e-4,
    dim=1024,
    n_blocks=6,
    gamma_0=-3.5,
    gamma_1=4.5
    ):

    lib.utils.print_args(locals())

    train_data, _, _ = lib.lm_datasets.books1()
    train_iterator = lib.lm_datasets.sequential_iterator(
        train_data, batch_size, seq_len, 0, True)

    def token_posteriors(z, alpha, sigma_squared):
        squared_dist = (alpha[:,None,None] - z)**2 - z**2 # up to a const.
        log_likelihood = -squared_dist/(2*sigma_squared[:,None,None])
        log_posterior = F.log_softmax(log_likelihood, dim=2)
        posterior = log_posterior.exp()
        return posterior, log_posterior

    def gaussian_kl(mu_p, sigma_squared_p, mu_q, sigma_squared_q):
        """KL(p||q)"""
        return (
            0.5*(sigma_squared_q.log() - sigma_squared_p.log())
            + (sigma_squared_p+(mu_p - mu_q)**2)/(2*sigma_squared_q)
            - 0.5
        )

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('pos_codes',
                lib.transformer.position_codes(dim, seq_len)) 
            self.input = nn.Linear(256, dim)
            self.blocks = nn.Sequential(*[
                lib.transformer.TransformerBlock(dim, 8)
                for _ in range(n_blocks)
            ])
            self.output_norm = nn.LayerNorm(dim)
            self.output = nn.Linear(dim, 256)

        def forward(self, z, alpha, sigma_squared):
            x, _ = token_posteriors(z, alpha, sigma_squared)
            x = self.input(x * float(np.sqrt(256)))
            x = x + self.pos_codes[None,:,:]
            x = self.blocks(x)
            x = self.output_norm(x)
            logits = self.output(x)
            x_pred = F.softmax(logits, dim=2)
            return x_pred, logits

    model = Model().cuda()
    lib.utils.print_model(model)

    nll_ema = 0.

    def forward():
        nonlocal nll_ema

        x = next(train_iterator).cuda().long()

        x_onehot = F.one_hot(x, num_classes=256).float()

        t = torch.empty([batch_size], device='cuda')
        # Use first two entries of t for reconstruction/prior losses
        t[0], t[1] = 0, 1
        # Low-discrepancy sampler for the remaining entries of t
        t[2:] = torch.arange(batch_size-2, device='cuda') / float(batch_size-2)
        t[2:] = (t[2:] + torch.rand(1, device='cuda')) % 1
        t.requires_grad = True
        gamma = gamma_0 + t*(gamma_1 - gamma_0)
        sigma_squared = torch.sigmoid(gamma)
        alpha = torch.sigmoid(-gamma).sqrt()
        snr = (-gamma).exp()
        snr_prime = autograd.grad(snr.sum(), [t], create_graph=True)[0]

        z = (
            (alpha[:,None,None] * x_onehot) + 
            (sigma_squared[:,None,None].sqrt() * torch.randn_like(x_onehot))
        )
        x_pred, logits = model(z, alpha, sigma_squared)

        reconst_loss = F.cross_entropy(logits[0], x[0])

        prior_loss = gaussian_kl(
            alpha[1:2,None,None] * x_onehot,
            sigma_squared[1:2,None,None],
            torch.tensor(0., device='cuda'),
            torch.tensor(1., device='cuda')
        ).sum(dim=2).mean()

        diffusion_loss = (x_pred - x_onehot).pow(2).sum(dim=2).mean(dim=1)
        diffusion_loss = -0.5*(snr_prime * diffusion_loss)[2:].mean()

        nll = reconst_loss + prior_loss + diffusion_loss

        heuristic = F.cross_entropy(logits.permute(0,2,1), x,
            reduction='none').mean(dim=1)
        heuristic = -0.5*(snr_prime * heuristic)[2:].mean()

        nll_ema = (.9999 * nll_ema) + (.0001 * nll.item())

        return heuristic, nll, reconst_loss, prior_loss, torch.tensor(nll_ema)

    def hook(_):
        torch.save(model.state_dict(), 'model.pt')

    opt = optim.Adam(
        list(model.parameters()), lr=lr)
    lib.utils.train_loop(forward, opt, steps,
        print_freq=print_freq, lr_warmup_steps=100, lr_cooldown_steps=steps//10,
        names=['nll', 'reconst', 'prior', 'nll_ema'],
        hook=hook, hook_freq=10_000
    )
    
    return nll_ema

if __name__ == '__main__':
    fire.Fire(main)