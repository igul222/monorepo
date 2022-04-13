"""
Character-level continuous-time variational diffusion model.

TODO:
- Implement sampling
- Scale the transformer residuals
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
    batch_size=128,
    seq_len=128,
    steps=1_000_000,
    print_freq=1000,
    lr=1e-4,
    dim=1024,
    n_blocks=6,
    n_heads=8,
    gamma_0=2.0,
    gamma_1=9.2,
    # gamma_0=-3.5,
    # gamma_1=4.5,
    grad_accumulation_steps=1,
    weights_path=None,
    sampling_timesteps=4096
    ):

    lib.utils.print_args(locals())

    train_data, _, _ = lib.lm_datasets.books1()
    train_iterator = lib.lm_datasets.sequential_iterator(
        train_data, batch_size, seq_len, 0, True)

    def token_posterior(z, alpha, sigma_squared, bias):
        squared_dist = (alpha[:,None,None] - z)**2 - z**2 # up to a const.
        log_probs = -squared_dist/(2*sigma_squared[:,None,None])
        return F.softmax(log_probs + bias, dim=2)

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
            self.input_bias = nn.Parameter(torch.zeros(256))
            self.input = nn.Linear(256, dim)
            self.blocks = nn.ModuleList([
                lib.transformer.TransformerBlock(dim, n_heads)
                for _ in range(n_blocks)
            ])
            self.output_norm = nn.LayerNorm(dim)
            self.output = nn.Linear(dim, 256)

        def forward(self, z, alpha, sigma_squared):
            # x = token_posterior(z, alpha, sigma_squared,
            #     self.input_bias[None,None,:]) * float(np.sqrt(256))
            x = z
            x = self.input(x) + self.pos_codes[None,:,:]
            # breakpoint()
            for i in range(len(self.blocks)):
                x = self.blocks[i](x)
            x = self.output_norm(x)
            logits = self.output(x)
            x_pred = F.softmax(logits, dim=2)
            return x_pred, logits

    def noise_schedule(t):
        gamma = gamma_0 + t*(gamma_1 - gamma_0)
        return gamma

    # x_scale = 1
    x_scale = float(np.sqrt(256))

    model = Model().cuda()
    lib.utils.print_model(model)
    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path))

    nll_ema = 0.

    def forward():
        nonlocal nll_ema

        x = next(train_iterator).cuda().long()

        x_onehot = F.one_hot(x, num_classes=256).float() * x_scale

        t = torch.empty([batch_size], device='cuda')
        # Use first two entries of t for reconstruction/prior losses
        t[0], t[1] = 0, 1
        # Low-discrepancy sampler for the remaining entries of t
        t[2:] = torch.arange(batch_size-2, device='cuda') / float(batch_size-2)
        t[2:] = (t[2:] + torch.rand(1, device='cuda')) % 1
        t.requires_grad = True
        gamma = noise_schedule(t)
        sigma_squared = torch.sigmoid(gamma)
        alpha = torch.sigmoid(-gamma).sqrt()
        snr = (-gamma).exp()
        snr_prime = autograd.grad(snr.sum(), [t], create_graph=True)[0]

        z = (
            (alpha[:,None,None] * x_onehot) + 
            (sigma_squared[:,None,None].sqrt() * torch.randn_like(x_onehot))
        )
        x_pred, logits = model(z, alpha, sigma_squared)
        x_pred = x_pred * x_scale

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
        hook=hook, hook_freq=10_000,
        grad_accumulation_steps=grad_accumulation_steps
    )
    
    # Sampling (implements Appendix A.4 eqn 33 in VDM). Most of the calculations
    # happen in float64 because I was too lazy to write numerically stable code.
    torch.set_default_dtype(torch.float64)
    with torch.no_grad():
        z = torch.randn((32, seq_len, 256), device='cuda')
        for t in tqdm.tqdm(torch.linspace(1., 0., sampling_timesteps)):
            t = t[None].cuda()
            s = t - 1. / sampling_timesteps
            gamma_s = noise_schedule(s)
            gamma_t = noise_schedule(t)
            alpha_squared_s = torch.sigmoid(-gamma_s)
            alpha_squared_t = torch.sigmoid(-gamma_t)
            sigma_squared_t = torch.sigmoid(gamma_t)
            sigma_t = sigma_squared_t.sqrt()
            with torch.cuda.amp.autocast():
                x_pred = model(z.float(),
                    alpha_squared_t.sqrt().float(),
                    sigma_squared_t.float()
                )[0]
            x_pred = x_pred.double() * x_scale
            x_samples = x_pred.argmax(dim=-1)
            if t > 0:
                c = -torch.expm1(gamma_s - gamma_t)
                z = z - c * (z - alpha_squared_t.sqrt() * x_pred)
                z *= (alpha_squared_s/alpha_squared_t).sqrt()
                z += (c * (1 - alpha_squared_s)).sqrt() * torch.randn_like(z)
    torch.set_default_dtype(torch.float32)

    for x in x_samples:
        x = x.detach().cpu().numpy().tobytes()
        # replace non-ascii bytes with '#'
        x = re.sub(rb'[^\x00-\x7F]', b'#', x)
        x = x.decode('utf-8')
        # replace newlines with '↵' symbol for easier printing
        print(x.replace("\n", "↵"))

    return nll_ema

if __name__ == '__main__':
    fire.Fire(main)