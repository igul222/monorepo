"""
MNIST VDM with black-box classifier guidance.
For fast training: keep hyperparams as-is.
For high quality: set batch size 256, 40K steps.
"""

import fire
import math
import numpy as np
import lib.datasets
import lib.transformer
import lib.utils
import torch
import torch.nn.functional as F
import tqdm
from torch import nn, optim, autograd

def main(**args):
    args = lib.utils.AttributeDict(args)
    args.setdefault('batch_size', 64)
    args.setdefault('dim', 512)
    args.setdefault('lr', 1e-3)
    args.setdefault('print_freq', 1000)
    args.setdefault('steps', 10_000)
    args.setdefault('sampling_timesteps', 1024)
    args.setdefault('gamma_0', -5.)
    args.setdefault('gamma_1', 5.)
    lib.utils.print_args(args)

    # Lots of annoying big/small numbers throughout this code, so we'll do
    # everything in fp64 by default and explicitly switch to fp16 where
    # appropriate.
    torch.set_default_dtype(torch.float64)

    X_train, y_train = lib.datasets.mnist('train')

    class Denoiser(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_conv = nn.Conv2d(1, args.dim, 4, 2, 1)
            self.register_buffer('pos_codes',
                lib.transformer.position_codes(args.dim, 14*14))
            self.transformer_blocks = nn.Sequential(*[
                lib.transformer.TransformerBlock(args.dim, 8, output_scale=0.5)
                for _ in range(4)
            ])
            self.output_norm = nn.LayerNorm(args.dim)
            self.output_conv = nn.ConvTranspose2d(args.dim, 1, 4, 2, 1)

        def forward(self, z, gamma):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                x = z.float().view(-1, 1, 28, 28)
                x = self.input_conv(x)
                x = F.gelu(x)
                x = x.view(-1, args.dim, 196).permute(0, 2, 1)
                x = x + self.pos_codes[None,:,:]
                x = self.transformer_blocks(x)
                x = self.output_norm(x)
                x = x.view(-1, 14, 14, args.dim).permute(0, 3, 1, 2)
                x = self.output_conv(x)
                x = x.view(-1, 784)
            x = x.double()
            alpha = torch.sigmoid(-gamma).sqrt()
            sigma = torch.sigmoid(gamma).sqrt()
            x = (z - (sigma * x)) / alpha
            return x

    # Train diffusion model
  
    denoiser = Denoiser().float().cuda()
    lib.utils.print_model(denoiser)

    def forward():
        x = lib.utils.get_batch(X_train, args.batch_size)
        t = torch.rand(x.shape[0], device='cuda')[:,None]
        gamma = args.gamma_0 + (t * (args.gamma_1 - args.gamma_0))
        alpha = torch.sigmoid(-gamma).sqrt()
        sigma = torch.sigmoid(gamma).sqrt()
        z = (alpha * x) + (sigma * torch.randn_like(x))
        x_reconst = denoiser(z, gamma)
        diffusion_loss = (-gamma).exp() * (x - x_reconst).pow(2)
        return diffusion_loss.mean()

    opt = optim.Adam(denoiser.parameters(), lr=args.lr, betas=(0.9, 0.99))
    lib.utils.train_loop(forward, opt, args.steps, print_freq=args.print_freq,
        lr_warmup_steps=100, lr_decay=True)

    # Train classifier

    classifier = nn.Sequential(
        nn.Linear(784, 1024),  nn.ReLU(),
        nn.Linear(1024, 1024), nn.ReLU(),
        nn.Linear(1024, 10)
    ).float().cuda()
    lib.utils.print_model(classifier)

    def forward():
        x, y = lib.utils.get_batch([X_train, y_train], 1024)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits = classifier(x.float())
        return F.cross_entropy(logits.double(), y)

    opt = optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=1.0)
    lib.utils.train_loop(forward, opt, 10_000, print_freq=1000,
        lr_warmup_steps=100, lr_decay=True)

    # Generate samples

    with torch.no_grad():
        z = torch.randn((64, 784), device='cuda')
        for t in tqdm.tqdm(torch.linspace(1., 0., args.sampling_timesteps)):
            t = t[None].cuda()
            s = t - 1. / args.sampling_timesteps
            gamma_s = args.gamma_0 + (s * (args.gamma_1 - args.gamma_0))
            gamma_t = args.gamma_0 + (t * (args.gamma_1 - args.gamma_0))
            alpha_squared_s = torch.sigmoid(-gamma_s)
            alpha_squared_t = torch.sigmoid(-gamma_t)
            with torch.enable_grad():
                z.requires_grad = True
                x_reconst = denoiser(z, gamma_t)
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    logits = classifier(x_reconst.float())
                logp_y_given_z = F.log_softmax(logits, dim=-1)[:,3]
                grad = autograd.grad(logp_y_given_z.sum(), [z])[0]
            # Guidance!
            sigma_squared_t = torch.sigmoid(gamma_t)
            x_reconst += 3.0 * grad * sigma_squared_t / alpha_squared_t.sqrt()
            if t > 0:
                c = -torch.expm1(gamma_s - gamma_t)
                z *= (1 - c) * alpha_squared_s.sqrt() / alpha_squared_t.sqrt()
                z += c * (alpha_squared_s.sqrt() * x_reconst.double())
                z += (c * (1 - alpha_squared_s)).sqrt() * torch.randn_like(z)
        x_samples = x_reconst
    lib.utils.save_image_grid(x_samples, f'samples.png')

if __name__ == '__main__':
    fire.Fire(main)