"""
Text VDM with embeddings.
"""

import collections
import fire
import math
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import lib.lm_datasets
import lib.transformer
import lib.utils
import re
import os
import time
import torch
import torch.nn.functional as F
import tqdm
import warnings
from torch import nn, optim, autograd

def main(**args):
    args = lib.utils.AttributeDict(args)
    args.setdefault('dataset', 'books1_char')
    args.setdefault('dim', 1024)
    args.setdefault('lr', 1e-4)
    args.setdefault('n_blocks', 9)
    args.setdefault('n_heads', 8)
    args.setdefault('noise_schedule_lr', 3e-3)
    args.setdefault('save_weights', False)
    args.setdefault('steps', 1_000_000)
    args.setdefault('print_freq', 1000)
    args.setdefault('sampling_timesteps', 4096)
    args.setdefault('weights_path', None)

    if args.dataset == 'books1_char':
        args.setdefault('gamma_0', 2.0)
        args.setdefault('gamma_1', 12.0)
        args.setdefault('batch_size', 128)
        args.setdefault('seq_len', 128)
        args.setdefault('vocab_size', 256)
    elif args.dataset == 'books1_word':
        args.setdefault('gamma_0', 5.0)
        args.setdefault('gamma_1', 15.0)
        args.setdefault('batch_size', 192)
        args.setdefault('seq_len', 64)
        args.setdefault('vocab_size', 8192)
    elif args.dataset == 'e2e':
        args.setdefault('gamma_0', 3.0)
        args.setdefault('gamma_1', 13.0)
        args.setdefault('batch_size', 256)
        args.setdefault('seq_len', 64)
        args.setdefault('vocab_size', 821)
    elif args.dataset == 'rocstories':
        args.setdefault('gamma_0', 3.0)
        args.setdefault('gamma_1', 13.0)
        args.setdefault('batch_size', 128)
        args.setdefault('seq_len', 64)
        args.setdefault('vocab_size', 11043)

    lib.utils.print_args(args)

    # Lots of annoying big/small numbers throughout this code, so we'll do
    # everything in fp32 by default and explicitly switch to fp16 where
    # appropriate.

    # Each dataset has slightly different loading code:

    if args.dataset in ['books1_char', 'books1_word']:

        dataset = lib.lm_datasets.books1()
        (train_data, _, test_data), (word2idx, idx2word) = dataset

        train_iterator = lib.lm_datasets.random_iterator(
            train_data, args.batch_size, args.seq_len)
        test_iterator = lib.lm_datasets.random_iterator(
            test_data, args.batch_size, args.seq_len)

    elif args.dataset == 'e2e':

        dataset = lib.lm_datasets.e2e(args.vocab_size)
        (train_data, _, test_data), (word2idx, idx2word) = dataset
        train_data = torch.cat(train_data)
        test_data = torch.cat(test_data)

        train_iterator = lib.lm_datasets.random_iterator(
            train_data, args.batch_size, args.seq_len)
        test_iterator = lib.lm_datasets.random_iterator(
            test_data, args.batch_size, args.seq_len)

    elif args.dataset == 'rocstories':

        dataset = lib.lm_datasets.rocstories(args.vocab_size)
        (train_data, _, test_data), (word2idx, idx2word) = dataset

        train_iterator = lib.lm_datasets.padded_random_iterator(
            train_data, args.batch_size, args.seq_len, word2idx[b'PAD'])
        test_iterator = lib.lm_datasets.padded_random_iterator(
            test_data, args.batch_size, args.seq_len, word2idx[b'PAD'])


    def gaussian_kl(mu_p, sigma_p, mu_q, sigma_q):
        """KL(p||q)"""
        # return (
        #     sigma_q.log() - sigma_p.log()
        #     + (sigma_p**2 + (mu_p - mu_q)**2)/(2*sigma_q**2)
        #     - 0.5
        # )
        result = mu_p.clone()
        result.sub_(mu_q)
        result.pow_(2)
        result.add_(sigma_p**2)
        result.div_(2 * sigma_q**2)
        result.add_(sigma_q.log() - sigma_p.log() - 0.5)
        return result

    def cross_entropy(logits, x, reduction):
        """Memory-efficient drop-in replacement for F.cross_entropy."""
        assert(reduction=='none')
        logits_logsumexp = torch.logsumexp(logits, dim=1)
        return logits_logsumexp - logits[
            torch.arange(x.shape[0], device='cuda')[:,None],
            x,
            torch.arange(x.shape[1], device='cuda')[None,:],
        ]

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = nn.Parameter(
                torch.randn(args.vocab_size, args.dim))
            self.embeddings_A = nn.Parameter(torch.eye(args.dim))
            self.register_buffer('pos_codes',
                lib.transformer.position_codes(args.dim, args.seq_len))
            self.blocks = nn.ModuleList([
                lib.transformer.TransformerBlock(args.dim, args.n_heads,
                    output_scale=float(1./np.sqrt(args.n_blocks))
                )
                for _ in range(args.n_blocks)
            ])
            self.output_norm = nn.LayerNorm(args.dim)
            self.output = nn.Linear(args.dim, args.dim)

            self.decoder_A = nn.Parameter(torch.eye(args.dim))
            self.decoder_bias = nn.Parameter(torch.zeros([args.vocab_size]))

        def forward(self, z, gamma):
            x = z
            x.add_(self.pos_codes[None,:,:])
            for i in range(len(self.blocks)):
                x = self.blocks[i](x)
            x = self.output_norm(x)
            return self.output(x)

        def decode(self, z):
            x = z
            x = x @ (self.decoder_A @ self.decoder_A.T)
            # x.div_(float(np.sqrt(args.dim)))
            x = x @ (self.embeddings_A @ self.embeddings_A.T)
            x = x @ self.embeddings.T
            x.add_(self.decoder_bias[None,None,:])
            return x

    class NoiseSchedule(nn.Module):
        def __init__(self):
            super().__init__()
            self.W1 = nn.Parameter(torch.randn(1024, 1))
            self.b1 = nn.Parameter(torch.randn(1024))
            self.W2 = nn.Parameter(torch.randn(1, 1024))
        def forward(self, t):
            """t.shape: [n]"""
            W1 = F.softplus(self.W1)
            W2 = 0.01 * F.softplus(self.W2)
            def gamma_tilde(t):
                h = t[:,None] - 0.5
                h = (h @ W1.T) + self.b1[None,:]
                h = torch.tanh(h)
                h = (h @ W2.T)[:,0]
                return h
            gamma_tilde_0 = gamma_tilde(torch.tensor([0.], device='cuda',
                dtype=torch.float32))
            gamma_tilde_1 = gamma_tilde(torch.tensor([1.], device='cuda',
                dtype=torch.float32))
            gamma_tilde_t = gamma_tilde(t)
            return args.gamma_0 + (
                (args.gamma_1 - args.gamma_0) *
                (gamma_tilde_t - gamma_tilde_0) /
                (gamma_tilde_1 - gamma_tilde_0)
            )

    model = Model().cuda()
    noise_schedule = NoiseSchedule().cuda()
    lib.utils.print_model(model)
    if args.weights_path is not None:
        model.load_state_dict(
            torch.load(os.path.join(args.weights_path, 'model.pt')))
        noise_schedule.load_state_dict(
            torch.load(os.path.join(args.weights_path, 'noise_schedule.pt')))

    def compute_losses(x):
        t = torch.empty([args.batch_size], device='cuda')
        # First two entries of t are used for reconst_loss and prior_loss below
        t[0], t[1] = 0, 1
        # Low-discrepancy sampler for the remaining entries of t
        t[2:] = torch.arange(args.batch_size-2, device='cuda')
        t[2:] /= float(args.batch_size-2)
        t[2:] = (t[2:] + torch.rand(1, device='cuda')) % 1
        t.requires_grad = True
        with torch.enable_grad():
            # Don't propagate grads for the first two entries of t
            gamma = torch.cat([
                noise_schedule(t[:2]).detach(),
                noise_schedule(t[2:])
            ])
            gamma_prime = autograd.grad(gamma.sum(), [t], create_graph=True)[0]
        # Manually edit gradients so that the noise schedule minimizes
        # E[nll^2] while the rest of the model minimizes E[nll].
        def set_grad_hook(tensor):
            if tensor.requires_grad:
                def grad_hook(grad):
                    handle.remove()
                    new_grad = torch.clone(grad.detach())
                    new_grad[2:] *= 2. * diffusion_loss[2:].detach()
                    return new_grad
                handle = tensor.register_hook(grad_hook)
        gamma = gamma.clone()
        set_grad_hook(gamma)
        set_grad_hook(gamma_prime)
        # Quantities derived from gamma and gamma_prime:
        alpha = torch.sigmoid(-gamma).sqrt()
        sigma = torch.sigmoid(gamma).sqrt()
        snr_prime = -(-gamma).exp() * gamma_prime # SNR = exp(-gamma)
        # Scaled by a constant to keep the loss in a reasonable range:
        snr_prime_scaled = -(-gamma + args.gamma_0).exp() * gamma_prime

        # Construct z (with reparam. trick gradients) using mostly in-place ops
        z0 = model.embeddings[x, :] @ (model.embeddings_A @ model.embeddings_A.T)
        z = torch.randn([x.shape[0], x.shape[1], args.dim],
            dtype=torch.float32, device='cuda')
        z.mul_(sigma[:,None,None])
        z.add_(alpha[:,None,None] * z0)

        # Model forward pass
        with torch.cuda.amp.autocast():
            z0_reconst_half = model(z, gamma)
        z0_reconst = z0_reconst_half.float()

        # NLL computation.
        logits = model.decode((z0 * alpha[0:1]) + (sigma[0:1] * torch.randn_like(z0)))
        reconst_loss = cross_entropy(
            logits.permute(0,2,1), x, reduction='none').mean()

        prior_loss = gaussian_kl(
            alpha[1] * z0,
            sigma[1],
            torch.tensor(0., device='cuda'),
            torch.tensor(1., device='cuda')
        ).sum(dim=2).mean()

        diffusion_loss = (z0 - z0_reconst).pow(2).sum(dim=2).mean(dim=1)
        diffusion_loss = -0.5*(snr_prime * diffusion_loss)

        nll = reconst_loss + prior_loss + diffusion_loss[2:].mean()

        return (
            nll,
            reconst_loss,
            prior_loss
        )

    opt = optim.Adam([
        {'params': model.parameters(), 'lr':args.lr},
        {'params': noise_schedule.parameters(), 'lr':args.noise_schedule_lr}
    ])

    # Train loop

    nll_ema = 0.
    def forward():
        nonlocal nll_ema
        x = next(train_iterator).cuda().long()
        nll, reconst, prior = compute_losses(x)
        nll_ema = (.9999 * nll_ema) + (.0001 * nll.item())
        return (nll, reconst, prior, torch.tensor(nll_ema))

    def hook(step):
        # Save weights
        if args.save_weights:
            torch.save(model.state_dict(), 'model.pt')
            torch.save(noise_schedule.state_dict(), 'noise_schedule.pt')
        # Save gamma plot
        t = torch.linspace(0., 1., 1024).cuda()
        gamma = noise_schedule(t)
        plt.clf()
        plt.plot(t.detach().cpu().numpy(), gamma.detach().cpu().numpy())
        plt.savefig(f'gamma_{step}.jpg')
        # Compute test NLL
        with torch.no_grad():
            x = next(test_iterator).cuda().long()
            losses = []
            for i in range(500):
                nll, _, _ = compute_losses(x)
                losses.append(nll.item())
            print(f'Test NLL: approx. {np.mean(losses)}')

    lib.utils.train_loop(forward, opt, args.steps,
        names=['reconst', 'prior', 'nll_ema'],
        hook=hook, hook_freq=2000,
        print_freq=args.print_freq, lr_warmup_steps=100,
        lr_cooldown_steps=args.steps//10,
        amp_autocast=False
    )

    # # Sampling (implements Appendix A.4 eqn 33 in VDM). Right now this needs
    # # float64 to work, but probably just because it was carelessly implemented.
    # torch.set_default_dtype(torch.float64)
    with torch.no_grad():
        z = torch.randn((32, args.seq_len, args.dim), device='cuda')
        for t in tqdm.tqdm(torch.linspace(1., 0., args.sampling_timesteps)):
            t = t[None].cuda()
            s = t - 1. / args.sampling_timesteps
            gamma_s = noise_schedule(s.float()).double()
            gamma_t = noise_schedule(t.float()).double()
            alpha_squared_s = torch.sigmoid(-gamma_s)
            alpha_squared_t = torch.sigmoid(-gamma_t)
            sigma_squared_t = torch.sigmoid(gamma_t)
            sigma_t = sigma_squared_t.sqrt()
            with torch.cuda.amp.autocast():
                z0_reconst_half = model(z.float(), gamma_t.float())
            z0_reconst = z0_reconst_half.double()
            # x_pred = F.softmax(logits, dim=2)
            # x_pred.mul_(x_scale)
            # x_samples = x_pred.argmax(dim=-1)
            if t > 0:
                c = -torch.expm1(gamma_s - gamma_t)
                z = z - c * (z - alpha_squared_t.sqrt() * z0_reconst)
                z *= (alpha_squared_s/alpha_squared_t).sqrt()
                z += (c * (1 - alpha_squared_s)).sqrt() * torch.randn_like(z)
        with torch.cuda.amp.autocast():
            logits_half = model.decode(z.float())
        x_pred = F.softmax(logits_half.double(), dim=2)
        x_samples = x_pred.argmax(dim=-1)

        for x in x_samples:
            x = x.tolist()
            x = b' '.join([idx2word[i] for i in x])
            x = x.decode('utf-8', 'ignore')
            # replace newlines with '↵' symbol for cleaner printing
            print(x.replace("\n", "↵"))

    return nll_ema

if __name__ == '__main__':
    fire.Fire(main)