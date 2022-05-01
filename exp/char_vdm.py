"""
Character-level continuous-time variational diffusion model.
"""

import collections
import fire
import math
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import lib.ddp
import lib.diffusion_lm_datasets
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
from torch.nn.parallel import DistributedDataParallel as DDP

def main(**args):
    args = lib.utils.AttributeDict(args)
    args.setdefault('dataset', 'rocstories_gpt')
    args.setdefault('dim', 1024)
    args.setdefault('lr', 1e-4)
    args.setdefault('grad_accumulation_steps', 1)
    args.setdefault('n_blocks', 9)
    args.setdefault('n_heads', 8)
    args.setdefault('noise_schedule_lr', 3e-3)
    args.setdefault('rank', 0)
    args.setdefault('reweighted_loss', False)
    args.setdefault('save_weights', False)
    args.setdefault('steps', 1_000_000)
    args.setdefault('print_freq', 1000)
    args.setdefault('sampling_timesteps', 4096)
    args.setdefault('weights_path', None)

    if args.dataset == 'books1_char':
        args.setdefault('gamma_0', 2.0)
        args.setdefault('gamma_1', 12.0)
        args.setdefault('batch_size', 128)
    elif args.dataset == 'books1_word':
        args.setdefault('gamma_0', 5.0)
        args.setdefault('gamma_1', 15.0)
        args.setdefault('batch_size', 192)
    elif args.dataset in ['e2e', 'e2e_gpt']:
        args.setdefault('gamma_0', 3.0)
        args.setdefault('gamma_1', 13.0)
        args.setdefault('batch_size', 256)
    elif args.dataset in ['rocstories', 'rocstories_gpt']:
        args.setdefault('gamma_0', 5.0)
        args.setdefault('gamma_1', 15.0)
        args.setdefault('batch_size', 96)
    lib.utils.print_args(args)

    # Lots of annoying big/small numbers throughout this code, so we'll do
    # everything in fp32 by default and explicitly switch to fp16 where
    # appropriate.

    dataset = lib.diffusion_lm_datasets.REGISTRY[args.dataset](args.batch_size)
    (train_iterator, test_iterator), (word2idx, idx2word) = dataset

    typical_seq_len = next(train_iterator)[0].shape[1]
    vocab_size = len(word2idx)
    print(f'typical_seq_len: {typical_seq_len}, vocab_size: {vocab_size}')

    x_scale = float(np.sqrt(vocab_size))

    def gaussian_kl(mu_p, sigma_p, mu_q, sigma_q):
        """KL(p||q)"""
        return (
            sigma_q.log() - sigma_p.log()
            + (sigma_p**2 + (mu_p - mu_q)**2)/(2*sigma_q**2)
            - 0.5
        )

    def cross_entropy(logits, x, reduction):
        """Memory-efficient drop-in replacement for F.cross_entropy."""
        assert(reduction=='none')
        logits_logsumexp = torch.logsumexp(logits, dim=1)
        return logits_logsumexp - logits[
            torch.arange(x.shape[0], device='cuda')[:,None],
            x,
            torch.arange(x.shape[1], device='cuda')[None,:],
        ]

    def sqrt_sigmoid(x):
        """Numerically stable sqrt(sigmoid(x))"""
        return torch.sigmoid(x.double()).sqrt().float()

    class InputPreprocessor(nn.Module):
        """Normalizes noisy simplex vectors at the input of the model."""
        def __init__(self):
            super().__init__()
            self.gamma_mlp = nn.Sequential(
                nn.Linear(1, 1024),
                nn.Tanh(),
                nn.Linear(1024, vocab_size+1)
            )

        def forward(self, z, gamma, input_weights):
            gamma = gamma - ((args.gamma_0 + args.gamma_1) / 2.)
            gamma = gamma / ((args.gamma_1 - args.gamma_0) / 2.)
            gamma_mlp_out = self.gamma_mlp(gamma[:,None])
            with torch.cuda.amp.autocast(enabled=False):
                gamma_mlp_out = gamma_mlp_out.float()
                bias = gamma_mlp_out[:,None,1:]
                temp = gamma_mlp_out[:,0,None,None]
                # z = log_p(xt|x0) up to a const.
                z.mul_(F.softplus(temp))
                z.add_(bias)
                result = F.softmax(z, dim=2)
                result = result @ input_weights.T
                result.mul_(x_scale)
            return result

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('pos_codes',
                lib.transformer.position_codes(args.dim))
            self.input_preprocessor = InputPreprocessor()
            self.blocks = nn.ModuleList([
                lib.transformer.TransformerBlock(args.dim, args.n_heads,
                    output_scale=float(1./np.sqrt(args.n_blocks))
                )
                for _ in range(args.n_blocks)
            ])
            self.output_norm = nn.LayerNorm(args.dim)
            self.output = nn.Linear(args.dim, vocab_size)
            self.output.bias.data.zero_()

        def forward(self, z, gamma):
            x = self.input_preprocessor(z, gamma, self.output.weight.T)
            x.add_(self.pos_codes[None,:x.shape[1],:])
            for i in range(len(self.blocks)):
                x = self.blocks[i](x)
            x = self.output_norm(x)
            return self.output(x)

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

    ddp_model = DDP(model)
    ddp_noise_schedule = DDP(noise_schedule)

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
                ddp_noise_schedule(t[:2]).detach(),
                ddp_noise_schedule(t[2:])
            ])
            gamma_prime = autograd.grad(gamma.sum(), [t], create_graph=True)[0]
        # Manually edit gradients so that the noise schedule minimizes
        # E[heuristic^2] while the rest of the model minimizes E[heuristic].
        def set_grad_hook(tensor):
            if tensor.requires_grad:
                def grad_hook(grad):
                    handle.remove()
                    new_grad = torch.clone(grad.detach())
                    new_grad[2:] *= 2. * heuristic[2:].detach()
                    return new_grad
                handle = tensor.register_hook(grad_hook)
        gamma = gamma.clone()
        set_grad_hook(gamma)
        set_grad_hook(gamma_prime)
        # Quantities derived from gamma and gamma_prime:
        alpha = sqrt_sigmoid(-gamma)
        sigma = sqrt_sigmoid(gamma)
        snr_prime = -(-gamma).exp() * gamma_prime # SNR = exp(-gamma)

        # Construct z (with reparam. trick gradients) using only in-place ops
        z = torch.randn([x.shape[0], x.shape[1], vocab_size],
            dtype=torch.float32, device='cuda')
        z.mul_(sigma[:,None,None])
        z.scatter_add_(
            2,
            x[:,:,None],
            (x_scale * alpha[:,None,None].expand(-1,x.shape[1],-1))
        )

        # Model forward pass
        with torch.cuda.amp.autocast():
            logits_half = ddp_model(z, gamma)
        logits = logits_half.float()

        # Training objective
        heuristic = cross_entropy(logits.permute(0,2,1), x, reduction='none')
        heuristic = gamma_prime * heuristic.mean(dim=1)
        if not args.reweighted_loss:
            heuristic *= (args.gamma_0 - gamma).exp()

        # NLL computation. Not used in training, but computed / printed for
        # convenience.
        with torch.no_grad():
            reconst_loss = F.cross_entropy(logits[0], x[0])

            prior_onehot = F.one_hot(x[1,0],num_classes=vocab_size).float()
            prior_loss = gaussian_kl(
                alpha[1] * x_scale * prior_onehot,
                sigma[1],
                torch.tensor(0., device='cuda'),
                torch.tensor(1., device='cuda')
            ).sum()
            
            diffusion_loss = F.softmax(logits_half, dim=2)
            diffusion_loss.scatter_add_(
                2,
                x[:,:,None],
                -torch.ones([x.shape[0], x.shape[1], 1], device='cuda',
                    dtype=torch.float16)
            )
            diffusion_loss.pow_(2)
            diffusion_loss = diffusion_loss.sum(dim=2).mean(dim=1).float()
            diffusion_loss.mul_(x_scale**2)
            diffusion_loss = -0.5*(snr_prime * diffusion_loss)[2:].mean()
            nll = reconst_loss + prior_loss + diffusion_loss

        return (
            heuristic[2:].mean(),
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
        x = next(train_iterator)[0].cuda().long()
        loss, nll, reconst, prior = compute_losses(x)
        nll_ema = (.9999 * nll_ema) + (.0001 * nll.item())
        return (loss, nll, reconst, prior, torch.tensor(nll_ema))

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
            x = next(test_iterator)[0].cuda().long()
            losses = []
            for i in range(500):
                _, nll, _, _ = compute_losses(x)
                losses.append(nll.item())
            print(f'Test NLL: approx. {np.mean(losses)}')

    lib.utils.train_loop(forward, opt, args.steps,
        names=['nll', 'reconst', 'prior', 'nll_ema'],
        hook=hook, hook_freq=2000,
        print_freq=args.print_freq, lr_warmup_steps=100,
        lr_cooldown_steps=args.steps//10,
        amp_autocast=False,
        grad_accumulation_steps=args.grad_accumulation_steps
    )

    # Sampling (implements Appendix A.4 eqn 33 in VDM). Right now this needs
    # float64 to work, but probably just because it was carelessly implemented.
    torch.set_default_dtype(torch.float64)
    with torch.no_grad():
        z = torch.randn((32, typical_seq_len, vocab_size), device='cuda')
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
                logits_half = model(z.float(), gamma_t.float())
            logits = logits_half.double()
            x_pred = F.softmax(logits, dim=2)
            x_pred.mul_(x_scale)
            x_samples = x_pred.argmax(dim=-1)
            if t > 0:
                c = -torch.expm1(gamma_s - gamma_t)
                z = z - c * (z - alpha_squared_t.sqrt() * x_pred)
                z *= (alpha_squared_s/alpha_squared_t).sqrt()
                z += (c * (1 - alpha_squared_s)).sqrt() * torch.randn_like(z)

        for x in x_samples:
            x = x.tolist()
            x = b' '.join([idx2word[i] for i in x])
            x = x.decode('utf-8', 'ignore')
            # replace newlines with '↵' symbol for cleaner printing
            print(x.replace("\n", "↵"))

    return nll_ema

if __name__ == '__main__':
    fire.Fire(lib.ddp.wrap_main(main))
