"""
Character-level continuous-time variational diffusion model.
"""

import collections
import copy
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
import sys
import time
import torch
import torch.nn.functional as F
import tqdm
import warnings
from torch import nn, optim, autograd
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import contextmanager

def main(**args):
    args = lib.utils.AttributeDict(args)
    args.setdefault('batch_size', 128)
    args.setdefault('dataset', 'rocstories_gpt')
    args.setdefault('ema_decay', 0.999)
    args.setdefault('grad_accumulation_steps', 1)
    args.setdefault('hook_freq', 5000)
    args.setdefault('lr', 1e-4)
    args.setdefault('lr_warmup_steps', 1000)
    args.setdefault('lr_cooldown_steps', 10_000)
    args.setdefault('model_size', 'small')
    args.setdefault('noise_schedule_lr', 3e-3)
    args.setdefault('print_freq', 1000)
    args.setdefault('rank', 0)
    args.setdefault('reweighted_loss', False)
    args.setdefault('save_weights', False)
    args.setdefault('steps', 1_000_000)
    args.setdefault('sampling_timesteps', 4096)
    args.setdefault('weights_path', None)
    args.setdefault('input_duplicates', 1)
    args.setdefault('truncate_p', 0.)
    args.setdefault('nucleus_start_t', 0.)
    args.setdefault('zero_unks', False)

    if args.model_size == 'small':
        args.setdefault('dim', 1024)
        args.setdefault('n_blocks', 9)
        args.setdefault('n_heads', 8)
    elif args.model_size == 'medium':
        args.setdefault('dim', 1536)
        args.setdefault('n_blocks', 12)
        args.setdefault('n_heads', 12)

    if args.dataset == 'books1_char':
        args.setdefault('gamma_0', 2.0)
        args.setdefault('gamma_1', 12.0)
    elif args.dataset == 'books1_word':
        args.setdefault('gamma_0', 5.0)
        args.setdefault('gamma_1', 15.0)
    elif args.dataset in ['e2e', 'e2e_gpt']:
        args.setdefault('gamma_0', 3.0)
        args.setdefault('gamma_1', 13.0)
    elif args.dataset in ['rocstories', 'rocstories_gpt']:
        args.setdefault('gamma_0', 5.0)
        args.setdefault('gamma_1', 15.0)

    lib.utils.print_args(args)

    # Lots of annoying big/small numbers throughout this code, so we'll do
    # everything in fp64 by default and explicitly switch to fp16 where
    # appropriate.
    torch.set_default_dtype(torch.float64)

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
            bias = gamma_mlp_out[:,None,1:]
            temp = gamma_mlp_out[:,None,:1]
            # z = log_p(xt|x0) up to transformations
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
            gamma_tilde_0 = gamma_tilde(torch.tensor([0.], device='cuda'))
            gamma_tilde_1 = gamma_tilde(torch.tensor([1.], device='cuda'))
            gamma_tilde_t = gamma_tilde(t)
            return args.gamma_0 + (
                (args.gamma_1 - args.gamma_0) *
                (gamma_tilde_t - gamma_tilde_0) /
                (gamma_tilde_1 - gamma_tilde_0)
            )

    model = Model().float().cuda()
    lib.utils.print_model(model)

    noise_schedule = NoiseSchedule().cuda()

    if args.weights_path is not None:
        model.load_state_dict(torch.load(
            os.path.join(args.weights_path, 'model.pt'),
            map_location=torch.device('cuda')
        ))
        noise_schedule.load_state_dict(torch.load(
            os.path.join(args.weights_path, 'noise_schedule.pt'),
            map_location=torch.device('cuda')
        ))

    model_ema = copy.deepcopy(model)

    ddp_model = DDP(model)
    ddp_noise_schedule = DDP(noise_schedule)

    def update_ema_params():
        model_params = [p for _,p in sorted(list(model.named_parameters()))]
        ema_params = [p for _,p in sorted(list(model_ema.named_parameters()))]
        for p1, p2 in zip(model_params, ema_params):
            p2.data.mul_(args.ema_decay)
            p2.data.add_((1 - args.ema_decay) * p1.data.float())

    @contextmanager
    def use_ema_params():
        model_params = [p for _,p in sorted(list(model.named_parameters()))]
        ema_params = [p for _,p in sorted(list(model_ema.named_parameters()))]
        old_data = [p.data.clone() for p in model_params]
        for p1, p2 in zip(model_params, ema_params):
            p1.data = p2.data.to(p1.data.dtype, copy=True)
        yield
        for param, old in zip(model_params, old_data):
            param.data.copy_(old)

    def compute_losses(x):
        x = torch.cat([x]*args.input_duplicates, dim=0)

        t = torch.empty([x.shape[0]], device='cuda')
        # First two entries of t are used for reconst_loss and prior_loss below
        t[0], t[1] = 0, 1
        # Low-discrepancy sampler for the remaining entries of t
        t[2:] = torch.arange(x.shape[0]-2, device='cuda')
        t[2:] /= float(x.shape[0]-2)
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
        alpha = torch.sigmoid(-gamma).sqrt()
        sigma = torch.sigmoid(gamma).sqrt()
        snr_prime = -(-gamma).exp() * gamma_prime # SNR = exp(-gamma)

        # Construct z (with reparam. trick gradients) using only in-place ops
        z = torch.randn([x.shape[0], x.shape[1], vocab_size],
            dtype=torch.float32, device='cuda')
        z.mul_(sigma[:,None,None])
        z.scatter_add_(
            2,
            x[:,:,None],
            (x_scale * alpha[:,None,None].expand(-1,x.shape[1],-1)).float()
        )

        # Model forward pass
        with torch.cuda.amp.autocast():
            logits = ddp_model(z, gamma.float())

        # Heuristic training loss
        heuristic = cross_entropy(
            logits.permute(0,2,1).float(), x, reduction='none'
        ).double()
        heuristic = gamma_prime * heuristic.mean(dim=1)
        if not args.reweighted_loss:
            heuristic *= (args.gamma_0 - gamma).exp()

        # NLL computation. Not used in training, but computed / printed for
        # convenience.
        with torch.no_grad():
            reconst_loss = F.cross_entropy(logits[0].float(), x[0])

            prior_onehot = F.one_hot(x[1,0],num_classes=vocab_size).double()
            prior_loss = gaussian_kl(
                alpha[1] * x_scale * prior_onehot,
                sigma[1],
                torch.tensor(0., device='cuda'),
                torch.tensor(1., device='cuda')
            ).sum()
            
            diffusion_loss = F.softmax(logits, dim=2)
            diffusion_loss.scatter_add_(
                2,
                x[:,:,None],
                -torch.ones([x.shape[0], x.shape[1], 1], device='cuda',
                    dtype=torch.float16)
            )
            diffusion_loss.pow_(2)
            diffusion_loss = diffusion_loss.mean(dim=1).double().sum(dim=1)
            diffusion_loss.mul_(x_scale**2)
            diffusion_loss = -0.5*(snr_prime * diffusion_loss)[2:].mean()
            nll = reconst_loss + prior_loss + diffusion_loss

        return (
            heuristic[2:].mean(),
            nll,
            reconst_loss,
            prior_loss
        )

    opt = optim.AdamW([
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
        update_ema_params()

        # Compute test NLL
        if step % args.hook_freq == (args.hook_freq - 1):
            with use_ema_params(), torch.no_grad():
                losses = []
                for i in range(40_000 // args.batch_size):
                    x = next(test_iterator)[0].cuda().long()
                    _, nll, _, _ = compute_losses(x)
                    losses.append(nll.item())
                print(f'Test NLL: approx. {np.mean(losses)}')

            if (args.rank == 0):
                if not (np.mean(losses) < 3.0):
                    print('Test NLL not less than 3.0; not saving weights!')
                    return
                # Save weights
                if args.save_weights:
                    torch.save(model_ema.state_dict(), 'model.pt')
                    torch.save(noise_schedule.state_dict(), 'noise_schedule.pt')

                # Save gamma plot
                t = torch.linspace(0., 1., 1024).cuda()
                gamma = noise_schedule(t)
                plt.clf()
                plt.plot(t.detach().cpu().numpy(), gamma.detach().cpu().numpy())
                plt.savefig(f'gamma_{step}.jpg')

    lib.utils.train_loop(forward, opt, args.steps,
        names=['nll', 'reconst', 'prior', 'nll_ema'],
        hook=hook, hook_freq=1,
        print_freq=args.print_freq, lr_warmup_steps=args.lr_warmup_steps,
        lr_cooldown_steps=args.lr_cooldown_steps,
        amp_autocast=False, amp_grad_scaler=True,
        grad_accumulation_steps=args.grad_accumulation_steps,
        ddp_models=[ddp_model, ddp_noise_schedule],
    )

    def nucleus(probs):
        probs_sort, probs_argsort = torch.sort(probs, dim=2)
        probs_mask = 1-(probs_sort.cumsum(dim=2) < args.truncate_p).half()
        probs_sort *= probs_mask
        probs_sort /= probs_sort.sum(dim=2, keepdim=True)
        probs_argsort_argsort = torch.zeros_like(probs_argsort)
        probs_argsort_argsort.scatter_(
            2, 
            probs_argsort,
            torch.arange(probs.shape[2], device='cuda')[None,None,:].expand(
                probs.shape[0], probs.shape[1], -1)
        )
        probs = probs_sort[
            torch.arange(probs.shape[0], device='cuda')[:,None,None],
            torch.arange(probs.shape[1], device='cuda')[None,:,None],
            probs_argsort_argsort
        ]
        return probs

    # Sampling (implements Appendix A.4 eqn 33 in VDM). Needs float64 to work.
    with use_ema_params(), torch.no_grad():
        z = torch.randn((32, typical_seq_len, vocab_size), device='cuda')
        for t in tqdm.tqdm(torch.linspace(1., 0., args.sampling_timesteps)):
            t = t[None].cuda()
            s = t - 1. / args.sampling_timesteps
            gamma_s = noise_schedule(s).double()
            gamma_t = noise_schedule(t).double()
            alpha_squared_s = torch.sigmoid(-gamma_s)
            alpha_squared_t = torch.sigmoid(-gamma_t)
            sigma_squared_s = torch.sigmoid(gamma_s)
            sigma_squared_t = torch.sigmoid(gamma_t)
            sigma_s = sigma_squared_s.sqrt()
            sigma_t = sigma_squared_t.sqrt()
            with torch.cuda.amp.autocast():
                logits = model(z.float(), gamma_t.float()).double()
            x_pred = F.softmax(logits, dim=2)
            if args.zero_unks:
                x_pred[:,:,word2idx[b'UNK']].zero_()
                x_pred.div_(x_pred.sum(dim=2, keepdim=True))
            if t < args.nucleus_start_t:
                x_pred = nucleus(x_pred)
            x_pred.mul_(x_scale)
            x_samples = x_pred.argmax(dim=-1)
            if t > 0:
                c = -torch.expm1(gamma_s - gamma_t)
                z *= (1 - c) * alpha_squared_s.sqrt() / alpha_squared_t.sqrt()
                z += c * (alpha_squared_s.sqrt() * x_pred)
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