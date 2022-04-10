"""
Character-level text DDPM.
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
from torch import nn, optim
import tqdm

def main(
    T=1024,
    batch_size=64,
    seq_len=256,
    steps=10_000,
    print_freq=1000,
    lr=1e-3,
    dim=512,
    n_heads=4,
    transformer_blocks=3,
    conv_blocks=0,
    grad_accumulation_steps=1,
    ):

    lib.utils.print_args(locals())

    # train_data, _, _ = lib.lm_datasets.enwik8()
    train_data, _, _ = lib.lm_datasets.books1()
    train_iterator = lib.lm_datasets.sequential_iterator(
        train_data, batch_size, seq_len, 0, True)

    # Variance of the forward process noise at each step
    beta_ = torch.linspace(1e-4, 0.01, T)
    # Noise floor
    beta_.data[0] = (0.15)**2
    log_beta = nn.Parameter(beta_.log().cuda())
    beta = log_beta.exp()
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    def preprocess(x, t, alpha_bar):
        x_scale = alpha_bar.sqrt()[t, None, None]
        squared_dist = (x_scale - x)**2 - x**2 # up to a constant
        eps_var = (1 - alpha_bar)[t, None, None]
        log_likelihood = -squared_dist/(2*eps_var)
        log_posterior = F.log_softmax(log_likelihood, dim=1)
        posterior = log_posterior.exp()
        return posterior * float(np.sqrt(256)), posterior, log_posterior

    # lib.utils.print_row('t', 'epsilon_scale', 'x_scale', 'entropy')
    # for t, x in enumerate((1-alpha_bar).sqrt().tolist()):
    #     if t % 10 == 0:
    #         X_rand = torch.randint(low=0, high=256, size=[1024,1024], device='cuda')
    #         X_rand = F.one_hot(X_rand, num_classes=256).permute(0,2,1)
    #         epsilon = torch.randn(X_rand.shape, device='cuda')
    #         X_rand = (alpha_bar[t].sqrt()*X_rand) + ((1-alpha_bar[t]).sqrt()*epsilon)
    #         _, posterior, log_posterior = preprocess(X_rand, torch.full([1024], t).cuda(), alpha_bar)
    #         entropy = (posterior*(-log_posterior)).sum(dim=1).mean()
    #         lib.utils.print_row(t, x, alpha_bar.sqrt()[t], entropy)
    # del X_rand, posterior, log_posterior, entropy, _

    class ConvResBlock(nn.Module):
        def __init__(self, dim, norm='group'):
            super().__init__()
            self.conv1 = nn.Conv1d(dim, dim, 5, padding='same')
            self.conv2 = nn.Conv1d(dim, dim, 5, padding='same')
            assert(norm in ['group', 'none'])
            if norm == 'group':
                self.norm1 = nn.GroupNorm(8, dim)
                self.norm2 = nn.GroupNorm(8, dim)
            elif norm == 'none':
                self.norm1 = (lambda x: x)
                self.norm2 = (lambda x: x)
        def forward(self, x):
            z = x
            z = self.conv1(F.relu(self.norm1(z)))
            z = self.conv2(F.relu(self.norm2(z)))
            x = x + z
            return x

    class Model(nn.Module):
        def __init__(self, alpha_bar):
            super().__init__()
            self.register_buffer('t_codes',
                lib.transformer.position_codes(dim, T))
            self.register_buffer('pos_codes',
                lib.transformer.position_codes(dim, seq_len)) 
            self.input = nn.Conv1d(256, dim, 1, 1, padding=0)
            self.conv_blocks = nn.Sequential(*[
                ConvResBlock(dim, norm=('none' if i==0 else 'group'))
                for i in range(conv_blocks)
            ])
            self.transformer_blocks = nn.Sequential(*[
                (lib.transformer.TransformerBlock(dim, n_heads))
                for _ in range(transformer_blocks)
            ])
            self.output_norm = nn.LayerNorm(dim)
            self.output = nn.Linear(dim, 256)
            self.log_scale = nn.Parameter(
                (1-alpha_bar).sqrt().log().clone().detach()
            )
            # self.register_buffer('alpha_bar', alpha_bar)

        def forward(self, x, t, alpha_bar):
            x_orig = x
            x, _, log_posterior = preprocess(x, t, alpha_bar)
            x = self.input(x)
            x = x + self.t_codes[t][:,:,None]
            x = self.conv_blocks(x)
            x = x.permute(0,2,1) # BCT -> BTC
            x = x + self.pos_codes[None,:,:]
            x = self.transformer_blocks(x)
            x = self.output_norm(x)
            x = self.output(x)
            logits = x.permute(0,2,1) # BTC -> BCT
            # logits = log_posterior.detach()
            x_pred = F.softmax(logits, dim=1).detach()
            scale_1 = alpha_bar.sqrt()[t][:,None,None]
            scale_2 = self.log_scale[t].exp()[:,None,None]
            epsilon_pred = (x_orig - (scale_1 * x_pred)) / scale_2
            return epsilon_pred, logits

    model = Model(alpha_bar).cuda()
    lib.utils.print_model(model)

    t_buckets = []

    def gaussian_kl(mu_p, sigma2_p, mu_q, sigma2_q):
        """KL(p||q)"""
        return (0.5*(sigma2_q.log() - sigma2_p.log())) + (sigma2_p+(mu_p - mu_q)**2)/(2*sigma2_q) - 0.5

    loss2_ema = torch.ones(T, device='cuda')

    def loss(X, X_discrete):
        nonlocal loss2_ema
        beta = log_beta.exp()
        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        beta_tilde = torch.cat([
            torch.ones([1], device='cuda'),
            beta[1:] * (1 - alpha_bar[:-1])/(1 - alpha_bar[1:])
        ], dim=0)

        # Low-discrepancy sampler for t
        # n_cycles = T // X.shape[0]
        # t = torch.arange(X.shape[0]).cuda() * n_cycles
        # t += torch.randint(low=0, high=n_cycles, size=[1])[0].cuda()
        loss_rms = loss2_ema.sqrt().detach()
        t = torch.multinomial(loss_rms, X.shape[0])
        importance_weights = loss_rms.mean(dim=0, keepdim=True) / loss_rms
        imp_weights_t = importance_weights[t]

        alpha_bar_t = alpha_bar[t][:,None,None]
        epsilon = torch.randn(X.shape, device='cuda')
        X_noised = (alpha_bar_t.sqrt()*X) + ((1-alpha_bar_t).sqrt()*epsilon)

        epsilon_pred, logits = model(
            X_noised,
            t,
            alpha_bar
        )
        mse = (epsilon - epsilon_pred).pow(2).sum(dim=1).mean()
        xent = F.cross_entropy(logits, X_discrete, reduction='none').mean(dim=1)
        xent_unweighted = (xent * imp_weights_t).mean()
        xent_by_t = []
        for t1, t2 in t_buckets:
            t_mask = (t >= t1).float() * (t < t2).float()
            xent_by_t.append((xent*t_mask).mean()/t_mask.mean())

        # Likelihood (per-token)
        l_0 = (xent * (t == 0).float() * imp_weights_t ).sum() * (T // X.shape[0])

        l_T = gaussian_kl(
            alpha_bar[-1].sqrt()*X,
            1-alpha_bar[-1], 
            torch.zeros([1,1,1], device='cuda'),
            torch.ones([1,1,1], device='cuda')
        ).sum(dim=1).mean()

        mu_tilde = ((beta[t]*alpha_bar[t-1].sqrt())/(1-alpha_bar[t]))[:,None,None] * X
        mu_tilde += (alpha[t,None,None].sqrt()*(1 - alpha_bar[t-1,None,None])/(1 - alpha_bar_t)) * X_noised

        l_t = gaussian_kl(
            mu_tilde,
            beta_tilde[t][:,None,None],
            (1. / alpha[t].sqrt())[:,None,None] * ( X_noised - (beta[t]/(1-alpha_bar[t]).sqrt())[:,None,None]*epsilon_pred ),
            beta[t,None,None]
        ).mean(dim=2).sum(dim=1) * (t > 0).float() * imp_weights_t * (T // X.shape[0])

        loss2_ema *= 0.99
        loss2_ema[0] += 0.01 * (l_0**2)
        for i, t in enumerate(t):
            loss2_ema[t] += 0.01 * (l_t[i]**2)
        loss2_ema = loss2_ema.detach()

        l_t = l_t.sum()

        nll = l_0 + l_T + l_t

        return xent_unweighted + mse, mse, xent_unweighted, nll, l_0, l_T, l_t, *xent_by_t

    def forward():
        X = next(train_iterator).cuda()
        X_onehot = F.one_hot(X.long(), num_classes=256).permute(0,2,1)
        return loss(X_onehot, X.long())

    def hook(step):
        pass
        # lib.utils.print_row(*list(range(T)))
        # lib.utils.print_row(*loss2_ema.sqrt().tolist())
        # torch.save(model.state_dict(), 'model.pt')

    opt = optim.Adam(list(model.parameters()), lr=lr)
    lib.utils.train_loop(forward, opt, steps, print_freq=print_freq,
        lr_warmup_steps=100, lr_cooldown_steps=steps//10,
        grad_accumulation_steps=grad_accumulation_steps,
        names=['mse', 'xent', 'nll', 'l0', 'lT', 'lt']+[f'xent_{t1}:{t2}' for t1,t2 in t_buckets],
        hook=hook, hook_freq=100
    )

    sample_batches = 64
    bs_multiple = 1

    all_X_samples = []
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for _ in tqdm.tqdm(range(sample_batches // bs_multiple)):
                X_samples = torch.randn(
                    [bs_multiple * batch_size, 256, seq_len]).cuda()
                # X_samples *= (1-alpha_bar).sqrt()[-1]
                for t in range(T)[::-1]:
                    X_samples = (
                        (1./alpha[t].sqrt()) *
                        (
                            X_samples -
                            (
                                (beta[t]/(1-alpha_bar[t]).sqrt()) * 
                                model(
                                    X_samples,
                                    torch.tensor(
                                        [t]*(bs_multiple*batch_size)
                                    ).long().cuda()
                                )[0]
                            )
                        )
                    )
                    if t > 1:
                        epsilon = torch.randn_like(X_samples).cuda()
                        X_samples += beta[t].sqrt() * epsilon
                all_X_samples.append(X_samples)

    with torch.no_grad():
        sample_freqs = torch.zeros([256])
        data_freqs = torch.zeros([256])
        for i, X_samples in enumerate(all_X_samples):
            X_samples = X_samples.argmax(dim=1)

            if i==0:
                print('Samples:')
                for x in X_samples[:10]:
                    x = x.detach().cpu().numpy().tobytes()
                    # replace non-ascii bytes with '#'
                    x = re.sub(rb'[^\x00-\x7F]', b'#', x)
                    x = x.decode('utf-8')
                    print(x)
                    print('---')

            for x in X_samples.view(-1):
                sample_freqs[x] += 1

        for _ in range(sample_batches):
            X = next(train_iterator)
            for x in X.view(-1).tolist():
                data_freqs[x] += 1
        
        sample_freqs /= sample_freqs.sum()
        data_freqs /= data_freqs.sum()
        l1 = (sample_freqs - data_freqs).abs().sum().item()
        print('Unigram L1:', l1)

if __name__ == '__main__':
    fire.Fire(main)