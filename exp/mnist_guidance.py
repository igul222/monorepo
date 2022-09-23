import fire
import math
import numpy as np
import lib.transformer
import lib.datasets
import lib.utils
import torch
import torch.nn.functional as F
from torch import nn, optim, autograd
import tqdm

def main(
    T=1024,
    batch_size=64,
    steps=8000,
    print_freq=100,
    lr=3e-4,
    dim=192,
    guidance_weight=2,
    n_guidance_samples=8):

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
            self.register_buffer(
                't_codes', lib.transformer.position_codes(dim, T))
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
            x = self.input(x) + self.t_codes[t][:,:,None,None]
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
        t = torch.randint(low=0, high=T, size=[X.shape[0]]).cuda()
        alpha_bar_t = alpha_bar[t][:,None]
        epsilon = torch.randn(X.shape, device='cuda')
        epsilon_pred = model(
            (alpha_bar_t.sqrt()*X) + ((1-alpha_bar_t).sqrt()*epsilon), t
        )
        return (epsilon - epsilon_pred).pow(2).sum(dim=1)

    def forward():
        X = lib.utils.get_batch(X_train, batch_size)
        return loss(X).mean()

    opt = optim.Adam(model.parameters(), lr=lr)
    lib.utils.train_loop(forward, opt, steps, print_freq=print_freq,
        lr_warmup_steps=100, lr_cooldown_steps=steps//10)

    # Step 2: Train an MNIST classifier
    classifier = nn.Sequential(
        nn.Linear(784, 1024), nn.ReLU(),
        nn.Linear(1024, 1024), nn.ReLU(),
        nn.Linear(1024, 10)
    ).cuda()
    def forward():
        X, y = lib.utils.get_batch([X_train, y_train], batch_size)
        logits = classifier(X)
        return F.cross_entropy(logits, y)
    opt = optim.Adam(classifier.parameters())
    lib.utils.train_loop(forward, opt, 10000, print_freq=1000)

    # Step 3: Sample
    def sample(t, x_t, guided, quiet=False):
        with torch.no_grad():
            x_t = x_t.clone()
            t_iterator = range(T)[::-1]
            if not quiet:
                t_iterator = tqdm.tqdm(t_iterator)
            for t in t_iterator:
                t_vector = torch.tensor([t]*x_t.shape[0]).long().cuda()
                with torch.cuda.amp.autocast():
                    epsilon_pred = model(x_t, t_vector)
                if guided:
                    x0_given_xt = sample(
                        t,
                        x_t.repeat_interleave(n_guidance_samples, dim=0),
                        False,
                        quiet=True
                    ).view(n_guidance_samples, -1, 784)
                    with torch.enable_grad():
                        x0_given_xt.requires_grad = True
                        logprobs = F.log_softmax(classifier(x0_given_xt), dim=-1)
                        logprobs = logprobs[:,:,3].logsumexp(dim=0) # Generate digit '4'
                        grads = autograd.grad(logprobs.sum(), [x0_given_xt])[0]
                        grad_xt = grads.sum(dim=0) * alpha_bar[t].sqrt()
                    epsilon_pred -= guidance_weight * beta[t].sqrt() * grad_xt
                x_t -= (
                    (beta[t]/(1-alpha_bar[t]).sqrt()) * epsilon_pred
                )
                x_t /= alpha[t].sqrt()
                if t > 1:
                    x_t += beta[t].sqrt() * torch.randn_like(x_t).cuda()
            return x_t

    print('Generating unconditional samples')
    samples = sample(T, torch.randn([8, 784]).cuda(), False)
    lib.utils.save_image_grid(samples, f'uncond_samples.png')

    print('Generating conditional samples')
    samples = sample(T, torch.randn([8, 784]).cuda(), True)
    lib.utils.save_image_grid(samples, f'cond_samples.png')

if __name__ == '__main__':
    fire.Fire(main)