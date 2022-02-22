"""
IRM on Colored MNIST.
"""

import fire
import lib.colored_mnist
import lib.utils
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim, autograd

def main(dim=256, lr=1e-3, penalty_anneal_iters=100,
    penalty_weight=1e5, steps=501):

    lib.utils.print_args(locals())

    envs = lib.colored_mnist.colored_mnist()
  
    model = nn.Sequential(*[
        nn.Linear(392, dim),
        nn.ReLU(),
        nn.Linear(dim, dim),
        nn.ReLU(),
        nn.Linear(dim, 1)
    ]).cuda()
    
    # Define loss function helpers
  
    def mean_nll(logits, y):
      return F.binary_cross_entropy_with_logits(logits, y)
  
    def mean_accuracy(logits, y):
      preds = (logits > 0.).float()
      return ((preds - y).abs() < 1e-2).float().mean()
  
    def penalty(logits, y):
      scale = torch.tensor(1.).cuda().requires_grad_()
      loss = mean_nll(logits * scale, y)
      grad = autograd.grad(loss, [scale], create_graph=True)[0]
      return torch.sum(grad**2)
  
    # Train loop
  
    opt = optim.Adam(model.parameters(), lr=lr)
    step = 0
    def forward():
        nonlocal step

        for env in envs:
            logits = model(env['images'])
            env['nll'] = mean_nll(logits, env['labels'])
            env['acc'] = mean_accuracy(logits, env['labels'])
            env['penalty'] = penalty(logits, env['labels'])
        train_nll = (envs[0]['nll'] + envs[1]['nll']) / 2.
        train_acc = (envs[0]['acc'] + envs[1]['acc']) / 2.
        train_penalty = (envs[0]['penalty'] + envs[1]['penalty']) / 2.

        if step >= penalty_anneal_iters:
            loss = train_nll + (penalty_weight * train_penalty)
            if penalty_weight > 1.0:
                loss /= penalty_weight
        else:
            loss = train_nll + train_penalty

        test_acc = envs[2]['acc']
        step += 1
        return loss, train_nll, train_acc, train_penalty, test_acc

    lib.utils.train_loop(forward, opt, steps, 
        names=['train nll', 'train acc', 'train penalty', 'test acc'],
        print_freq=100)

if __name__ == '__main__':
    fire.Fire(main)