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

def main(dim=256, z_dim=32, lr_enc=1e-3, lr_clf=1e-3, steps=1000, lambda_mi=0.,
    lambda_nll=1.):

    lib.utils.print_args(locals())

    envs = lib.colored_mnist.colored_mnist()
  
    # enc = nn.Sequential(*[
    #     nn.Linear(392, z_dim),
    #     # nn.ReLU(),
    #     # nn.Linear(dim, dim),
    #     # nn.ReLU(),
    #     # nn.Linear(dim, z_dim)
    # ]).cuda()
    
    enc = nn.Linear(392, z_dim).cuda()
    nn.init.orthogonal_(enc.weight)

    clf = nn.Sequential(*[
        nn.Linear(z_dim, dim),
        nn.ReLU(),
        nn.Linear(dim, dim),
        nn.ReLU(),
        nn.Linear(dim, 4) # ye=00,01,10,11
    ]).cuda()

    # Helpers

    def mean_accuracy(logits, y):
        return logits.argmax(dim=1).eq(y.long()).float().mean()

    def losses(logits, labels, env_idx):
        logprobs = F.log_softmax(logits, dim=1)
        y_logprobs = torch.logsumexp(logprobs.view(-1,2,2), dim=2)
        y_nll = -y_logprobs[
            torch.arange(y_logprobs.shape[0]),
            labels
        ].mean()
        acc = mean_accuracy(y_logprobs, labels)
        if env_idx < 2:
            e_logprobs = torch.logsumexp(logprobs.view(-1,2,2), dim=1)
            y_and_e_logprobs = logprobs.view(-1,2,2)[:,:, env_idx]
            y_and_e_nll = -y_and_e_logprobs[
                torch.arange(y_and_e_logprobs.shape[0]),
                labels
            ].mean()
            y_given_e_logprobs = y_and_e_logprobs - e_logprobs[:, None, env_idx]
            y_given_e_nll = -y_given_e_logprobs[
                torch.arange(y_given_e_logprobs.shape[0]),
                labels
            ].mean()
            mi = y_nll - y_given_e_nll
            return acc, y_nll, mi, y_and_e_nll
        else:
            return acc, None, None, None

    # Train loop

    opt = optim.Adam([
        {'params': enc.parameters(), 'lr': lr_enc, 'weight_decay': 1e-4},
        {'params': clf.parameters(), 'lr': lr_clf, 'weight_decay': 1e-3}
    ], betas=(0.5, 0.99))

    def forward():
        for env_idx, env in enumerate(envs):
            z = enc(env['images'])
            labels = env['labels'].long()[:,0]

            logits = clf(z)
            acc, y_nll, mi, _ = losses(logits, labels, env_idx)

            logits_detach = clf(z.detach())
            _, y_nll_detach, mi_detach, y_and_e_nll_detach = \
                losses(logits_detach, labels, env_idx)

            env['acc'] = acc

            if env_idx < 2:
                enc_loss = (lambda_mi*mi) + (lambda_nll * y_nll)
                enc_loss_detach = (lambda_mi*mi_detach) + (lambda_nll * y_nll_detach)
                disc_loss_detach = y_and_e_nll_detach
                loss = enc_loss - enc_loss_detach + disc_loss_detach
                env['mi'] = mi
                env['loss'] = loss

        train_loss = (envs[0]['loss'] + envs[1]['loss']) / 2.
        train_mi = (envs[0]['mi'] + envs[1]['mi']) / 2.
        train_acc = (envs[0]['acc'] + envs[1]['acc']) / 2.

        W = enc.weight
        orth_loss = ((W @ W.T) - torch.eye(z_dim).cuda()).pow(2).sum()
        train_loss = train_loss + (1.0 * orth_loss)

        test_acc = envs[2]['acc']
        return train_loss, orth_loss, train_mi, train_acc, test_acc

    lib.utils.train_loop(forward, opt, steps, 
        names=['orth loss', 'train mi', 'train acc', 'test acc'],
        print_freq=10)

if __name__ == '__main__':
    fire.Fire(main)