"""
CIFAR-10 vision transformer. 86% test acc in about 10 mins on a Titan V.
"""

import fire
import lib.datasets
import lib.transformer
import lib.utils
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms
from torch import nn, optim

def main(**args):
    args = lib.utils.AttributeDict(args)
    args.setdefault('batch_size', 256)
    args.setdefault('grad_accumulation_steps', 1)
    args.setdefault('lr', 1e-3)
    args.setdefault('dim', 256)
    args.setdefault('n_blocks', 4)
    args.setdefault('n_heads', 8)
    args.setdefault('steps', 10_000)
    lib.utils.print_args(args)

    X_train, y_train = lib.datasets.cifar10('train')
    X_test, y_test = lib.datasets.cifar10('test')

    augment = lib.datasets.make_batch_transform(
        torchvision.transforms.RandomCrop(32, padding=[4,4]),
        torchvision.transforms.RandomHorizontalFlip()
    ).cuda()

    class VisionTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            # self.input = nn.Sequential(
            #     nn.Conv2d(3, args.dim//2, 5, stride=2, padding=2),
            #     nn.ReLU(),
            #     nn.Conv2d(args.dim//2, args.dim, 5, stride=2, padding=2)
            # )
            # Convs work better, but uncomment this if you insist on patches.
            self.input = nn.Conv2d(3, args.dim, 1, stride=1, padding=0, bias=False)

            self.n_patches = (32 // 1)**2

            self.register_buffer('pos_codes',
                lib.transformer.position_codes(args.dim, self.n_patches)
            )

            self.blocks = nn.Sequential(*[
                lib.transformer.TransformerBlock(args.dim, args.n_heads)
                for _ in range(args.n_blocks)
            ])

        def forward(self, x):
            x = self.input(8*(x-0.5))
            x = x.view(x.shape[0], args.dim, self.n_patches).permute(0,2,1)
            x = x + self.pos_codes[None,:,:]
            x = self.blocks(x)
            x = x[:,0,:10]
            return x

    model = VisionTransformer().cuda()
    lib.utils.print_model(model)

    def forward():
        X, y = lib.utils.get_batch([X_train, y_train], 
            args.batch_size // args.grad_accumulation_steps)
        X, y = X.cuda(), y.cuda()
        X = augment(X)
        logits = model(X)
        return F.cross_entropy(logits, y)

    def hook(_):
        model.eval()
        def acc_fn(X, y):
            X, y = X.cuda(), y.cuda()
            return model(X).argmax(dim=1).eq(y).float()
        train_acc = lib.utils.batch_apply(acc_fn, X_train,y_train, batch_size=args.batch_size // args.grad_accumulation_steps).mean().item()
        test_acc = lib.utils.batch_apply(acc_fn, X_test, y_test, batch_size=args.batch_size // args.grad_accumulation_steps).mean().item()
        print(f'Acc: {train_acc} train, {test_acc} test')
        model.train()
        return test_acc

    opt = optim.Adam(model.parameters())
    lib.utils.train_loop(forward, opt, args.steps, print_freq=100,
        lr_cooldown_steps=args.steps//10,
        grad_accumulation_steps=args.grad_accumulation_steps
    )

    return hook(None)

if __name__ == '__main__':
    fire.Fire(main)