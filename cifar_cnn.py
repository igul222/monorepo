"""
CIFAR-10 CNN classifier. Gets ~92% test accuracy in about 10 minutes.
"""

import fire
import lib.datasets
import lib.utils
import torch
import torch.nn.functional as F
import torchvision.transforms
from torch import nn, optim

def main(**args):
    args = lib.utils.AttributeDict(args)
    args.setdefault('lr', 1e-3)
    args.setdefault('batch_size', 128)
    args.setdefault('steps', 10_000)
    lib.utils.print_args(args)

    X_train, y_train = lib.datasets.cifar10('train')
    X_test, y_test = lib.datasets.cifar10('test')

    augment = lib.datasets.make_batch_transform(
        torchvision.transforms.RandomCrop(32, padding=[4,4]),
        torchvision.transforms.RandomHorizontalFlip()
    ).cuda()

    class ResBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
            self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
            self.norm1 = nn.BatchNorm2d(dim)
            self.norm2 = nn.BatchNorm2d(dim)
        def forward(self, x):
            x_shortcut = x
            x = self.conv1(F.gelu(self.norm1(x)))
            x = self.conv2(F.gelu(self.norm2(x)))
            return x_shortcut + x

    class WideResnet(nn.Module):
        """
        Wide ResNet. I handle the downsampling slightly differently for
        simplicity. The default settings (N=1, k=4) correspond roughly
        to WRN-16-8.
        """
        def __init__(self, N=2, k=8, dim_in=3, dim_out=None):
            super().__init__()
            self.input = nn.Conv2d(dim_in, 16*k, 1, padding='same')
            self.conv2 = nn.Sequential(*[ResBlock(16*k) for _ in range(N)])
            self.pre_conv3 = nn.Conv2d(16*k, 32*k, 1, stride=2, bias=False)
            self.conv3 = nn.Sequential(*[ResBlock(32*k) for _ in range(N)])
            self.pre_conv4 = nn.Conv2d(32*k, 64*k, 1, stride=2, bias=False)
            self.conv4 = nn.Sequential(*[ResBlock(64*k) for _ in range(N)])
            self.output_norm = nn.BatchNorm1d(64*k)
            if dim_out is not None:
                self.output = nn.Linear(64*k, dim_out)
            else:
                self.output = None
        def forward(self, x):
            x = self.input(x)
            x = self.conv2(x)
            x = self.pre_conv3(x)
            x = self.conv3(x)
            x = self.pre_conv4(x)
            x = self.conv4(x)
            x = x.mean(dim=[2,3])
            x = self.output_norm(x)
            if self.output is not None:
                x = self.output(x)
            return x

    model = WideResnet(dim_out=10).cuda()
    lib.utils.print_model(model)

    def forward():
        X, y = lib.utils.get_batch([X_train, y_train], args.batch_size)
        X, y = X.cuda(), y.cuda()
        X = (2*augment(X)) - 1
        logits = model(X)
        return F.cross_entropy(logits, y)

    def hook(_):
        model.eval()
        def acc_fn(X, y):
            X, y = X.cuda(), y.cuda()
            X = (2*X) - 1
            return model(X).argmax(dim=1).eq(y).float()
        train_acc = lib.utils.batch_apply(acc_fn, X_train, y_train).mean()
        test_acc = lib.utils.batch_apply(acc_fn, X_test, y_test).mean()
        print(f'Acc: {train_acc.item()} train, {test_acc.item()} test')
        model.train()

    opt = optim.Adam(model.parameters())
    lib.utils.train_loop(forward, opt, args.steps, print_freq=100,
        lr_cooldown_steps=args.steps//10, hook=hook, hook_freq=1000)

if __name__ == '__main__':
    fire.Fire(main)