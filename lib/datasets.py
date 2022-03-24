import numpy as np
import os
import socket
import torch
import torchvision.datasets
from torch import nn

DATA_DIR = os.path.realpath(
    os.path.expanduser(f'~/local/{socket.gethostname()}/data'))

def _parallel_shuffle(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

def mnist(split):
    """
    split: 'train', test'
    """
    assert(split in ['train', 'test'])
    mnist = torchvision.datasets.MNIST(DATA_DIR,
        train=(split=='train'), download=True)
    X, y = mnist.data.clone(), mnist.targets.clone()
    _parallel_shuffle(X.numpy(), y.numpy())
    X = (X.float() / 256.)
    return X.view(-1, 784).cuda(), y.cuda()

def cifar10(split):
    assert(split in ['train', 'test'])
    cifar = torchvision.datasets.CIFAR10(DATA_DIR,
        train=(split=='train'), download=True)
    X, y = torch.tensor(cifar.data).clone(), torch.tensor(cifar.targets).clone()
    _parallel_shuffle(X.numpy(), y.numpy())
    X = X / 256.
    X = X.permute(0,3,1,2) # BHWC -> BCHW
    return X, y

def make_batch_transform(*transforms):
    """
    Return a module which applies the given torchvision transforms to each
    item in a batch.
    TODO: half-precision, disable grads
    """
    class _BatchTransform(nn.Module):
        def __init__(self, transform):
            super().__init__()
            self.transform = transform
        def forward(self, x):
            result = [self.transform(x_) for x_ in x]
            return torch.stack(result, dim=0)
    transform = nn.Sequential(*transforms)
    return torch.jit.script(_BatchTransform(transform))