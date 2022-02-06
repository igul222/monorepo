"""
Masked conv modules for PixelCNNs.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class CausalConv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, mask_present):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn([dim_out, dim_in, kernel_size//2 + 1, kernel_size]))
        self.bias = nn.Parameter(torch.zeros([dim_out]))
        assert(kernel_size % 2 == 1)
        self.padding = (kernel_size//2, kernel_size//2)
        self.weight.data /= float(np.sqrt(kernel_size**2 * dim_in))
        mask = torch.ones(
            (dim_out, dim_in, kernel_size//2 + 1, kernel_size))
        if mask_present:
            mask[:,:,-1,kernel_size//2:] *= 0.
        else:
            mask[:,:,-1,(kernel_size//2)+1:] *= 0.
        self.register_buffer('mask', mask, persistent=False)
        self.kernel_size = kernel_size

    def forward(self, x):
        weight = self.weight * self.mask
        x = F.conv2d(x.contiguous(), weight, self.bias, padding=self.padding)
        x = x[:,:,:-(self.kernel_size//2),:]
        return x

class CausalResBlock(nn.Module):
    def __init__(self, dim, kernel_size=5):
        super().__init__()
        self.conv1 = CausalConv(dim, dim, kernel_size, False)
        self.conv2 = CausalConv(dim, dim, kernel_size, False)
        self.conv1 = torch.nn.utils.weight_norm(self.conv1)
        self.conv2 = torch.nn.utils.weight_norm(self.conv2)

    def forward(self, x):
        x_shortcut = x
        x = self.conv1(F.relu(x))
        x = self.conv2(F.relu(x))
        return x_shortcut + x
