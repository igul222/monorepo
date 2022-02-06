"""
MNIST PixelCNN. Trains in about 5 minutes.
"""

import fire
import lib.datasets
import lib.pixelcnn
import lib.utils
import torch
import torch.nn.functional as F
from torch import nn, optim

class PixelCNN(nn.Module):
    def __init__(self, dim, n_blocks):
        super().__init__()
        self.embedding = nn.Embedding(256, 32)
        self.input = lib.pixelcnn.CausalConv(32, dim, 7, True)
        self.blocks = nn.Sequential(*[
            lib.pixelcnn.CausalResBlock(dim)
            for _ in range(n_blocks)
        ])
        self.output = nn.Conv2d(dim, 256, 1)
    def forward(self, x):
        x = self.embedding(x).permute(0,3,1,2)
        x = self.input(x)
        x = self.blocks(x)
        x = self.output(x)
        return x

def quantize(X):
    with torch.no_grad():
        return ((X*256)+0.5).long()

def main(
    batch_size=64,
    steps=8000,
    print_freq=100,
    dim=128,
    n_blocks=7,
    lr=1e-3):

    lib.utils.print_args(locals())

    X_train, y_train = lib.datasets.mnist('train')
    X_test, y_test = lib.datasets.mnist('test')

    X_train = quantize(X_train)
    X_test = quantize(X_test)

    model = PixelCNN(dim, n_blocks).cuda()
    lib.utils.print_model(model)

    def forward():
        X = lib.utils.get_batch(X_train, batch_size)
        X = X.view(batch_size, 28, 28)
        logits = model(X)
        return F.cross_entropy(logits, X)

    opt = optim.Adam(model.parameters(), lr=lr)
    lib.utils.train_loop(forward, opt, steps, print_freq=print_freq,
        lr_cooldown_steps=steps//10)

    def eval_loss(X):
        X = X.view(-1, 28, 28)
        return F.cross_entropy(model(X), X, reduction='none')
    print('Test loss:', lib.utils.batch_apply(eval_loss, X_test).mean().item())

    with torch.no_grad():
        X_samples = torch.zeros([64, 28, 28], dtype=torch.int64, device='cuda')
        for x in range(28):
            for y in range(28):
                logits = model(X_samples)
                probs = F.softmax(logits[:,:,x,y], dim=1)
                X_samples[:,x,y] = torch.multinomial(probs, 1)[:,0]
        X_samples = torch.stack([X_samples]*3, dim=3).byte()
        lib.utils.save_image_grid(X_samples, 'samples.png')

if __name__ == '__main__':
    fire.Fire(main)