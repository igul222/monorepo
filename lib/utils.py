import argparse
import collections
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import torch
import types
import warnings
from torch import optim

class AttributeDict(dict):
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, value):
        self[attr] = value

def print_args(args, title=None):
    if isinstance(args, argparse.Namespace):
        args = vars(args)
    if title:
        print(f'{title} args:')
    else:
        print('Args:')
    for k, v in sorted(args.items()):
        print(f'\t{k}: {v}')

def print_model(model):
    print('Parameters:')
    total_params = 0
    for name, param in model.named_parameters():
        print(f"\t{name}: {list(param.shape)}")
        total_params += param.view(-1).shape[0]
    print(f'Total parameters: {total_params:,}')

def print_tensor(label, tensor):
    """Print a tensor with a given label."""
    torch.set_printoptions(precision=3, linewidth=119, sci_mode=False)
    print(f'{label}:')
    for line in str(tensor).splitlines():
        print(f"\t{line}")
    torch.set_printoptions(profile='default')

def print_row(*row, colwidth=16):
    """Print a row of values."""
    def format_val(x):
        if isinstance(x, torch.Tensor):
            x = x.item()
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str( x).ljust(colwidth)[:colwidth]
    print("  ".join([format_val(x) for x in row]))

def train_loop(
    forward,
    opt,
    steps,
    names=[],
    hook=None,
    hook_freq=1000,
    print_freq=1000,
    quiet=False,
    resume_step=None,
    lr_cooldown_steps=0,
    lr_warmup_steps=0,
    time_limit=None,
    fp16=True):

    def lr_fn(step):
        if step < lr_warmup_steps:
            return float(step+1) / lr_warmup_steps
        elif step >= (steps - lr_cooldown_steps):
            return 0.1
        else:
            return 1.0
    scheduler = optim.lr_scheduler.LambdaLR(opt, lr_fn)

    if not quiet:
        print_row('step', 'step time', 'loss', *names)
    histories = collections.defaultdict(lambda: [])
    scaler = torch.cuda.amp.GradScaler()
    start_time = time.time()
    for step in range(steps):

        if (resume_step is not None) and (step < resume_step):
            scheduler.step()
            continue

        with torch.cuda.amp.autocast(enabled=fp16):
            forward_vals = forward()
            if not isinstance(forward_vals, tuple):
                forward_vals = (forward_vals,)

        scaler.scale(forward_vals[0]).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scheduler.step()

        histories['loss'].append(forward_vals[0].item())
        for name, val in zip(names, forward_vals[1:]):
            histories[name].append(val.item())

        if (step==0) or (step % print_freq == (print_freq - 1)):
            if not quiet:
                print_row(
                    step,
                    (time.time() - start_time) / (step+1),
                    np.mean(histories['loss']),
                    *[np.mean(histories[name]) for name in names]
                )
            histories.clear()

        if step % hook_freq == (hook_freq - 1) and hook is not None:
            hook(step)

        del forward_vals

        if (time_limit is not None) and ((time.time()-start_time) > time_limit):
            print('Time limit exceeded!')
            break

def save_image_grid(images, path):
    """
    Given a tensor representing a batch of images, arrange them into a
    rectangular grid and save the images to the given path. The specific
    preprocessing is inferred based on the image shape, dtype, and values.
    Supported image formats:
    MNIST: float, shape (N, 784), values in [0, 1]
    Colored MNIST: float, shape (N, 2*784), channel-minor, values in [0, 1]
    All others: byte, shape (N, H, W, C), values in [0, 255]
    """
    assert(torch.is_tensor(images))

    if (images.shape[-1] == 784):
        # MNIST
        images = images.reshape((-1, 28, 28, 1))
        images = images.expand(-1, -1, -1, 3)
        images = images.clamp(min=0.001, max=0.999)
        images = (images * 256).byte()
    elif (images.shape[-1] == 2*784):
        # Colored MNIST
        images = images.reshape((-1, 28, 28, 2))
        images = torch.cat([images, torch.zeros_like(images[:,:,:,:1])], dim=3)
        images = images.clamp(min=0.001, max=0.999)
        images = (images * 256).byte()

    assert(images.ndim == 4) # BHWC
    assert(images.dtype == torch.uint8)
    images = images.detach().cpu().numpy()
    n_images = images.shape[0]
    n_rows = int(np.sqrt(n_images))
    while n_images % n_rows != 0:
        n_rows -= 1
    n_cols = n_images//n_rows
    # Copy each image into its spot in the grid
    height, width = images[0].shape[:2]
    grid_image = np.zeros((height*n_rows, width*n_cols, 3), dtype='uint8')
    for n, image in enumerate(images):
        j = n // n_cols
        i = n % n_cols
        grid_image[j*height:j*height+height, i*width:i*width+width] = image
    plt.imsave(path, grid_image)

def infinite_iterator(iterator):
    while True:
        yield from iterator

def get_batch(x, batch_size):
    if isinstance(x, list):
        idx = torch.randint(low=0, high=len(x[0]), size=(batch_size,))
        return [v[idx] for v in x]
    else:
        idx = torch.randint(low=0, high=x.shape[0], size=(batch_size,))
        return x[idx]

def batch_apply(fn, *inputs, batch_size=128):
    for X in inputs:
        assert(len(X) == len(inputs[0]))
    with torch.no_grad():
        results = []
        for i in range(0, len(inputs[0]), batch_size):
            results.append(fn(*[X[i:i+batch_size] for X in inputs]))
        return torch.cat(results, dim=0)