import torch

def get_batch(x, batch_size):
    if isinstance(x, list):
        idx = torch.randint(low=0, high=len(x[0]), size=(batch_size,))
        return [v[idx] for v in x]
    else:
        idx = torch.randint(low=0, high=x.shape[0], size=(batch_size,))
        return x[idx]