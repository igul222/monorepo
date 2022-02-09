"""
Datasets and helpers for language modeling.
"""

import numpy as np
import os
import socket
import torch
import tqdm
import warnings
import zipfile

DATA_DIR = os.path.realpath(
    os.path.expanduser(f'~/local/{socket.gethostname()}/data'))

def enwik8():
    data = zipfile.ZipFile(
        os.path.join(DATA_DIR, 'enwik8.zip'), 'r').read('enwik8')
    data = torch.tensor(np.frombuffer(data, dtype='uint8'))
    return data[:90_000_000], data[90_000_000:95_000_000], data[95_000_000:]

def books1():
    print('Loading books1:')
    path = os.path.join(DATA_DIR, 'books1.txt')
    data = torch.zeros([os.path.getsize(path)], dtype=torch.uint8)
    chunk_len = 128*1024*1024 # 128MB
    with open(path, 'rb') as f:
        for i in tqdm.tqdm(range(0, len(data), chunk_len)):
            f.seek(i)
            chunk = np.frombuffer(f.read(chunk_len), dtype='uint8')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data[i:i+chunk_len] = torch.from_numpy(chunk)
    return data

def _get_slices(data, offsets, seq_len):
    # https://stackoverflow.com/q/46091111
    all_indx = offsets[:, None] + torch.arange(seq_len)
    return data[all_indx]

def sequential_iterator(data, batch_size, seq_len, overlap, infinite):
    offsets = torch.randint(low=0, high=data.shape[0]-seq_len+1,
        size=[batch_size], dtype=torch.int64)
    offsets[0] = 0
    while True:
        yield _get_slices(data, offsets, seq_len)
        offsets = (offsets + seq_len - overlap) % (data.shape[0] - seq_len + 1)
        if (offsets[0] < seq_len - overlap) and (not infinite):
            break

def random_iterator(data, batch_size, seq_len):
    while True:
        offsets = torch.randint(low=0, high=data.shape[0]-seq_len+1,
            size=[batch_size], dtype=torch.int64)
        yield _get_slices(data, offsets, seq_len)