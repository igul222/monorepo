"""
Datasets and helpers for language modeling.
"""

import collections
import numpy as np
import os
import random
import re
import socket
import torch
import torch.nn.functional as F
import tqdm
import warnings
import zipfile

# DATA_DIR = os.path.realpath(
#     os.path.expanduser(f'~/local/{socket.gethostname()}/data'))
DATA_DIR = os.path.expanduser('~/data')

def tokenize(bytes):
    """Fast-enough, good-enough, tokenizer."""
    return re.compile(b'[A-Za-z]+|\S').findall(bytes)

def _get_slices(data, offsets, seq_len):
    # https://stackoverflow.com/q/46091111
    all_indx = offsets[:, None] + torch.arange(seq_len)
    return data[all_indx]

def sequential_iterator(data, batch_size, seq_len, overlap, infinite):
    offsets = torch.arange(start=0, end=data.shape[0],
        step=data.shape[0]//batch_size, dtype=torch.int64)[:batch_size]
    while True:
        yield _get_slices(data, offsets, seq_len)
        offsets = (offsets + seq_len - overlap)
        if (offsets[-1] + seq_len > data.shape[0]) and (not infinite):
            break
        offsets = offsets % (data.shape[0] - seq_len + 1)

def random_iterator(data, batch_size, seq_len):
    while True:
        offsets = torch.randint(low=0, high=data.shape[0]-seq_len+1,
            size=[batch_size], dtype=torch.int64)
        yield _get_slices(data, offsets, seq_len)

def padded_random_iterator(data, batch_size, seq_len, pad_token):
    while True:
        x = random.choices(data, k=batch_size)
        # Manually pad the first element of x to the full seq_len
        x0 = F.pad(x[0], (0, max(0, seq_len - x[0].shape[0])), value=pad_token)
        x[0] = x0
        x = torch.nn.utils.rnn.pad_sequence(
            x, batch_first=True, padding_value=pad_token
        )
        yield x[:, :seq_len]

def enwik8():
    data = zipfile.ZipFile(
        os.path.join(DATA_DIR, 'enwik8.zip'), 'r').read('enwik8')
    data = torch.tensor(np.frombuffer(data, dtype='uint8'))
    return data[:90_000_000], data[90_000_000:95_000_000], data[95_000_000:]

def books1():
    print('Loading books1:')
    path = os.path.join(DATA_DIR, 'books1.txt')
    data = torch.zeros([os.path.getsize(path)], dtype=torch.uint8)
    read_chunk_len = 128*1024*1024 # 128MB
    with open(path, 'rb') as f:
        for i in tqdm.tqdm(range(0, len(data), read_chunk_len)):
            f.seek(i)
            chunk = np.frombuffer(f.read(read_chunk_len), dtype='uint8')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data[i:i+read_chunk_len] = torch.from_numpy(chunk)
    # Split into 100KB chunks and shuffle.
    shuffle_chunk_len = 100_000
    data = data[:shuffle_chunk_len*(data.shape[0]//shuffle_chunk_len)]
    np.random.RandomState(0).shuffle(data.view(shuffle_chunk_len, -1).numpy())
    idx2word = [bytes([i]) for i in range(256)]
    word2idx = {word:idx for idx, word in enumerate(idx2word)}
    return (
        (data[:-10_000_000], data[-10_000_000:-5_000_000], data[-5_000_000:]),
        (word2idx, idx2word)
    )

def books1_wordlevel(vocab_size, force_rebuild=False):
    path = os.path.join(DATA_DIR, 'books1.txt')
    cached_path = os.path.join(
        DATA_DIR, f'books1_wordlevel_cached_{vocab_size}.pt'
    )

    if force_rebuild or (not os.path.exists(cached_path)):
        # Build vocab on the first 512MB
        print('books1_wordlevel: Building vocab...')
        vocab = collections.Counter()
        with open(path, 'rb') as f:
            text = f.read(512*1024*1024)
            vocab.update(tokenize(text))
        idx2word = [w for w, _ in vocab.most_common(vocab_size-1)]
        idx2word.append(b'UNK')
        word2idx = {word:idx for idx, word in enumerate(idx2word)}
        del vocab
        print('books1_wordlevel: Least-frequent words:', idx2word[-5:])

        # Load / tokenize the entire file in chunks
        read_chunk_len = 128*1024*1024 # 128MB
        data = torch.zeros([os.path.getsize(path)//2], dtype=torch.int16)
        data_idx = 0
        unk = word2idx[b'UNK']
        with open(path, 'rb') as f:
            for i in tqdm.tqdm(range(0, os.path.getsize(path), read_chunk_len)):
                f.seek(i)
                words = f.read(read_chunk_len)
                words = [word2idx.get(w, unk) for w in tokenize(words)]
                words = torch.tensor(words, dtype=torch.int16)
                data[data_idx:data_idx+len(words)] = words
                data_idx += len(words)
        data = data[:data_idx].clone()
        # Split into 20K-token chunks and shuffle.
        shuffle_chunk_len = 20_000
        data = data[:shuffle_chunk_len*(data.shape[0]//shuffle_chunk_len)]
        np.random.RandomState(0).shuffle(
            data.view(shuffle_chunk_len, -1).numpy()
        )
        torch.save((data, word2idx, idx2word), cached_path)
    else:
        print('books1_wordlevel: Loading from cache...')
        data, word2idx, idx2word = torch.load(cached_path)
    return (
        (data[:-2_000_000], data[-2_000_000:-1_000_000], data[-1_000_000:]),
        (word2idx, idx2word)
    )