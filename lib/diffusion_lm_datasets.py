"""
Datasets and helpers for language modeling.
"""

import collections
import csv
import json
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
from spacy.lang.en import English

# DATA_DIR = os.path.realpath(
#     os.path.expanduser(f'~/local/{socket.gethostname()}/data'))
DATA_DIR = os.path.expanduser('~/data')

def _tokenize(bytes):
    """Fast-enough, good-enough, tokenizer."""
    return re.compile(b'[A-Za-z]+|\S').findall(bytes)

def _get_slices(data, offsets, seq_len):
    # https://stackoverflow.com/q/46091111
    all_indx = offsets[:, None] + torch.arange(seq_len)
    return data[all_indx]

def _random_iterator(data, batch_size, seq_len):
    lens = torch.tensor([seq_len for _ in range(batch_size)])
    while True:
        offsets = torch.randint(low=0, high=data.shape[0]-seq_len+1,
            size=[batch_size], dtype=torch.int64)
        yield _get_slices(data, offsets, seq_len), lens

def _padded_random_iterator(data, batch_size, pad_token):
    while True:
        np.random.shuffle(data)
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            if len(batch) < batch_size:
                continue
            lens = torch.tensor([len(x) for x in batch])
            batch = torch.nn.utils.rnn.pad_sequence(
                batch, batch_first=True, padding_value=pad_token)
            yield batch, lens

def books1_char(batch_size):
    seq_len = 128

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

    train_data = data[:-10_000_000]
    test_data = data[-5_000_000:]
    train_iterator = _random_iterator(train_data, batch_size, seq_len)
    test_iterator = _random_iterator(test_data, batch_size, seq_len)

    return (train_iterator, test_iterator), (word2idx, idx2word)

def books1_word(batch_size, force_rebuild=False):
    seq_len = 64
    vocab_size = 8192

    path = os.path.join(DATA_DIR, 'books1.txt')
    cache_path = os.path.join(
        DATA_DIR, f'books1_wordlevel_cached_{vocab_size}.pt'
    )

    if force_rebuild or (not os.path.exists(cache_path)):
        # Build vocab on the first 512MB
        print('books1_wordlevel: Building vocab...')
        vocab = collections.Counter()
        with open(path, 'rb') as f:
            text = f.read(512*1024*1024)
            vocab.update(_tokenize(text))
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
                words = [word2idx.get(w, unk) for w in _tokenize(words)]
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
        torch.save((data, word2idx, idx2word), cache_path)
    else:
        print('books1_wordlevel: Loading from cache...')
        data, word2idx, idx2word = torch.load(cache_path)


    train_data = data[:-2_000_000]
    test_data = data[-1_000_000:]
    train_iterator = _random_iterator(train_data, batch_size, seq_len)
    test_iterator = _random_iterator(test_data, batch_size, seq_len)

    return (train_iterator, test_iterator), (word2idx, idx2word)

def _e2e(batch_size, use_gpt_training_data):
    seq_len = 64
    vocab_size = 821 # Matches Lisa's setup

    nlp = English()
    tokenizer = nlp.tokenizer

    if use_gpt_training_data:
        splits = [
            '/sailhome/igul/jobs/2022_04_17_211456_gpt2/all_samples.txt',
            '/u/scr/xlisali/e2e_data/src1_test.txt'
        ]
    else:
        splits = [
            '/u/scr/xlisali/e2e_data/src1_train.txt',
            '/u/scr/xlisali/e2e_data/src1_test.txt'
        ]

    vocab = collections.Counter()
    with open(splits[0], 'r') as f:
        for line in f:
            if len(line.split('||')) < 2:
                continue
            text = line.split('||')[1]
            text = [x.text.encode('utf-8', 'ignore') for x in tokenizer(text)]
            vocab.update(text)
    idx2word = [w for w, _ in vocab.most_common(vocab_size-4)]
    idx2word.extend([b'START', b'END', b'UNK', b'PAD'])
    word2idx = {word:idx for idx, word in enumerate(idx2word)}

    splits_data = []
    for split in splits:
        split_data = []
        with open(split, 'r') as f:
            for line in f:
                if len(line.split('||')) < 2:
                    continue # Malformed line
                text = line.split('||')[1]
                text = [
                    x.text.encode('utf-8', 'ignore')
                    for x in tokenizer(text)
                ]
                text = [
                    word2idx[b'START'],
                    *[word2idx.get(word, word2idx[b'UNK']) for word in text],
                    word2idx[b'END']
                ]
                split_data.append(torch.tensor(text))
        splits_data.append(split_data)
    
    train_data = torch.cat(splits_data[0])
    test_data = torch.cat(splits_data[1])
    train_iterator = _random_iterator(train_data, batch_size, seq_len)
    test_iterator = _random_iterator(test_data, batch_size, seq_len)

    return (train_iterator, test_iterator), (word2idx, idx2word)

def e2e(batch_size):
    return _e2e(batch_size, False)

def e2e_gpt(batch_size):
    return _e2e(batch_size, True)

def rocstories(batch_size):
    vocab_size = 8192

    splits = [
        '/juice/scr/xlisali/diffusion_lm/ROCstory/roc_train.json',
        '/juice/scr/xlisali/diffusion_lm/ROCstory/roc_valid.json',
    ]

    vocab = collections.Counter()
    with open(splits[0], 'rb') as f:
        for line in f:
            # lines are formatted as: ["Fred was playing basketball."]\n
            text = _tokenize(line[2:-3])
            vocab.update(text)
    idx2word = [w for w, _ in vocab.most_common(vocab_size-4)]
    idx2word.extend([b'START', b'END', b'UNK', b'PAD'])
    word2idx = {word:idx for idx, word in enumerate(idx2word)}

    splits_data = []
    for split in splits:
        split_data = []
        with open(split, 'rb') as f:
            for line in f:
                text = _tokenize(line[2:-3])
                text = [
                    word2idx[b'START'],
                    *[word2idx.get(word, word2idx[b'UNK']) for word in text],
                    word2idx[b'END']
                ]
                split_data.append(torch.tensor(text))
        splits_data.append(split_data)

    train_data, test_data = splits_data
    pad_idx = word2idx[b'PAD']
    train_iterator = _padded_random_iterator(train_data, batch_size, pad_idx)
    test_iterator = _padded_random_iterator(test_data, batch_size, pad_idx)

    return (train_iterator, test_iterator), (word2idx, idx2word)

def rocstories_gpt(batch_size):
    # Inherit the vocab and test split from rocstories()
    (_, test_iterator), (word2idx, idx2word) = rocstories(batch_size)

    train_data = []
    with open('/sailhome/igul/data/rocstories_gptj.txt', 'rb') as f:
        for line in tqdm.tqdm(f, mininterval=10):
            text = _tokenize(line[:-1])
            text = [
                word2idx[b'START'],
                *[word2idx.get(word, word2idx[b'UNK']) for word in text],
                word2idx[b'END']
            ]
            train_data.append(torch.tensor(text))

    pad_idx = word2idx[b'PAD']
    train_iterator = _padded_random_iterator(train_data, batch_size, pad_idx)
    return (train_iterator, test_iterator), (word2idx, idx2word)


REGISTRY = {
    'books1_char': books1_char,
    'books1_word': books1_word,
    'e2e': e2e,
    'e2e_gpt': e2e_gpt,
    'rocstories': rocstories,
    'rocstories_gpt': rocstories_gpt
}