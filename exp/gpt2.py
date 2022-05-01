"""
GPT-2 Finetuning.
"""

import argparse
import json
import numpy as np
import lib.lm_datasets
import lib.utils
import os
import socket
import torch
import torch.nn.functional as F
import tqdm
from torch import nn, optim
from transformers import GPT2TokenizerFast
from transformers import GPT2LMHeadModel
from transformers.optimization import Adafactor
from transformers import AutoConfig, AutoTokenizer, GPTJForCausalLM, AutoModel

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--seq_len', type=int, default=64)
parser.add_argument('--sample_len', type=int, default=10_000_000)
parser.add_argument('--grad_accumulation_steps', type=int, default=8)
# gpt2, gpt2-medium, gpt2-large, gpt2-xl, gptj
parser.add_argument('--model', type=str, default='gpt2-xl')
parser.add_argument('--from_scratch', action='store_true')
parser.add_argument('--steps', type=int, default=800)
parser.add_argument('--dataset', type=str, default='e2e')
args = parser.parse_args()

DATA_DIR = os.path.realpath(
    os.path.expanduser(f'~/local/{socket.gethostname()}/data'))

print('Loading model...')

if args.model == 'gptj':
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    model = GPTJForCausalLM.from_pretrained(
        "EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16,
        low_cpu_mem_usage=True, cache_dir=os.path.join(
            DATA_DIR, 'huggingface/transformers')
    ).cuda()
    # device_map = {
    #     0: list(range(12)),
    #     1: list(range(12,28))
    # }
    # model.parallelize(device_map)
else:
    tokenizer = GPT2TokenizerFast.from_pretrained(args.model)
    if args.from_scratch:
        config = AutoConfig.from_pretrained(args.model)
        model = GPT2LMHeadModel(config).cuda()
    else:
        model = GPT2LMHeadModel.from_pretrained(
            args.model,
            cache_dir=os.path.join(DATA_DIR, 'huggingface/transformers')
        ).cuda().half()
lib.utils.print_model(model)

if args.dataset == 'e2e':
    def load_splits():
        sents = []
        with open(f'/u/scr/xlisali/e2e_data/src1_train.txt', 'r') as f:
            for line in f:
                sents.append(line.split('||')[1])
        np.random.RandomState(0).shuffle(sents)
        test_sents = sents[:len(sents)//10]
        train_sents = sents[len(sents)//10:]
        return train_sents, test_sents
    train_data, test_data = load_splits()
    def _make_train_iterator():
        sents = [
            tokenizer('||'+s, return_tensors='pt')['input_ids'][0]
            for s in train_data
        ]
        while True:
            np.random.shuffle(sents)
            cat_sents = torch.cat(sents)
            for i in range(0, len(cat_sents), args.batch_size*(args.seq_len+1)):
                batch = cat_sents[i:i+(args.batch_size*(args.seq_len+1))]
                if len(batch) < args.batch_size*(args.seq_len+1):
                    continue
                yield batch.view(args.batch_size, args.seq_len+1)
            print('Epoch!')
    train_iterator = _make_train_iterator()
    test_data = tokenizer('||'.join(test_data), return_tensors='pt')['input_ids'][0]
    test_iterator = lib.lm_datasets.random_iterator(
        test_data, args.batch_size, args.seq_len+1)
elif args.dataset == 'rocstories':
    splits = [
        '/juice/scr/xlisali/diffusion_lm/ROCstory/roc_train.json',
        '/juice/scr/xlisali/diffusion_lm/ROCstory/roc_valid.json',
    ]
    split_data = []
    for split in splits:
        with open(split, 'r') as f:
            lines = [json.loads(line)[0].strip() for line in f]
            split_data.append(lines)
    train_data, test_data = split_data
    def _make_iterator(data, verbose):
        sents = [
            tokenizer('\n' + s, return_tensors='pt')['input_ids'][0]
            for s in data
        ]
        while True:
            np.random.shuffle(sents)
            cat_sents = torch.cat(sents)
            for i in range(0, len(cat_sents), args.batch_size*(args.seq_len+1)):
                batch = cat_sents[i:i+(args.batch_size*(args.seq_len+1))]
                if len(batch) < args.batch_size*(args.seq_len+1):
                    continue
                yield batch.view(args.batch_size, args.seq_len+1)
            if verbose:
                print('Epoch!')
    train_iterator = _make_iterator(train_data, True)
    test_iterator = _make_iterator(test_data, False)

# Finetuning

def cross_entropy(logits, x, reduction):
    """Memory-efficient drop-in replacement for F.cross_entropy."""
    assert(reduction=='none')
    logits_logsumexp = torch.logsumexp(logits, dim=1)
    return logits_logsumexp - logits[
        torch.arange(x.shape[0], device='cuda'),
        x
    ]

def forward():
    X = next(train_iterator).cuda()
    logits = model(X[:,:-1]).logits
    loss = cross_entropy(
        logits.view(-1, logits.shape[-1]),
        X[:,1:].reshape(-1),
        reduction='none'
    ).float().mean()
    return loss * 512, loss

opt = Adafactor(model.parameters(), lr=args.lr, relative_step=False,
    warmup_init=False, scale_parameter=False)

def hook(_):
    model.eval()
    test_losses = []
    with torch.no_grad():
        for i in range(100 * args.grad_accumulation_steps):
            X = next(test_iterator).cuda()
            logits = model(X[:,:-1]).logits
            logits = logits.float()
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                X[:,1:].reshape(-1)
            )
            test_losses.append(loss.item())
            del X, logits, loss
    print('Test loss:', np.mean(test_losses))
    model.train()

lib.utils.train_loop(forward, opt, args.steps,
    print_freq=10, hook=hook, hook_freq=100, lr_cooldown_steps=100,
    lr_warmup_steps=100, grad_accumulation_steps=args.grad_accumulation_steps,
    amp_grad_scaler=False, amp_autocast=False, names=['scaled_loss']
)

# Sampling

sample_bs = 16
model.eval()
if args.dataset == 'e2e':
    prefix = '||'
else:
    prefix = '\n'
samples = [
    torch.full([sample_bs], inp_id, dtype=torch.int64)
    for inp_id in tokenizer(prefix)['input_ids']
]
with torch.no_grad():
    for i in tqdm.tqdm(range(args.sample_len), mininterval=10):
        context = torch.stack(samples[-args.seq_len:], dim=1).cuda()
        logits = model(context).logits[:, -1]
        probs = F.softmax(logits, dim=1)
        tokens = torch.multinomial(probs, 1)[:, 0]
        samples.append(tokens.cpu())
        if i % 1000 == 999:
            for j in range(sample_bs):
                text = tokenizer.decode([x[j] for x in samples[:-args.seq_len]])
                with open(f'samples_{j}.txt', 'a') as f:
                    f.write(text)
            samples = samples[-args.seq_len:]