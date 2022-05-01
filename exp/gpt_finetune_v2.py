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
parser.add_argument('--lr', type=float, default=3e-6)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--n_samples', type=int, default=100_000)
parser.add_argument('--grad_accumulation_steps', type=int, default=8)
# gpt2, gpt2-medium, gpt2-large, gpt2-xl, gptj
parser.add_argument('--model', type=str, default='gptj')
parser.add_argument('--from_scratch', action='store_true')
parser.add_argument('--steps', type=int, default=1400)
parser.add_argument('--dataset', type=str, default='rocstories')
parser.add_argument('--save_pretrained', action='store_true')
parser.add_argument('--pretrained_path', type=str, default=None)
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--truncate_p', type=float, default=0.2)
args = parser.parse_args()

DATA_DIR = os.path.realpath(
    os.path.expanduser(f'~/local/{socket.gethostname()}/data'))

print('Loading model...')

if args.pretrained_path:
    if args.model == 'gptj':
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        model = GPTJForCausalLM.from_pretrained(args.pretrained_path,
            torch_dtype=torch.float16, low_cpu_mem_usage=True).cuda()
    else:
        tokenizer = GPT2TokenizerFast.from_pretrained(args.model)
        model = GPT2LMHeadModel.from_pretrained(
            args.pretrained_path).cuda().half()
else:
    if args.model == 'gptj':
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        model = GPTJForCausalLM.from_pretrained(
            "EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16,
            low_cpu_mem_usage=True, cache_dir=os.path.join(
                DATA_DIR, 'huggingface/transformers')
        )
        device_map = {
            0: list(range(12)),
            1: list(range(12,28))
        }
        model.parallelize(device_map)
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
    raise Exception('unsupported')
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
            torch.cat([
                torch.tensor([tokenizer.bos_token_id]),
                tokenizer(s, return_tensors='pt')['input_ids'][0],
                torch.tensor([tokenizer.eos_token_id])
            ])
            for s in data
        ]
        while True:
            np.random.shuffle(sents)
            for i in range(0, len(sents),
                args.batch_size * args.grad_accumulation_steps):

                j = i + (args.batch_size * args.grad_accumulation_steps)
                sents[i:j] = sorted(sents[i:j], key=len)

            for i in range(0, len(sents), args.batch_size):
                batch = sents[i:i+args.batch_size]
                if len(batch) < args.batch_size:
                    continue
                lens = torch.tensor([len(x) for x in batch])
                batch = torch.nn.utils.rnn.pad_sequence(
                    batch, batch_first=True, padding_value=0)
                yield batch, lens
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
    X, lens = next(train_iterator)
    X, lens = X.cuda(), lens.cuda()
    logits = model(X[:,:-1]).logits
    losses = cross_entropy(
        logits.view(-1, logits.shape[-1]),
        X[:,1:].reshape(-1),
        reduction='none'
    ).reshape(X.shape[0], X.shape[1]-1).float()
    mask = (torch.arange(1, X.shape[1]).cuda()[None,:] < lens[:,None]).float()
    loss = (losses * mask).sum(dim=1).mean(dim=0)
    token_count = mask.sum(dim=1).mean(dim=0)
    return loss * 16, loss, token_count

opt = Adafactor(model.parameters(), lr=args.lr, relative_step=False,
    warmup_init=False, scale_parameter=False)

def hook(_):
    model.eval()
    test_losses = []
    with torch.no_grad():
        for i in range(100 * args.grad_accumulation_steps):
            X, lens = next(test_iterator)
            X, lens = X.cuda(), lens.cuda()
            logits = model(X[:,:-1]).logits
            losses = cross_entropy(
                logits.view(-1, logits.shape[-1]),
                X[:,1:].reshape(-1),
                reduction='none'
            ).reshape(X.shape[0], X.shape[1]-1).float()
            mask = (torch.arange(1, X.shape[1]).cuda()[None,:] < lens[:,None]).float()
            loss = (losses * mask).sum(dim=1).mean(dim=0)
            test_losses.append(loss.item())
            del X, logits, loss
    print('Test loss:', np.mean(test_losses))
    model.train()

lib.utils.train_loop(forward, opt, args.steps,
    print_freq=10, hook=hook, hook_freq=100, lr_cooldown_steps=100,
    lr_warmup_steps=100, grad_accumulation_steps=args.grad_accumulation_steps,
    amp_grad_scaler=False, amp_autocast=False, names=['scaled_loss', 'token_count']
)

if args.save_pretrained:
    print('Saving pretrained model...')
    model.save_pretrained('model')

# Sampling
print('Sampling...')
SAMPLE_BS = 8
MAX_SEQ_LEN = 100
model.eval()
sents = []
with torch.no_grad():
    for i in tqdm.tqdm(range(args.n_samples), mininterval=10):
        samples = [[tokenizer.bos_token_id] for _ in range(SAMPLE_BS)]
        finished = [False for _ in range(SAMPLE_BS)]
        for j in range(MAX_SEQ_LEN):
            X = torch.tensor(samples, device='cuda')
            logits = model(X).logits[:, -1]
            logits.div_(args.temperature)
            probs = F.softmax(logits, dim=1)

            # Nucleus sampling
            probs_sort, probs_argsort = torch.sort(probs, dim=1)
            probs_mask = 1-(probs_sort.cumsum(dim=1) < args.truncate_p).half()
            probs_sort *= probs_mask
            probs_sort /= probs_sort.sum(dim=1, keepdim=True)
            probs_argsort_argsort = torch.zeros_like(probs_argsort)
            probs_argsort_argsort.scatter_(
                1, 
                probs_argsort,
                torch.arange(probs.shape[1], device='cuda')[None,:].expand(probs.shape[0], -1)
            )
            probs = probs_sort[
                torch.arange(probs.shape[0], device='cuda')[:,None],
                probs_argsort_argsort
            ]

            tokens = torch.multinomial(probs, 1)[:, 0]
            for k in range(SAMPLE_BS):
                samples[k].append(tokens[k].item())
                last_token_is_eos = samples[k][-1] == tokenizer.eos_token_id
                if last_token_is_eos and not finished[k]:
                    sent = tokenizer.decode(samples[k][1:-1])
                    sents.append(sent)
                    finished[k] = True
            if all(finished):
                break
        if len(sents) >= 100:
            with open(f'samples.txt', 'a') as f:
                for sent in sents:
                    f.write(sent + "\n")
            sents = []