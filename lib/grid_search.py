"""
Run a grid search. Features:
- multi-GPU via fire replicas
- preemption
- various kinds of summary reports
"""

import fire
import functools
import importlib
import itertools
import lib.utils
import numpy as np
import os
import pickle
import time
import torch
import traceback

RESULTS_FILENAME = 'grid_search_results.pkl'

def read_results():
    if os.path.exists(RESULTS_FILENAME):
        with open(RESULTS_FILENAME, 'rb') as f:
            results = pickle.load(f)
        return results
    else:
        return {}

def write_result(key, val):
    results = read_results()
    results[key] = val
    with open(RESULTS_FILENAME, 'wb') as f:
            pickle.dump(results, f)

def format_row(*row, colwidth=16):
    """Return a row of values as a string."""
    def format_val(x):
        if isinstance(x, torch.Tensor):
            x = x.item()
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str( x).ljust(colwidth)[:colwidth]
    return "  ".join([format_val(x) for x in row]) + "\n"

def write_sorted_trials_report(results, grid_keys):
    sorted_trials = sorted(results.items(), key=lambda x: x[1])
    with open('sorted_trials.txt', 'w') as f:
        f.write(format_row('result', *grid_keys))
        for vals, result in sorted_trials:
            f.write(format_row(result, *vals))

def write_single_key_report(results, grid_keys, grid_vals, key_idx):
    results_by_val = {}
    for vals, result in results.items():
        val = vals[key_idx]
        if val not in results_by_val:
            results_by_val[val] = []
        results_by_val[val].append(result)

    with open(f'{grid_keys[key_idx]}.txt', 'w') as f:
        for reduce_name, reduce_fn in [('min', np.min), ('max', np.max)]:
            f.write(f'{reduce_name} over everything else:\n')
            f.write(format_row(grid_keys[key_idx], 'result'))
            for val in grid_vals[key_idx]:
                f.write(format_row(val, reduce_fn(results_by_val.get(val))))
            f.write('\n')

def write_two_key_report(results, grid_keys, grid_vals, key1_idx, key2_idx):
    results_by_val_pair = {}
    for vals, result in results.items():
        val_pair = (vals[key1_idx], vals[key2_idx])
        if val_pair not in results_by_val_pair:
            results_by_val_pair[val_pair] = []
        results_by_val_pair[val_pair].append(result)

    with open(f'{grid_keys[key1_idx]}_{grid_keys[key2_idx]}.txt', 'w') as f:
        for reduce_name, reduce_fn in [('min', np.min), ('max', np.max)]:
            f.write(f'{reduce_name} over everything else:\n')
            f.write(
                f'rows: {grid_keys[key1_idx]}, cols: {grid_keys[key2_idx]}\n'
            )
            f.write(format_row('', *grid_vals[key2_idx]))
            for val1 in grid_vals[key1_idx]:
                row = [
                    reduce_fn(results_by_val_pair.get((val1, val2)))
                    for val2 in grid_vals[key2_idx]
                ]
                f.write(format_row(val1, *row))
            f.write('\n')

def search(fn, grid, replica_idx, n_replicas):
    grid_keys = list(grid.keys())
    grid_vals = list(grid.values())
    
    grid_combinations = list(itertools.product(*grid_vals))
    np.random.RandomState(0).shuffle(grid_combinations)
    grid_combinations = grid_combinations[replica_idx::n_replicas]

    for vals in grid_combinations:
        if vals not in read_results():
            try:
                result = fn(**dict(zip(grid_keys, vals)))
                write_result(vals, result)
            except Exception:
                traceback.print_exc()

        results = read_results()
        write_sorted_trials_report(results, grid_keys)

        for key_idx in range(len(grid_keys)):
            write_single_key_report(results, grid_keys, grid_vals, key_idx)

        for i in range(len(grid_keys)):
            for j in range(len(grid_keys)):
                    if i != j:
                        write_two_key_report(results, grid_keys, grid_vals, i,j)

def main(module_name, **kwargs):
    replica_idx = int(os.getenv('FIRE_REPLICA_IDX', default='0'))
    n_replicas = int(os.getenv('FIRE_N_REPLICAS', default='1'))
    time.sleep(10 * replica_idx) # Avoid race conditions
    print(f'grid_search: replica {replica_idx} of {n_replicas} starting!')

    module = importlib.import_module(module_name)

    grid = {}
    static_kwargs = {}
    for k, v in kwargs.items():
        if isinstance(v, list):
            grid[k] = v
        else:
            static_kwargs[k] = v

    fn = functools.partial(module.main, **static_kwargs)
    search(fn, grid, replica_idx, n_replicas)

if __name__ == '__main__':
    fire.Fire(main)