"""
Run a grid search and print a results table. Can be used as a library or
a script.
"""

import fire
import functools
import importlib
import itertools
import lib.utils
import traceback

def search(fn, grid):
    grid_keys = list(grid.keys())
    grid_vals = list(grid.values())
    results = {}
    for vals in itertools.product(*grid_vals):
        try:
            result = fn(**dict(zip(grid_keys, vals)))
        except Exception:
            traceback.print_exc()
            result = 'err'
        results[vals] = result

    if len(grid_keys) == 1:
        key = grid_keys[0]
        for val in grid_vals[0]:
            print(f'{key}={val}:', results[(val,)])
    else:
        x_key, y_key = grid_keys[-2:]
        print (f'Rows: {y_key}, cols: {x_key}')
        for prefix_vals in itertools.product(*grid_vals[:-2]):
            if len(prefix_vals):
                prefix_items = zip(grid_keys[:-2], prefix_vals)
                print(', '.join([f'{k}={v}' for k,v in prefix_items])+':')
            lib.utils.print_row('', *grid[x_key])
            for y_val in grid[y_key]:
                results_row = [
                    results[prefix_vals + (x_val, y_val)]
                    for x_val in grid[x_key]
                ]
                lib.utils.print_row(y_val, *results_row)

def main(module_name, **kwargs):
    module = importlib.import_module(module_name)

    grid = {}
    static_kwargs = {}
    for k, v in kwargs.items():
        if isinstance(v, list):
            grid[k] = v
        else:
            static_kwargs[k] = v

    fn = functools.partial(module.main, **static_kwargs)
    search(fn, grid)


if __name__ == '__main__':
    fire.Fire(main)