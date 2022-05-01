import sys
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import random

def _worker_fn(rank, world_size, main_fn, args_dict):
    # Setup
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', rank=rank,
        world_size=world_size)

    if rank != 0:
        sys.stdout = open('/dev/null', 'w')

    # Main function
    main_fn(**args_dict, rank=rank)

    # Cleanup
    dist.destroy_process_group()


def wrap_main(main_fn):
    """
    Usage: instead of calling main() directly, call wrap_main(main)().
    main should take only kwargs.
    """

    def main(**args):
        os.environ['PYTHONUNBUFFERED'] = '1'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(random.randint(1024, 65536))
        world_size = torch.cuda.device_count()

        mp.spawn(
            _worker_fn,
            (world_size, main_fn, args),
            nprocs=world_size,
            join=True
        )

    return main
