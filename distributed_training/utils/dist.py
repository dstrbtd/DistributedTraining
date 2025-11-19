import torch.distributed as dist
import torch


def gloabl_dist_checkpoint(check: bool, process_group):
    """
    Returns True iff *all ranks* reported ok=True.
    Performs a distributed MIN reduction.
    """
    t = torch.tensor([1 if check else 0], dtype=torch.int, device="cpu")
    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=process_group)
    return t.item() == 1
