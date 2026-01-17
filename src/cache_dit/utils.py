import gc
import time
import torch
import diffusers
import builtins as __builtin__
import contextlib

from cache_dit.logger import init_logger
from .platforms import current_platform

logger = init_logger(__name__)


def dummy_print(*args, **kwargs):
    pass


@contextlib.contextmanager
def disable_print():
    origin_print = __builtin__.print
    __builtin__.print = dummy_print
    yield
    __builtin__.print = origin_print


@torch.compiler.disable
def is_diffusers_at_least_0_3_5() -> bool:
    return diffusers.__version__ >= "0.35.0"


@torch.compiler.disable
def maybe_empty_cache():
    try:
        time.sleep(1)
        gc.collect()
        current_platform.empty_cache()
        current_platform.ipc_collect()
        time.sleep(1)
        gc.collect()
        current_platform.empty_cache()
        current_platform.ipc_collect()
    except Exception:
        pass


def print_tensor(
    x: torch.Tensor,
    name: str,
    dim: int = 1,
    no_dist_shape: bool = True,
    disable: bool = True,
):
    if disable:
        return

    if x is None:
        print(f"{name} is None")
        return

    if not isinstance(x, torch.Tensor):
        print(f"{name} is not a tensor, type: {type(x)}")
        return

    x = x.contiguous()
    if torch.distributed.is_initialized():
        # all gather hidden_states and check values mean
        gather_x = [torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(gather_x, x)
        gather_x = torch.cat(gather_x, dim=dim)

        if not no_dist_shape:
            x_shape = gather_x.shape
        else:
            x_shape = x.shape

        rank = torch.distributed.get_rank()
        print(
            f"\nrank: {rank}, {name}, mean: {gather_x.float().mean().item()}, "
            f"std: {gather_x.float().std().item()}, shape: {x_shape}",
            flush=True,
        )
    else:
        print(
            f"{name}, mean: {x.float().mean().item()}, "
            f"std: {x.float().std().item()}, shape: {x.shape}"
        )
