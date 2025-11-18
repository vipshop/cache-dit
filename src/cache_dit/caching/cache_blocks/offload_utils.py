import torch
import asyncio
import logging
from contextlib import contextmanager
from typing import Generator, Optional, List
from diffusers.hooks.group_offloading import _is_group_offload_enabled
from cache_dit.logger import init_logger

logger = init_logger(__name__)


@torch.compiler.disable
@contextmanager
def maybe_onload(
    block: torch.nn.Module,
    reference_tensor: torch.Tensor,
    pending_tasks: List[asyncio.Task] = [],
) -> Generator:

    if not _is_group_offload_enabled(block):
        yield block
        return

    original_devices: Optional[List[torch.device]] = None
    if hasattr(block, "parameters"):
        params = list(block.parameters())
        if params:
            original_devices = [param.data.device for param in params]

    target_device: torch.device = reference_tensor.device
    move_task: Optional[asyncio.Task] = None
    need_restore: bool = False

    try:
        if original_devices is not None:
            unique_devices = list(set(original_devices))
            if len(unique_devices) > 1 or unique_devices[0] != target_device:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Onloading from {unique_devices} to {target_device}")

                has_meta_params = any(dev.type == "meta" for dev in original_devices)
                if has_meta_params:  # compatible with sequential cpu offload
                    block = block.to_empty(device=target_device)
                else:
                    block = block.to(target_device, non_blocking=False)
                need_restore = True
        yield block
    finally:
        if need_restore and original_devices:

            async def restore_device():
                for param, original_device in zip(block.parameters(), original_devices):
                    param.data = await asyncio.to_thread(
                        lambda p, d: p.to(d, non_blocking=True),
                        param.data,  # type: torch.Tensor
                        original_device,  # type: torch.device
                    )  # type: ignore[assignment]

            loop = get_event_loop()
            move_task = loop.create_task(restore_device())
            if move_task:
                pending_tasks.append(move_task)


def get_event_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

    if not loop.is_running():

        def run_loop() -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        import threading

        if not any(t.name == "_my_loop" for t in threading.enumerate()):
            threading.Thread(target=run_loop, name="_my_loop", daemon=True).start()

    return loop


@torch.compiler.disable
def maybe_offload(
    pending_tasks: List[asyncio.Task],
) -> None:
    if not pending_tasks:
        return

    loop = get_event_loop()

    async def gather_tasks():
        return await asyncio.gather(*pending_tasks)

    future = asyncio.run_coroutine_threadsafe(gather_tasks(), loop)
    try:
        future.result(timeout=30.0)
    except Exception as e:
        logger.error(f"May Offload Error: {e}")

    pending_tasks.clear()
