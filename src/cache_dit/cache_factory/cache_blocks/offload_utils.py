import torch
from contextlib import contextmanager
import asyncio
from typing import Generator, Optional, Tuple, List


@contextmanager
def sync_block_device(
    block: torch.nn.Module,
    reference_tensor: torch.Tensor,
    pending_tasks: List[asyncio.Task] = [],
) -> Generator[Tuple[torch.nn.Module, Optional[asyncio.Task]], None, None]:
    original_device: Optional[torch.device] = None
    if hasattr(block, "parameters"):
        params = list(block.parameters())
        if params:
            original_device = params[0].device

    target_device: torch.device = reference_tensor.device
    move_task: Optional[asyncio.Task] = None
    need_restore: bool = False

    try:
        if original_device is not None and original_device != target_device:
            block = block.to(target_device, non_blocking=False)
            need_restore = True
        yield block
    finally:
        if need_restore and original_device is not None:

            async def restore_device():
                block.to(original_device, non_blocking=True)
                for param in block.parameters():
                    await asyncio.to_thread(lambda: param.data.device)

            move_task = asyncio.create_task(restore_device())
            if move_task:
                pending_tasks.append(move_task)


def get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if not loop.is_running():

        def run_loop():
            asyncio.set_event_loop(loop)
            loop.run_forever()

        import threading

        threading.Thread(target=run_loop, daemon=True).start()

    return loop


def wait_for_async_tasks(
    pending_tasks: List[asyncio.Task], loop: asyncio.AbstractEventLoop
) -> None:
    if not pending_tasks:
        return

    future: asyncio.Future = asyncio.run_coroutine_threadsafe(
        asyncio.gather(*pending_tasks), loop
    )
    future.result()
