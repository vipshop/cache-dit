import torch
import functools
from typing import Optional, Callable
from cache_dit.platforms import current_platform
from cache_dit.logger import init_logger

logger = init_logger(__name__)
_original_torch_compile: Optional[Callable] = None


def _get_mindiesd_backend():
    try:
        from mindiesd.compilation import MindieSDBackend

        _backend = MindieSDBackend()
    except ImportError:
        _backend = None

    return _backend


def maybe_wrap_torch_compile():
    global _original_torch_compile

    # Avoid duplicate patch
    if _original_torch_compile is not None:
        return

    _original_torch_compile = torch.compile

    # MindIESD Backend Available
    mindiesd_backend = _get_mindiesd_backend()

    @functools.wraps(_original_torch_compile)
    def patched_compile(*args, **kwargs):
        if "backend" not in kwargs and "npu" in current_platform.device_type:
            if mindiesd_backend:
                logger.warning(
                    "NPU platform detected with MindIE-SD available. "
                    "torch.compile will default to MindIESDBackend. "
                    "Override it with torch.compile(backend=...) if needed."
                )
                kwargs["backend"] = mindiesd_backend
            else:
                logger.warning(
                    "NPU platform detected but MindIE-SD not found. "
                    "Run `pip install mindiesd` for better NPU performance on Compilation."
                )

        return _original_torch_compile(*args, **kwargs)

    # Patch Torch Compile
    torch.compile = patched_compile
