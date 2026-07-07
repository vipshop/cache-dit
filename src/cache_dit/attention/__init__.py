from ..envs import ENV
from ..logger import init_logger
from .backends.register import _AttnBackend, _AttnBackendRegistry, _dispatch_attention_fn
from ._interface import set_attn_backend

from ..distributed.core import (
  _ContextParallelConfig,
  _enable_context_parallelism,
  _all_to_all_o_async_fn,
  _all_to_all_qkv_async_fn,
  _init_comm_metadata,
)
from ._diffusers_bridge import _register_cache_dit_attn_backends_to_diffusers
from ._backend_selector import AttnBackendSelector

logger = init_logger(__name__)

__all__ = [
  "_AttnBackend",
  "_AttnBackendRegistry",
  "_dispatch_attention_fn",
  "set_attn_backend",
  "_maybe_register_custom_attn_backends",
  "AttnBackendSelector",
]


def _maybe_register_custom_attn_backends():
  """Maybe re-register native attention backend to enable context parallelism."""
  if ENV.CACHE_DIT_CUSTOM_ATTN_ALREADY_DISPATCH:
    return

  try:
    # Import backend modules to ensure they are registered
    from .backends import native, cudnn, sage, flash, flash_varlen, npu

    registered = _register_cache_dit_attn_backends_to_diffusers()
    ENV.CACHE_DIT_CUSTOM_ATTN_ALREADY_DISPATCH = True

    logger.info(f"Registered custom attn backends: {', '.join(registered)}.")
  except ImportError as e:
    raise e
