"""Vendored HY-Nomas (segment-autoregressive video world model) transformer and KV cache.

The release is not a diffusers checkpoint (the repo ships weights plus a 3-byte ``config.json``), so
the model code lives here and is imported lazily by the CLI example and the block adapter; importing
``cache_dit`` never pulls it.
"""
from __future__ import annotations

__all__ = [
  "NomasAttentionPlan",
  "NomasCache",
  "NomasDecodeOutput",
  "NomasParallelInfo",
  "NomasPipeline",
  "NomasPipelineOutput",
  "NomasPixelUMMDecoder",
  "NomasTransformer3DModel",
  "PixelUMMArch",
]


def __getattr__(name: str):
  """Import the heavy modules on first attribute access.

  :param name: Requested attribute.
  :returns: The resolved object.
  """
  if name == "NomasTransformer3DModel":
    from .transformer import NomasTransformer3DModel
    return NomasTransformer3DModel
  if name in ("NomasAttentionPlan", "NomasCache"):
    from . import kv_cache
    return getattr(kv_cache, name)
  if name in ("NomasDecodeOutput", "NomasPixelUMMDecoder", "PixelUMMArch"):
    from . import components
    return getattr(components, name)
  if name in ("NomasPipeline", "NomasPipelineOutput"):
    from .pipeline import NomasPipeline, NomasPipelineOutput
    return {"NomasPipeline": NomasPipeline, "NomasPipelineOutput": NomasPipelineOutput}[name]
  if name == "NomasParallelInfo":
    from .attention import NomasParallelInfo
    return NomasParallelInfo
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
