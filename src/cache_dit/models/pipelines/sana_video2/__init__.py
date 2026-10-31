"""Vendored SANA-Video 2.0 (5B, 720p) pipeline and transformer.

The release is not a diffusers checkpoint (merged `.pth` + `config.yaml`), so the model code lives
here and is imported lazily by the CLI example and the block adapter; importing ``cache_dit`` never
pulls it.
"""
from __future__ import annotations

__all__ = ["SanaVideo2Pipeline", "SanaVideo2PipelineOutput", "SanaVideo2Transformer3DModel"]


def __getattr__(name: str):
  """Import the heavy modules on first attribute access.

  :param name: Requested attribute.
  :returns: The resolved object.
  """
  if name in ("SanaVideo2Pipeline", "SanaVideo2PipelineOutput"):
    from .pipeline import SanaVideo2Pipeline, SanaVideo2PipelineOutput
    return {
      "SanaVideo2Pipeline": SanaVideo2Pipeline,
      "SanaVideo2PipelineOutput": SanaVideo2PipelineOutput,
    }[name]
  if name == "SanaVideo2Transformer3DModel":
    from .transformer import SanaVideo2Transformer3DModel
    return SanaVideo2Transformer3DModel
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
