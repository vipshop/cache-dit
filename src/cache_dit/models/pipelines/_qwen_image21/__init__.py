"""Second-pass, cache-dit optimised variants of diffusers models (``_`` prefix).

Unlike the vendored models in this package, nothing here is copied from upstream: both
classes are thin subclasses of the diffusers implementation plus cache-dit's fused
operators, so the vendored-code conventions do not apply. Import is lazy so that
``import cache_dit`` never pulls in diffusers pipelines.
"""

from typing import Any

__all__ = ["_QwenImage21Pipeline", "_QwenImage21Transformer2DModel"]


def __getattr__(name: str) -> Any:
  """Import the subclasses on first access.

  :param name: Attribute requested on the package.
  :returns: The requested class.
  :raises AttributeError: When the attribute is not part of the public surface.
  """

  if name == "_QwenImage21Transformer2DModel":
    from .transformer_qwen_image21 import _QwenImage21Transformer2DModel

    return _QwenImage21Transformer2DModel
  if name == "_QwenImage21Pipeline":
    from .pipeline_qwenimage21 import _QwenImage21Pipeline

    return _QwenImage21Pipeline
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
