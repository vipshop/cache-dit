"""Second-pass, cache-dit optimised variants of the diffusers FLUX.2-Klein model (``_`` prefix).

Unlike the vendored models in this package, nothing here is copied from upstream: both classes
are thin subclasses of the diffusers implementation plus cache-dit's fused operators, so the
vendored-code conventions do not apply. Import is lazy so that ``import cache_dit`` never pulls
in diffusers pipelines.
"""

from typing import Any

__all__ = ["_Flux2KleinPipeline", "_Flux2KleinTransformer2DModel"]


def __getattr__(name: str) -> Any:
  """Import the subclasses on first access.

  :param name: Attribute requested on the package.
  :returns: The requested class.
  :raises AttributeError: When the attribute is not part of the public surface.
  """

  if name == "_Flux2KleinTransformer2DModel":
    from .transformer_flux2_klein import _Flux2KleinTransformer2DModel

    return _Flux2KleinTransformer2DModel
  if name == "_Flux2KleinPipeline":
    from .pipeline_flux2klein import _Flux2KleinPipeline

    return _Flux2KleinPipeline
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
