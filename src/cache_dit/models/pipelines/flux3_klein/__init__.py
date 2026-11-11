"""Second-pass, cache-dit optimised pass over diffusers' FLUX.3 transformer (``_`` prefix).

Nothing here is copied from upstream: `_Flux3KleinTransformer2DModel` is a thin subclass of the
diffusers implementation plus cache-dit's fused operators, so the vendored-code conventions do
not apply. FLUX.3 is a `ModularPipeline`, so there is no pipeline subclass either - the CLI
example swaps the transformer class in its `post_init_hook`. Import is lazy so that
``import cache_dit`` never pulls in diffusers pipelines.
"""

from typing import Any

__all__ = ["_Flux3KleinTransformer2DModel"]


def __getattr__(name: str) -> Any:
  """Import the subclass on first access.

  :param name: Attribute requested on the package.
  :returns: The requested class.
  :raises AttributeError: When the attribute is not part of the public surface.
  """

  if name == "_Flux3KleinTransformer2DModel":
    from .transformer_flux3_klein import _Flux3KleinTransformer2DModel

    return _Flux3KleinTransformer2DModel
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
