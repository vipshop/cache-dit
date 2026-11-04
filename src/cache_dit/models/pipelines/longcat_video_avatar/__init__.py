"""LongCat-Video-Avatar: audio-driven human video generation.

The release ships its own pipeline and transformer code, so both are vendored here instead of being
imported from `diffusers`. See `transformer.py` and `pipeline.py` for the list of intentional
differences from the upstream copy.
"""

from .pipeline import LongCatVideoAvatarPipeline
from .transformer import LongCatAvatarSingleStreamBlock, LongCatVideoAvatarTransformer3DModel

__all__ = [
  "LongCatAvatarSingleStreamBlock",
  "LongCatVideoAvatarTransformer3DModel",
  "LongCatVideoAvatarPipeline",
]
