"""MiniMax-H3-Image helpers that are not part of the vendored pipeline.

The transformer, the pipeline and the scheduler come from the installed diffusers; what lives here is
the prompt-expansion stage, which is a *separate* checkpoint (`MiniMax-H3-Image-PE`) plus the
ratio-to-canvas arithmetic the two CLI examples need.

The module is imported lazily by the examples, so importing `cache_dit` never pulls it in.
"""

from .prompt_enhancer import (
  aspect_ratio_to_size,
  expand_prompt,
)

__all__ = ["aspect_ratio_to_size", "expand_prompt"]
