"""Vendored LingBot-Video Dense (Robbyant) implementation.

The checkpoint's `model_index.json` points `transformer` and `scheduler` at the repo's own
`lingbot_video` module, so diffusers cannot load it without `trust_remote_code`. The classes are
imported lazily: importing `cache_dit` never pulls the model code in.
"""

__all__ = [
  "LingBotVideoPipeline",
  "LingBotVideoTransformer3DModel",
  "FlowUniPCMultistepScheduler",
]


def __getattr__(name):
  if name == "LingBotVideoPipeline":
    from .pipeline import LingBotVideoPipeline

    return LingBotVideoPipeline
  if name == "LingBotVideoTransformer3DModel":
    from .transformer import LingBotVideoTransformer3DModel

    return LingBotVideoTransformer3DModel
  if name == "FlowUniPCMultistepScheduler":
    from .scheduler import FlowUniPCMultistepScheduler

    return FlowUniPCMultistepScheduler
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
