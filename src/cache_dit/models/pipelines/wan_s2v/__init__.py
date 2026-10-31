"""Self-contained Wan2.2-S2V (speech-to-video) model.

The model is not part of the diffusers library (upstream PR huggingface/diffusers#12258 was
closed unmerged), so cache-dit owns the transformer and the pipeline instead of patching
diffusers. The implementation is ported from that PR (Apache-2.0) and keeps the public class
names, the `blocks` module list and every state-dict key, which is what the block adapter, the
CP/TP planners and `from_pretrained` rely on.

Import is lazy: `import cache_dit` must not pull the model code.
"""

__all__ = [
  "S2VAudioContext",
  "WanS2VTransformer3DModel",
  "WanSpeechToVideoPipeline",
]


def __getattr__(name):
  if name == "WanSpeechToVideoPipeline":
    from .pipeline import WanSpeechToVideoPipeline

    return WanSpeechToVideoPipeline
  if name in ("WanS2VTransformer3DModel", "S2VAudioContext"):
    from .transformer import S2VAudioContext, WanS2VTransformer3DModel

    return {
      "WanS2VTransformer3DModel": WanS2VTransformer3DModel,
      "S2VAudioContext": S2VAudioContext,
    }[name]
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
