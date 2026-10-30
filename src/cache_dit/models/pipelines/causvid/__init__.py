"""CausVid-Wan: causal few-step student of `Wan-AI/Wan2.1-T2V-1.3B`.

The release ships a single native-Wan `model.pt` and its own inference code, so the pipeline,
transformer and sigma grid are vendored here instead of patching diffusers. Class contracts that
cache-dit relies on: `CausVidPipeline` is a `DiffusionPipeline`, the transformer is a diffusers
`ModelMixin`, the block stack is a top-level `blocks` ModuleList declared in `_repeated_blocks`, and
the text encoder / VAE are the diffusers and transformers modules of the Wan2.1 base checkpoint (so
TE-P and VAE-P keep working).
"""

from .pipeline import CausVidPipeline, CausVidPipelineOutput
from .scheduler import CausVidFlowMatchScheduler
from .transformer import (
  CausVidSelfAttentionKVCache,
  CausVidTransformer3DModel,
  CausVidTransformerBlock,
)

__all__ = [
  "CausVidPipeline",
  "CausVidPipelineOutput",
  "CausVidFlowMatchScheduler",
  "CausVidSelfAttentionKVCache",
  "CausVidTransformer3DModel",
  "CausVidTransformerBlock",
]
