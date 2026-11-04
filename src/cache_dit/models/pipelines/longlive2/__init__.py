"""LongLive 2.0: chunk-causal autoregressive generator on the Wan2.2-TI2V-5B backbone.

The release ships a single native-Wan `model_bf16.pt` plus its own inference code, so the pipeline,
transformer and causal KV cache are vendored here instead of patching diffusers. Class contracts that
cache-dit relies on: `LongLive2Pipeline` is a `DiffusionPipeline`, the transformer is a diffusers
`ModelMixin`, the block stack is a top-level `blocks` ModuleList declared in `_repeated_blocks`, and
the text encoder / VAE are the diffusers and transformers modules of the Wan2.2-TI2V-5B checkpoint
(so TE-P and VAE-P keep working).
"""

from .pipeline import LongLive2Pipeline, LongLive2PipelineOutput
from .transformer import (
  LongLive2CausalAttnProcessor,
  LongLive2RotaryPosEmbed,
  LongLive2SelfAttentionKVCache,
  LongLive2Transformer3DModel,
  LongLive2TransformerBlock,
)

__all__ = [
  "LongLive2CausalAttnProcessor",
  "LongLive2Pipeline",
  "LongLive2PipelineOutput",
  "LongLive2RotaryPosEmbed",
  "LongLive2SelfAttentionKVCache",
  "LongLive2Transformer3DModel",
  "LongLive2TransformerBlock",
]
