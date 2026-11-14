"""MiniMax-H3 profiling boundaries using the upstream computation blocks."""

from diffusers.modular_pipelines.minimax_h3.decoders import (
  MiniMaxH3AudioDecodeStep,
  MiniMaxH3VideoDecodeStep,
)
from diffusers.modular_pipelines.minimax_h3.denoise import (
  MiniMaxH3DenoiseLoopWrapper,
  MiniMaxH3LoopDenoiser,
  MiniMaxH3LoopSchedulerStep,
)
from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3TextEncoderStep
from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import MiniMaxH3Blocks

from ....profile.nvtx import stage_range
from ....utils import profiled_denoise_loop


class _TextEncoder(MiniMaxH3TextEncoderStep):
  __call__ = stage_range("encode_text")(MiniMaxH3TextEncoderStep.__call__)


class _Denoiser(MiniMaxH3LoopDenoiser):
  __call__ = stage_range("predict")(MiniMaxH3LoopDenoiser.__call__)


class _SchedulerStep(MiniMaxH3LoopSchedulerStep):
  __call__ = stage_range("scheduler_step")(MiniMaxH3LoopSchedulerStep.__call__)


class _VideoDecoder(MiniMaxH3VideoDecodeStep):
  __call__ = stage_range("decode_video")(MiniMaxH3VideoDecodeStep.__call__)


class _AudioDecoder(MiniMaxH3AudioDecodeStep):
  __call__ = stage_range("decode_audio")(MiniMaxH3AudioDecodeStep.__call__)


class MiniMaxH3DenoiseLoop(MiniMaxH3DenoiseLoopWrapper):
  """Bind the shared loop to the upstream MiniMax-H3 component declarations."""

  __call__ = stage_range("denoise_loop")(profiled_denoise_loop)


def minimax_h3_blocks():
  """Bind stage boundaries before workflow selection flattens the block tree."""
  blocks = MiniMaxH3Blocks()
  blocks.sub_blocks["text_encoder"].sub_blocks["t2va"] = _TextEncoder()
  blocks.sub_blocks["decode"].sub_blocks["video"] = _VideoDecoder()
  blocks.sub_blocks["decode"].sub_blocks["audio"] = _AudioDecoder()

  for core in blocks.sub_blocks["denoise"].sub_blocks.values():
    denoiser = core.sub_blocks["denoise"].sub_blocks["denoiser"]
    loop_blocks = {
      "denoiser": _Denoiser(transformer_name=denoiser.transformer_name),
      "update": _SchedulerStep(),
    }
    core.sub_blocks["denoise"] = MiniMaxH3DenoiseLoop.from_blocks_dict(loop_blocks)

  return blocks
