"""MiniMax-H3 profiling boundaries using the upstream computation blocks."""

import torch

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

from cache_dit.profile import profile_steps
from cache_dit.profile.nvtx import stage_range
from cache_dit.cli.registers import ExampleRegister


def create_example(args):
  """Create a registered H3 example with explicit profiling blocks.

  :param args: Arguments for an H3 workflow.
  :returns: The example configured before pipeline construction.
  """
  example = ExampleRegister.get_example(args, args.example)
  example.init_config.modular_blocks_factory = minimax_h3_blocks
  return example


@torch.no_grad()
def profiled_denoise_loop(self, components, state):
  """Run the original loop with complete profiling steps.

  :param self: Loop block providing the original state and computation methods.
  :param components: Pipeline components used by the original computation blocks.
  :param state: Pipeline state to read and update.
  :returns: The components and updated pipeline state.
  """
  block_state = self.get_block_state(state)
  with self.progress_bar(total=len(block_state.timesteps)) as progress_bar:
    with profile_steps(block_state.timesteps) as steps:
      for i, t in steps:
        with stage_range(f"denoise_step {i}"):
          components, block_state = self.loop_step(components, block_state, i=i, t=t)
          progress_bar.update()
  self.set_block_state(state, block_state)
  return components, state


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
