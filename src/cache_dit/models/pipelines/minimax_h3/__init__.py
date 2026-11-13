"""MiniMax-H3 complete-step boundaries using the upstream computation blocks."""

from diffusers.modular_pipelines.minimax_h3.denoise import MiniMaxH3DenoiseLoopWrapper
from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import MiniMaxH3Blocks

from ....utils import profiled_denoise_loop


class MiniMaxH3DenoiseLoop(MiniMaxH3DenoiseLoopWrapper):
  """Bind the shared loop to the upstream MiniMax-H3 component declarations."""

  __call__ = profiled_denoise_loop


def minimax_h3_blocks():
  """Bind complete-step boundaries before workflow selection flattens the block tree."""
  blocks = MiniMaxH3Blocks()
  for core in blocks.sub_blocks["denoise"].sub_blocks.values():
    loop = core.sub_blocks["denoise"]
    core.sub_blocks["denoise"] = MiniMaxH3DenoiseLoop.from_blocks_dict(dict(loop.sub_blocks))
  return blocks
