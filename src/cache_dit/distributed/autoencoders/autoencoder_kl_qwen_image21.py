import torch
from diffusers import AutoencoderKLQwenImage21

from ..config import ParallelismConfig
from .autoencoder_kl_qwen_image import AutoencoderKLQwenImageDataParallelismPlanner
from .register import AutoEncoderDataParallelismPlannerRegister


@AutoEncoderDataParallelismPlannerRegister.register("AutoencoderKLQwenImage21")
class AutoencoderKLQwenImage21DataParallelismPlanner(AutoencoderKLQwenImageDataParallelismPlanner):
  """Data-parallel tiled VAE for Qwen-Image 2.1.

  `AutoencoderKLQwenImage21` is a standalone class (it does not inherit from
  `AutoencoderKLQwenImage`), but it keeps the same tiling API, so the tiled encode/decode
  implementation is reused: tiles are distributed across ranks, gathered on rank 0 and blended
  there, then the result is propagated back through the ranks.

  The 2.1 decoder differs in two ways: it flags the first temporal chunk of a tile with
  `first_chunk`, and it clamps the decoded output to `[-1, 1]`. `patch_size` is not supported by
  the data-parallel path.
  """

  # Without `first_chunk` the decoder treats a tile start as a continuation chunk and raises
  # `Expected 3D or 4D input to conv2d`.
  decoder_first_chunk: bool = True
  clamp_decoded_output: bool = True

  def _apply(
    self,
    auto_encoder: torch.nn.Module,
    parallelism_config: ParallelismConfig,
    **kwargs,
  ) -> torch.nn.Module:
    assert isinstance(auto_encoder, AutoencoderKLQwenImage21), (
      "AutoencoderKLQwenImage21DataParallelismPlanner can only be applied to AutoencoderKLQwenImage21"
    )
    assert getattr(auto_encoder.config, "patch_size", None) is None, (
      "Tiled data-parallel VAE decoding does not implement patch_size; disable --parallel-vae "
      "for this checkpoint.")
    dp_mesh = self.mesh(parallelism_config=parallelism_config)
    auto_encoder = self.parallelize_tiling(
      auto_encoder=auto_encoder,
      dp_mesh=dp_mesh,
    )

    return auto_encoder
