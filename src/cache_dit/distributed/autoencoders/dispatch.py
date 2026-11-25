from typing import Optional

import torch

from ...logger import init_logger
from ..config import ParallelismConfig
from .planners import _activate_auto_encoder_dp_planners
from .register import AutoEncoderDataParallelismPlannerRegister

logger = init_logger(__name__)

# TODO: Extend batch-local tile decoding to other VAE models.
def set_minimax_h3_vae_batch_local_tiles(vae: torch.nn.Module, enabled: bool = True) -> None:
  """Set the tile batching preference on an installed MiniMax-H3 video VAE planner.

  Call on every rank before inference. The preference applies during multi-rank tiled decoding.

  :param vae: Video VAE instance parallelized by cache-dit.
  :param enabled: Decode each temporal window's local tiles in one batch.
  """
  from diffusers import AutoencoderKLMiniMaxH3

  if not isinstance(enabled, bool):
    raise TypeError("enabled must be a boolean.")
  if not isinstance(vae, AutoencoderKLMiniMaxH3):
    raise TypeError("Expected an AutoencoderKLMiniMaxH3 video VAE.")
  if "_h3_vae_batch_local_tiles" not in vars(vae):
    raise RuntimeError("Install the MiniMax-H3 video VAE planner before configuring tile batching.")
  vae._h3_vae_batch_local_tiles = enabled


def _ensure_auto_encoder_dp_planners_activated() -> None:
  if AutoEncoderDataParallelismPlannerRegister.supported_planners()[0] == 0:
    _activate_auto_encoder_dp_planners()


def _parallelize_autoencoder_dp(
  auto_encoder: torch.nn.Module,
  parallelism_config: Optional[ParallelismConfig],
) -> torch.nn.Module:
  assert isinstance(
    auto_encoder, torch.nn.Module
  ), f"auto_encoder must be an instance of torch.nn.Module, but got {type(auto_encoder)}"

  if parallelism_config is None:
    return auto_encoder

  _ensure_auto_encoder_dp_planners_activated()

  # We don't check backend here because auto encoder may use different
  # parallelism backend with transformer.
  return AutoEncoderDataParallelismPlannerRegister.get_planner(auto_encoder)().apply(
    auto_encoder=auto_encoder,
    parallelism_config=parallelism_config,
  )


def parallelize_autoencoder(
  auto_encoder: torch.nn.Module,
  parallelism_config: Optional[ParallelismConfig],
) -> torch.nn.Module:
  """Parallelize one autoencoder with the configured DP strategy.

  :param auto_encoder: Autoencoder module to parallelize.
  :param parallelism_config: Parallelism configuration shared with the transformer.
  :returns: The parallelized autoencoder.
  """
  assert isinstance(
    auto_encoder, torch.nn.Module
  ), f"auto_encoder must be an instance of torch.nn.Module, but got {type(auto_encoder)}"
  if getattr(auto_encoder, "_is_parallelized", False):
    logger.warning("The auto encoder is already parallelized. Skipping parallelism enabling.")
    return auto_encoder

  if parallelism_config is None:
    return auto_encoder

  auto_encoder = _parallelize_autoencoder_dp(
    auto_encoder=auto_encoder,
    parallelism_config=parallelism_config,
  )

  auto_encoder._is_parallelized = True  # type: ignore[attr-defined]
  auto_encoder._parallelism_config = parallelism_config  # type: ignore[attr-defined]

  logger.info(f"Parallelize Auto Encoder: {auto_encoder.__class__.__name__}, "
              f"id:{id(auto_encoder)}, {parallelism_config.strify(True, vae=True)}")

  return auto_encoder
