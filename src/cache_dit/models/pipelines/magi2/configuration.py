"""Frozen configuration objects for the released MAGI-2 Preview checkpoint.

Copied from ``vllm-omni @ 7f217a1`` ``vllm_omni/diffusion/models/magi2/configuration_magi2.py``
(Apache-2.0, Copyright (c) 2026 SandAI). Intentional differences:

* formatted to cache-dit style (2-space indent, Sphinx docstrings);
* ``Magi2PreviewConfig`` gained ``from_dict`` / ``to_dict`` so the pipeline can rebuild the
  architecture from a checkpoint ``config.json`` without importing the upstream dataclass module.

The default values are the released checkpoint's architecture, not tunables: changing them makes the
released weights unloadable.
"""

import dataclasses
from typing import Any, Dict

import torch

__all__ = [
  "MAGI2_GENERATION_CONFIG",
  "MAGI2_PREVIEW_CONFIG",
  "Magi2GenerationConfig",
  "Magi2MHCConfig",
  "Magi2MoEConfig",
  "Magi2PreviewConfig",
]


@dataclasses.dataclass(frozen=True)
class Magi2MHCConfig:
  """Multi-head hyper-connection (mHC) settings.

  :param enabled: The released checkpoint requires mHC; disabling it is rejected by ``validate``.
  :param num_streams: Number of residual streams carried through every layer.
  :param alpha_init: Initial value of the learnable scalar gate per coefficient.
  :param sinkhorn_iterations: Sinkhorn-Knopp iterations for the residual matrix.
  :param sinkhorn_epsilon: Numerical epsilon of the Sinkhorn normalization.
  """

  enabled: bool = True
  num_streams: int = 4
  alpha_init: float = 0.01
  sinkhorn_iterations: int = 20
  sinkhorn_epsilon: float = 1e-12


@dataclasses.dataclass(frozen=True)
class Magi2MoEConfig:
  """Multi-head MoE settings (heads are sharded like an expert axis).

  :param num_heads: MoE heads; ``hidden_size`` is split evenly across them.
  :param num_experts: Experts per head.
  :param top_k: Experts activated per token and head.
  :param expert_intermediate_size: Hidden width of one expert.
  :param shared_expert_intermediate_size: Hidden width of the always-on shared expert.
  :param modality_shared_expert_intermediate_size: Hidden width of the per-modality shared expert.
  :param score_function: Routing score function; the released checkpoint uses ``sigmoid``.
  :param normalize_routing_weights: L1-normalize the top-k probabilities.
  :param routing_scale: Constant scale applied to the routing probabilities.
  :param layers: Layer indices whose MLP is an MoE layer (the rest are dense).
  """

  num_heads: int = 12
  num_experts: int = 256
  top_k: int = 6
  expert_intermediate_size: int = 1280
  shared_expert_intermediate_size: int = 1280
  modality_shared_expert_intermediate_size: int = 1280
  score_function: str = "sigmoid"
  normalize_routing_weights: bool = True
  routing_scale: float = 4.9
  layers: tuple = tuple(range(2, 38))

  def __post_init__(self) -> None:
    if self.num_heads <= 0 or self.num_experts <= 0:
      raise ValueError("MAGI-2 MoE head and expert counts must be positive")
    if not 0 < self.top_k <= self.num_experts:
      raise ValueError("MAGI-2 MoE top_k must be in [1, num_experts]")
    if self.score_function != "sigmoid":
      raise ValueError("The released MAGI-2 checkpoint uses sigmoid routing")


@dataclasses.dataclass(frozen=True)
class Magi2PreviewConfig:
  """Architecture of the 114B-parameter Preview transformer.

  :param num_layers: Transformer depth.
  :param hidden_size: Single-stream hidden width; the model carries ``mhc.num_streams`` of them.
  :param head_dim: Attention head width.
  :param num_query_groups: KV head count (equal to the Q head count in the release).
  :param video_in_channels: Video latent channels entering the pre-adapter.
  :param audio_in_channels: Audio latent channels entering the pre-adapter.
  :param text_in_channels: Text embedding width entering the pre-adapter.
  :param intermediate_factor: Dense MLP expansion factor (the released formula rounds to 128).
  :param multimodal_layers: Layers with per-modality grouped projections.
  :param params_dtype: Weight dtype of the released checkpoint.
  :param attention_softcap: Logit softcap; ``<= 0`` disables it.
  :param attention_sink_tokens: Learned sink logits per head.
  :param attention_gating: Per-head sigmoid gate on the attention output.
  :param mhc: mHC settings.
  :param moe: MoE settings.
  """

  num_layers: int = 40
  hidden_size: int = 3072
  head_dim: int = 128
  num_query_groups: int = 24
  video_in_channels: int = 48
  audio_in_channels: int = 64
  text_in_channels: int = 5120
  intermediate_factor: float = 4.0
  multimodal_layers: tuple = (0, 1, 38, 39)
  params_dtype: torch.dtype = torch.bfloat16
  attention_softcap: float = -1.0
  attention_sink_tokens: int = 1
  attention_gating: bool = True
  mhc: Magi2MHCConfig = dataclasses.field(default_factory=Magi2MHCConfig)
  moe: Magi2MoEConfig = dataclasses.field(default_factory=Magi2MoEConfig)

  @property
  def num_attention_heads(self) -> int:
    """Q/KV head count, derived from ``hidden_size`` and ``head_dim``."""

    if self.hidden_size % self.head_dim:
      raise ValueError("hidden_size must be divisible by head_dim")
    return self.hidden_size // self.head_dim

  @property
  def num_heads_q(self) -> int:
    """Query head count."""

    return self.num_attention_heads

  @property
  def num_heads_kv(self) -> int:
    """Key/value head count."""

    return self.num_query_groups

  @property
  def virtual_width(self) -> int:
    """Flattened width of the mHC stream stack."""

    return self.hidden_size * self.mhc.num_streams

  def validate(self) -> None:
    """Reject configurations the released checkpoint cannot run.

    :raises ValueError: When mHC, attention gating or the sink is disabled, or an index is invalid.
    """

    if not self.mhc.enabled:
      raise ValueError("The released MAGI-2 Preview checkpoint requires mHC")
    if not self.attention_gating:
      raise ValueError("The released MAGI-2 Preview checkpoint requires attention gating")
    if self.attention_sink_tokens < 1:
      raise ValueError("The released MAGI-2 Preview checkpoint requires an attention sink")
    if self.hidden_size % self.moe.num_heads:
      raise ValueError("hidden_size must be divisible by the number of MoE heads")
    layer_ids = set(range(self.num_layers))
    if not set(self.multimodal_layers).issubset(layer_ids):
      raise ValueError("multimodal layer index is outside the transformer")
    if not set(self.moe.layers).issubset(layer_ids):
      raise ValueError("MoE layer index is outside the transformer")

  @classmethod
  def from_dict(cls, config: Dict[str, Any]) -> "Magi2PreviewConfig":
    """Build a config from a plain dict, as stored next to the released weights.

    :param config: Mapping with the dataclass field names; absent keys keep their defaults.
    :returns: A validated config instance.
    """

    config = dict(config or {})
    mhc = config.pop("mhc", None)
    moe = config.pop("moe", None)
    dtype = config.get("params_dtype")
    if isinstance(dtype, str):
      config["params_dtype"] = getattr(torch, dtype)
    if config.get("multimodal_layers") is not None:
      config["multimodal_layers"] = tuple(config["multimodal_layers"])
    if moe is not None:
      moe = dict(moe)
      if "layers" in moe and moe["layers"] is not None:
        moe["layers"] = tuple(moe["layers"])
      config["moe"] = Magi2MoEConfig(**moe)
    if mhc is not None:
      config["mhc"] = Magi2MHCConfig(**dict(mhc))
    known = {field.name for field in dataclasses.fields(cls)}
    config = {key: value for key, value in config.items() if key in known}
    instance = cls(**config)
    instance.validate()
    return instance


@dataclasses.dataclass(frozen=True)
class Magi2GenerationConfig:
  """Sampling contract of the released 10-second audio-video clip.

  :param duration_seconds: Only 10 s is supported by the released weights.
  :param fps: Public output frame rate (the DiT/decoder timeline runs at 25 fps).
  :param output_frames: ``duration_seconds * fps``.
  :param preview_steps: Released preview-stage step count.
  :param shift: Flow-matching schedule shift.
  :param video_guidance_scale: Video CFG scale.
  :param audio_guidance_scale: Audio CFG scale.
  :param audio_latent_fps: Audio latent frame rate.
  :param video_vae_stride: Video VAE (T, H, W) stride.
  :param patch_size: Patch size (the release patches 1x1x1).
  :param video_latent_channels: Video latent channels.
  :param audio_latent_channels: Audio latent channels.
  :param audio_sample_rate: Output sample rate.
  """

  duration_seconds: float = 10.0
  fps: float = 12.5
  output_frames: int = 125
  preview_steps: int = 100
  shift: float = 7.0
  video_guidance_scale: float = 5.0
  audio_guidance_scale: float = 7.0
  audio_latent_fps: float = 25.0
  video_vae_stride: tuple = (8, 16, 16)
  patch_size: tuple = (1, 1, 1)
  video_latent_channels: int = 48
  audio_latent_channels: int = 64
  audio_sample_rate: int = 44100


MAGI2_PREVIEW_CONFIG = Magi2PreviewConfig()
MAGI2_GENERATION_CONFIG = Magi2GenerationConfig()
