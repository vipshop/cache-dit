"""SANA-Video 2.0 denoising transformer with grouped Attention Residuals.

Copied from NVlabs/Sana (Apache-2.0, Copyright 2026 NVIDIA CORPORATION & AFFILIATES),
``diffusion/model/nets/sana_video2.py`` + ``sana_video2_blocks.py`` (``SanaVideo2Block``), and
restructured so that cache-dit can cache it. Parameter names, the block math and the
Attention-Residual aggregation are unchanged; the released checkpoint loads with
``strict=True``.

Intentional differences from the upstream copy:

* **One block call per loop iteration, one tensor in / one tensor out.** Upstream runs two
  ``attend`` aggregations and the group flush in ``transformer.forward``'s loop body, which
  cache-dit's cached-block wrapper cannot replay (it only threads ``hidden_states``). The two
  aggregations, the running group residual and the flush therefore moved *into*
  :class:`SanaVideo2Block`, so a block maps "running group residual" to "running group residual".
  The shared ``value_buffer`` / ``key_buffer`` and the two aggregation weights are passed as
  arguments, which the wrapper does forward unchanged to every block.
* **``values`` / ``keys`` live across denoising steps.** They are lazily allocated and kept on the
  transformer, because a cached step skips whole Attention-Residual groups: the skipped slots keep
  the previous step's group residual, which is exactly the block-level approximation DBCache asks
  for. Slot 0 (the patch-embedded input) is rewritten on every forward.
* ``Fn_compute_blocks`` / ``Bn_compute_blocks`` must stay a multiple of ``attn_res_block_size``
  (the DBCache default of 8 matches the released ``attn_res_block_size=8``): a group is cached or
  recomputed as a whole, so a group start block has to be the first block of a cache region.
  ``forward`` checks the resolved cache config and warns when it is not, because the adapter is
  created before the cache config exists.
* Context parallelism shards the token axis after ``x_embedder`` (through the ``sequence_split``
  marker) and gathers before the latent grid is rebuilt, because the aggregation buffer of this
  model lives inside the block loop and would otherwise stay full length while the blocks see
  shards. TI2V's per-frame timestep is rejected under CP, see ``forward``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from torch import nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin

from ....logger import init_logger
from .attention import GatedLinearAttention, GatedSoftmaxAttention, MultiHeadCrossAttention
from .components import (
  CaptionEmbedder,
  DepthRMSNorm,
  PatchEmbedMS3D,
  T2IFinalLayer,
  TimestepEmbedder,
  WanRotaryPosEmbed,
  _RMSNorm,
  t2i_modulate,
)


def get_softmax_layer_indices(depth: int, softmax_ratio: float = 0.25) -> list[int]:
  """Distribute the dense-attention anchor layers evenly through the stack.

  :param depth: Number of transformer blocks.
  :param softmax_ratio: Fraction of blocks that use dense softmax attention.
  :returns: Zero-based indices of the anchor blocks.
  """
  if depth <= 0:
    raise ValueError(f"depth must be positive, got {depth}.")
  if not 0.0 < softmax_ratio <= 1.0:
    raise ValueError(f"softmax_ratio must be in (0, 1], got {softmax_ratio}.")
  anchor_count = max(1, int(depth * softmax_ratio))
  step = depth / anchor_count
  return [int((index + 1) * step) - 1 for index in range(anchor_count)]


logger = init_logger(__name__)


class BlockAttentionResidual(nn.Module):
  """Shared, timestep-independent aggregation over the depth representations.

  :param hidden_size: Hidden size of the residual stream.
  :param eps: Epsilon of the depth RMS normalization.
  """

  def __init__(self, hidden_size: int, eps: float = 1e-6):
    super().__init__()
    self.attn_proj = nn.Linear(hidden_size, 1, bias=False)
    self.mlp_proj = nn.Linear(hidden_size, 1, bias=False)
    self.final_proj = nn.Linear(hidden_size, 1, bias=False)
    self.key_norm = DepthRMSNorm(eps=eps)
    nn.init.zeros_(self.attn_proj.weight)
    nn.init.zeros_(self.mlp_proj.weight)
    nn.init.zeros_(self.final_proj.weight)


def attend_buffer(proj_weight: torch.Tensor, key_norm: nn.Module, value_buffer: torch.Tensor,
                  key_buffer: torch.Tensor, active_count: int,
                  partial_block: Optional[torch.Tensor]) -> torch.Tensor:
  """Depth aggregation over preallocated slots (upstream ``attend_buffer``).

  :param proj_weight: ``attn_proj`` / ``mlp_proj`` / ``final_proj`` weight of shape `[1, C]`.
  :param key_norm: Parameter-free depth RMS normalization.
  :param value_buffer: Depth slots `[slots, B, S, C]`.
  :param key_buffer: Depth-normalized slots of the same shape.
  :param active_count: Number of flushed slots.
  :param partial_block: Running group residual, or `None` at a group start.
  :returns: Aggregated tokens `[B, S, C]`.
  """
  if partial_block is not None:
    value_buffer[active_count] = partial_block
    key_buffer[active_count] = key_norm(partial_block.unsqueeze(0)).squeeze(0)
    source_count = active_count + 1
  else:
    source_count = active_count
  if source_count == 1:
    return value_buffer[0]
  values = value_buffer[:source_count]
  keys = key_buffer[:source_count]
  logits = torch.einsum("d,nbtd->nbt", proj_weight.squeeze(0), keys)
  return torch.einsum("nbt,nbtd->btd", logits.softmax(0), values)


class SanaVideo2Block(nn.Module):
  """One hybrid-attention Sana-Video 2.0 layer, mapping a group residual to a group residual.

  :param hidden_size: Hidden size.
  :param num_heads: Number of cross-attention heads.
  :param attention: ``"linear"`` or ``"softmax"`` self-attention.
  :param linear_head_dim: Head dimension of the gated linear attention.
  :param softmax_head_dim: Head dimension of the dense anchor attention.
  :param mlp_ratio: SwiGLU expansion ratio.
  :param qk_norm: Whether the self-attention q/k projections are RMS-normalized.
  :param cross_norm: Whether the cross-attention q/k projections are RMS-normalized.
  :param fp32_attention: Accumulate the self-attention matmuls in fp32.
  :param depth: Number of blocks in the stack, needed for the last-group flush.
  :param attn_res_block_size: Attention-Residual group size.
  :param index: Zero-based position of this block inside the stack.
  """

  def __init__(self,
               hidden_size: int,
               num_heads: int,
               attention: str,
               linear_head_dim: int,
               softmax_head_dim: int,
               mlp_ratio: float = 4.0,
               qk_norm: bool = True,
               cross_norm: bool = True,
               fp32_attention: bool = True,
               depth: int = 32,
               attn_res_block_size: int = 8,
               index: int = 0):
    super().__init__()
    self.index = int(index)
    self.attention_type = str(attention)
    self.attn_res_block_size = int(attn_res_block_size)
    self.is_group_start = self.index % self.attn_res_block_size == 0
    self.is_group_end = ((self.index + 1) % self.attn_res_block_size == 0
                         or self.index + 1 == int(depth))
    self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    if attention == "linear":
      self.attn = GatedLinearAttention(hidden_size,
                                       linear_head_dim,
                                       qk_norm=qk_norm,
                                       fp32_attention=fp32_attention)
    elif attention == "softmax":
      self.attn = GatedSoftmaxAttention(hidden_size,
                                        softmax_head_dim,
                                        qk_norm=qk_norm,
                                        fp32_attention=fp32_attention)
    else:
      raise ValueError(f"Unsupported attention type: {attention!r}.")
    self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, qk_norm=cross_norm)
    self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.mlp = _SwiGLU(hidden_size, int(hidden_size * mlp_ratio), hidden_size)
    self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)
    # Parameter-free, so registering one per block does not touch the state dict.
    self.key_norm = DepthRMSNorm(eps=1e-6)

  def _modulation(self, t: torch.Tensor, batch: int) -> Tuple[torch.Tensor, ...]:
    if t.ndim <= 2:
      modulation = self.scale_shift_table.unsqueeze(0) + t.reshape(batch, 6, -1)
      return modulation.chunk(6, dim=1)
    token_groups = t.shape[2]
    modulation = self.scale_shift_table[None, None] + t.reshape(batch, token_groups, 6, -1)
    return modulation.chunk(6, dim=-2)

  def forward(self,
              hidden_states: torch.Tensor,
              encoder_hidden_states: Optional[torch.Tensor] = None,
              temb: Optional[torch.Tensor] = None,
              encoder_attention_mask: Optional[torch.Tensor] = None,
              rotary_emb: Optional[Dict[str, torch.Tensor]] = None,
              value_buffer: Optional[torch.Tensor] = None,
              key_buffer: Optional[torch.Tensor] = None,
              attn_proj_weight: Optional[torch.Tensor] = None,
              mlp_proj_weight: Optional[torch.Tensor] = None,
              cp_config: Any = None) -> torch.Tensor:
    """Run one block of the depth-aggregated loop.

    :param hidden_states: Running group residual of the previous block; ignored at a group start,
      where the aggregation only sees the embedded input and the flushed groups.
    :param encoder_hidden_states: Projected caption features `[B, L, C]`.
    :param temb: ``t_block`` modulation input `[B, 6C]` or `[B, 1, F, 6C]`.
    :param encoder_attention_mask: Optional binary caption mask `[B, L]`.
    :param rotary_emb: Complex RoPE tables keyed by attention type; the block picks its own.
    :param value_buffer: Shared depth slots `[slots, B, S, C]`.
    :param key_buffer: Shared depth-normalized slots.
    :param attn_proj_weight: ``attn_res.attn_proj`` weight, shape `[1, C]`.
    :param mlp_proj_weight: ``attn_res.mlp_proj`` weight, shape `[1, C]`.
    :param cp_config: cache-dit context-parallel config, or `None`.
    :returns: The updated running group residual.
    """
    active_count = 1 + self.index // self.attn_res_block_size
    partial = None if self.is_group_start else hidden_states
    hidden = attend_buffer(attn_proj_weight, self.key_norm, value_buffer, key_buffer, active_count,
                           partial)
    attn_delta = self.forward_attn_sublayer(hidden, encoder_hidden_states, temb,
                                            encoder_attention_mask, rotary_emb[self.attention_type],
                                            cp_config)
    carry = attn_delta if partial is None else partial + attn_delta
    hidden = attend_buffer(mlp_proj_weight, self.key_norm, value_buffer, key_buffer, active_count,
                           carry)
    carry = carry + self.forward_mlp_sublayer(hidden, temb)
    if self.is_group_end:
      value_buffer[active_count] = carry
      key_buffer[active_count] = self.key_norm(carry.unsqueeze(0)).squeeze(0)
    return carry

  def forward_attn_sublayer(self,
                            hidden: torch.Tensor,
                            y: torch.Tensor,
                            t: torch.Tensor,
                            mask: Optional[torch.Tensor] = None,
                            rotary_emb: Optional[torch.Tensor] = None,
                            cp_config: Any = None) -> torch.Tensor:
    """Return the self- and cross-attention delta without accumulating it.

    :param hidden: Aggregated tokens `[B, S, C]`.
    :param y: Projected caption features.
    :param t: Modulation input.
    :param mask: Optional caption mask.
    :param rotary_emb: Complex RoPE table.
    :param cp_config: cache-dit context-parallel config, or `None`.
    :returns: Attention delta `[B, S, C]`.
    """
    batch, tokens, channels = hidden.shape
    shift_msa, scale_msa, gate_msa, _, _, _ = self._modulation(t, batch)
    if t.ndim <= 2:
      attn_input = t2i_modulate(self.norm1(hidden), shift_msa, scale_msa)
      attn_delta = gate_msa * self.attn(attn_input, rotary_emb=rotary_emb, cp_config=cp_config)
    else:
      token_groups = t.shape[2]
      grouped = self.norm1(hidden).reshape(batch, token_groups, -1, channels)
      attn_input = t2i_modulate(grouped, shift_msa, scale_msa).reshape(batch, tokens, channels)
      attn_delta = self.attn(attn_input, rotary_emb=rotary_emb, cp_config=cp_config)
      attn_delta = (gate_msa * attn_delta.reshape(batch, token_groups, -1, channels)).reshape(
        batch, tokens, channels)
    cross_delta = self.cross_attn(hidden + attn_delta, y, mask=mask, cp_config=cp_config)
    return attn_delta + cross_delta

  def forward_mlp_sublayer(self, hidden: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Return the MLP delta without accumulating it.

    :param hidden: Aggregated tokens `[B, S, C]`.
    :param t: Modulation input.
    :returns: MLP delta `[B, S, C]`.
    """
    batch, tokens, channels = hidden.shape
    _, _, _, shift_mlp, scale_mlp, gate_mlp = self._modulation(t, batch)
    if t.ndim <= 2:
      mlp_input = t2i_modulate(self.norm2(hidden), shift_mlp, scale_mlp)
      return gate_mlp * self.mlp(mlp_input)
    token_groups = t.shape[2]
    grouped = self.norm2(hidden).reshape(batch, token_groups, -1, channels)
    mlp_input = t2i_modulate(grouped, shift_mlp, scale_mlp).reshape(batch, tokens, channels)
    delta = self.mlp(mlp_input).reshape(batch, token_groups, -1, channels)
    return (gate_mlp * delta).reshape(batch, tokens, channels)


class _SwiGLU(nn.Module):
  """SwiGLU feed-forward network, parameter names ``gate_proj`` / ``up_proj`` / ``down_proj``.

  :param in_features: Input size.
  :param hidden_features: Expansion size.
  :param out_features: Output size.
  :param bias: Whether the projections keep a bias.
  """

  def __init__(self,
               in_features: int,
               hidden_features: Optional[int] = None,
               out_features: Optional[int] = None,
               bias: bool = True):
    super().__init__()
    hidden_features = hidden_features or 4 * in_features
    out_features = out_features or in_features
    self.gate_proj = nn.Linear(in_features, hidden_features, bias=bias)
    self.up_proj = nn.Linear(in_features, hidden_features, bias=bias)
    self.down_proj = nn.Linear(hidden_features, out_features, bias=bias)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class SanaVideo2Transformer3DModel(ModelMixin, ConfigMixin):
  """Hybrid linear/softmax video diffusion transformer with Attention Residuals.

  Note:
    ``blocks`` is the denoising stack cache-dit caches, and ``blocks[i].index`` carries the block
    position so a single call per block reproduces the upstream Attention-Residual loop exactly;
    the shared aggregation buffers are passed to every block, so a cached step keeps the group
    residuals of the skipped groups.
  """

  _repeated_blocks = ["SanaVideo2Block"]
  _supports_gradient_checkpointing = False
  _no_split_modules = ["SanaVideo2Block"]

  @register_to_config
  def __init__(self,
               in_channels: int = 128,
               hidden_size: int = 2560,
               depth: int = 32,
               num_heads: int = 20,
               mlp_ratio: float = 4.0,
               caption_channels: int = 2304,
               model_max_length: int = 300,
               qk_norm: bool = True,
               cross_norm: bool = True,
               y_norm: bool = True,
               norm_eps: float = 1e-5,
               y_norm_scale_factor: float = 0.01,
               linear_head_dim: int = 128,
               softmax_head_dim: int = 256,
               softmax_ratio: float = 0.25,
               attn_res_block_size: int = 8,
               patch_size: Sequence[int] = (1, 1, 1),
               input_size: int = 15,
               fp32_attention: bool = True,
               timestep_norm_scale_factor: float = 1.0):
    super().__init__()
    patch_size = tuple(int(value) for value in patch_size)
    if patch_size[0] != 1:
      raise ValueError(f"patch_size must begin with 1, got {patch_size}.")
    if hidden_size % linear_head_dim or hidden_size % softmax_head_dim:
      raise ValueError("hidden_size must be divisible by both attention head dimensions.")
    self.hidden_size = hidden_size
    self.depth = depth
    self.num_heads = num_heads
    self.patch_size = patch_size
    self.attn_res_block_size = attn_res_block_size
    self.in_channels = in_channels
    self.timestep_norm_scale_factor = timestep_norm_scale_factor
    # Legacy positional table: unused by `forward` (the model is rope-based), kept only so the
    # released checkpoint loads with `strict=True`.
    self.register_buffer("pos_embed", torch.zeros(1, input_size * input_size, hidden_size))
    self.x_embedder = PatchEmbedMS3D(patch_size, in_channels, hidden_size, bias=True)
    self.t_embedder = TimestepEmbedder(hidden_size)
    self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
    self.y_embedder = CaptionEmbedder(caption_channels, hidden_size, model_max_length)
    if y_norm:
      self.attention_y_norm = _RMSNorm(hidden_size, y_norm_scale_factor, norm_eps)
    self.rope_linear = WanRotaryPosEmbed(linear_head_dim, patch_size, 1024)
    self.rope_softmax = WanRotaryPosEmbed(softmax_head_dim, patch_size, 1024)
    self.softmax_layer_indices = get_softmax_layer_indices(depth, softmax_ratio)
    anchor_set = set(self.softmax_layer_indices)
    self.block_attention_types = [
      "softmax" if index in anchor_set else "linear" for index in range(depth)
    ]
    self.blocks = nn.ModuleList([
      SanaVideo2Block(
        hidden_size,
        num_heads,
        self.block_attention_types[index],
        linear_head_dim,
        softmax_head_dim,
        mlp_ratio=mlp_ratio,
        qk_norm=qk_norm,
        cross_norm=cross_norm,
        fp32_attention=fp32_attention,
        depth=depth,
        attn_res_block_size=attn_res_block_size,
        index=index,
      ) for index in range(depth)
    ])
    self.final_layer = T2IFinalLayer(hidden_size, patch_size, in_channels * (1))
    self.attn_res = BlockAttentionResidual(hidden_size)
    # Context-parallel markers: identity modules that give the CP hooks a boundary for the token
    # split (and the two RoPE tables), and for the gather before the latent grid is rebuilt.
    self.sequence_split = nn.Identity()
    self.rope_linear_split = nn.Identity()
    self.rope_softmax_split = nn.Identity()
    self.sequence_gather = nn.Identity()
    self._attn_res_values: Optional[torch.Tensor] = None
    self._attn_res_keys: Optional[torch.Tensor] = None

  @property
  def dtype(self) -> torch.dtype:
    return self.x_embedder.proj.weight.dtype

  @dtype.setter
  def dtype(self, value: torch.dtype) -> None:
    self.to(dtype=value)

  def post_load_weights(self) -> None:
    """Rebuild the non-persistent RoPE tables when the model was built on the meta device."""
    for rope in (self.rope_linear, self.rope_softmax):
      if rope.freqs.is_meta:
        rope.rebuild_freqs()

  @torch.compiler.disable
  def _check_cache_group_alignment(self) -> None:
    """Warn once when a cache region boundary cuts an Attention-Residual group in half."""
    if getattr(self, "_cache_alignment_checked", False):
      return
    context_kwargs = getattr(self, "_context_kwargs", None)
    if not isinstance(context_kwargs, dict):
      return
    cache_config = context_kwargs.get("cache_config")
    if cache_config is None:
      return
    self._cache_alignment_checked = True
    for name in ("Fn_compute_blocks", "Bn_compute_blocks"):
      blocks = int(getattr(cache_config, name, 0) or 0)
      if blocks and blocks % self.attn_res_block_size:
        logger.warning(
          f"{name}={blocks} is not a multiple of attn_res_block_size="
          f"{self.attn_res_block_size}: a cache region that starts or ends inside an "
          "Attention-Residual group reuses a stale group residual for the remaining layers of "
          "that group. Use 8 (the default) or a multiple of it.")

  def _aggregation_weights(self, device: torch.device) -> Tuple[torch.Tensor, ...]:
    """Return the three depth-aggregation weights on the compute device.

    `--layerwise-offload` hooks only onload modules that are called with token inputs, and these
    weights are read as plain tensors, so the hooks never see them. Keeping a device copy avoids a
    host-to-device copy per denoising step; the copy is refreshed when the parameter version or the
    device changes.

    :param device: Device the tokens live on.
    :returns: ``(attn_proj, mlp_proj, final_proj)`` weights, each `[1, C]`.
    """
    projections = (self.attn_res.attn_proj, self.attn_res.mlp_proj, self.attn_res.final_proj)
    cache = getattr(self, "_aggregation_weight_cache", None)
    signature = (device, tuple(projection.weight._version for projection in projections))
    if cache is None or cache[0] != signature:
      weights = tuple(projection.weight.to(device=device) for projection in projections)
      cache = (signature, weights)
      self._aggregation_weight_cache = cache
    return cache[1]

  def _attn_res_buffers(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the depth-slot buffers for this batch/geometry, reusing them across steps.

    :param x: Embedded tokens `[B, S, C]`.
    :returns: ``(values, keys)`` of shape `[groups + 2, B, S, C]`.
    """
    slots = -(-self.depth // self.attn_res_block_size) + 2
    cached = (self._attn_res_values is not None
              and self._attn_res_values.shape == (slots, x.shape[0], x.shape[1], x.shape[2])
              and self._attn_res_values.dtype == x.dtype
              and self._attn_res_values.device == x.device)
    if not cached:
      self._attn_res_values = torch.empty(slots,
                                          x.shape[0],
                                          x.shape[1],
                                          x.shape[2],
                                          device=x.device,
                                          dtype=x.dtype)
      self._attn_res_keys = torch.empty_like(self._attn_res_values)
    return self._attn_res_values, self._attn_res_keys

  def forward(self,
              hidden_states: torch.Tensor,
              timestep: torch.Tensor,
              encoder_hidden_states: torch.Tensor,
              encoder_attention_mask: Optional[torch.Tensor] = None,
              return_dict: bool = True,
              **kwargs: Any):
    """Predict the flow-matching velocity for one batch of noisy latents.

    :param hidden_states: Noisy latents `[B, in_channels, F, H, W]`.
    :param timestep: Scalar timesteps `[B]`, or per-frame timesteps `[B, 1, F, 1, 1]` for TI2V.
    :param encoder_hidden_states: Caption features `[B, L, caption_channels]`.
    :param encoder_attention_mask: Optional binary caption mask `[B, L]`.
    :param return_dict: Kept for API compatibility; the output is a plain tensor.
    :returns: Predicted velocity with the shape of ``hidden_states``.
    """
    cp_config = getattr(self, "_cp_config", None)
    batch, _, frames, height, width = hidden_states.shape
    per_frame_timestep = timestep.ndim == 5
    if per_frame_timestep:
      if cp_config is not None:
        # The per-frame branch reshapes the *local* token shard by the *global* frame count, which
        # only lines up when the shard boundary lands on a frame boundary and every rank holds a
        # whole number of frames. Neither holds in general (13 latent frames across 2 ranks), and
        # the wrong pairing does not raise: it silently swaps the modulation of neighbouring frames.
        raise NotImplementedError(
          "SANA-Video 2.0 TI2V (per-frame timestep) does not support context parallelism: the "
          "per-frame modulation indexes frames, but CP shards the token axis inside a frame. Run "
          "TI2V without `--parallel ulysses`/`--parallel ring`.")
      timestep = timestep.reshape(batch, 1, frames)
    if self.timestep_norm_scale_factor != 1.0:
      timestep = timestep.float() / self.timestep_norm_scale_factor
    else:
      timestep = timestep.long().float()
    x = self.x_embedder(hidden_states.to(self.dtype))
    patch_f, patch_h, patch_w = self.patch_size
    f, h, w = frames // patch_f, height // patch_h, width // patch_w
    device = x.device
    ropes = {
      "linear": self.rope_linear((f, h, w), device),
      "softmax": self.rope_softmax((f, h, w), device),
    }
    # Context-parallel markers: the CP hooks split the token axis and the two RoPE tables with the
    # same boundaries, and gather the shard back before the latent grid is rebuilt. They are plain
    # identities when CP is off.
    x = self.sequence_split(x)
    ropes["linear"] = self.rope_linear_split(ropes["linear"])
    ropes["softmax"] = self.rope_softmax_split(ropes["softmax"])
    t = self.t_embedder(timestep.flatten())
    t0 = self.t_block(t).unflatten(0, timestep.shape)
    t = t.unflatten(0, timestep.shape)
    y = self.y_embedder(encoder_hidden_states.to(self.dtype)).reshape(batch, -1, self.hidden_size)
    if self.config.y_norm:
      y = self.attention_y_norm(y)
    mask = encoder_attention_mask
    if mask is not None:
      mask = mask.reshape(batch, -1)
    self._check_cache_group_alignment()
    # The aggregation weights are read as plain tensors, so `--layerwise-offload` hooks cannot
    # onload them: they belong to `attn_res`, not to a module that is called with token inputs.
    attn_proj_weight, mlp_proj_weight, final_proj_weight = self._aggregation_weights(x.device)
    values, keys = self._attn_res_buffers(x)
    values[0] = x
    keys[0] = self.attn_res.key_norm(x.unsqueeze(0)).squeeze(0)
    carry = torch.zeros_like(x)
    for block in self.blocks:
      # The loop body must not read any per-block attribute: under cache-dit `self.blocks` holds a
      # single wrapper module that replays the real stack internally, so `block.index` / `block.attn`
      # do not exist here. Each block picks its own RoPE table from the dict.
      carry = block(
        carry,
        encoder_hidden_states=y,
        temb=t0,
        encoder_attention_mask=mask,
        rotary_emb=ropes,
        value_buffer=values,
        key_buffer=keys,
        attn_proj_weight=attn_proj_weight,
        mlp_proj_weight=mlp_proj_weight,
        cp_config=cp_config,
      )
    groups = -(-self.depth // self.attn_res_block_size)
    x = attend_buffer(final_proj_weight, self.attn_res.key_norm, values, keys, groups + 1, None)
    x = self.sequence_gather(x)
    out = self.final_layer(x, t)
    out = self._unpatchify(out, f, h, w)
    return Transformer2DModelOutput(sample=out) if return_dict else (out, )

  def _unpatchify(self, x: torch.Tensor, f: int, h: int, w: int) -> torch.Tensor:
    """Rebuild the latent grid from the token stream.

    :param x: Tokens `[B, F*H*W, patch_vol * out_channels]`.
    :param f: Latent frames.
    :param h: Latent height.
    :param w: Latent width.
    :returns: Latents `[B, out_channels, F, H, W]`.
    """
    patch_f, patch_h, patch_w = self.patch_size
    out_channels = self.config.in_channels
    x = x.reshape(x.shape[0], f, h, w, patch_f, patch_h, patch_w, out_channels)
    x = torch.einsum("nfhwopqc->ncfohpwq", x)
    return x.reshape(x.shape[0], out_channels, f * patch_f, h * patch_h, w * patch_w)


__all__ = [
  "BlockAttentionResidual",
  "SanaVideo2Block",
  "SanaVideo2Transformer3DModel",
  "attend_buffer",
  "get_softmax_layer_indices",
]
