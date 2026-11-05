"""Cross-modal bridge of MOVA, vendored from OpenMOSS/MOVA ``interactionv2.py``.

The released bridge conditions the visual tower on the audio hidden states and vice versa, once per
layer, before that layer's blocks run. Both directions read the towers' states from *before* the
layer (``apply_conditional_control`` is handed the incoming ``visual_hidden_states`` twice), which is
why :meth:`DualTowerConditionalBridge.forward` keeps ``visual_in`` around.

Vendored changes:

* attention goes through cache-dit's dispatcher instead of the four-way backend fallback;
* ``cp_config`` is threaded through and the audio->visual direction (audio queries over video keys)
  all-gathers its key/value shards before attending, so context parallelism stays exact while only
  the video stream is sharded. The caller rotates the *local* key with the *local* rotary slice
  first, so the gathered key needs no re-positioning.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange

from .attention import AttentionModule
from torch.nn import RMSNorm

__all__ = [
  "ConditionalCrossAttention",
  "ConditionalCrossAttentionBlock",
  "CrossModalInteractionController",
  "DualTowerConditionalBridge",
  "PerFrameAttentionPooling",
  "RotaryEmbedding",
  "apply_bridge_directions",
]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
  """Rotate half of the hidden dims (upstream helper).

  :param x: Tensor whose last axis is the head width.
  :returns: The rotated tensor.
  """

  x1 = x[..., :x.shape[-1] // 2]
  x2 = x[..., x.shape[-1] // 2:]
  return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                  head_dim: int) -> torch.Tensor:
  """Apply rotary embedding to a ``[B, L, H*D]`` tensor.

  :param x: Query or key projection output.
  :param cos: Cosine table, ``[1, L, head_dim]``.
  :param sin: Sine table, ``[1, L, head_dim]``.
  :param head_dim: Head width.
  :returns: The rotated tensor, same shape as ``x``.
  """

  x = rearrange(x, "b l (h d) -> b l h d", d=head_dim)
  cos = cos.to(x.dtype).to(x.device).unsqueeze(2)
  sin = sin.to(x.dtype).to(x.device).unsqueeze(2)
  x = (x * cos) + (rotate_half(x) * sin)
  return rearrange(x, "b l h d -> b l (h d)")


def _cp_all_gather(x: torch.Tensor, dim: int, cp_config) -> torch.Tensor:
  """Concatenate one tensor's shards along ``dim`` across the CP group.

  The CP runtime equipartitions the video sequence, so the shards are equally sized and a plain
  ``all_gather`` into preallocated buffers is enough.

  :param x: Local shard.
  :param dim: Dimension to concatenate on.
  :param cp_config: Armed context-parallel config.
  :returns: The gathered tensor.
  """

  group = cp_config._flattened_mesh.get_group()
  x = x.contiguous()
  chunks = [torch.empty_like(x) for _ in range(dist.get_world_size(group))]
  dist.all_gather(chunks, x, group=group)
  return torch.cat(chunks, dim=dim)


class RotaryEmbedding(nn.Module):
  """Standard rotary embedding built from ``position_ids`` (upstream copy).

  :param base: RoPE base.
  :param dim: Head width.
  :param device: Device of the inverse-frequency buffer.
  """

  def __init__(self, base: float, dim: int, device=None):
    super().__init__()
    self.base = base
    self.dim = dim
    self.attention_scaling = 1.0

    inv_freq = 1.0 / (base ** (
      torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    self.register_buffer("inv_freq", inv_freq, persistent=False)
    self.original_inv_freq = self.inv_freq

  @torch.no_grad()
  def forward(self, x: torch.Tensor,
              position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the cosine / sine tables for the given positions.

    :param x: Tensor whose dtype the tables take.
    :param position_ids: ``[1, L]`` positions, in audio-step units.
    :returns: ``(cos, sin)`` of shape ``[1, L, dim]``.
    """

    inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1,
                                                                    1).to(x.device)
    position_ids_expanded = position_ids[:, None, :].float()
    with torch.autocast(device_type=x.device.type, enabled=False):
      freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
      emb = torch.cat((freqs, freqs), dim=-1)
      cos = emb.cos() * self.attention_scaling
      sin = emb.sin() * self.attention_scaling
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class PerFrameAttentionPooling(nn.Module):
  """Attention-pool the ``H*W`` tokens of every frame into one token (only for ``pooled_adaln``).

  :param dim: Hidden width.
  :param num_heads: Attention head count.
  :param eps: LayerNorm epsilon.
  """

  def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
    super().__init__()
    assert dim % num_heads == 0, "dim must be divisible by num_heads"
    self.dim = dim
    self.num_heads = num_heads
    self.probe = nn.Parameter(torch.randn(1, 1, dim))
    nn.init.normal_(self.probe, std=0.02)
    self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
    self.layernorm = nn.LayerNorm(dim, eps=eps)

  def forward(self, x: torch.Tensor, grid_size: Tuple[int, int, int]) -> torch.Tensor:
    """Pool each frame separately.

    :param x: ``[B, T*H*W, D]``.
    :param grid_size: ``(T, H, W)``.
    :returns: ``[B, T, D]``.
    """

    b, length, d = x.shape
    t, h, w = grid_size
    assert d == self.dim, f"Channel dimension mismatch: D={d} vs dim={self.dim}"
    assert length == t * h * w, f"Flattened length mismatch: L={length} vs T*H*W={t * h * w}"
    spatial = h * w
    x_bt = x.view(b, t, spatial, d).contiguous().view(b * t, spatial, d)
    probe = self.probe.expand(b * t, -1, -1)
    pooled = self.attention(probe, x_bt, x_bt, need_weights=False)[0].squeeze(1)
    return self.layernorm(pooled.view(b, t, d))


class CrossModalInteractionController:
  """Decide which layers interact, in both directions.

  :param visual_layers: Number of video layers.
  :param audio_layers: Number of audio layers.
  """

  def __init__(self, visual_layers: int = 30, audio_layers: int = 30):
    self.visual_layers = visual_layers
    self.audio_layers = audio_layers
    self.min_layers = min(visual_layers, audio_layers)

  def get_interaction_layers(self,
                             strategy: str = "shallow_focus") -> Dict[str, List[Tuple[int, int]]]:
    """Build the per-direction layer mapping.

    :param strategy: ``shallow_focus`` / ``distributed`` / ``progressive`` / ``custom`` / ``full``.
    :returns: ``{"v2a": [(i, i), ...], "a2v": [...]}``.
    """

    if strategy == "shallow_focus":
      num_interact = min(10, self.min_layers // 3)
      interact_layers = list(range(0, num_interact))
    elif strategy == "distributed":
      interact_layers = list(range(0, self.min_layers, 3))
    elif strategy == "progressive":
      shallow = list(range(0, min(8, self.min_layers)))
      deep = list(range(8, self.min_layers, 3)) if self.min_layers > 8 else []
      interact_layers = shallow + deep
    elif strategy == "custom":
      interact_layers = [i for i in [0, 2, 4, 6, 8, 12, 16, 20] if i < self.min_layers]
    elif strategy == "full":
      interact_layers = list(range(0, self.min_layers))
    else:
      raise ValueError(f"Unknown interaction strategy: {strategy}")

    return {
      "v2a": [(i, i) for i in interact_layers],
      "a2v": [(i, i) for i in interact_layers],
    }

  def should_interact(self, layer_idx: int, direction: str,
                      interaction_mapping: Dict[str, List[Tuple[int, int]]]) -> bool:
    """Whether ``layer_idx`` interacts in ``direction``.

    :param layer_idx: Zero-based layer index.
    :param direction: ``"v2a"`` or ``"a2v"``.
    :param interaction_mapping: Mapping returned by :meth:`get_interaction_layers`.
    :returns: True when the layer interacts.
    """

    if direction not in interaction_mapping:
      return False
    return any(src == layer_idx for src, _ in interaction_mapping[direction])


class ConditionalCrossAttention(nn.Module):
  """Cross attention between the two towers' hidden states, with optional per-side RoPE.

  :param dim: Query width (the primary tower).
  :param kv_dim: Key/value width (the conditioning tower).
  :param num_heads: Head count (divided by tensor parallelism).
  :param eps: RMSNorm epsilon.
  """

  def __init__(self, dim: int, kv_dim: int, num_heads: int, eps: float = 1e-6):
    super().__init__()
    self.q_dim = dim
    self.kv_dim = kv_dim
    self.num_heads = num_heads
    self.head_dim = self.q_dim // num_heads

    self.q = nn.Linear(dim, dim)
    self.k = nn.Linear(kv_dim, dim)
    self.v = nn.Linear(kv_dim, dim)
    self.o = nn.Linear(dim, dim)
    self.norm_q = RMSNorm(dim, eps=eps)
    self.norm_k = RMSNorm(dim, eps=eps)

    self.attn = AttentionModule(self.num_heads)

  def forward(self,
              x: torch.Tensor,
              y: torch.Tensor,
              x_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
              y_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
              cp_config=None,
              gather_kv: bool = False) -> torch.Tensor:
    """Attend from the primary tower to the conditioning tower.

    :param x: Primary hidden states, ``[B, S, dim]``.
    :param y: Conditioning hidden states, ``[B, L, kv_dim]``.
    :param x_freqs: Optional ``(cos, sin)`` for the query.
    :param y_freqs: Optional ``(cos, sin)`` for the key.
    :param cp_config: Armed context-parallel config, or None.
    :param gather_kv: Whether the key/value shards must be gathered before attending (the
      audio->visual direction, whose keys come from the sharded video stream).
    :returns: Attention output, ``[B, S, dim]``.
    """

    q = self.norm_q(self.q(x))
    k = self.norm_k(self.k(y))
    v = self.v(y)
    if x_freqs is not None:
      q = _apply_rotary(q, x_freqs[0], x_freqs[1], self.head_dim)
    if y_freqs is not None:
      k = _apply_rotary(k, y_freqs[0], y_freqs[1], self.head_dim)
    if gather_kv and cp_config is not None and cp_config._world_size > 1:
      k = _cp_all_gather(k, dim=1, cp_config=cp_config)
      v = _cp_all_gather(v, dim=1, cp_config=cp_config)
    return self.o(self.attn(q, k, v))


class AdaLayerNorm(nn.Module):
  """AdaLN modulation driven by the pooled conditioning features (only for ``pooled_adaln``).

  :param embedding_dim: Conditioning width.
  :param output_dim: Output width of the modulation MLP.
  :param norm_eps: LayerNorm epsilon.
  :param chunk_dim: Axis the modulation is chunked on.
  """

  def __init__(
    self,
    embedding_dim: int,
    output_dim: Optional[int] = None,
    norm_eps: float = 1e-5,
    chunk_dim: int = 2,
  ):
    super().__init__()
    self.chunk_dim = chunk_dim
    output_dim = output_dim or embedding_dim * 2
    self.silu = nn.SiLU()
    self.linear = nn.Linear(embedding_dim, output_dim)
    self.norm = nn.LayerNorm(output_dim // 2, norm_eps, False)

  def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
    """Apply the modulation.

    :param x: Hidden states.
    :param temb: Pooled conditioning features.
    :returns: Modulated hidden states.
    """

    temb = self.linear(self.silu(temb))
    scale, shift = temb.chunk(2, dim=self.chunk_dim)
    return self.norm(x) * (1 + scale) + shift


class ConditionalCrossAttentionBlock(nn.Module):
  """Layer-norm the conditioning tower, then run the conditional cross attention.

  :param dim: Query width.
  :param kv_dim: Key/value width.
  :param num_heads: Head count.
  :param eps: Normalization epsilon.
  :param pooled_adaln: Whether to modulate the query with per-frame pooled features.
  """

  def __init__(self,
               dim: int,
               kv_dim: int,
               num_heads: int,
               eps: float = 1e-6,
               pooled_adaln: bool = False):
    super().__init__()
    self.y_norm = nn.LayerNorm(kv_dim, eps=eps)
    self.inner = ConditionalCrossAttention(dim=dim, kv_dim=kv_dim, num_heads=num_heads, eps=eps)
    self.pooled_adaln = pooled_adaln
    if pooled_adaln:
      self.per_frame_pooling = PerFrameAttentionPooling(kv_dim, num_heads=num_heads, eps=eps)
      self.adaln = AdaLayerNorm(kv_dim, output_dim=dim * 2, chunk_dim=2)

  def forward(self,
              x: torch.Tensor,
              y: torch.Tensor,
              x_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
              y_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
              video_grid_size: Optional[Tuple[int, int, int]] = None,
              cp_config=None,
              gather_kv: bool = False) -> torch.Tensor:
    """Run the conditional cross attention.

    ``y_norm`` stays replicated under tensor parallelism: it normalizes the conditioning tower's
    width, which is never sharded along the primary tower's feature axis.

    :param x: Primary hidden states.
    :param y: Conditioning hidden states.
    :param x_freqs: Optional ``(cos, sin)`` for the query.
    :param y_freqs: Optional ``(cos, sin)`` for the key.
    :param video_grid_size: ``(T, H, W)``, required when ``pooled_adaln`` is set.
    :param cp_config: Armed context-parallel config, or None.
    :param gather_kv: Whether the key/value shards must be gathered.
    :returns: Attention output for the primary tower.
    """

    if self.pooled_adaln:
      assert video_grid_size is not None, "video_grid_size must not be None"
      pooled_y = self.per_frame_pooling(y, video_grid_size)
      if pooled_y.shape[1] != x.shape[1]:
        pooled_y = F.interpolate(
          pooled_y.permute(0, 2, 1),
          size=x.shape[1],
          mode="linear",
          align_corners=False,
        ).permute(0, 2, 1)
      x = self.adaln(x, temb=pooled_y)
    return self.inner(
      x=x,
      y=self.y_norm(y),
      x_freqs=x_freqs,
      y_freqs=y_freqs,
      cp_config=cp_config,
      gather_kv=gather_kv,
    )


def apply_bridge_directions(a2v_conditioner,
                            v2a_conditioner,
                            condition_scale: float,
                            visual_hidden_states: torch.Tensor,
                            audio_hidden_states: torch.Tensor,
                            *,
                            visual_rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                            audio_rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                            a2v_condition_scale: Optional[float] = None,
                            v2a_condition_scale: Optional[float] = None,
                            video_grid_size: Optional[Tuple[int, int, int]] = None,
                            cp_config=None) -> Tuple[torch.Tensor, torch.Tensor]:
  """Condition both towers for one layer, in the upstream order.

  Both directions read the states that *enter* the layer: the video->audio direction must not see the
  audio->visual update of the same layer. This is the single implementation shared by
  :meth:`DualTowerConditionalBridge.forward` and the fused block, so the cached path and the plain
  path cannot drift apart.

  :param a2v_conditioner: Audio->video conditioner of this layer, or None.
  :param v2a_conditioner: Video->audio conditioner of this layer, or None.
  :param condition_scale: Fallback conditioning strength.
  :param visual_hidden_states: Video hidden states entering the layer.
  :param audio_hidden_states: Audio hidden states entering the layer.
  :param visual_rope: ``(cos, sin)`` of the video stream.
  :param audio_rope: ``(cos, sin)`` of the audio stream.
  :param a2v_condition_scale: Audio->video strength override.
  :param v2a_condition_scale: Video->audio strength override.
  :param video_grid_size: ``(T, H, W)``, required when ``pooled_adaln`` is set.
  :param cp_config: Armed context-parallel config, or None.
  :returns: ``(visual, audio)`` after conditioning.
  """

  visual_conditioned = visual_hidden_states
  if a2v_conditioner is not None:
    scale = a2v_condition_scale if a2v_condition_scale is not None else condition_scale
    visual_conditioned = visual_hidden_states + a2v_conditioner(
      x=visual_hidden_states,
      y=audio_hidden_states,
      x_freqs=visual_rope,
      y_freqs=audio_rope,
      video_grid_size=video_grid_size,
      cp_config=cp_config,
      gather_kv=False,
    ) * scale

  audio_conditioned = audio_hidden_states
  if v2a_conditioner is not None:
    scale = v2a_condition_scale if v2a_condition_scale is not None else condition_scale
    audio_conditioned = audio_hidden_states + v2a_conditioner(
      x=audio_hidden_states,
      y=visual_hidden_states,
      x_freqs=audio_rope,
      y_freqs=visual_rope,
      video_grid_size=video_grid_size,
      cp_config=cp_config,
      gather_kv=True,
    ) * scale

  return visual_conditioned, audio_conditioned


class DualTowerConditionalBridge(ModelMixin, ConfigMixin):
  """Per-layer bidirectional conditioning between the video and audio towers.

  :param visual_layers: Number of video layers.
  :param audio_layers: Number of audio layers.
  :param visual_hidden_dim: Video hidden width.
  :param audio_hidden_dim: Audio hidden width.
  :param audio_fps: Audio latent frame rate.
  :param head_dim: Attention head width used to derive both head counts.
  :param interaction_strategy: Layer selection strategy.
  :param apply_cross_rope: Whether the cross attentions are rotary-embedded.
  :param apply_first_frame_bias_in_rope: Whether the first video frame gets the FPS bias.
  :param trainable_condition_scale: Whether the condition scale is a parameter.
  :param pooled_adaln: Whether the audio->video direction uses pooled adaLN.
  """

  _repeated_blocks = ("ConditionalCrossAttentionBlock", )

  @register_to_config
  def __init__(self,
               visual_layers: int = 30,
               audio_layers: int = 30,
               visual_hidden_dim: int = 3072,
               audio_hidden_dim: int = 1536,
               audio_fps: float = 44100.0 / 2048.0,
               head_dim: int = 128,
               interaction_strategy: str = "shallow_focus",
               apply_cross_rope: bool = False,
               apply_first_frame_bias_in_rope: bool = False,
               trainable_condition_scale: bool = False,
               pooled_adaln: bool = False):
    super().__init__()

    self.visual_hidden_dim = visual_hidden_dim
    self.audio_hidden_dim = audio_hidden_dim
    self.audio_fps = audio_fps
    self.head_dim = head_dim
    self.apply_cross_rope = apply_cross_rope
    self.apply_first_frame_bias_in_rope = apply_first_frame_bias_in_rope
    self.trainable_condition_scale = trainable_condition_scale
    self.pooled_adaln = pooled_adaln
    if self.trainable_condition_scale:
      self.condition_scale = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    else:
      self.condition_scale = 1.0

    self.controller = CrossModalInteractionController(visual_layers, audio_layers)
    self.interaction_mapping = self.controller.get_interaction_layers(interaction_strategy)

    self.audio_to_video_conditioners = nn.ModuleDict()
    self.video_to_audio_conditioners = nn.ModuleDict()

    self.rotary = RotaryEmbedding(base=10000.0, dim=head_dim)
    for v_layer, _ in self.interaction_mapping["a2v"]:
      self.audio_to_video_conditioners[str(v_layer)] = ConditionalCrossAttentionBlock(
        dim=visual_hidden_dim,
        kv_dim=audio_hidden_dim,
        num_heads=visual_hidden_dim // head_dim,
        pooled_adaln=False,
      )
    for a_layer, _ in self.interaction_mapping["v2a"]:
      self.video_to_audio_conditioners[str(a_layer)] = ConditionalCrossAttentionBlock(
        dim=audio_hidden_dim,
        kv_dim=visual_hidden_dim,
        num_heads=audio_hidden_dim // head_dim,
        pooled_adaln=self.pooled_adaln,
      )

  @torch.no_grad()
  def build_aligned_freqs(
    self,
    video_fps: float,
    grid_size: Tuple[int, int, int],
    audio_steps: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
  ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Build the cross-modal rotary tables, aligning video frames to audio steps.

    :param video_fps: Request frame rate.
    :param grid_size: Video token grid ``(f, h, w)``.
    :param audio_steps: Number of audio latent frames.
    :param device: Device of the tables.
    :param dtype: Dtype of the tables.
    :returns: ``((cos_v, sin_v), (cos_a, sin_a))``.
    """

    f_v, h, w = grid_size
    length_v = f_v * h * w
    length_a = int(audio_steps)

    device = device or next(self.parameters()).device
    dtype = dtype or torch.float32

    audio_pos = torch.arange(length_a, device=device, dtype=torch.float32).unsqueeze(0)
    if self.apply_first_frame_bias_in_rope:
      video_effective_fps = float(video_fps) / 4.0
      if f_v > 0:
        t_starts = torch.zeros((f_v, ), device=device, dtype=torch.float32)
        if f_v > 1:
          t_starts[1:] = (1.0 / float(video_fps)) + torch.arange(
            f_v - 1, device=device, dtype=torch.float32) * (1.0 / video_effective_fps)
      else:
        t_starts = torch.zeros((0, ), device=device, dtype=torch.float32)
      video_pos_per_frame = t_starts * float(self.audio_fps)
    else:
      scale = float(self.audio_fps) / float(video_fps / 4.0)
      video_pos_per_frame = torch.arange(f_v, device=device, dtype=torch.float32) * scale
    video_pos = video_pos_per_frame.repeat_interleave(h * w).unsqueeze(0)

    dummy_v = torch.zeros((1, length_v, self.head_dim), device=device, dtype=dtype)
    dummy_a = torch.zeros((1, length_a, self.head_dim), device=device, dtype=dtype)
    cos_v, sin_v = self.rotary(dummy_v, position_ids=video_pos)
    cos_a, sin_a = self.rotary(dummy_a, position_ids=audio_pos)
    return (cos_v, sin_v), (cos_a, sin_a)

  def should_interact(self, layer_idx: int, direction: str) -> bool:
    """Whether ``layer_idx`` interacts in ``direction``.

    :param layer_idx: Zero-based layer index.
    :param direction: ``"v2a"`` or ``"a2v"``.
    :returns: True when the layer interacts.
    """

    return self.controller.should_interact(layer_idx, direction, self.interaction_mapping)

  def apply_conditional_control(self,
                                layer_idx: int,
                                direction: str,
                                primary_hidden_states: torch.Tensor,
                                condition_hidden_states: torch.Tensor,
                                x_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                                y_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                                condition_scale: Optional[float] = None,
                                video_grid_size: Optional[Tuple[int, int, int]] = None,
                                cp_config=None) -> torch.Tensor:
    """Condition one tower on the other, in one direction.

    :param layer_idx: Zero-based layer index.
    :param direction: ``"a2v"`` or ``"v2a"``.
    :param primary_hidden_states: States of the tower being conditioned.
    :param condition_hidden_states: States of the conditioning tower.
    :param x_freqs: ``(cos, sin)`` for the primary tower.
    :param y_freqs: ``(cos, sin)`` for the conditioning tower.
    :param condition_scale: Overrides the module's own scale when given.
    :param video_grid_size: ``(T, H, W)``, required when ``pooled_adaln`` is set.
    :param cp_config: Armed context-parallel config, or None.
    :returns: The conditioned states of the primary tower.
    """

    if not self.controller.should_interact(layer_idx, direction, self.interaction_mapping):
      return primary_hidden_states

    if direction == "a2v":
      conditioner = self.audio_to_video_conditioners[str(layer_idx)]
    elif direction == "v2a":
      conditioner = self.video_to_audio_conditioners[str(layer_idx)]
    else:
      raise ValueError(f"Invalid direction: {direction}")

    conditioned = conditioner(
      x=primary_hidden_states,
      y=condition_hidden_states,
      x_freqs=x_freqs,
      y_freqs=y_freqs,
      video_grid_size=video_grid_size,
      cp_config=cp_config,
      gather_kv=direction == "v2a",
    )
    scale = condition_scale if condition_scale is not None else self.condition_scale
    return primary_hidden_states + conditioned * scale

  def forward(self,
              layer_idx: int,
              visual_hidden_states: torch.Tensor,
              audio_hidden_states: torch.Tensor,
              *,
              x_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
              y_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
              a2v_condition_scale: Optional[float] = None,
              v2a_condition_scale: Optional[float] = None,
              condition_scale: Optional[float] = None,
              video_grid_size: Optional[Tuple[int, int, int]] = None,
              cp_config=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply both directions to one layer.

    Both directions read the *incoming* states: the visual->audio direction must not see the
    audio->visual update of the same layer.

    :param layer_idx: Zero-based layer index.
    :param visual_hidden_states: Video hidden states entering this layer.
    :param audio_hidden_states: Audio hidden states entering this layer.
    :param x_freqs: ``(cos, sin)`` for the video stream.
    :param y_freqs: ``(cos, sin)`` for the audio stream.
    :param a2v_condition_scale: Audio->video strength.
    :param v2a_condition_scale: Video->audio strength.
    :param condition_scale: Fallback strength for both directions.
    :param video_grid_size: ``(T, H, W)``, required when ``pooled_adaln`` is set.
    :param cp_config: Armed context-parallel config, or None.
    :returns: ``(visual, audio)`` after conditioning.
    """

    key = str(layer_idx)
    a2v = (self.audio_to_video_conditioners[key] if self.should_interact(layer_idx, "a2v")
           and key in self.audio_to_video_conditioners else None)
    v2a = (self.video_to_audio_conditioners[key] if self.should_interact(layer_idx, "v2a")
           and key in self.video_to_audio_conditioners else None)
    return apply_bridge_directions(
      a2v,
      v2a,
      self.condition_scale if condition_scale is None else condition_scale,
      visual_hidden_states,
      audio_hidden_states,
      visual_rope=x_freqs,
      audio_rope=y_freqs,
      a2v_condition_scale=a2v_condition_scale,
      v2a_condition_scale=v2a_condition_scale,
      video_grid_size=video_grid_size,
      cp_config=cp_config,
    )
