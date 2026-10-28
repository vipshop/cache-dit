"""Built-in HiDream-O1-Image transformer.

HiDream-O1-Image has no diffusers implementation: the checkpoint is a Qwen3-VL-8B decoder repurposed
as a pixel-level DiT (`architectures=["Qwen3VLForConditionalGeneration"]`, `model_type="qwen3_vl"`),
so `DiffusionPipeline.from_pretrained` cannot import it. This module is the vendored denoiser.

Architecture notes that the cache-dit wiring depends on:

* **Flat block stack.** `layers` is a top-level `nn.ModuleList`, so `mock_transformer` can replace it
  with a one-element list and the block loop replays exactly one block call per iteration.
* **Timestep injection is outside the block loop.** The timestep is encoded by `t_embedder1` and
  written into the `<|tms_token|>` row of the token embeddings; blocks carry no modulation. The
  first block therefore still sees the timestep through its input, which is what makes residual-diff
  caching meaningful here, but there is no per-block `temb` to keep in sync.
* **Attention is mixed causal/full.** Text rows are causal, the timestep row and all image rows
  attend to the whole sequence. `attention_mask` is a dense additive `[B, 1, S, S]` tensor built by
  the pipeline (the release falls back to exactly this form when the backend has no piecewise-span
  support) and it is passed to every block unchanged.
* **No VAE and no text encoder.** Raw RGB patches go in through `x_embedder`, `final_layer2` emits
  `patch_size * patch_size * 3` values per token, and the token ids are embedded by this module's own
  `embed_tokens`. `--parallel-vae` / `--parallel-text` therefore do not apply.
* **GQA is widened before attention.** `num_key_value_heads=8` vs `num_attention_heads=32`; K/V are
  repeated to the query head count and attention runs as MHA with `enable_gqa=False`, because
  cache-dit's templated context-parallel attention rejects `enable_gqa=True`.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
  """Rotate the last dimension in half so `q * cos + rotate_half(q) * sin` applies RoPE.

  :param x: Tensor whose last dimension is `head_dim`.
  :returns: The half-rotated tensor.
  """
  x1 = x[..., :x.shape[-1] // 2]
  x2 = x[..., x.shape[-1] // 2:]
  return torch.cat((-x2, x1), dim=-1)


class HiDreamO1ImageRMSNorm(nn.Module):
  """RMSNorm with the reference's fp32 statistics and dtype-cast before the weight multiply."""

  def __init__(self, dim: int, eps: float = 1e-6):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(dim))
    self.variance_epsilon = eps

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Normalize the last dimension.

    :param hidden_states: Input states `[..., dim]`.
    :returns: Normalized states with the input dtype.
    """
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)


class HiDreamO1ImageRotaryEmbedding(nn.Module):
  """Interleaved multi-modal RoPE table used by the Qwen3-VL backbone.

  `position_ids` has shape `[3, B, S]` and holds one row per `(t, h, w)` axis; the three axes are
  interleaved into a single frequency vector with `mrope_section`. The table is packed as
  `cat(cos, sin, -1)` so that context parallelism can split one plain tensor (a `(cos, sin)` tuple
  cannot be hooked).
  """

  def __init__(
      self,
      head_dim: int,
      rope_theta: float = 5_000_000.0,
      mrope_section=(24, 20, 20),
  ):
    super().__init__()
    exponent = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
    inv_freq = 1.0 / (float(rope_theta) ** exponent)
    self.register_buffer("inv_freq", inv_freq, persistent=False)
    self.mrope_section = tuple(mrope_section)

  def forward(self, position_ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Build the packed RoPE table.

    :param position_ids: Position ids `[3, B, S]`.
    :param dtype: Output dtype.
    :returns: Packed table `[B, S, 2 * head_dim]`.
    """
    if position_ids.ndim == 2:
      position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    inv_freq = self.inv_freq[None, None, :, None].float().to(position_ids.device)
    inv_freq = inv_freq.expand(3, position_ids.shape[1], -1, 1)
    positions = position_ids[:, :, None, :].float()
    with torch.autocast(device_type=position_ids.device.type, enabled=False):
      freqs = (inv_freq @ positions).transpose(2, 3)
      freqs_t = freqs[0]
      for dim, offset in enumerate((1, 2), start=1):
        length = self.mrope_section[dim] * 3
        freqs_t[..., offset:length:3] = freqs[dim, ..., offset:length:3]
      emb = torch.cat((freqs_t, freqs_t), dim=-1)
      cos = emb.cos()
      sin = emb.sin()
    return torch.cat((cos, sin), dim=-1).to(dtype)


class HiDreamO1ImageAttention(nn.Module):
  """Self-attention over the joint `[text, timestep, image]` sequence.

  Q/K/V stay in `[B, S, heads, head_dim]`; cache-dit's dispatcher expects that layout and transposes
  internally, while feeding it SDPA's `[B, heads, S, head_dim]` silently produces a coherent but
  wrong image.
  """

  def __init__(
    self,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rms_norm_eps: float = 1e-6,
  ):
    super().__init__()
    self.num_heads = num_heads
    self.num_kv_heads = num_kv_heads
    self.head_dim = head_dim
    self.scaling = head_dim ** -0.5
    self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
    self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
    self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
    self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
    self.q_norm = HiDreamO1ImageRMSNorm(head_dim, eps=rms_norm_eps)
    self.k_norm = HiDreamO1ImageRMSNorm(head_dim, eps=rms_norm_eps)

  def forward(
    self,
    hidden_states: torch.Tensor,
    rope: torch.Tensor,
    attention_mask: torch.Tensor = None,
    cp_config=None,
  ) -> torch.Tensor:
    """Attend over the joint sequence.

    :param hidden_states: Joint states `[B, S, hidden]`.
    :param rope: Packed RoPE table `[B, S, 2 * head_dim]` (local chunk under CP).
    :param attention_mask: Dense additive mask `[B, 1, S_full, S_full]`.
    :param cp_config: cache-dit context-parallel config, `None` for plain attention.
    :returns: Attention output `[B, S, hidden]`.
    """
    from ....attention import _dispatch_attention_fn

    batch_size, seq_len, _ = hidden_states.shape
    if attention_mask.shape[-1] != seq_len and cp_config is None:
      # The mask always spans the full joint sequence and is never split, so a shorter local
      # sequence without a CP config means the context-parallel split leaked in (or `_cp_config`
      # was lost): attention would otherwise happily mask full-length keys against local queries.
      raise RuntimeError(f"attention mask length {attention_mask.shape[-1]} does not match the "
                         f"local sequence {seq_len} without a context-parallel config.")
    query = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
    key = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
    value = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

    # Per-head QK normalization runs after the head reshape, so it always sees `head_dim` (never the
    # tensor-parallel shard width).
    query = self.q_norm(query)
    key = self.k_norm(key)

    cos, sin = rope.chunk(2, dim=-1)
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    query = query * cos + _rotate_half(query) * sin
    key = key * cos + _rotate_half(key) * sin

    # Widen GQA to MHA: cache-dit's templated CP attention rejects `enable_gqa=True`, and the
    # repeated-KV path is the faster SDPA kernel for this head ratio.
    if self.num_kv_heads != self.num_heads:
      repeat = self.num_heads // self.num_kv_heads
      key = key.repeat_interleave(repeat, dim=2)
      value = value.repeat_interleave(repeat, dim=2)

    attn_output = _dispatch_attention_fn(
      query,
      key,
      value,
      attn_mask=attention_mask,
      dropout_p=0.0,
      is_causal=False,
      scale=self.scaling,
      cp_config=cp_config,
    )
    return self.o_proj(attn_output.reshape(batch_size, seq_len, -1))


class HiDreamO1ImageMLP(nn.Module):
  """SwiGLU feed-forward used by the Qwen3-VL decoder."""

  def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str = "silu"):
    super().__init__()
    if hidden_act != "silu":
      raise ValueError(f"Unsupported hidden_act={hidden_act!r}; HiDream-O1-Image uses `silu`.")
    self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
    self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
    self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Apply the gated MLP.

    :param hidden_states: Input states `[B, S, hidden]`.
    :returns: Output states `[B, S, hidden]`.
    """
    return self.down_proj(
      nn.functional.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class HiDreamO1ImageDecoderLayer(nn.Module):
  """One pre-norm decoder block: attention then SwiGLU MLP, both residual."""

  def __init__(
    self,
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rms_norm_eps: float = 1e-6,
    hidden_act: str = "silu",
  ):
    super().__init__()
    self.self_attn = HiDreamO1ImageAttention(
      hidden_size=hidden_size,
      num_heads=num_heads,
      num_kv_heads=num_kv_heads,
      head_dim=head_dim,
      rms_norm_eps=rms_norm_eps,
    )
    self.mlp = HiDreamO1ImageMLP(hidden_size, intermediate_size, hidden_act=hidden_act)
    self.input_layernorm = HiDreamO1ImageRMSNorm(hidden_size, eps=rms_norm_eps)
    self.post_attention_layernorm = HiDreamO1ImageRMSNorm(hidden_size, eps=rms_norm_eps)

  def forward(
    self,
    hidden_states: torch.Tensor,
    rope: torch.Tensor,
    attention_mask: torch.Tensor = None,
    cp_config=None,
  ) -> torch.Tensor:
    """Run one block over the joint sequence.

    :param hidden_states: Joint states `[B, S, hidden]`.
    :param rope: Packed RoPE table `[B, S, 2 * head_dim]`.
    :param attention_mask: Dense additive mask `[B, 1, S_full, S_full]`.
    :param cp_config: cache-dit context-parallel config, `None` for plain attention.
    :returns: Joint states `[B, S, hidden]`.
    """
    residual = hidden_states
    hidden_states = self.self_attn(
      self.input_layernorm(hidden_states),
      rope=rope,
      attention_mask=attention_mask,
      cp_config=cp_config,
    )
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
    return residual + hidden_states


class HiDreamO1ImageTimestepEmbedder(nn.Module):
  """Sinusoidal timestep embedding followed by a two-layer MLP."""

  def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(frequency_embedding_size, hidden_size, bias=True),
      nn.SiLU(),
      nn.Linear(hidden_size, hidden_size, bias=True),
    )
    self.frequency_embedding_size = frequency_embedding_size

  @staticmethod
  def timestep_embedding(timestep: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Build the sinusoidal embedding.

    :param timestep: Timesteps `[B]`.
    :param dim: Embedding width.
    :param max_period: Maximum period of the sinusoids.
    :returns: Embedding `[B, dim]`.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) *
                      torch.arange(0, half, dtype=torch.float32, device=timestep.device) / half)
    args = timestep[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
      embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

  def forward(self, timestep: torch.Tensor) -> torch.Tensor:
    """Embed the denoising timestep.

    :param timestep: Timesteps in `[0, 1]`, as consumed by the reference (`t * 1000` internally).
    :returns: Embedding `[B, hidden]`.
    """
    embedding = self.timestep_embedding(timestep * 1000, self.frequency_embedding_size)
    return self.mlp(embedding.to(self.mlp[0].weight.dtype))


class HiDreamO1ImagePatchEmbed(nn.Module):
  """Bottleneck projection of flattened `patch_size * patch_size * in_channels` RGB patches."""

  def __init__(self, hidden_size: int, patch_size: int = 32, in_channels: int = 3):
    super().__init__()
    bottleneck_dim = hidden_size // 4
    self.proj1 = nn.Linear(patch_size * patch_size * in_channels, bottleneck_dim, bias=False)
    self.proj2 = nn.Linear(bottleneck_dim, hidden_size, bias=True)

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Project pixel patches into the hidden space.

    :param hidden_states: Patches `[B, S, patch_size * patch_size * in_channels]`.
    :returns: Token embeddings `[B, S, hidden]`.
    """
    return self.proj2(self.proj1(hidden_states))


class HiDreamO1ImageFinalLayer(nn.Module):
  """Output head mapping hidden states back to flattened RGB patches."""

  def __init__(self, hidden_size: int, patch_size: int = 32, out_channels: int = 3):
    super().__init__()
    self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Predict the clean pixel patches.

    :param hidden_states: Joint states `[B, S, hidden]`.
    :returns: Predicted patches `[B, S, patch_size * patch_size * out_channels]`.
    """
    return self.linear(hidden_states)


class HiDreamO1ImageTransformer2DModel(ModelMixin, ConfigMixin):
  """Pixel-space denoiser of HiDream-O1-Image.

  Note:
    `layers` is the denoising stack cache-dit caches. `rope_split` and `sequence_gather` are the
    context-parallel identity markers: the RoPE table is output-split after being built on the full
    sequence, and the joint sequence is gathered back before the output head. Neither marker holds
    parameters, so the state dict is unaffected.
  """

  _supports_gradient_checkpointing = False
  # Denoising stack discovered by regional quantization and `compile_repeated_blocks()`.
  _repeated_blocks = ["HiDreamO1ImageDecoderLayer"]
  # cache-dit's Fn-residual replay re-runs the first block, so the CP input split lives there.

  @register_to_config
  def __init__(
    self,
    hidden_size: int = 4096,
    intermediate_size: int = 12288,
    num_layers: int = 36,
    num_attention_heads: int = 32,
    num_key_value_heads: int = 8,
    head_dim: int = 128,
    vocab_size: int = 151936,
    rms_norm_eps: float = 1e-6,
    hidden_act: str = "silu",
    rope_theta: float = 5_000_000.0,
    mrope_section=(24, 20, 20),
    patch_size: int = 32,
    in_channels: int = 3,
    tms_token_id: int = 151673,
    frequency_embedding_size: int = 256,
  ):
    super().__init__()
    if hidden_size // num_attention_heads != head_dim:
      raise ValueError(f"hidden_size={hidden_size} and num_attention_heads={num_attention_heads} "
                       f"contradict head_dim={head_dim}.")
    if num_attention_heads % num_key_value_heads:
      raise ValueError(f"num_attention_heads={num_attention_heads} must be a multiple of "
                       f"num_key_value_heads={num_key_value_heads}.")

    self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
    self.layers = nn.ModuleList([
      HiDreamO1ImageDecoderLayer(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_heads=num_attention_heads,
        num_kv_heads=num_key_value_heads,
        head_dim=head_dim,
        rms_norm_eps=rms_norm_eps,
        hidden_act=hidden_act,
      ) for _ in range(num_layers)
    ])
    self.norm = HiDreamO1ImageRMSNorm(hidden_size, eps=rms_norm_eps)
    self.rotary_emb = HiDreamO1ImageRotaryEmbedding(
      head_dim=head_dim,
      rope_theta=rope_theta,
      mrope_section=mrope_section,
    )
    self.t_embedder1 = HiDreamO1ImageTimestepEmbedder(hidden_size, frequency_embedding_size)
    self.x_embedder = HiDreamO1ImagePatchEmbed(
      hidden_size,
      patch_size=patch_size,
      in_channels=in_channels,
    )
    self.final_layer2 = HiDreamO1ImageFinalLayer(
      hidden_size,
      patch_size=patch_size,
      out_channels=in_channels,
    )

    # Context-parallel markers (see the class docstring).
    self.rope_split = nn.Identity()
    self.sequence_gather = nn.Identity()

  @property
  def num_blocks(self) -> int:
    """Depth of the denoising stack, taken from the config (the stack may be mocked to length 1)."""
    return self.config.num_layers

  def forward(
    self,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    vinputs: torch.Tensor,
    timestep: torch.Tensor,
    attention_mask: torch.Tensor = None,
    return_dict: bool = True,
  ):
    """Predict the clean pixel patches for one denoising step.

    :param input_ids: Text side token ids `[B, T]`, ending with `<|boi_token|>` and the timestep
      token.
    :param position_ids: Interleaved mRoPE ids `[3, B, T + L]` covering the joint sequence.
    :param vinputs: Noisy pixel patches `[B, L, patch_size * patch_size * 3]`.
    :param timestep: Denoising timestep in `[0, 1]`, shape `[B]` or `[1]`.
    :param attention_mask: Dense additive mask `[B, 1, T + L, T + L]`; text rows are causal, the
      timestep row and image rows are full.
    :param return_dict: Return a `Transformer2DModelOutput` instead of a bare tensor.
    :returns: Predicted patches for the whole joint sequence.
    """
    if attention_mask is None:
      # Omitting the mask silently turns every row bidirectional, which lets text tokens attend to
      # future tokens and to the whole image: coherent-looking but wrong output.
      raise ValueError("HiDream-O1-Image requires the mixed causal/full attention mask.")
    dtype = self.embed_tokens.weight.dtype
    inputs_embeds = self.embed_tokens(input_ids)

    timestep_embeds = self.t_embedder1(timestep.reshape(-1).to(inputs_embeds.device))
    timestep_mask = (input_ids == self.config.tms_token_id).unsqueeze(-1)
    inputs_embeds = torch.where(
      timestep_mask,
      timestep_embeds.unsqueeze(1).expand_as(inputs_embeds),
      inputs_embeds,
    )

    image_embeds = self.x_embedder(vinputs.to(dtype))
    hidden_states = torch.cat([inputs_embeds, image_embeds], dim=1)

    # The table is built on the full sequence and then output-split by the CP hook, so every block
    # receives the RoPE of its own chunk.
    rope = self.rope_split(self.rotary_emb(position_ids, dtype))

    cp_config = getattr(self, "_cp_config", None)
    for layer in self.layers:
      hidden_states = layer(
        hidden_states,
        rope=rope,
        attention_mask=attention_mask,
        cp_config=cp_config,
      )

    hidden_states = self.sequence_gather(hidden_states)
    x_pred = self.final_layer2(self.norm(hidden_states))
    if not return_dict:
      return (x_pred, )
    return Transformer2DModelOutput(sample=x_pred)
