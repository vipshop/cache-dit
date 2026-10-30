"""Vendored LingBot-Video dense transformer.

Ported from the official `lingbot_video/transformer_lingbot_video.py`
(https://github.com/Robbyant/lingbot-video, Apache-2.0, Copyright Ant Group Co., Ltd.).
Only the **dense** variant
(`num_experts == 0`) is vendored: the MoE router, the triton/sglang expert kernels and the
FlashAttention-3 `packed_indices` path are dropped, because
  * the released Dense 1.3B checkpoint is dense (`transformer/config.json: num_experts=0`), and
  * cache-dit always runs one sample per forward (the pipeline issues one forward per CFG branch),
    so the packed-varlen branch is dead code that would otherwise pin a FA-3 dependency.

Note: keep the constraints below intact, cache-dit's dispatch depends on them.
  * `self.blocks` stays a top-level `nn.ModuleList` of `LingBotVideoBlock`.
  * `LingBotVideoBlock.forward` keeps `hidden_states` as its first positional argument and returns a
    single tensor (cache-dit's `Pattern_3` contract), with the residual added inside the block.
  * `temb6` (token-wise modulation) is computed *before* the block loop from `timestep`, so a cached
    step still gets a fresh modulation signal.
  * `_keep_in_fp32_modules` and the `to()` override stay: the norms, the timestep MLP and the
    per-block `scale_shift_table` must stay fp32 or the sample drifts.
  * The four `nn.Identity()` markers (`cp_rotary`, `cp_temb_input`, `cp_temb6`, `cp_out`) are the
    context-parallel split/gather boundaries; they carry no parameters.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin

from ....attention import _dispatch_attention_fn

LINGBOT_VIDEO_FP32_MODULES = (
  "time_embedder",
  "time_modulation",
  "scale_shift_table",
  "norm",
  "norm1",
  "norm2",
  "norm_q",
  "norm_k",
  "norm_post_attn",
  "norm_post_ffn",
  "norm_out",
  "norm_out_modulation",
)


def should_keep_in_fp32(name: str) -> bool:
  return any(module_name in name.split(".") for module_name in LINGBOT_VIDEO_FP32_MODULES)


class LingBotVideoRMSNorm(nn.Module):
  """RMSNorm with fp32 accumulation."""

  def __init__(self, dim: int, eps: float = 1e-6):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(dim))
    self.variance_epsilon = eps

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return (self.weight * hidden_states).to(input_dtype)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
  """Apply complex RoPE to `(B, S, H, D)` attention tensors."""
  with torch.amp.autocast("cuda", enabled=False):
    x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    out = torch.view_as_real(x_c * freqs_cis.unsqueeze(2)).flatten(3)
    return out.type_as(x)


class LingBotVideoRotaryEmbedding(nn.Module):
  """Complex64 RoPE table indexed by 3-D position ids."""

  def __init__(self, axes_dims: Tuple[int, ...], axes_lens: Tuple[int, ...], theta: float):
    super().__init__()
    self.axes_dims = tuple(axes_dims)
    self.axes_lens = list(axes_lens)
    self.theta = theta
    self.freqs_cis = None

  @staticmethod
  def precompute_freqs_cis(dim: Tuple[int, ...], end: Tuple[int, ...], theta: float):
    freqs_cis = []
    for d, e in zip(dim, end):
      freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d))
      timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
      freqs = torch.outer(timestep, freqs).float()
      freqs_cis.append(torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64))
    return freqs_cis

  def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
    # position_ids: (S, 3) int -> (S, head_dim / 2) complex64
    device = position_ids.device
    max_vals = position_ids.max(dim=0).values.tolist()
    needs_rebuild = self.freqs_cis is None or any(m >= l for m, l in zip(max_vals, self.axes_lens))
    if needs_rebuild:
      for i in range(len(self.axes_lens)):
        if max_vals[i] >= self.axes_lens[i]:
          self.axes_lens[i] = int(max_vals[i] * 1.5) + 1
      self.freqs_cis = self.precompute_freqs_cis(
        self.axes_dims,
        tuple(self.axes_lens),
        theta=self.theta,
      )
      self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]
    elif self.freqs_cis[0].device != device:
      self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]

    return torch.cat(
      [self.freqs_cis[i][position_ids[:, i]] for i in range(len(self.axes_dims))],
      dim=-1,
    )


def make_joint_position_ids(
  text_len: int,
  grid_t: int,
  grid_h: int,
  grid_w: int,
  device: torch.device,
) -> torch.Tensor:
  """3-D positions in `[video; text]` order.

  Text tokens occupy `t = 1..text_len`; the video `t` axis starts at `text_len + 1`, so the two
  modalities never share a rotary phase.

  :param text_len: Number of valid text tokens for this sample.
  :param grid_t: Latent frames after patching.
  :param grid_h: Latent height after patching.
  :param grid_w: Latent width after patching.
  :param device: Device of the returned index tensor.
  :returns: `(grid_t * grid_h * grid_w + text_len, 3)` int32 position ids.
  """
  tt = torch.arange(grid_t, device=device, dtype=torch.int32) + (text_len + 1)
  hh = torch.arange(grid_h, device=device, dtype=torch.int32)
  ww = torch.arange(grid_w, device=device, dtype=torch.int32)
  grid = torch.stack(torch.meshgrid(tt, hh, ww, indexing="ij"), dim=-1).flatten(0, 2)
  text_t = torch.arange(text_len, device=device, dtype=torch.int32) + 1
  text_pos = torch.stack([text_t, torch.zeros_like(text_t), torch.zeros_like(text_t)], dim=-1)
  return torch.cat([grid, text_pos], dim=0)


class LingBotVideoTextEmbedder(nn.Module):
  """Text conditioner: RMSNorm -> Linear -> SiLU -> Linear."""

  def __init__(self, text_dim: int, hidden_size: int):
    super().__init__()
    self.norm = LingBotVideoRMSNorm(text_dim, eps=1e-6)
    self.linear_1 = nn.Linear(text_dim, hidden_size, bias=True)
    self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.norm(x)
    return self.linear_2(F.silu(self.linear_1(x)))


class LingBotVideoAttention(nn.Module):
  """Joint self-attention with per-head q/k RMSNorm and complex RoPE.

  The tensor keeps the `(B, S, H, D)` layout: cache-dit's context-parallel dispatcher takes that
  layout as-is, while SDPA wants `(B, H, S, D)` and is transposed inside the backend.
  """

  def __init__(self, hidden_size, num_heads, norm_eps, qkv_bias, out_bias):
    super().__init__()
    self.num_heads = num_heads
    self.head_dim = hidden_size // num_heads
    self.to_q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
    self.to_k = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
    self.to_v = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
    self.norm_q = LingBotVideoRMSNorm(self.head_dim, norm_eps)
    self.norm_k = LingBotVideoRMSNorm(self.head_dim, norm_eps)
    self.to_out = nn.Linear(hidden_size, hidden_size, bias=out_bias)

  def forward(self, x, rotary_emb, cp_config=None):
    q = self.to_q(x).unflatten(2, (self.num_heads, self.head_dim))
    k = self.to_k(x).unflatten(2, (self.num_heads, self.head_dim))
    v = self.to_v(x).unflatten(2, (self.num_heads, self.head_dim))
    q = apply_rotary_emb(self.norm_q(q), rotary_emb)
    k = apply_rotary_emb(self.norm_k(k), rotary_emb)
    out = _dispatch_attention_fn(
      q,
      k,
      v,
      dropout_p=0.0,
      is_causal=False,
      cp_config=cp_config,
    )
    return self.to_out(out.flatten(2, 3).type_as(x))


class LingBotVideoMLP(nn.Module):

  def __init__(self, hidden_size, intermediate_size):
    super().__init__()
    self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
    self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
    self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

  def forward(self, x):
    return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LingBotVideoBlock(nn.Module):
  """One denoising block: joint attention + gated FFN, both scale-shifted by `temb6`.

  `temb6` is token-wise `(B * S, 6 * D)`: the 6-way modulation is computed once per forward from
  the timestep and passed in, so the block itself is a pure function of `(hidden_states, temb6,
  rotary_emb)` — which is what makes the cached replay safe.
  """

  def __init__(
    self,
    hidden_size,
    num_attention_heads,
    intermediate_size,
    norm_eps,
    qkv_bias,
    out_bias,
    layer_idx,
  ):
    super().__init__()
    # Kept for debugging and for per-layer policies; the dense stack itself is uniform.
    self.layer_idx = layer_idx
    h = hidden_size
    self.scale_shift_table = nn.Parameter(torch.zeros(1, 6 * h))
    self.norm1 = LingBotVideoRMSNorm(h, norm_eps)
    self.attn = LingBotVideoAttention(h, num_attention_heads, norm_eps, qkv_bias, out_bias)
    self.norm_post_attn = LingBotVideoRMSNorm(h, norm_eps)
    self.norm2 = LingBotVideoRMSNorm(h, norm_eps)
    self.ffn = LingBotVideoMLP(h, intermediate_size)
    self.norm_post_ffn = LingBotVideoRMSNorm(h, norm_eps)

  def forward(
    self,
    hidden_states,
    temb6,
    rotary_emb,
    cp_config=None,
  ):
    expected_tokens = hidden_states.shape[0] * hidden_states.shape[1]
    if temb6.ndim != 2 or temb6.shape[0] != expected_tokens:
      raise ValueError(
        "LingBotVideoBlock expects token-level temb6 with shape "
        f"(B*S, 6D); got {tuple(temb6.shape)} for hidden states {tuple(hidden_states.shape)}.")

    mod = temb6.view(hidden_states.shape[0], hidden_states.shape[1], -1)
    mod = mod + self.scale_shift_table.unsqueeze(0)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)
    gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
    scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

    # `to_q.weight` describes the activation dtype only while it is a plain float weight: a quantized
    # (fp8 / int4) projection reports its storage dtype, and casting the modulated activations to it
    # would quantize them a second time without a scale, in front of torchao's own dynamic
    # quantization. Fall back to the residual-stream dtype in that case.
    bulk_dtype = self.attn.to_q.weight.dtype
    if not bulk_dtype.is_floating_point or bulk_dtype.itemsize < 2:
      bulk_dtype = hidden_states.dtype
    attn_in = (self.norm1(hidden_states) * scale_msa + shift_msa).to(bulk_dtype)
    attn_out = self.attn(attn_in, rotary_emb, cp_config=cp_config)
    attn_normed = self.norm_post_attn(attn_out)
    hidden_states = hidden_states + (gate_msa * attn_normed).to(hidden_states.dtype)

    ffn_in = (self.norm2(hidden_states) * scale_mlp + shift_mlp).to(bulk_dtype)
    ffn_normed = self.norm_post_ffn(self.ffn(ffn_in))
    hidden_states = hidden_states + (gate_mlp * ffn_normed).to(hidden_states.dtype)
    return hidden_states


class LingBotVideoTransformer3DModel(ModelMixin, ConfigMixin):
  """Dense LingBot-Video DiT.

  :param patch_size: Latent patch size `(pT, pH, pW)`.
  :param in_channels: Latent channels consumed by the patch embedder.
  :param out_channels: Velocity channels produced by the output projection.
  :param hidden_size: Residual stream width.
  :param num_attention_heads: Attention heads; `hidden_size // num_attention_heads` must equal
    `sum(axes_dims)`.
  :param depth: Number of `LingBotVideoBlock`s.
  :param intermediate_size: Gated-FFN width.
  :param text_dim: Text-encoder hidden size (Qwen3-VL last hidden state).
  :param axes_dims: RoPE frequencies per axis `(t, h, w)`, summed to the head dimension.
  :param axes_lens: Initial RoPE table length per axis; the table grows on demand.
  """

  _supports_gradient_checkpointing = False
  _no_split_modules = ["LingBotVideoBlock"]
  _keep_in_fp32_modules = list(LINGBOT_VIDEO_FP32_MODULES)
  _repeated_blocks = ["LingBotVideoBlock"]

  def to(self, *args, **kwargs):
    device, dtype, non_blocking, _ = torch._C._nn._parse_to(*args, **kwargs)
    if dtype is None or dtype == torch.float32:
      return super().to(*args, **kwargs)

    dtype_is_floating = torch.is_floating_point(torch.empty((), dtype=dtype))
    if not dtype_is_floating:
      return super().to(*args, **kwargs)

    if device is not None:
      super().to(device=device, non_blocking=non_blocking)

    for name, param in self.named_parameters():
      if not torch.is_floating_point(param):
        continue
      target_dtype = torch.float32 if should_keep_in_fp32(name) else dtype
      param.data = param.data.to(dtype=target_dtype, non_blocking=non_blocking)
      if param.grad is not None:
        param.grad.data = param.grad.data.to(dtype=target_dtype, non_blocking=non_blocking)

    for name, buffer in self.named_buffers():
      if not torch.is_floating_point(buffer):
        continue
      target_dtype = torch.float32 if should_keep_in_fp32(name) else dtype
      buffer.data = buffer.data.to(dtype=target_dtype, non_blocking=non_blocking)

    return self

  @register_to_config
  def __init__(
    self,
    patch_size: Tuple[int, int, int] = (1, 2, 2),
    in_channels: int = 16,
    out_channels: int = 16,
    hidden_size: int = 2048,
    num_attention_heads: int = 16,
    depth: int = 24,
    intermediate_size: int = 6144,
    text_dim: int = 2560,
    freq_dim: int = 256,
    norm_eps: float = 1e-6,
    rope_theta: float = 256.0,
    axes_dims: Tuple[int, int, int] = (32, 48, 48),
    axes_lens: Tuple[int, int, int] = (8192, 1024, 1024),
    qkv_bias: bool = False,
    out_bias: bool = True,
    patch_embed_bias: bool = True,
    timestep_mlp_bias: bool = True,
    num_experts: int = 0,
    num_experts_per_tok: int = 8,
    moe_intermediate_size: int = 512,
    decoder_sparse_step: int = 1,
    mlp_only_layers: Tuple[int, ...] = (),
    n_shared_experts: Optional[int] = None,
    score_func: str = "sigmoid",
    norm_topk_prob: bool = True,
    n_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    routed_scaling_factor: float = 1.0,
  ):
    super().__init__()
    if num_experts != 0:
      raise NotImplementedError(
        f"LingBot-Video MoE transformers are not supported by cache-dit yet (num_experts={num_experts}). "
        "Use the Dense 1.3B checkpoint.")
    head_dim = hidden_size // num_attention_heads
    assert head_dim == sum(axes_dims), f"head_dim {head_dim} != sum(axes_dims) {sum(axes_dims)}"

    self.patch_embedder = nn.Linear(
      in_channels * math.prod(patch_size),
      hidden_size,
      bias=patch_embed_bias,
    )
    self.time_proj = Timesteps(freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
    self.time_embedder = TimestepEmbedding(
      freq_dim,
      hidden_size,
      act_fn="silu",
      sample_proj_bias=timestep_mlp_bias,
    )
    self.time_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
    self.text_embedder = LingBotVideoTextEmbedder(text_dim, hidden_size)
    self.rope = LingBotVideoRotaryEmbedding(axes_dims, axes_lens, rope_theta)
    self.blocks = nn.ModuleList([
      LingBotVideoBlock(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        norm_eps=norm_eps,
        qkv_bias=qkv_bias,
        out_bias=out_bias,
        layer_idx=i,
      ) for i in range(depth)
    ])
    self.norm_out = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=norm_eps)
    self.norm_out_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
    self.proj_out = nn.Linear(hidden_size, math.prod(patch_size) * out_channels)

    # Context-parallel split/gather boundaries. `hidden_states` is split on `blocks.0` (cache-dit's
    # Fn-residual replay reads that split hook); the token-wise companions are split here. The gather
    # sits after `proj_out`, where the token width is `prod(patch_size) * out_channels` = 64 instead
    # of `hidden_size`, i.e. 32x less communication than gathering the residual stream.
    self.cp_rotary = nn.Identity()
    self.cp_temb_input = nn.Identity()
    self.cp_temb6 = nn.Identity()
    self.cp_out = nn.Identity()

  def forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    return_dict: bool = True,
  ):
    """Predict the flow-matching velocity.

    :param hidden_states: Noisy latents `(B, C, T, H, W)`.
    :param timestep: Sigma scaled to `[0, 1000]`, shape `(B,)`.
    :param encoder_hidden_states: Text features `(B, L, text_dim)`, already truncated to the valid
      token count (padding is rejected below).
    :param encoder_attention_mask: Optional `(B, L)` mask, `1` = valid token. It is only used to
      derive the text length: a padded mask raises, because the joint sequence and the RoPE table are
      built from that length.
    :param return_dict: Return a `Transformer2DModelOutput` instead of a tuple.
    :returns: `Transformer2DModelOutput` (or a tuple) holding the velocity.
    """
    B, C, T, H, W = hidden_states.shape
    if B != 1:
      raise NotImplementedError(
        "LingBot-Video runs one sample per forward in cache-dit: the joint `[video; text]` sequence "
        f"assumes a single prompt, got batch size {B}.")
    pF, pH, pW = self.config.patch_size
    gt, gh, gw = T // pF, H // pH, W // pW
    n_video = gt * gh * gw
    L = encoder_hidden_states.shape[1]
    device = hidden_states.device

    if encoder_attention_mask is not None:
      text_lens = encoder_attention_mask.sum(dim=-1).long()
    else:
      text_lens = torch.full((B, ), L, dtype=torch.long, device=device)
    text_len = int(text_lens[0].item())
    if text_len < L:
      raise NotImplementedError(
        "LingBot-Video requires text embeddings truncated to the valid length: the joint sequence "
        "length and the RoPE table are built from the text length, so a padding mask cannot be "
        "reconciled afterwards (and context parallelism would have to permute it the same way). "
        "Pass `prompt_embeds` already sliced to the valid length.")
    cp_config = getattr(self, "_cp_config", None)

    # patchify: token order (f h w), feature order (pf ph pw c)
    patch_tokens = hidden_states.reshape(B, C, gt, pF, gh, pH, gw, pW)
    patch_tokens = patch_tokens.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(
      B,
      n_video,
      pF * pH * pW * C,
    )
    x = self.patch_embedder(patch_tokens)
    text = self.text_embedder(encoder_hidden_states)
    joint = torch.cat([x, text], dim=1)  # [video; text]
    joint_seq_len = joint.shape[1]

    rotary = self.rope(make_joint_position_ids(text_len, gt, gh, gw, device)).unsqueeze(0)

    timestep_proj = self.time_proj(timestep.float())
    t_emb = self.time_embedder(timestep_proj)  # (B, D)
    temb_input = t_emb.unsqueeze(1).expand(B, joint_seq_len, -1)
    temb6 = self.time_modulation(temb_input.reshape(B * joint_seq_len, -1))
    temb6 = temb6.reshape(B, joint_seq_len, -1)  # (B, S, 6D)

    rotary = self.cp_rotary(rotary)
    temb_input = self.cp_temb_input(temb_input)
    temb6 = self.cp_temb6(temb6)
    temb6 = temb6.reshape(temb6.shape[0] * temb6.shape[1], -1)

    for block in self.blocks:
      joint = block(joint, temb6, rotary, cp_config=cp_config)

    local_seq_len = joint.shape[1]
    final_mod = self.norm_out_modulation(temb_input.reshape(B * local_seq_len, -1))
    shift, scale = final_mod.reshape(B, local_seq_len, -1).chunk(2, dim=-1)
    final_hidden = self.norm_out(joint) * (1.0 + scale) + shift
    projected = self.proj_out(final_hidden.to(self.proj_out.weight.dtype))
    projected = self.cp_out(projected)

    x = projected[:, :n_video]

    # unpatchify
    Cout = self.config.out_channels
    x = x.reshape(B, gt, gh, gw, pF, pH, pW, Cout)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, Cout, T, H, W)

    if not return_dict:
      return (x, )
    return Transformer2DModelOutput(sample=x)
