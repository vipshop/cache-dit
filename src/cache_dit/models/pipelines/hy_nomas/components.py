"""PixelUMM pixel decoder of HY-Nomas.

Copied from ``sglang.multimodal_gen.runtime.models.vaes.nomas`` (Apache-2.0), adapted from
Tencent-Hunyuan/Hy-Nomas.  The network, the sliding-tile attention layout and the temporal window
merge are unchanged; ``msgspec`` and the sglang decode-parallel group are replaced by
dataclasses and a plain serial decode.

Intentional differences from the sglang copy:

* **The fused elementwise kernels are not ported.**  Upstream compiles ``_modulate`` / ``_rotate`` /
  ``_qk_norm_mrope`` with ``emulate_precision_casts=True`` so the fused kernels are bit-identical to
  the eager sequence; running the eager sequence therefore keeps the same values.  The QK
  normalisation still chunks over tokens to bound the float32 footprint.
* **Decode is serial.**  sglang spreads independent temporal windows over its decode-parallel group;
  cache-dit's VAE-P dispatch is keyed on diffusers autoencoders, which this decoder is not, so VAE-P
  stays unsupported (see the model plan) and every window is decoded locally.
* **Checkpoint loading is owned here**, including the fp8 variant: the big projections ship as
  ``*.fp8_weight`` (float8_e4m3fn) plus ``*.scale`` (float32) and are dequantised on load.  All keys
  carry a ``decoder.`` prefix, which is stripped.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

TUBELET = (4, 16, 16)
TUBE_DIM = TUBELET[0] * TUBELET[1] * TUBELET[2] * 3
LATENT_CHANNELS = 32
# A DCAE latent cell is (4, 32, 32) pixels: 1 tubelet in time, 2 x 2 in space.
LATENT_TOKEN_GRID = 2
# Target float32 bytes per QK-normalisation chunk.
_QK_CHUNK_BYTES = 256 * 2 ** 20


@dataclass(frozen=True)
class PixelUMMArch:
  """Qwen3-8B-shaped expert (hidden 4096) with PixelUMM's T/H/W MRoPE.

  :param hidden_size: Residual width.
  :param intermediate_size: SwiGLU hidden width.
  :param num_layers: Layer count.
  :param num_heads: Query heads.
  :param num_kv_heads: Key/value heads (GQA).
  :param head_dim: Per-head width.
  :param rms_norm_eps: RMSNorm epsilon.
  :param rope_theta_t: RoPE theta of the temporal axis.
  :param rope_theta_hw: RoPE theta of the two spatial axes.
  :param local_attention_tile: Sliding-tile size; ``0`` keeps full attention.
  :param temporal_window_latents: Window length in latents.
  :param temporal_overlap_latents: Window overlap in latents.
  :param num_steps: DMD steps (the released checkpoint is 1).
  :param denoising_step_list: Descending DMD step ladder.
  :param seed: Seed of the frame-indexed decoder noise.
  """

  hidden_size: int = 4096
  intermediate_size: int = 12288
  num_layers: int = 36
  num_heads: int = 32
  num_kv_heads: int = 8
  head_dim: int = 128
  rms_norm_eps: float = 1e-6
  rope_theta_t: float = 1_000_000.0
  rope_theta_hw: float = 10_000.0
  local_attention_tile: int = 8
  temporal_window_latents: int = 10
  temporal_overlap_latents: int = 5
  num_steps: int = 1
  denoising_step_list: Tuple[float, ...] = (0.999, 0.866, 0.634, 0.342)
  seed: int = 42

  def __post_init__(self):
    if self.head_dim % 4 or self.num_heads % self.num_kv_heads:
      raise ValueError("head_dim must be divisible by 4 and heads by kv heads")
    if not 0 <= self.temporal_overlap_latents < self.temporal_window_latents:
      raise ValueError("PixelUMM requires 0 <= overlap < window")
    steps = self.denoising_step_list[:self.num_steps]
    if not 1 <= self.num_steps <= len(self.denoising_step_list) or not all(
        0 < b <= 1 and b < a for a, b in zip((1.01, *steps[:-1]), steps)):
      raise ValueError("PixelUMM requires descending DMD steps in (0, 1]")

  @property
  def rope_sections(self) -> Tuple[int, int, int]:
    """Head-dim slices rotated by the T, H and W position respectively."""
    return self.head_dim // 2, self.head_dim // 4, self.head_dim // 4


def patchify(video: torch.Tensor) -> torch.Tensor:
  """``[B,3,T,H,W] -> [B, (T/4)(H/16)(W/16), 3072]``; token order T,H,W, values (t,p,q,c)."""
  if video.ndim != 5 or video.shape[1] != 3:
    raise ValueError(f"expected RGB video [B,3,T,H,W], got {tuple(video.shape)}")
  pt, ph, pw = TUBELET
  batch, channels, frames, height, width = video.shape
  if frames % pt or height % ph or width % pw:
    raise ValueError(f"video {tuple(video.shape[2:])} must be a multiple of {TUBELET}")
  x = video.reshape(batch, channels, frames // pt, pt, height // ph, ph, width // pw, pw)
  return x.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(batch, -1, TUBE_DIM)


def unpatchify(tokens: torch.Tensor, grid: Tuple[int, int, int]) -> torch.Tensor:
  """Inverse of :func:`patchify` for one ``(T, H, W)`` token grid."""
  batch, count, dim = tokens.shape
  nt, nh, nw = grid
  pt, ph, pw = TUBELET
  if count != nt * nh * nw or dim != TUBE_DIM:
    raise ValueError(f"tokens {tuple(tokens.shape)} do not match grid {grid}")
  x = tokens.reshape(batch, nt, nh, nw, pt, ph, pw, 3)
  return x.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(batch, 3, nt * pt, nh * ph, nw * pw)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
  half = x.shape[-1] // 2
  return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def mrope_tables(arch: PixelUMMArch, grid: Tuple[int, int, int], device):
  """``cos``/``sin`` ``[1, T*H*W, 1, head_dim]`` for a THW-ordered grid, laid out ``T | H | W``."""
  positions = torch.meshgrid(*(torch.arange(n, device=device) for n in grid), indexing="ij")
  thetas = (arch.rope_theta_t, arch.rope_theta_hw, arch.rope_theta_hw)
  cos_parts, sin_parts = [], []
  for position, dim, theta in zip(positions, arch.rope_sections, thetas):
    inv_freq = theta ** (-torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    angles = position.flatten().float()[:, None] * inv_freq[None]
    angles = torch.cat((angles, angles), dim=-1)
    cos_parts.append(angles.cos())
    sin_parts.append(angles.sin())
  return (
    torch.cat(cos_parts, dim=-1)[None, :, None, :],
    torch.cat(sin_parts, dim=-1)[None, :, None, :],
  )


def apply_mrope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, sections) -> torch.Tensor:
  """``x`` ``[B,N,H,D]``; each section is rotated within itself (rotate_half per axis)."""
  rotated = torch.cat([_rotate_half(part) for part in x.split(tuple(sections), dim=-1)], dim=-1)
  return x * cos + rotated * sin


class PixelUMMRMSNorm(nn.Module):
  """Qwen3RMSNorm: normalise in float32, cast back to the input dtype, then scale."""

  def __init__(self, dim: int, eps: float):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(dim))
    self.eps = eps

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
    return self.weight * xf.to(dtype)


def _split_norm(x: torch.Tensor, norm_t: PixelUMMRMSNorm, norm_hw: PixelUMMRMSNorm):
  t_part, hw_part = x.chunk(2, dim=-1)
  return torch.cat((norm_t(t_part), norm_hw(hw_part)), dim=-1)


def _prepare_qk_chunked(x, cos, sin, norm_t, norm_hw, dtype, sections):
  """Float32 QK-norm + MRoPE, chunked over tokens to bound the float32 footprint."""
  result = torch.empty_like(x, dtype=dtype)
  chunk = max(1, _QK_CHUNK_BYTES // max(1, x.shape[0] * x.shape[2] * x.shape[3] * 4))
  for first in range(0, x.shape[1], chunk):
    part = x[:, first:first + chunk]
    normalized = _split_norm(part.float(), norm_t, norm_hw)
    result[:, first:first + chunk] = apply_mrope(normalized, cos[:, first:first + chunk],
                                                 sin[:, first:first + chunk], sections).to(dtype)
  return result


def local_attention_layout(grid: Tuple[int, int, int], tile: int, device):
  """``(perm, inverse, block_mask)`` of the sliding-tile attention, or ``None`` for tiny planes."""
  if tile <= 0:
    return None
  _, height, width = grid
  if -(-height // tile) <= 3 and -(-width // tile) <= 3:
    return None
  return _cached_local_layout(tuple(grid), tile, str(device))


@lru_cache(maxsize=8)
def _cached_local_layout(grid, tile, device):
  from torch.nn.attention.flex_attention import create_block_mask

  frames, height, width = grid
  tiles_h, tiles_w = -(-height // tile), -(-width // tile)
  t, h, w = torch.meshgrid(*(torch.arange(n, device=device) for n in grid), indexing="ij")
  tile_h, tile_w = (h // tile).flatten(), (w // tile).flatten()
  order = (((tile_h * tiles_w + tile_w) * frames + t.flatten()) * tile +
           (h % tile).flatten()) * tile
  perm = torch.argsort(order + (w % tile).flatten())
  tile_h, tile_w = tile_h[perm], tile_w[perm]
  center_h = tile_h.clamp(1, max(tiles_h - 2, 1))
  center_w = tile_w.clamp(1, max(tiles_w - 2, 1))

  def mask_mod(batch, head, q_idx, kv_idx):
    return (((center_h[q_idx] - tile_h[kv_idx]).abs() <= 1)
            & ((center_w[q_idx] - tile_w[kv_idx]).abs() <= 1))

  count = frames * height * width
  on_cuda = torch.device(device).type == "cuda"
  if not on_cuda and count * count > 16_777_216:
    raise ValueError("PixelUMM CPU reference mask is too large; use CUDA")
  builder = _block_mask_builder() if on_cuda else create_block_mask
  block_mask = builder(mask_mod, None, None, count, count, device=device)
  return perm, torch.argsort(perm), block_mask


@lru_cache(maxsize=1)
def _block_mask_builder():
  from torch.nn.attention.flex_attention import create_block_mask

  return torch.compile(create_block_mask, dynamic=True, fullgraph=True)


@lru_cache(maxsize=1)
def _compiled_flex():
  from torch.nn.attention.flex_attention import flex_attention

  # Never allow an eager quadratic-memory fallback after graph specialization.
  return torch.compile(flex_attention, dynamic=True, fullgraph=True)


def _flex_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_mask) -> torch.Tensor:
  """``[B,heads,N,D]`` x GQA ``[B,kv,N,D]``; compiled on CUDA because eager flex is quadratic."""
  from torch.nn.attention.flex_attention import flex_attention

  if q.device.type != "cuda" or torch.compiler.is_compiling():
    return flex_attention(q, k, v, block_mask=block_mask, enable_gqa=True)
  return _compiled_flex()(q, k, v, block_mask=block_mask, enable_gqa=True)


class PixelUMMAttention(nn.Module):
  """GQA with split T / HW QK-norm and T/H/W MRoPE; full or sliding-tile attention."""

  def __init__(self, arch: PixelUMMArch):
    super().__init__()
    self.arch = arch
    hidden, heads, kv, dim = (arch.hidden_size, arch.num_heads, arch.num_kv_heads, arch.head_dim)
    self.q_proj = nn.Linear(hidden, heads * dim, bias=False)
    self.k_proj = nn.Linear(hidden, kv * dim, bias=False)
    self.v_proj = nn.Linear(hidden, kv * dim, bias=False)
    self.o_proj = nn.Linear(heads * dim, hidden, bias=False)
    self.q_norm_t = PixelUMMRMSNorm(dim // 2, arch.rms_norm_eps)
    self.q_norm_hw = PixelUMMRMSNorm(dim // 2, arch.rms_norm_eps)
    self.k_norm_t = PixelUMMRMSNorm(dim // 2, arch.rms_norm_eps)
    self.k_norm_hw = PixelUMMRMSNorm(dim // 2, arch.rms_norm_eps)

  def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, block_mask=None):
    arch = self.arch
    batch, count, _ = x.shape
    q = self.q_proj(x).view(batch, count, arch.num_heads, arch.head_dim)
    k = self.k_proj(x).view(batch, count, arch.num_kv_heads, arch.head_dim)
    v = self.v_proj(x).view(batch, count, arch.num_kv_heads, arch.head_dim)
    # PixelUMM keeps q/k in float32 through QK-norm and RoPE.
    q = _prepare_qk_chunked(q, cos, sin, self.q_norm_t, self.q_norm_hw, v.dtype, arch.rope_sections)
    k = _prepare_qk_chunked(k, cos, sin, self.k_norm_t, self.k_norm_hw, v.dtype, arch.rope_sections)
    if block_mask is not None:
      out = _flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask)
    else:
      repeat = arch.num_heads // arch.num_kv_heads
      k, v = k.repeat_interleave(repeat, dim=2), v.repeat_interleave(repeat, dim=2)
      out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    return self.o_proj(out.transpose(1, 2).reshape(batch, count, arch.num_heads * arch.head_dim))


class PixelUMMMLP(nn.Module):
  """SwiGLU feed-forward."""

  def __init__(self, arch: PixelUMMArch):
    super().__init__()
    self.gate_proj = nn.Linear(arch.hidden_size, arch.intermediate_size, bias=False)
    self.up_proj = nn.Linear(arch.hidden_size, arch.intermediate_size, bias=False)
    self.down_proj = nn.Linear(arch.intermediate_size, arch.hidden_size, bias=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class PixelUMMLayer(nn.Module):
  """Pre-norm attention + MLP layer."""

  def __init__(self, arch: PixelUMMArch):
    super().__init__()
    self.input_norm = PixelUMMRMSNorm(arch.hidden_size, arch.rms_norm_eps)
    self.attn = PixelUMMAttention(arch)
    self.post_attn_norm = PixelUMMRMSNorm(arch.hidden_size, arch.rms_norm_eps)
    self.mlp = PixelUMMMLP(arch)

  def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, block_mask=None):
    x = x + self.attn(self.input_norm(x), cos, sin, block_mask)
    return x + self.mlp(self.post_attn_norm(x))


class PixelUMMHead(nn.Module):
  """``RMSNorm(eps=1e-6) -> Linear(hidden, 3072)``."""

  def __init__(self, hidden: int):
    super().__init__()
    self.norm = nn.RMSNorm(hidden, eps=1e-6)
    self.linear = nn.Linear(hidden, TUBE_DIM, bias=True)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.linear(self.norm(x))


class PixelUMMDecoder(nn.Module):
  """``forward(latent [B,32,L,h,w], xt [B,3,4L,32h,32w]) -> x0``: one DMD step."""

  def __init__(self, arch: PixelUMMArch):
    super().__init__()
    self.arch = arch
    hidden = arch.hidden_size
    self.embed = nn.Linear(TUBE_DIM, hidden)
    self.cond_proj = nn.Linear(LATENT_CHANNELS, LATENT_TOKEN_GRID ** 2 * hidden)
    self.layers = nn.ModuleList(PixelUMMLayer(arch) for _ in range(arch.num_layers))
    self.final_norm = PixelUMMRMSNorm(hidden, arch.rms_norm_eps)
    self.head = PixelUMMHead(hidden)

  @staticmethod
  def pixel_shape(latent: torch.Tensor) -> Tuple[int, int, int, int, int]:
    batch, _, frames, height, width = latent.shape
    return batch, 3, frames * TUBELET[0], height * 32, width * 32

  def _condition_tokens(self, latent: torch.Tensor) -> torch.Tensor:
    batch, _, frames, height, width = latent.shape
    g = LATENT_TOKEN_GRID
    x = self.cond_proj(latent.permute(0, 2, 3, 4, 1))  # [B,L,h,w,4*hidden]
    x = x.view(batch, frames, height, width, g, g, self.arch.hidden_size)
    return x.permute(0, 1, 2, 4, 3, 5, 6).reshape(batch, -1, self.arch.hidden_size)

  def forward(self, latent: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
    if latent.ndim != 5 or latent.shape[1] != LATENT_CHANNELS:
      raise ValueError(f"expected DCAE latent [B,{LATENT_CHANNELS},L,h,w], "
                       f"got {tuple(latent.shape)}")
    if tuple(xt.shape) != self.pixel_shape(latent):
      raise ValueError(f"xt {tuple(xt.shape)} != expected {self.pixel_shape(latent)}")
    dtype = self.embed.weight.dtype
    context = (torch.autocast(device_type=latent.device.type, dtype=dtype)
               if dtype in (torch.bfloat16, torch.float16) else _nullcontext())
    _, _, frames, height, width = latent.shape
    grid = (frames, height * LATENT_TOKEN_GRID, width * LATENT_TOKEN_GRID)
    with context:
      tokens = self.embed(patchify(xt.to(dtype))) + self._condition_tokens(latent.to(dtype))
      cos, sin = mrope_tables(self.arch, grid, latent.device)
      layout = local_attention_layout(grid, self.arch.local_attention_tile, latent.device)
      block_mask = None
      if layout is not None:
        perm, inverse, block_mask = layout
        tokens, cos, sin = tokens[:, perm], cos[:, perm], sin[:, perm]
      for layer in self.layers:
        tokens = layer(tokens, cos, sin, block_mask)
      tokens = self.final_norm(tokens)
      if layout is not None:
        tokens = tokens[:, inverse]
    # The source runs the head after leaving autocast.
    return unpatchify(self.head(tokens), grid)


def _nullcontext():
  import contextlib

  return contextlib.nullcontext()


@dataclass
class NomasDecodeOutput:
  """Decoder output; ``.sample`` mirrors the diffusers decode contract."""

  sample: torch.Tensor


def _dequant_blockwise(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
  """Dequantise a block-wise float8 weight: ``w[i, j] * scale[i // bh, j // bw]``.

  :param weight: ``[out, in]`` float8 tensor.
  :param scale: ``[out // bh, in // bw]`` float32 scales.
  :returns: The float32 dequantised weight.
  """
  if weight.ndim != 2 or scale.ndim != 2:
    raise ValueError("block-wise dequant needs a 2-D weight and a 2-D scale")
  out_f, in_f = weight.shape
  s0, s1 = scale.shape
  if out_f % s0 or in_f % s1:
    raise ValueError(f"scale {tuple(scale.shape)} does not tile weight {tuple(weight.shape)}")
  view = weight.to(torch.float32).view(s0, out_f // s0, s1, in_f // s1)
  return (view * scale[:, None, :, None]).reshape(out_f, in_f)


_PIXELUMM_PREFIX = "decoder."


def load_pixelumm_state_dict(path: str,
                             dtype: torch.dtype = torch.bfloat16) -> Dict[str, torch.Tensor]:
  """Read either PixelUMM checkpoint variant into this module's state-dict names.

  Handles the ``decoder.`` prefix, the fp8 variant (``*.fp8_weight`` + ``*.fp8_scale`` block-wise
  scales) and the ``.pt`` container, so the pipeline never touches checkpoint layout.

  :param path: ``.safetensors`` or ``.pt`` checkpoint of the decoder.
  :param dtype: Dtype of the returned floating-point tensors.
  :returns: A state dict keyed by :class:`PixelUMMDecoder`'s parameter names (the checkpoint's
    ``decoder.`` prefix is stripped, and fp8 pairs collapse into ``*.weight``).
  """
  from safetensors import safe_open

  # The released file bundles the DCAE-V **encoder** (convolutional, `encoder.*`) with the PixelUMM
  # decoder (`decoder.*`).  Text-to-video never encodes, so only the decoder half is read: with
  # `safe_open` the encoder tensors are never materialised (the whole file is 8.51 GiB).
  if path.endswith(".safetensors"):
    with safe_open(path, framework="pt", device="cpu") as handle:
      decoder_keys = {
        key: handle.get_tensor(key)
        for key in handle.keys() if key.startswith(_PIXELUMM_PREFIX)
      }
  else:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
      state = state["state_dict"]
    decoder_keys = {key: value for key, value in state.items() if key.startswith(_PIXELUMM_PREFIX)}
  if not decoder_keys:
    raise ValueError(f"{path} carries no `{_PIXELUMM_PREFIX}` tensors; this loader expects the "
                     f"released DCAE-V + PixelUMM bundle")
  scales = {
    key[:-len(".fp8_scale")]: value
    for key, value in decoder_keys.items() if key.endswith(".fp8_scale")
  }
  out: Dict[str, torch.Tensor] = {}
  for key, value in decoder_keys.items():
    if key.endswith(".fp8_scale") or key.endswith(".fp8_weight"):
      continue
    name = key[len(_PIXELUMM_PREFIX):]
    out[name] = value.to(dtype) if value.is_floating_point() else value
  for key, value in decoder_keys.items():
    if not key.endswith(".fp8_weight"):
      continue
    base = key[:-len("fp8_weight")]
    scale = scales.get(base[:-1])
    if scale is None:
      raise ValueError(f"fp8 weight {key} has no matching scale")
    out[base[len(_PIXELUMM_PREFIX):] + "weight"] = _dequant_blockwise(value, scale).to(dtype)
  return out


def _load_pixelumm_weights(module: "NomasPixelUMMDecoder", path: str, dtype: torch.dtype) -> None:
  """Load a decoder checkpoint, refusing to run on an unconverted module.

  ``load_state_dict`` copies into the **existing** parameter dtype, so loading a bf16 checkpoint
  into a freshly built float32 module silently keeps 2x the memory (and the memory estimate in the
  plan becomes wrong).  The conversion is therefore enforced instead of documented.

  :param module: Wrapper created in the target dtype.
  :param path: Checkpoint path.
  :param dtype: Expected parameter dtype.
  :raises RuntimeError: When the module was not converted first.
  """
  actual = next(module.parameters()).dtype
  if actual != dtype:
    raise RuntimeError(f"convert the decoder to {dtype} before loading: `load_state_dict` casts "
                       f"into the existing parameter dtype, so a {actual} module would silently "
                       f"keep {actual} weights")
  # The loader returns the inner decoder's names, i.e. without the wrapper's `decoder.` level.
  module.decoder.load_state_dict(load_pixelumm_state_dict(path, dtype=dtype), strict=True)


class NomasPixelUMMDecoder(nn.Module):
  """Windowed single-step PixelUMM decode with the diffusers ``.decode(...)`` contract.

  :param arch: Decoder architecture; defaults to the released PixelUMM expert.
  """

  def __init__(self, arch: Optional[PixelUMMArch] = None):
    super().__init__()
    self.arch = arch or PixelUMMArch()
    self.decoder = PixelUMMDecoder(self.arch)
    # Absolute latent index of the first decoded frame; streaming decodes slices of a longer
    # timeline whose noise follows that index.
    self.noise_frame_offset = 0

  @classmethod
  def from_checkpoint(cls,
                      path: str,
                      dtype: torch.dtype = torch.bfloat16,
                      arch: Optional[PixelUMMArch] = None) -> "NomasPixelUMMDecoder":
    """Build the decoder and load either checkpoint variant with ``strict=True``.

    :param path: ``.safetensors`` / ``.pt`` decoder checkpoint.
    :param dtype: Parameter dtype.
    :param arch: Optional architecture override.
    :returns: The loaded decoder in eval mode with gradients disabled.
    """
    module = cls(arch).to(dtype)
    _load_pixelumm_weights(module, path, dtype)
    return module.eval().requires_grad_(False)

  @property
  def dtype(self) -> torch.dtype:
    """Dtype of the decoder weights; diffusers' `DiffusionPipeline.to()` duck-types this."""
    return next(self.parameters()).dtype

  @property
  def device(self) -> torch.device:
    """Device of the decoder weights; diffusers' `DiffusionPipeline.to()` duck-types this."""
    return next(self.parameters()).device

  @contextmanager
  def timeline_offset(self, frame_offset: int):
    """Decode slice of a longer timeline whose noise is keyed by absolute frame index."""
    previous, self.noise_frame_offset = self.noise_frame_offset, frame_offset
    try:
      yield
    finally:
      self.noise_frame_offset = previous

  def _noise(self, latent: torch.Tensor, start: int, end: int) -> torch.Tensor:
    batch, _, _, height, width = latent.shape
    return torch.cat(
      [
        torch.randn(
          (batch, 3, 4, height * 32, width * 32),
          device=latent.device,
          dtype=torch.float32,
          generator=torch.Generator(device=latent.device).manual_seed(self.arch.seed * 1_000_003 +
                                                                      index),
        ) for index in range(start, end)
      ],
      dim=2,
    )

  @torch.inference_mode()
  def decode(self, latent: torch.Tensor, return_dict: bool = True, generator=None):
    """Decode ``[1, 32, L, h, w]`` latents to ``[1, 3, 4L, 32h, 32w]`` pixels.

    :param latent: DCAE-V latents, batch size 1.
    :param return_dict: Whether to wrap the tensor in :class:`NomasDecodeOutput`.
    :param generator: Unused; the decoder noise is frame-index seeded by design.
    :returns: The decoded video.
    """
    if latent.ndim != 5 or latent.shape[1] != LATENT_CHANNELS:
      raise ValueError("PixelUMM expects [B,32,T,H,W] latents")
    arch = self.arch
    length = latent.shape[2]
    starts = window_starts(length, arch.temporal_window_latents,
                           arch.temporal_window_latents - arch.temporal_overlap_latents)
    steps = arch.denoising_step_list[:arch.num_steps]
    offset = self.noise_frame_offset
    # A single step generates only one window of noise at a time.
    state = self._noise(latent, offset, offset + length) if len(steps) > 1 else None
    video = None
    for t_cur, t_next in zip(steps, (*steps[1:], 0.0)):

      def decode_window(start):
        end = min(start + arch.temporal_window_latents, length)
        noise = (self._noise(latent, offset + start, offset +
                             end) if state is None else state[:, :, start * 4:end * 4])
        return self.decoder(latent[:, :, start:end], noise)

      video = merge_windows([decode_window(start) for start in starts], [s * 4 for s in starts])
      if t_next:
        state = video.float() + (t_next / t_cur) * (state - video.float())
    return NomasDecodeOutput(video) if return_dict else (video, )

  def forward(self, latent: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
    return self.decoder(latent, xt)


def window_starts(length: int, window: int, stride: int):
  """Window start indices covering ``[0, length)`` with the last window flush to the end."""
  if length < 1 or not 1 <= stride <= window:
    raise ValueError("Invalid PixelUMM temporal window")
  if length <= window:
    return [0]
  starts = list(range(0, length - window + 1, stride))
  if starts[-1] != length - window:
    starts.append(length - window)
  return starts


def merge_windows(parts, starts):
  """Blend overlapping windows along the temporal axis with a cosine ramp."""
  result = parts[0]
  for part, start in zip(parts[1:], starts[1:]):
    overlap = result.shape[2] - start
    if overlap == 0:
      result = torch.cat((result, part), dim=2)
      continue
    if not 0 < overlap < part.shape[2]:
      raise ValueError("PixelUMM windows must extend the decoded timeline")
    position = torch.arange(1, overlap + 1, device=part.device, dtype=torch.float32)
    weight = 0.5 - 0.5 * torch.cos(math.pi * position / (overlap + 1))
    weight = weight.view(1, 1, -1, 1, 1).to(part.dtype)
    result = torch.cat(
      (
        result[:, :, :-overlap],
        result[:, :, -overlap:] * (1 - weight) + part[:, :, :overlap] * weight,
        part[:, :, overlap:],
      ),
      dim=2,
    )
  return result
