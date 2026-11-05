"""HY-Nomas segment-autoregressive video DiT, restructured so cache-dit can cache it.

Copied from ``sglang.multimodal_gen.runtime.models.dits.nomas`` + ``nomas_layers`` (Apache-2.0),
themselves adapted from Tencent-Hunyuan/Hy-Nomas (``hymm/models/dit/nomas_cpt.py``).  The block
math, the RoPE tables, the camera branch and the KV-cache protocol are unchanged; the released
checkpoint loads with ``strict=True`` and **no key remapping**.

Intentional differences from the sglang copy:

* **One block call per loop iteration, one tensor pair in / one tensor pair out.** Upstream's loop
  body applies ``cache.layers[index]``, ``action_modules[index]`` and the K/V collection, none of
  which the cached-block wrapper can replay (it only threads ``hidden_states`` /
  ``encoder_hidden_states`` and a constant ``*args``).  Each block therefore owns its ``index``, a
  non-registered reference to its camera module and its slice of the segment cache; the per-forward
  constants travel in one :class:`NomasBlockContext`, so the block maps ``(img, txt)`` to
  ``(img, txt)`` and matches ``ForwardPattern.Pattern_0`` exactly.
* **Checkpoint naming is kept verbatim** (``img_mlp.fc1/fc2``, ``img_mod.linear``,
  ``final_layer.adaLN_modulation.1``), so ``load_state_dict(..., strict=True)`` needs no
  ``param_names_mapping`` and a renamed projection cannot slip through.
* **The KV write never goes through the cached ``forward``.** ``predict(..., collect_cache=True)``
  is the only entry that produces entries; it looks through cache-dit's one-element block wrapper
  and raises instead of silently writing partial history.
* **Context parallelism is not wired yet.**  The sglang copy slices ``img[:, lo:hi]`` by hand and
  gathers with ``sp_all_gather``; here the plan is to let a hook plan do both (split on
  ``double_blocks.0``, gather on ``final_layer``, which sits outside the cached block container),
  and to slice the RoPE table per rank because a dataclass cannot enter a CP plan.  Until
  ``distributed/transformers/hy_nomas.py`` exists, :meth:`NomasTransformer3DModel.set_parallel`
  rejects ``sp_size > 1`` and the exchange helpers are exercised only by tests.
* ``msgspec`` structs and the sglang parallel/linear layers are replaced by dataclasses,
  ``nn.Linear`` and the Ulysses helpers of :mod:`.attention`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import PixArtAlphaTextProjection
from diffusers.models.modeling_utils import ModelMixin

from ....attention import _dispatch_attention_fn
from ....logger import init_logger
from .attention import (
  NomasParallelInfo,
  local_token_range,
  sp_all_gather,
  ulysses_all_to_all,
)
from .kv_cache import NomasAttentionPlan, NomasCache

logger = init_logger(__name__)


def get_meshgrid_nd(start, stop, device=None):
  """``[len(start), *sizes]`` grid stepping by 1 from ``start`` up to ``stop``."""
  axes = [
    torch.linspace(a, b, b - a + 1, dtype=torch.float32, device=device)[:b - a]
    for a, b in zip(start, stop)
  ]
  return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=0)


def get_1d_rotary_pos_embed(dim: int, pos: torch.Tensor, theta: float = 10000.0):
  """``(cos, sin)`` of shape ``[S, dim]`` for float positions ``pos`` ``[S]``."""
  freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=pos.device)[:dim // 2].float() / dim))
  freqs = torch.outer(pos, freqs)
  return freqs.cos().repeat_interleave(2, dim=1), freqs.sin().repeat_interleave(2, dim=1)


def get_nd_rotary_pos_embed(rope_dim_list, start, stop, theta, device=None):
  """n-D RoPE tables ``(cos, sin)`` of shape ``[prod(stop - start), sum(rope_dim_list)]``."""
  grid = get_meshgrid_nd(start, stop, device=device)
  embs = [
    get_1d_rotary_pos_embed(dim, grid[i].reshape(-1), theta) for i, dim in enumerate(rope_dim_list)
  ]
  return torch.cat([e[0] for e in embs], dim=1), torch.cat([e[1] for e in embs], dim=1)


def timestep_embedding(t: torch.Tensor, dim: int = 256, max_period: int = 10000) -> torch.Tensor:
  """Sinusoidal embedding of fractional timesteps, ``(*t.shape, dim)``; ``dim`` is even."""
  half = dim // 2
  freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) /
                    half).to(device=t.device)
  args = t.flatten()[:, None].float() * freqs[None]
  embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
  return embedding.view(*t.shape, dim)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
  """AdaLN modulation, applied per batch row."""
  return x * (1 + scale[:, None]) + shift[:, None]


def modulate_for_linear(x, shift, scale):
  """``modulate`` followed by the cast the next ``nn.Linear`` would apply under autocast."""
  if not (x.dtype == torch.float32 and x.is_cuda and torch.is_autocast_enabled(x.device.type)):
    return modulate(x, shift, scale)
  dtype = torch.get_autocast_dtype(x.device.type)
  return modulate(x, shift, scale).to(dtype)


def apply_gate(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
  return x * gate.unsqueeze(1)


def gated_residual(residual: torch.Tensor, update: torch.Tensor, gate: torch.Tensor):
  """``residual + update * gate`` in the residual's storage dtype."""
  return (residual + update * gate.unsqueeze(1)).to(residual.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
  x_real, x_imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
  return torch.stack([-x_imag, x_real], dim=-1).flatten(3)


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
  return (x.float() * cos + rotate_half(x.float()) * sin).type_as(x)


def apply_rotary_emb_single(x: torch.Tensor, freqs_cis: Tuple[torch.Tensor, torch.Tensor]):
  """Rotate one ``[B, S, H, D]`` tensor with ``(cos, sin)`` tables of shape ``[S, D]``."""
  cos, sin = freqs_cis
  if cos.shape != (x.shape[1], x.shape[-1]):
    raise ValueError(f"freqs_cis shape {cos.shape} does not match x shape {x.shape}")
  shape = [d if i in (1, x.ndim - 1) else 1 for i, d in enumerate(x.shape)]
  return _rotate(x, cos.view(*shape).to(x.device), sin.view(*shape).to(x.device))


class NomasRMSNorm(nn.Module):
  """RMSNorm with the upstream float32 reduction and a learned per-channel weight."""

  def __init__(self, dim: int, eps: float = 1e-6):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


def rms_norm_rotate(norm: NomasRMSNorm, x, freqs_cis, dtype):
  """``apply_rotary_emb_single(norm(x).to(dtype), freqs_cis)``.

  The mean-square reduction stays in float32, matching upstream's normalisation order.
  """
  xf = x.float()
  normed = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + norm.eps)).type_as(x) * norm.weight
  return apply_rotary_emb_single(normed.to(dtype), freqs_cis)


class NomasMLP(nn.Module):
  """Two-layer MLP named after the checkpoint (``fc1`` / ``fc2``)."""

  def __init__(self, dim: int, hidden_dim: int, act: str = "gelu_tanh"):
    super().__init__()
    if act not in ("gelu_tanh", "silu"):
      raise ValueError(f"Unsupported MLP activation {act!r}.")
    self.act = act
    self.fc1 = nn.Linear(dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.fc1(x)
    x = F.gelu(x, approximate="tanh") if self.act == "gelu_tanh" else F.silu(x)
    return self.fc2(x)


class NomasModulate(nn.Module):
  """``SiLU`` then ``Linear``, kept at ``.linear`` to match the checkpoint."""

  def __init__(self, dim: int, factor: int = 6):
    super().__init__()
    self.linear = nn.Linear(dim, factor * dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.linear(F.silu(x))


class NomasPatchEmbed(nn.Module):
  """Conv3d patchify; ``[B, C, T, H, W] -> [B, T*H*W, D]``."""

  def __init__(self, patch_size: Sequence[int], in_chans: int, embed_dim: int):
    super().__init__()
    self.patch_size = tuple(patch_size)
    self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.proj(x).flatten(2).transpose(1, 2)


class NomasTimestepEmbedder(nn.Module):
  """Sinusoidal timestep embedding followed by the checkpoint's two-layer ``mlp``."""

  def __init__(self, hidden_size: int):
    super().__init__()
    self.mlp = nn.Sequential(nn.Linear(256, hidden_size), nn.SiLU(),
                             nn.Linear(hidden_size, hidden_size))

  def forward(self, t: torch.Tensor) -> torch.Tensor:
    return self.mlp(timestep_embedding(t).type(self.mlp[0].weight.dtype))


class NomasFinalLayer(nn.Module):
  """AdaLN final layer; ``c`` is the per-request timestep vector ``[B, D]``."""

  def __init__(self, hidden_size: int, patch_size: Sequence[int], out_channels: int):
    super().__init__()
    self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.linear = nn.Linear(hidden_size, math.prod(patch_size) * out_channels)
    # `nn.Sequential` keeps the checkpoint's `.1.weight` key.
    self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

  def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
    return self.linear(modulate(self.norm_final(x), shift, scale))


def masked_self_attention(q, k, v, attn_mask):
  """Masked SDPA over the text stream (replicated: no cache, no sequence sharding).

  Goes through cache-dit's dispatcher with ``parallel_config=None`` because the mask is a plain
  dense tensor here, unlike the segment plan's varlen / block masks.  ``backend="native"`` is pinned
  at the call site: the registered ``_flash_3`` backend has no ``attn_mask`` parameter and the
  dispatcher *silently drops* unsupported arguments (``register.py``), which would attend over the
  padded text keys.  The text stream is two blocks deep, so the backend choice costs nothing.
  """
  out = _dispatch_attention_fn(
    q,
    k,
    v,
    attn_mask=attn_mask[:, None, None, :],
    backend="native",
    parallel_config=None,
  )
  # Padded text rows must come out as zeros, matching the upstream `attention` helper.
  return out.flatten(2) * attn_mask[:, :, None]


class NomasTokenRefinerBlock(nn.Module):
  """One masked self-attention + MLP block of the text refiner."""

  def __init__(self, hidden_size: int, heads_num: int):
    super().__init__()
    self.heads_num = heads_num
    self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
    self.self_attn_qkv = nn.Linear(hidden_size, hidden_size * 3)
    self.self_attn_proj = nn.Linear(hidden_size, hidden_size)
    self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
    self.mlp = NomasMLP(hidden_size, 4 * hidden_size, act="silu")
    self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

  def forward(self, x: torch.Tensor, c: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    gate_msa, gate_mlp = self.adaLN_modulation(c).chunk(2, dim=1)
    qkv = self.self_attn_qkv(self.norm1(x))
    b, length, _ = qkv.shape
    q, k, v = qkv.view(b, length, 3, self.heads_num, -1).unbind(2)
    x = x + apply_gate(self.self_attn_proj(masked_self_attention(q, k, v, attn_mask)), gate_msa)
    return x + apply_gate(self.mlp(self.norm2(x)), gate_mlp)


class NomasTokenRefiner(nn.Module):
  """Stack of :class:`NomasTokenRefinerBlock`."""

  def __init__(self, hidden_size: int, heads_num: int, depth: int):
    super().__init__()
    self.blocks = nn.ModuleList(
      [NomasTokenRefinerBlock(hidden_size, heads_num) for _ in range(depth)])

  def forward(self, x: torch.Tensor, c: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.clone().bool()
    mask[:, 0] = True  # avoid an all-masked row producing NaN
    for block in self.blocks:
      x = block(x, c, mask)
    return x


class NomasTextRefiner(nn.Module):
  """Projects text features to the residual width and refines them per token."""

  def __init__(self, config):
    super().__init__()
    d = config.hidden_size
    self.input_embedder = nn.Linear(config.text_states_dim, d)
    self.t_embedder = NomasTimestepEmbedder(d)
    self.c_embedder = PixArtAlphaTextProjection(config.text_states_dim, d, act_fn="silu")
    self.individual_token_refiner = NomasTokenRefiner(d, config.num_attention_heads, depth=2)

  def forward(self, x: torch.Tensor, t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(x.dtype).unsqueeze(-1)
    pooled = (x * weights).sum(1) / weights.sum(1)
    c = self.t_embedder(t) + self.c_embedder(pooled)
    return self.individual_token_refiner(self.input_embedder(x), c, mask)


def _camera_rope_tables(device=None):
  return get_nd_rotary_pos_embed([8, 28, 28], (0, 0, 0), (7500, 1, 1), theta=256, device=device)


class NomasCameraAttention(nn.Module):
  """Per-block camera conditioning: each video token reads its frame's extrinsics value."""

  def __init__(self, config):
    super().__init__()
    self.chunk_size = config.camera_chunk_size
    self.keyboard_embed = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 128))
    self.key_attn_q_norm, self.key_attn_k_norm = NomasRMSNorm(64), NomasRMSNorm(64)
    self.img_attn_q = nn.Linear(config.hidden_size, 1024, bias=False)
    self.keyboard_attn_kv = nn.Linear(128, 2048, bias=False)
    self.proj_keyboard = nn.Linear(1024, config.hidden_size, bias=False)
    if self.chunk_size != 1:
      # Only the chunked branch rotates; `[7500, 64]` twice per layer is ~4 MiB, i.e. ~207 MiB over
      # 54 layers, and the released checkpoint never takes that branch.
      self._build_rope_tables(torch.device("cpu"))

  def _build_rope_tables(self, device) -> None:
    cos, sin = _camera_rope_tables(device)
    self.register_buffer("freqs_cos", cos, persistent=False)
    self.register_buffer("freqs_sin", sin, persistent=False)

  def forward(self, x, camera, spatial_tokens, has_camera, token_offset=0):
    """``x`` holds video tokens ``[token_offset, token_offset + n)`` of the segment."""
    b, n, _ = x.shape
    frames = camera.shape[1]
    if self.chunk_size == 1:
      # Softmax over a single key is exactly 1, so each token reads its frame's value.
      v = self.keyboard_attn_kv(self.keyboard_embed(camera))
      v = v.view(b, frames, 2, 16, 64)[:, :, 1]
      frame = (torch.arange(n, device=x.device) + token_offset) // spatial_tokens
      out = self.proj_keyboard(v[:, frame].flatten(2))
    else:
      q = self.img_attn_q(x).view(b, n, 16, 64)
      k, v = self.keyboard_attn_kv(self.keyboard_embed(camera)).view(b, frames, 2, 16, 64).unbind(2)
      tokens = torch.arange(n, device=x.device) + token_offset
      qids = (tokens // spatial_tokens).clamp(max=7499)
      kids = torch.arange(frames, device=x.device).clamp(max=7499)
      q = rms_norm_rotate(self.key_attn_q_norm, q, (self.freqs_cos[qids], self.freqs_sin[qids]),
                          v.dtype)
      k = rms_norm_rotate(self.key_attn_k_norm, k, (self.freqs_cos[kids], self.freqs_sin[kids]),
                          v.dtype)
      chunks = []
      # Each camera chunk covers a contiguous token range; intersect with ours.
      for start in range(0, frames, self.chunk_size):
        end = min(start + self.chunk_size, frames)
        lo = max(start * spatial_tokens, token_offset) - token_offset
        hi = min(end * spatial_tokens, token_offset + n) - token_offset
        if hi <= lo:
          continue
        out = F.scaled_dot_product_attention(
          q[:, lo:hi].transpose(1, 2),
          k[:, start:end].transpose(1, 2),
          v[:, start:end].transpose(1, 2),
        )
        chunks.append(out.transpose(1, 2).flatten(2))
      out = self.proj_keyboard(torch.cat(chunks, 1))
    if has_camera is not None:
      # Upstream intentionally promotes this residual to float32.
      out = out * has_camera.float()[:, None, None]
    return x + out


class NomasEntrySink:
  """Per-layer K/V sink the blocks write into while collecting a segment's history.

  Deliberately **not** a list/dict/dataclass: cache-dit's layerwise CPU offload walks the block
  arguments with ``_map_tensors`` and rebuilds every container it meets (``dataclasses.replace``
  for dataclasses, a fresh comprehension for lists), so a plain ``[None] * num_layers`` would be
  replaced by a copy and every K/V written by a block would be silently dropped -- which the
  pipeline only notices as ``Nomas K/V collection skipped N of 54 blocks`` after the whole segment
  has been computed.  A plain object falls through ``_map_tensors`` unchanged, keeping the identity
  of the sink across the offload hook.
  """

  def __init__(self, size: int):
    self.entries: list = [None] * size

  def __setitem__(self, index: int, value) -> None:
    self.entries[index] = value

  def __len__(self) -> int:
    return len(self.entries)


@dataclass(frozen=True, eq=False)
class NomasBlockContext:
  """Per-forward constants shared by every block of one transformer call.

  ``eq=False`` keeps dataclass ``__eq__`` away from the tensors and the mutable cache: an
  accidental comparison would raise "Boolean value of Tensor is ambiguous".

  ``entries`` stays ``None`` unless this call collects K/V; a mutable list would make Dynamo guard
  its length and recompile every step under ``--compile``.

  :param vec: Timestep embedding ``[1, D]``.
  :param rope: This rank's ``(cos, sin)`` tables.
  :param valid_length: Committed rows of ``cache`` visible to this segment.
  :param frame_offset: Absolute latent-frame offset of this segment (RoPE position base).
  :param attention: The segment's attention plan.
  :param cache: The segment KV cache.
  :param camera_embedding: Optional ``[1, F, 64]`` camera embedding.
  :param camera_has_mask: Optional ``[1]`` camera gate.
  :param spatial_tokens: Tokens per latent frame.
  :param token_offset: Absolute index of this rank's first video token.
  :param collect_cache: Whether blocks should write K/V into ``entries``.
  :param entries: The :class:`NomasEntrySink` to fill when collecting, else ``None``.
  :param parallel: Shard geometry (tensor × context parallel).
  """

  vec: torch.Tensor
  rope: Tuple[torch.Tensor, torch.Tensor]
  valid_length: int
  frame_offset: int
  attention: NomasAttentionPlan
  cache: NomasCache
  camera_embedding: Optional[torch.Tensor]
  camera_has_mask: Optional[torch.Tensor]
  spatial_tokens: int
  token_offset: int
  collect_cache: bool
  entries: Optional[NomasEntrySink]
  parallel: NomasParallelInfo


class NomasDoubleStreamBlock(nn.Module):
  """One double-stream (video + text) block with its own cache slot and camera module."""

  def __init__(self, config, index: int = 0):
    super().__init__()
    d, heads = config.hidden_size, config.num_attention_heads
    self.index = index
    # Heads after TP; Ulysses splits these again around attention.
    self.heads = heads
    self.attn_heads = heads
    for stream in ("img", "txt"):
      setattr(self, f"{stream}_mod", NomasModulate(d, factor=6))
      for i in (1, 2):
        setattr(self, f"{stream}_norm{i}", nn.LayerNorm(d, eps=1e-6, elementwise_affine=False))
      for name in ("q", "k", "v"):
        setattr(self, f"{stream}_attn_{name}", nn.Linear(d, d, bias=True))
      setattr(self, f"{stream}_attn_proj", nn.Linear(d, d, bias=True))
      for name in ("q", "k"):
        setattr(self, f"{stream}_attn_{name}_norm", NomasRMSNorm(d // heads))
      setattr(self, f"{stream}_mlp", NomasMLP(d, 4 * d, act="gelu_tanh"))
    # Set by the transformer: a non-registered reference to `action_modules[index]`, so the camera
    # weights stay under their checkpoint names while the block can call them.
    self.action = None

  def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor,
              ctx: NomasBlockContext) -> tuple[torch.Tensor, torch.Tensor]:
    img, txt = hidden_states, encoder_hidden_states
    info = ctx.parallel
    qkv, mods = [], []
    for stream, x in (("img", img), ("txt", txt)):
      mod = getattr(self, f"{stream}_mod")(ctx.vec).chunk(6, -1)
      mods.append(mod)
      x = modulate_for_linear(getattr(self, f"{stream}_norm1")(x), shift=mod[0], scale=mod[1])
      q, k, v = [
        getattr(self, f"{stream}_attn_{name}")(x).unflatten(-1, (self.heads, -1))
        for name in ("q", "k", "v")
      ]
      q_norm = getattr(self, f"{stream}_attn_q_norm")
      k_norm = getattr(self, f"{stream}_attn_k_norm")
      if stream == "img":
        # QK norm, cast and RoPE fused; values are unchanged.
        q = rms_norm_rotate(q_norm, q, ctx.rope, v.dtype)
        k = rms_norm_rotate(k_norm, k, ctx.rope, v.dtype)
        # Local tokens, all heads -> all tokens, local heads.
        q, k, v = (ulysses_all_to_all(t, info, 2, 1) for t in (q, k, v))
      else:
        q, k = q_norm(q).to(v), k_norm(k).to(v)
        # Text is replicated; keep this rank's head slice.
        lo = info.sp_rank * self.attn_heads
        q, k, v = (t[:, :, lo:lo + self.attn_heads] for t in (q, k, v))
      qkv.append((q, k, v))
    q, k, v = [torch.cat((qkv[0][i], qkv[1][i]), 1) for i in range(3)]
    if ctx.collect_cache:
      ctx.entries[self.index] = {"k": k, "v": v}
    video_tokens = q.shape[1] - txt.shape[1]
    img_attn, txt_attn = ctx.attention(q, k, v, ctx.cache.layers[self.index],
                                       ctx.valid_length).split((video_tokens, txt.shape[1]), 1)
    img_attn = ulysses_all_to_all(img_attn, info, 1, 2)
    txt_attn = sp_all_gather(txt_attn, info, 2)
    outputs = []
    for stream, x, attn, mod in zip(("img", "txt"), (img, txt), (img_attn, txt_attn), mods):
      x = gated_residual(x, getattr(self, f"{stream}_attn_proj")(attn.flatten(2)), mod[2])
      x = gated_residual(
        x,
        getattr(self, f"{stream}_mlp")(modulate_for_linear(getattr(self, f"{stream}_norm2")(x),
                                                           shift=mod[3],
                                                           scale=mod[4])),
        mod[5],
      )
      outputs.append(x)
    img, txt = outputs
    if ctx.camera_embedding is not None:
      img = self.action(img,
                        ctx.camera_embedding,
                        ctx.spatial_tokens,
                        ctx.camera_has_mask,
                        token_offset=ctx.token_offset)
    return img, txt


class NomasTransformer3DModel(ModelMixin, ConfigMixin):
  """HY-Nomas segment-autoregressive video transformer.

  Note:
    ``double_blocks`` is the denoising stack cache-dit caches.  Each block carries its ``index`` and
    its own cache slot, so a single block call per iteration reproduces the upstream loop exactly.
    ``predict(collect_cache=True)`` is the only entry that writes K/V and never runs under the
    cached wrapper.
  """

  _repeated_blocks = ["NomasDoubleStreamBlock"]
  _supports_gradient_checkpointing = False
  _no_split_modules = ["NomasDoubleStreamBlock"]

  @register_to_config
  def __init__(
    self,
    hidden_size: int = 2048,
    num_attention_heads: int = 16,
    num_layers: int = 54,
    in_channels: int = 32,
    out_channels: int = 32,
    text_states_dim: int = 3584,
    patch_size: Sequence[int] = (1, 1, 1),
    rope_dim_list: Sequence[int] = (16, 56, 56),
    rope_theta: float = 256.0,
    camera_chunk_size: int = 1,
    attention_backend: str = "flex",
  ):
    super().__init__()
    patch_size = tuple(int(value) for value in patch_size)
    rope_dim_list = tuple(int(value) for value in rope_dim_list)
    if hidden_size % num_attention_heads:
      raise ValueError("hidden size must be divisible by head count")
    if sum(rope_dim_list) != hidden_size // num_attention_heads:
      raise ValueError("RoPE dimensions must sum to head dimension")
    if len(rope_dim_list) != 3 or any(d <= 0 or d % 2 for d in rope_dim_list):
      raise ValueError("RoPE requires three positive even dimensions")
    if patch_size != (1, 1, 1):
      raise ValueError("Nomas requires patch_size=(1, 1, 1)")
    if attention_backend not in ("fa3", "flex", "sdpa"):
      raise ValueError("Nomas attention backend must be fa3, flex or sdpa")
    if camera_chunk_size != 1:
      # That branch keeps an upstream-faithful raw SDPA over a camera chunk instead of going
      # through cache-dit's dispatcher; it must be converted before it is ever enabled.
      logger.warning(
        "HY-Nomas: camera_chunk_size=%d uses a raw SDPA camera branch that bypasses "
        "cache-dit's attention dispatcher (upstream parity only; the released "
        "checkpoint uses 1). Convert it to `_dispatch_attention_fn` before enabling.",
        camera_chunk_size)
    self.hidden_size = hidden_size
    self.num_attention_heads = num_attention_heads
    self.local_attention_heads = num_attention_heads
    self.num_channels_latents = in_channels
    self.attention_backend = attention_backend
    # Written by `set_parallel` before the first forward.
    self.parallel = NomasParallelInfo()
    self.img_in = NomasPatchEmbed(patch_size, in_channels, hidden_size)
    self.txt_in = NomasTextRefiner(self.config)
    self.time_in = NomasTimestepEmbedder(hidden_size)
    self.double_blocks = nn.ModuleList(
      [NomasDoubleStreamBlock(self.config, index=index) for index in range(num_layers)])
    self.final_layer = NomasFinalLayer(hidden_size, patch_size, out_channels)
    self.action_modules = nn.ModuleList(
      [NomasCameraAttention(self.config) for _ in range(num_layers)])
    self.extrinsics_embedding_camctrl = nn.Linear(64, 64)
    self.c2ws_hidden_states_layer1 = nn.Linear(64, 64)
    self.c2ws_hidden_states_layer2 = nn.Linear(64, 64)
    for block, action in zip(self.double_blocks, self.action_modules):
      # Not registered as a child: the weights must stay under `action_modules.{i}.*`.
      object.__setattr__(block, "action", action)

  def set_parallel(self, parallel: NomasParallelInfo) -> "NomasTransformer3DModel":
    """Adopt a shard geometry before the first forward.

    Owns **both** head constants: ``block.heads`` is what the projections' ``unflatten`` uses (heads
    per rank after tensor parallelism) and ``block.attn_heads`` is what the replicated text stream
    slices and the persistent cache allocates (heads per rank after tensor × context parallelism).
    A tensor-parallel planner must therefore *not* additionally call ``shard_div_attr`` on these
    blocks.

    :param parallel: Target geometry.
    :returns: ``self``, for chaining.
    """
    heads = self.config.num_attention_heads
    # Checked before `local_heads` so a tensor-parallel mismatch names tp_size, not tp*sp.
    if heads % parallel.tp_size:
      raise ValueError(f"{heads} attention heads are not divisible by "
                       f"tp_size={parallel.tp_size}")
    local = parallel.local_heads(heads)
    self.parallel = parallel
    self.local_attention_heads = local
    for block in self.double_blocks:
      block.heads = heads // parallel.tp_size
      block.attn_heads = local
    return self

  def set_tensor_parallel(self, tp_size: int) -> "NomasTransformer3DModel":
    """Adopt a tensor-parallel degree, keeping the context-parallel geometry in place.

    :param tp_size: Tensor-parallel size; ``1`` disables sharding.
    :returns: ``self``, for chaining.
    """
    current = self.parallel
    return self.set_parallel(
      NomasParallelInfo(
        tp_size=tp_size,
        sp_size=current.sp_size,
        sp_rank=current.sp_rank,
        group=current.group,
      ))

  def set_context_parallel(self, sp_size: int) -> "NomasTransformer3DModel":
    """Adopt a context-parallel degree, keeping the tensor-parallel geometry in place.

    The process group is filled in lazily by :meth:`_sync_context_parallel` once cache-dit has
    installed `_cp_config` (the planner runs before the mesh exists).

    :param sp_size: Ulysses degree; ``1`` disables sharding.
    :returns: ``self``, for chaining.
    """
    current = self.parallel
    return self.set_parallel(
      NomasParallelInfo(
        tp_size=current.tp_size,
        sp_size=sp_size,
        sp_rank=current.sp_rank,
        group=current.group,
      ))

  def _sync_context_parallel(self) -> "NomasTransformer3DModel":
    """Bind this rank's CP group/rank from the runtime `_cp_config`, once.

    The context-parallel planner runs *before* cache-dit creates the mesh, so it can only record
    the degree.  Every collective of this model takes that group explicitly: the global process
    group would divide by ``tp_size * sp_size`` on a hybrid run.

    :returns: ``self``, for chaining.
    """
    if self.parallel.sp_size <= 1 or self.parallel.group is not None:
      return self
    cp_config = getattr(self, "_cp_config", None)
    if cp_config is None:
      raise RuntimeError("context parallelism is enabled but `_cp_config` is missing; the CP "
                         "runtime must arm the transformer before the first forward.")
    mesh = getattr(cp_config, "_flattened_mesh", None)
    if mesh is None:
      raise RuntimeError("context parallelism is enabled but the CP mesh is missing")
    self._cp_group = mesh.get_group()
    self._cp_rank = torch.distributed.get_rank(self._cp_group)
    self._cp_world_size = torch.distributed.get_world_size(self._cp_group)
    if self._cp_world_size != self.parallel.sp_size:
      raise RuntimeError(f"CP group has {self._cp_world_size} ranks but the planner recorded "
                         f"sp_size={self.parallel.sp_size}")
    return self.set_parallel(
      NomasParallelInfo(
        tp_size=self.parallel.tp_size,
        sp_size=self.parallel.sp_size,
        sp_rank=self._cp_rank,
        group=self._cp_group,
      ))

  def post_load_weights(self) -> "NomasTransformer3DModel":
    """Rebuild the camera RoPE tables, which the checkpoint does not ship."""
    if self.config.camera_chunk_size == 1:
      return self
    cos, sin = _camera_rope_tables(torch.device("cpu"))
    for module in self.action_modules:
      param = module.img_attn_q.weight
      module.freqs_cos = cos.to(device=param.device, dtype=param.dtype)
      module.freqs_sin = sin.to(device=param.device, dtype=param.dtype)
    return self

  @staticmethod
  def layerwise_offload_filter(name: str, module: nn.Module) -> bool:
    """Stream only the denoising blocks; the camera/text/head modules run every step.

    :param name: Module name inside the transformer.
    :param module: Candidate submodule.
    :returns: Whether ``module`` should be streamed.
    """
    return isinstance(module, NomasDoubleStreamBlock)

  def resident_to(self, device) -> "NomasTransformer3DModel":
    """Move the never-streamed parts onto the compute device.

    :param device: Compute device.
    :returns: ``self``, for chaining.
    """
    for module in (self.img_in, self.txt_in, self.time_in, self.final_layer, self.action_modules,
                   self.extrinsics_embedding_camctrl, self.c2ws_hidden_states_layer1,
                   self.c2ws_hidden_states_layer2):
      module.to(device)
    return self

  def allocate_cache(self, capacity: int) -> NomasCache:
    """Allocate this segment session's KV cache.

    :param capacity: Row capacity in tokens.
    :returns: A zeroed cache sized for this rank's heads.
    """
    p = next(self.parameters())
    heads = self.parallel.local_heads(self.config.num_attention_heads)
    if heads != self.local_attention_heads:
      raise RuntimeError(f"shard geometry is stale: parallel says {heads} heads per rank while the "
                         f"blocks carry {self.local_attention_heads}; call `set_parallel` first")
    return NomasCache.allocate(
      self.config,
      capacity,
      device=p.device,
      dtype=p.dtype,
      num_heads=heads,
    )

  def embed_camera(self, norm_extrinsics: torch.Tensor) -> torch.Tensor:
    """Already-normalized ``[1, F, 4, 4]`` extrinsics; pad the front to 4 frames."""
    if (norm_extrinsics.ndim != 4 or norm_extrinsics.shape[0] != 1
        or norm_extrinsics.shape[2:] != (4, 4) or norm_extrinsics.shape[1] == 0):
      raise ValueError("normalized extrinsics must have shape [1, F>0, 4, 4]")
    p = self.extrinsics_embedding_camctrl.weight
    x = norm_extrinsics.to(device=p.device, dtype=p.dtype)
    pad = (-x.shape[1]) % 4
    if pad:
      x = torch.cat((x[:, :1].expand(-1, pad, -1, -1), x), 1)
    x = self.extrinsics_embedding_camctrl(x.reshape(1, -1, 64))
    return x + self.c2ws_hidden_states_layer2(F.silu(self.c2ws_hidden_states_layer1(x)))

  def prepare_segment_context(
    self,
    latent: torch.Tensor,
    text_mask: torch.Tensor,
    cache: NomasCache,
    *,
    text_offsets=None,
    video_windows=None,
    hide_cached_action_tails: bool = False,
    rope=None,
  ):
    """Build timestep-independent metadata for one segment.

    :param latent: ``[1, C, T, H, W]`` latents of the segment.
    :param text_mask: ``[1, S]`` boolean text mask.
    :param cache: The segment KV cache.
    :param text_offsets: Optional head/action text partition.
    :param video_windows: Optional action window tiling.
    :param hide_cached_action_tails: Whether to hide action tails of earlier segments.
    :param rope: Optional prebuilt RoPE tables that survive an eviction.
    :returns: ``(attention_plan, local_rope, (t, h, w))``.
    """
    self._sync_context_parallel()
    t, h, w = latent.shape[2:]
    c = self.config
    attention = NomasAttentionPlan(
      text_mask,
      cache.history_mask(hide_cached_action_tails),
      t,
      h * w,
      text_offsets,
      video_windows,
      self.attention_backend,
    )
    if rope is None:
      offset = cache.frame_offset
      # Upstream builds offset tables on the CPU; keep its construction.
      rope = get_nd_rotary_pos_embed(
        c.rope_dim_list,
        (offset, 0, 0),
        (offset + t, h, w),
        c.rope_theta,
        device=latent.device if offset == 0 else None,
      )
      rope = tuple(x.to(latent.device) for x in rope)
      lo, hi = local_token_range(t * h * w, self.parallel)
      rope = tuple(x[lo:hi] for x in rope)
    else:
      # A caller-supplied table must already be this rank's slice: slicing it again under context
      # parallelism would drop half of every rank's positions.
      expected = t * h * w // self.parallel.sp_size
      if rope[0].shape[0] != expected:
        raise ValueError(f"prebuilt RoPE table has {rope[0].shape[0]} rows but this rank owns "
                         f"{expected} tokens; pass the full table instead")
    return attention, rope, (t, h, w), cache.frame_offset

  def _block_list(self, force_full_compute: bool):
    """Resolve the block list, looking through cache-dit's one-element wrapper when needed.

    :param force_full_compute: Whether the caller must run every block.
    :returns: The list of blocks to iterate.
    :raises RuntimeError: When a wrapper cannot be looked through.
    """
    blocks = self.double_blocks
    if len(blocks) != 1:
      return blocks
    if isinstance(blocks[0], NomasDoubleStreamBlock):
      return blocks
    if not force_full_compute:
      # The expected cached path: one wrapper replays the stack and may skip middle blocks.
      return blocks
    raw = getattr(blocks[0], "transformer_blocks", None)
    if raw is None:
      raise RuntimeError(
        f"Cannot bypass the block-skip wrapper of type {type(blocks[0]).__name__}: it exposes no "
        f"`transformer_blocks` list, and the K/V write must run every block. Please report this "
        f"with the cache-dit version.")
    if len(raw) != self.config.num_layers:
      raise RuntimeError(
        f"Cannot bypass the block-skip wrapper {type(blocks[0]).__name__}: its block list has "
        f"{len(raw)} entries, expected {self.config.num_layers}. The K/V write must run every "
        f"block, so it cannot go through a partially replayed stack.")
    return raw

  def _run(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    text_mask: torch.Tensor,
    *,
    cache: NomasCache,
    camera_embedding=None,
    camera_has_mask=None,
    text_offsets=None,
    video_windows=None,
    hide_cached_action_tails: bool = False,
    collect_cache: bool = False,
    context=None,
  ):
    """Shared body of :meth:`forward` and :meth:`predict`."""
    if self.training or torch.is_grad_enabled():
      raise RuntimeError("Nomas cache inference requires eval() and inference_mode()")
    c = self.config
    self._check_inputs(hidden_states, timestep, encoder_hidden_states, text_mask, cache)
    _, _, t, h, w = hidden_states.shape
    self._check_camera(camera_embedding, camera_has_mask, t, hidden_states.dtype)
    if context is None:
      context = self.prepare_segment_context(
        hidden_states,
        text_mask,
        cache,
        text_offsets=text_offsets,
        video_windows=video_windows,
        hide_cached_action_tails=hide_cached_action_tails,
      )
    attention, rope, shape, frame_offset = context
    # All three have to match: an eviction can restore the same `valid_length` while the RoPE
    # position base moved on, and reusing the old context would then place the segment at the wrong
    # absolute frames.
    if (shape != (t, h, w) or cache.valid_length != attention.history_length
        or cache.frame_offset != frame_offset):
      raise ValueError("stale Nomas segment context; rebuild after cache changes")
    lo, _ = local_token_range(t * h * w, self.parallel)
    entries = NomasEntrySink(c.num_layers) if collect_cache else None
    with torch.autocast(
        device_type=hidden_states.device.type,
        dtype=hidden_states.dtype,
        enabled=hidden_states.dtype in (torch.bfloat16, torch.float16),
    ):
      # `img_in` runs on the full sequence (cheap, no attention): under context parallelism the hook
      # plan splits the first block's `hidden_states` argument.
      img = self.img_in(hidden_states)
      txt = self.txt_in(encoder_hidden_states, timestep, text_mask)
      vec = self.time_in(timestep)
      ctx = NomasBlockContext(
        vec=vec,
        rope=rope,
        valid_length=cache.valid_length,
        frame_offset=frame_offset,
        attention=attention,
        cache=cache,
        camera_embedding=camera_embedding,
        camera_has_mask=camera_has_mask,
        spatial_tokens=h * w,
        token_offset=lo,
        collect_cache=collect_cache,
        entries=entries,
        parallel=self.parallel,
      )
      for block in self._block_list(force_full_compute=collect_cache):
        img, txt = block(img, txt, ctx)
      img = self.final_layer(img, vec)
    if collect_cache and any(entry is None for entry in entries.entries):
      missing = [index for index, entry in enumerate(entries.entries) if entry is None]
      raise RuntimeError(
        f"Nomas K/V collection skipped {len(missing)} of {c.num_layers} blocks (first: "
        f"{missing[0]}); committing a partial history would corrupt every later segment.")
    img = img.view(1, t, h, w, c.out_channels).permute(0, 4, 1, 2, 3).contiguous()
    if collect_cache:
      return img, entries.entries
    return img, None

  def forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    *,
    text_mask: torch.Tensor,
    cache: NomasCache,
    camera_embedding=None,
    camera_has_mask=None,
    text_offsets=None,
    video_windows=None,
    hide_cached_action_tails: bool = False,
    collect_cache: bool = False,
    context=None,
  ):
    """Denoising forward; the cached runtime patches this method.

    :param hidden_states: ``[1, C, T, H, W]`` latents.
    :param encoder_hidden_states: ``[1, S, text_states_dim]`` text features.
    :param timestep: ``[1]`` fractional timestep scaled by 1000.
    :param text_mask: ``[1, S]`` boolean text mask.
    :param cache: The segment KV cache.
    :param camera_embedding: Optional ``[1, F, 64]`` camera embedding.
    :param camera_has_mask: Optional ``[1]`` camera gate.
    :param text_offsets: Optional head/action text partition.
    :param video_windows: Optional action window tiling.
    :param hide_cached_action_tails: Whether to hide action tails of earlier segments.
    :param collect_cache: Rejected here; K/V collection must use :meth:`predict`.
    :param context: Optional prebuilt segment context.
    :returns: The velocity prediction ``[1, C, T, H, W]``.
    """
    if collect_cache:
      raise RuntimeError("Nomas K/V collection must run through predict(collect_cache=True): the "
                         "cached forward may skip blocks and would write partial history.")
    video, _ = self._run(
      hidden_states,
      timestep,
      encoder_hidden_states,
      text_mask,
      cache=cache,
      camera_embedding=camera_embedding,
      camera_has_mask=camera_has_mask,
      text_offsets=text_offsets,
      video_windows=video_windows,
      hide_cached_action_tails=hide_cached_action_tails,
      collect_cache=False,
      context=context,
    )
    return video

  def predict(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    text_mask: torch.Tensor,
    *,
    cache: NomasCache,
    collect_cache: bool = False,
    **kwargs,
  ):
    """Read-only prediction that may also collect this segment's K/V.

    Never wrapped by the cached-block runtime, and it always runs every block, so the entries it
    returns are complete.

    :param hidden_states: ``[1, C, T, H, W]`` latents.
    :param timestep: ``[1]`` fractional timestep scaled by 1000.
    :param encoder_hidden_states: ``[1, S, text_states_dim]`` text features.
    :param text_mask: ``[1, S]`` boolean text mask.
    :param cache: The segment KV cache.
    :param collect_cache: Whether to return this segment's K/V entries.
    :param kwargs: Additional arguments forwarded to :meth:`_run`.
    :returns: ``(video, entries)`` where ``entries`` is ``None`` unless ``collect_cache``.
    """
    return self._run(
      hidden_states,
      timestep,
      encoder_hidden_states,
      text_mask,
      cache=cache,
      collect_cache=collect_cache,
      **kwargs,
    )

  def _check_inputs(self, hidden_states, timestep, encoder_hidden_states, text_mask, cache) -> None:
    c = self.config
    dtype = next(self.parameters()).dtype
    if (hidden_states.ndim != 5 or hidden_states.shape[:2] != (1, c.in_channels)
        or min(hidden_states.shape[2:]) <= 0):
      raise ValueError("latents must have shape [1, in_channels, T>0, H>0, W>0]")
    if text_mask.ndim != 2 or text_mask.shape[0] != 1:
      raise ValueError("text mask must have shape [1, S]")
    if encoder_hidden_states.shape != (1, text_mask.shape[-1], c.text_states_dim):
      raise ValueError("text feature shape does not match mask/model")
    if timestep.shape != (1, ):
      raise ValueError("timestep must have shape [1]")
    if len(cache.layers) != c.num_layers or cache.visible.device != hidden_states.device:
      raise ValueError("cache layer count or device mismatch")
    if cache.layers[0]["k"].shape[2] != self.local_attention_heads:
      raise ValueError(f"cache holds {cache.layers[0]['k'].shape[2]} heads but this rank owns "
                       f"{self.local_attention_heads}; allocate the cache after parallelism")
    for tensor in (hidden_states, encoder_hidden_states):
      if tensor.device != self.device or tensor.dtype != dtype:
        raise ValueError("latent/text dtype and device must match model weights")
    if text_mask.device != self.device or timestep.device != self.device:
      raise ValueError("mask and timestep device must match model weights")

  def _check_camera(self, camera_embedding, camera_has_mask, frames, dtype) -> None:
    if camera_embedding is not None:
      if camera_embedding.shape != (1, frames, 64):
        raise ValueError("camera/latent alignment mismatch")
      if camera_embedding.device != self.device or camera_embedding.dtype != dtype:
        raise ValueError("camera embedding dtype/device mismatch")
    if camera_has_mask is not None and (camera_has_mask.shape != (1, )
                                        or camera_has_mask.device != self.device):
      raise ValueError("camera mask must have shape [1] on model device")
