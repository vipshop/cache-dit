"""Ming-Image DiT: Z-Image transformer plus reference-latent / direct-condition inputs.

`Ming-Image-0.1-Design` and `Ming-Image-0.1-Design-Layer` are the same Z-Image stack with two extra
input paths, so this module subclasses the diffusers `ZImageTransformer2DModel` instead of vendoring
a second copy of the block/attention/RoPE/adaLN code:

* reference latents (`ref_latents`) are appended to the noisy latents along the frame axis, which
  lets the context refiners and the main blocks attend to the reference frames;
* direct conditions (`direct_condition`) are appended to the prompt features along the sequence
  axis, i.e. they take part in the same RoPE axis as the text tokens. They are appended *after*
  `cap_embedder`, exactly like the reference implementation: the connector already projects them
  into hidden space, so they must not go through the prompt embedder again.

Both extra inputs are explicit `forward` arguments (no global forward context), and the returned
latents are cropped back to the generated frames so callers never see reference frames.

The two checkpoints differ only in how partially filled padding slots are represented:

* `alignment_padding_mode="zero_masked"` (`Design`): sequence padding is zeroed, no
  `x_pad_token` / `cap_pad_token` weights exist in the checkpoint.
* `alignment_padding_mode="learned"` (`Design-Layer`): padding uses learned `x_pad_token` /
  `cap_pad_token` embeddings.

Note:
  Keep the constraints below intact, cache-dit's dispatch depends on them:

  * The class must stay a `ZImageTransformer2DModel` subclass with the parent's module tree
    (`noise_refiner`, `context_refiner`, `layers`, `all_x_embedder`, `all_final_layer`), so the
    `layers` stack keeps working with `ForwardPattern.Pattern_3` and with Z-Image context/tensor
    parallelism.
  * `_repeated_blocks` is inherited (`ZImageTransformerBlock`) so regional quantization and
    `compile_repeated_blocks()` still find the denoising stack.
  * Exactly one block is called per loop iteration inside `super().forward()`, which is what the
    cached-block wrapper replaces; this subclass must not reimplement that loop.
  * Sequence padding is still applied by the parent `_prepare_sequence`, so the zero-masked
    variant exposes `x_pad_token` / `cap_pad_token` as non-persistent zero buffers: the parent
    keeps running, while `state_dict()` and strict checkpoint loading stay free of those keys.
  * `cap_embedder` is rebuilt as `_DirectCapEmbedder` from the parent's own children, so the
    `cap_embedder.0.*` / `cap_embedder.1.*` weights and the `_skip_layerwise_casting_patterns`
    entry are untouched. Its `direct_rows` payload is set and cleared around the single parent
    `forward` call, which keeps `forward` non-reentrant per instance (one model per process).
  * Zero-masked padding rows are dropped from the attention mask by two small parent overrides,
    `_prepare_sequence` for the refiners and `_build_unified_sequence` for the main blocks. Both
    masks stay full-length: the context-parallel attention paths collect full K/V before applying
    the mask, so the mask is not sharded along with the sequence.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from diffusers.configuration_utils import register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_z_image import (
  SEQ_MULTI_OF,
  ZImageTransformer2DModel,
)

ZERO_MASKED_PADDING = "zero_masked"
LEARNED_PADDING = "learned"

# Ming-Image keeps Z-Image's axes_dims but the frame axis holds a whole batch of frames.
MING_AXES_LENS = [20480, 512, 512]


class _DirectCapEmbedder(nn.Sequential):
  """Cap embedder that splices already-embedded direct conditions into its output.

  Ming-Image appends direct conditions to the prompt features *after* `cap_embedder`, i.e. they are
  already in hidden space and must skip the `RMSNorm` + `Linear` pair. Rebuilding the parent's
  `nn.Sequential` from its own children keeps the `cap_embedder.0.*` / `cap_embedder.1.*` parameter
  names, and the owning transformer hands over the placeholder row ranges to replace through
  `direct_rows`, together with the expected row count of the concatenated prompt batch.

  :param base: The parent `cap_embedder`, a `RMSNorm` + `Linear` sequence.
  """

  def __init__(self, base: nn.Sequential) -> None:
    super().__init__(*base)
    self.direct_rows: list[tuple[int, int, torch.Tensor]] = []
    self.direct_total = 0

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    out = super().forward(x)
    if not self.direct_rows:
      return out
    if out.shape[0] != self.direct_total:
      raise RuntimeError(
        f"Prompt batch has {out.shape[0]} rows but {self.direct_total} direct-condition rows were "
        "planned; the parent prompt padding must have changed.")

    pieces, cursor = [], 0
    for start, end, rows in self.direct_rows:
      pieces.append(out[cursor:start])
      pieces.append(rows.to(device=out.device, dtype=out.dtype))
      cursor = end
    pieces.append(out[cursor:])
    return torch.cat(pieces, dim=0)


class MingImageTransformer2DModel(ZImageTransformer2DModel):
  """Ming-Image joint DiT, reusing the Z-Image transformer stack.

  :param all_patch_size: Spatial patch sizes; only the first entry is used by Ming-Image.
  :param all_f_patch_size: Temporal patch sizes, aligned with `all_patch_size`.
  :param in_channels: Latent channels of the noisy and reference latents.
  :param dim: Hidden size.
  :param n_layers: Number of main transformer blocks.
  :param n_refiner_layers: Number of noise/context refiner blocks.
  :param n_heads: Number of attention heads.
  :param n_kv_heads: Number of key/value heads (GQA).
  :param norm_eps: RMSNorm epsilon.
  :param qk_norm: Whether to apply QK normalization in attention.
  :param cap_feat_dim: Channel count of the prompt features before `cap_embedder`.
  :param siglip_feat_dim: Unused by Ming-Image, kept for Z-Image signature compatibility.
  :param rope_theta: RoPE base frequency.
  :param t_scale: Timestep scaling factor.
  :param axes_dims: RoPE dimensions of the (frame, height, width) axes; the sum must equal
    `dim // n_heads`.
  :param axes_lens: RoPE table lengths per axis. Ming-Image ships `[20480, 512, 512]`, the first
    axis has to cover every prompt token plus every latent token of the concatenated frame axis.
  :param alignment_padding_mode: `"zero_masked"` for a zero padding token without checkpoint
    weights, `"learned"` for learned `x_pad_token` / `cap_pad_token` embeddings.
  :param multi_frame_output: Whether the checkpoint emits more than one frame per sample. Single
    frame checkpoints are cropped to one frame on output, matching the upstream `unpatchify()`.
  """

  @register_to_config
  def __init__(
    self,
    all_patch_size=(2, ),
    all_f_patch_size=(1, ),
    in_channels: int = 16,
    dim: int = 3840,
    n_layers: int = 30,
    n_refiner_layers: int = 2,
    n_heads: int = 30,
    n_kv_heads: int = 30,
    norm_eps: float = 1e-5,
    qk_norm: bool = True,
    cap_feat_dim: int = 2560,
    siglip_feat_dim: int | None = None,
    rope_theta: float = 256.0,
    t_scale: float = 1000.0,
    axes_dims: tuple[int, ...] | list[int] = (32, 48, 48),
    axes_lens: tuple[int, ...] | list[int] = tuple(MING_AXES_LENS),
    alignment_padding_mode: str = ZERO_MASKED_PADDING,
    multi_frame_output: bool = False,
  ) -> None:
    if alignment_padding_mode not in {ZERO_MASKED_PADDING, LEARNED_PADDING}:
      raise ValueError(f"Unsupported Ming-Image alignment padding mode: {alignment_padding_mode!r}")
    if not isinstance(multi_frame_output, bool):
      raise TypeError(f"`multi_frame_output` must be a bool, got {type(multi_frame_output)!r}")

    super().__init__(
      all_patch_size=all_patch_size,
      all_f_patch_size=all_f_patch_size,
      in_channels=in_channels,
      dim=dim,
      n_layers=n_layers,
      n_refiner_layers=n_refiner_layers,
      n_heads=n_heads,
      n_kv_heads=n_kv_heads,
      norm_eps=norm_eps,
      qk_norm=qk_norm,
      cap_feat_dim=cap_feat_dim,
      siglip_feat_dim=siglip_feat_dim,
      rope_theta=rope_theta,
      t_scale=t_scale,
      axes_dims=axes_dims,
      axes_lens=axes_lens,
    )

    self.alignment_padding_mode = alignment_padding_mode
    self.multi_frame_output = multi_frame_output
    # Direct conditions are hidden-space tensors appended after the prompt embedder.
    self.cap_embedder = _DirectCapEmbedder(self.cap_embedder)
    # Per-item (latent, prompt) token counts without padding, set for one forward pass only.
    self._inner_valid: list[tuple[int, int]] = []

    if alignment_padding_mode == ZERO_MASKED_PADDING:
      del self.x_pad_token
      del self.cap_pad_token
      self.register_buffer("x_pad_token", torch.zeros(1, dim), persistent=False)
      self.register_buffer("cap_pad_token", torch.zeros(1, dim), persistent=False)

  @staticmethod
  def _as_per_sample(items, batch_size: int, name: str, item_ndim: int) -> list[torch.Tensor]:
    """Split a batched tensor or validate a list into one tensor per batch item.

    :param items: A batched `(B, ...)` tensor, a single per-item tensor, or a list of per-item
      tensors.
    :param batch_size: Expected batch size.
    :param name: Argument name used in error messages.
    :param item_ndim: Rank of a single batch item, used to tell a batched tensor from a lone one.
    :returns: One tensor per batch item.
    """
    if isinstance(items, torch.Tensor):
      per_sample = list(items.unbind(dim=0)) if items.dim() == item_ndim + 1 else [items]
    else:
      per_sample = list(items)

    if len(per_sample) != batch_size:
      raise ValueError(
        f"`{name}` has {len(per_sample)} item(s) but the batch size is {batch_size}.")
    for item in per_sample:
      if item.dim() != item_ndim:
        raise ValueError(f"`{name}` items must have {item_ndim} dims, got {tuple(item.shape)}.")
    return per_sample

  def _masks_inner_padding(self, pad_token) -> bool:
    """Whether padding rows inside a sequence must be removed from the attention mask.

    Only the zero-masked variant masks them: its padding rows are plain zeros, so they have to be
    excluded as attention keys. The learned variant keeps them attendable, exactly like Z-Image.

    :param pad_token: The padding embedding the parent is about to broadcast.
    :returns: `True` when the sequence built with `pad_token` must be masked on its padding rows.
    """
    if self.alignment_padding_mode != ZERO_MASKED_PADDING:
      return False
    return pad_token is self.x_pad_token or pad_token is self.cap_pad_token

  def _prepare_sequence(
    self,
    feats,
    pos_ids,
    inner_pad_mask,
    pad_token,
    noise_mask=None,
    device=None,
  ):
    """Parent sequence preparation with zero-masked padding rows dropped from the attention mask.

    The parent only masks the cross-item batch padding, so an all-equal-length batch returns
    `attn_mask=None` even when every item carries inner padding rows. Zero-masked Ming-Image needs
    those rows masked out, so the mask is rebuilt from `inner_pad_mask` (True = padding row).

    :param feats: Per-item feature tensors.
    :param pos_ids: Per-item RoPE position ids.
    :param inner_pad_mask: Per-item boolean padding masks, True marks a padding row.
    :param pad_token: Padding embedding, also used to tell the latent path from the prompt path.
    :param noise_mask: Optional per-token noise mask, forwarded unchanged.
    :param device: Device for the rebuilt mask.
    :returns: `(feats, freqs_cis, attn_mask, item_seqlens, noise_mask_tensor)` like the parent.
    """
    outputs = super()._prepare_sequence(feats, pos_ids, inner_pad_mask, pad_token, noise_mask,
                                        device)
    if not self._masks_inner_padding(pad_token):
      return outputs

    feats, freqs_cis, _, item_seqlens, noise_mask_tensor = outputs
    attn_mask = torch.zeros((len(item_seqlens), feats.shape[1]), dtype=torch.bool, device=device)
    for index, (seq_len, inner_pad) in enumerate(zip(item_seqlens, inner_pad_mask)):
      attn_mask[index, :seq_len] = ~inner_pad
    return feats, freqs_cis, attn_mask, item_seqlens, noise_mask_tensor

  def _build_unified_sequence(
    self,
    x,
    x_freqs,
    x_seqlens,
    x_noise_mask,
    cap,
    cap_freqs,
    cap_seqlens,
    cap_noise_mask,
    siglip,
    siglip_freqs,
    siglip_seqlens,
    siglip_noise_mask,
    omni_mode,
    device,
  ):
    """Parent unified sequence builder with the same zero-masked padding handling.

    The refined latents and prompts keep their padding rows, so the main blocks need a joint mask
    that attends to the generated tokens and to the real prompt tokens only.

    :returns:`(unified, unified_freqs, attn_mask, noise_mask_tensor)` like the parent.
    """
    outputs = super()._build_unified_sequence(
      x,
      x_freqs,
      x_seqlens,
      x_noise_mask,
      cap,
      cap_freqs,
      cap_seqlens,
      cap_noise_mask,
      siglip,
      siglip_freqs,
      siglip_seqlens,
      siglip_noise_mask,
      omni_mode,
      device,
    )
    if omni_mode or not self._inner_valid:
      return outputs

    unified, unified_freqs, _, noise_mask_tensor = outputs
    attn_mask = torch.zeros((unified.shape[0], unified.shape[1]), dtype=torch.bool, device=device)
    for index, (x_valid, cap_valid) in enumerate(self._inner_valid):
      attn_mask[index, :x_valid] = True
      cap_start = x_seqlens[index]
      attn_mask[index, cap_start:cap_start + cap_valid] = True
    return unified, unified_freqs, attn_mask, noise_mask_tensor

  def forward(
    self,
    x: list[torch.Tensor],
    t: torch.Tensor,
    cap_feats: list[torch.Tensor],
    return_dict: bool = True,
    patch_size: int = 2,
    f_patch_size: int = 1,
    joint_attention_kwargs: dict[str, Any] | None = None,
    ref_latents: list[torch.Tensor] | torch.Tensor | None = None,
    direct_condition: list[torch.Tensor] | torch.Tensor | None = None,
  ):
    """Run one denoising step on Ming-Image inputs.

    :param x: Latents of the frames being generated, one `(C, F, H, W)` tensor per batch item.
    :param t: Timestep per batch item, in Z-Image's normalized unit interval.
    :param cap_feats: Prompt features, one `(T, cap_feat_dim)` tensor per batch item.
    :param return_dict: Whether to return a `Transformer2DModelOutput` instead of a tuple.
    :param patch_size: Spatial patch size, must be in `all_patch_size`.
    :param f_patch_size: Temporal patch size, must be in `all_f_patch_size`.
    :param joint_attention_kwargs: LoRA scale holder forwarded to the parent `forward`.
    :param ref_latents: Optional reference latents, `(B, C, F_ref, H, W)`, `(C, F_ref, H, W)` or one
      `(C, F_ref, H, W)` tensor per item. They are concatenated after the generated frames.
    :param direct_condition: Optional direct conditions in hidden space, one row per prompt token,
      as `(B, S, dim)`, `(S, dim)` or one `(S, dim)` tensor per item. They are concatenated after
      the embedded prompt features, as in the reference implementation, so the prompt sequence
      length grows by `S` and the latent RoPE range shifts accordingly.
    :returns: Latents of the generated frames only, with the same container type as `return_dict`
      implies. The sign convention is the parent's, this method never negates the output.
    """
    if isinstance(x[0], (list, tuple)) or isinstance(cap_feats[0], (list, tuple)):
      raise ValueError("Ming-Image supports flat Z-Image mode only, not nested (omni) inputs.")
    if len(x) != len(cap_feats):
      raise ValueError(f"`x` has {len(x)} item(s) but `cap_feats` has {len(cap_feats)}.")

    generated_frames = []
    for item in x:
      if item.dim() != 4:
        raise ValueError(f"Expected `x` items shaped (C, F, H, W), got {tuple(item.shape)}.")
      generated_frames.append(int(item.shape[1]))

    if ref_latents is not None:
      refs = self._as_per_sample(ref_latents, len(x), "ref_latents", item_ndim=4)
      for item, ref in zip(x, refs):
        if ref.shape[0] != item.shape[0] or ref.shape[2:] != item.shape[2:]:
          raise ValueError(f"`ref_latents` item {tuple(ref.shape)} is incompatible with "
                           f"latent {tuple(item.shape)}.")
      x = [
        torch.cat([item, ref.to(device=item.device, dtype=item.dtype)], dim=1)
        for item, ref in zip(x, refs)
      ]

    direct_rows: list[tuple[int, int, torch.Tensor]] = []
    if direct_condition is not None:
      directs = self._as_per_sample(direct_condition,
                                    len(cap_feats),
                                    "direct_condition",
                                    item_ndim=2)
      cap_inputs, cursor = [], 0
      for cap, direct in zip(cap_feats, directs):
        if direct.shape[-1] != self.dim:
          raise ValueError(f"`direct_condition` item {tuple(direct.shape)} must be in hidden space "
                           f"({self.dim}), got {direct.shape[-1]} channels.")
        # Placeholder rows keep the prompt sequence length (and its RoPE range) honest; the embedder
        # replaces them with `direct` after `cap_embedder`.
        item = torch.cat([cap, cap[-1:].repeat(direct.shape[0], 1)], dim=0)
        cap_inputs.append(item)
        # `patchify_and_embed` pads each prompt item to a multiple of SEQ_MULTI_OF around the direct
        # rows, so row ranges advance by the padded length rather than by the raw one.
        start = cursor + cap.shape[0]
        direct_rows.append((start, start + direct.shape[0], direct))
        cursor += item.shape[0] + (-item.shape[0]) % SEQ_MULTI_OF
      cap_feats = cap_inputs
      self.cap_embedder.direct_total = cursor

    if self.alignment_padding_mode == ZERO_MASKED_PADDING:
      self._inner_valid = [(
        (item.shape[1] // f_patch_size) * (item.shape[2] // patch_size) *
        (item.shape[3] // patch_size),
        int(cap_item.shape[0]),
      ) for item, cap_item in zip(x, cap_feats)]

    self.cap_embedder.direct_rows = direct_rows
    output = super().forward(
      x,
      t,
      cap_feats,
      return_dict=return_dict,
      patch_size=patch_size,
      f_patch_size=f_patch_size,
      joint_attention_kwargs=joint_attention_kwargs,
    )
    self.cap_embedder.direct_rows = []
    self.cap_embedder.direct_total = 0
    self._inner_valid = []

    samples = output.sample if return_dict else output[0]
    keep_frames = generated_frames if self.multi_frame_output else [1] * len(generated_frames)
    samples = [sample[:, :frames] for sample, frames in zip(samples, keep_frames)]

    if return_dict:
      return Transformer2DModelOutput(sample=samples)
    return (samples, )


__all__ = [
  "MingImageTransformer2DModel",
  "ZERO_MASKED_PADDING",
  "LEARNED_PADDING",
  "MING_AXES_LENS",
]
