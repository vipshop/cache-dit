"""Packing helpers between dense modality tensors and the MAGI-2 packed ABI.

Copied from ``vllm-omni @ 7f217a1`` ``vllm_omni/diffusion/models/magi2/preview_data_proxy.py``
(Apache-2.0, Copyright (c) 2026 SandAI). Intentional differences:

* no ``vllm_omni.platforms`` import: ``cu_seqlens`` always lives on the compute device;
* ``VarlenHandler`` is imported from :mod:`.attention`;
* cache-dit style (2-space indent, Sphinx docstrings).

The released transformer consumes one packed varlen sequence per sample, laid out
VIDEO -> AUDIO -> TEXT and followed by optional reference-image special-token/image pairs.
"""

import dataclasses
import math
from enum import IntEnum
from itertools import chain
from typing import Literal

import torch
from einops import rearrange
from torch.nn import functional as F

from .attention import VarlenHandler

__all__ = [
  "Magi2DataProxy",
  "Magi2PreviewDataProxyConfig",
  "ModelInput",
  "Modality",
  "PackedModelInput",
  "SimplePackedData",
  "SingleData",
  "get_coords",
  "sinusoidal_embedding_1d",
]


class Modality(IntEnum):
  """Checkpoint modality ids used by the pre/post adapters."""

  VIDEO = 0
  AUDIO = 1
  TEXT = 2
  # TIME is remapped to TEXT by the transformer after the time tokens are extracted.
  TIME = 3


@dataclasses.dataclass(frozen=True)
class Magi2PreviewDataProxyConfig:
  """Packing configuration of the released preview checkpoint.

  :param t_patch_size: Temporal patch size.
  :param patch_size: Spatial patch size.
  :param spatial_rope_interpolation: ``extra`` keeps the native reference grid, ``inter`` pins 32x32.
  :param add_time_token: Append one diffusion-time token to the packed sequence.
  :param time_channel_dim: Width of the per-token time channel written into the hidden states.
  :param time_aligned_rope: Reserved by the release; unused here.
  :param audio_latent_fps: Audio latent frame rate.
  :param time_pos_fps: Temporal position scale of the coordinate grid.
  :param vae_first_latent_is_image: Whether the first video latent frame is an image.
  :param video_fps: Internal video timeline frame rate.
  """

  t_patch_size: int = 1
  patch_size: int = 1
  spatial_rope_interpolation: Literal["inter", "extra"] = "extra"
  add_time_token: bool = False
  time_channel_dim: int = 64
  time_aligned_rope: bool = False
  audio_latent_fps: float = 25.0
  time_pos_fps: float = 3.125
  vae_first_latent_is_image: bool = True
  video_fps: float = 25.0

  def __post_init__(self) -> None:
    if self.t_patch_size <= 0 or self.patch_size <= 0:
      raise ValueError("MAGI-2 patch sizes must be positive")
    if self.time_channel_dim < 0:
      raise ValueError("time_channel_dim must be non-negative")


@dataclasses.dataclass
class ModelInput:
  """Unpacked tensors for one native preview-transformer invocation.

  :param x_t: ``[B, C, T, H, W]`` video latents.
  :param audio_x_t: ``[B, L, C]`` audio latents.
  :param audio_feat_len: Per-sample audio length.
  :param txt_feat: ``[B, L, D]`` text features.
  :param txt_feat_len: Per-sample text length.
  :param t: Per-sample normalized diffusion timestep.
  :param ref_audio_feat: Optional ``[B, L, C]`` reference audio latents.
  :param ref_audio_feat_len: Per-sample reference audio length.
  :param ref_video_feat: Optional ``[B, C, T, H, W]`` reference video latents.
  :param ref_video_feat_len: Per-sample reference video length.
  :param per_token_video_t: Optional per-token video timestep.
  :param per_token_audio_t: Optional per-token audio timestep.
  :param ref_image_feat: Optional ``[B, M, C, 1, H, W]`` reference-image latents.
  :param ref_image_feat_len: Optional ``[B, M, 2]`` reference-image latent grids.
  :param ref_image_special_token_embedding: Optional ``[B, M, D]`` reference-image markers.
  """

  x_t: torch.Tensor
  audio_x_t: torch.Tensor
  audio_feat_len: object
  txt_feat: torch.Tensor
  txt_feat_len: object
  t: torch.Tensor
  ref_audio_feat: torch.Tensor = None
  ref_audio_feat_len: object = None
  ref_video_feat: torch.Tensor = None
  ref_video_feat_len: object = None
  per_token_video_t: torch.Tensor = None
  per_token_audio_t: torch.Tensor = None
  ref_image_feat: torch.Tensor = None
  ref_image_feat_len: torch.Tensor = None
  ref_image_special_token_embedding: torch.Tensor = None


def _to_int(value) -> int:
  """Read one length as a Python int.

  :param value: Python int or 0-d/1-element tensor.
  :returns: The integer value.
  """

  if isinstance(value, torch.Tensor):
    return int(value.detach().reshape(-1)[0].item())
  return int(value)


def _length_at(value, index: int) -> int:
  """Read the ``index``-th length.

  :param value: Sequence of lengths or a scalar length.
  :param index: Sample index.
  :returns: The integer length.
  """

  if isinstance(value, torch.Tensor) and value.ndim == 0:
    if index:
      raise IndexError("a scalar length only describes one sample")
    return _to_int(value)
  return _to_int(value[index])


def _pad_cat(tensors: list, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
  """Concatenate token blocks, padding narrow ones with zeros.

  :param tensors: Token blocks ``[tokens_i, channels_i]``.
  :param device: Output device.
  :param dtype: Output dtype.
  :returns: ``[sum(tokens), max(channels)]`` tensor.
  """

  if not tensors:
    return torch.empty(0, 0, device=device, dtype=dtype)
  max_channel = max(tensor.shape[-1] for tensor in tensors)
  return torch.cat([
    F.pad(tensor.to(device=device, dtype=dtype), (0, max_channel - tensor.shape[-1]))
    for tensor in tensors
  ],
                   dim=0)


def _segment_paint(values: list, seqlens: list, *, dtype: torch.dtype, device: torch.device,
                   output_size: int) -> torch.Tensor:
  """Paint one value per segment into a flat tensor.

  :param values: Value of each segment.
  :param seqlens: Length of each segment.
  :param dtype: Output dtype.
  :param device: Output device.
  :param output_size: Total length.
  :returns: The painted tensor.
  """

  mapping = torch.empty(output_size, dtype=dtype, device=device)
  offset = 0
  for value, seqlen in zip(values, seqlens, strict=True):
    mapping[offset:offset + seqlen] = int(value)
    offset += seqlen
  if offset != output_size:
    raise ValueError(f"segment lengths total {offset}, expected {output_size}")
  return mapping


def _seqlens_to_cu_seqlens(seqlens: list, device: torch.device = None) -> torch.Tensor:
  """Convert per-sample lengths into cumulative lengths.

  :param seqlens: Per-sample lengths.
  :param device: Output device.
  :returns: ``[len(seqlens) + 1]`` cumulative lengths.
  """

  lengths = torch.tensor(seqlens, dtype=torch.int32, device=device)
  return F.pad(torch.cumsum(lengths, dim=0), (1, 0))


def _len_to_list(value: torch.Tensor) -> list:
  """Convert a length tensor into a Python list.

  :param value: Length tensor.
  :returns: The lengths as ints.
  """

  return [int(item) for item in value.detach().to(torch.long).reshape(-1).tolist()]


def get_coords(
  shape: tuple,
  ref_feat_shape: tuple,
  offset_thw: tuple = (0, 0, 0),
  *,
  device: torch.device = torch.device("cpu"),
  dtype: torch.dtype = torch.float32,
  time_positions: torch.Tensor = None,
) -> torch.Tensor:
  """Build MAGI-2's ``(t,h,w,T,H,W,ref_T,ref_H,ref_W)`` coordinate rows.

  :param shape: ``(t, h, w)`` grid of this segment.
  :param ref_feat_shape: ``(ref_t, ref_h, ref_w)`` normalization grid.
  :param offset_thw: Offset added to the ``(t, h, w)`` coordinates.
  :param device: Output device.
  :param dtype: Output dtype.
  :param time_positions: Optional explicit temporal positions.
  :returns: ``[prod(shape), 9]`` coordinates.
  """

  ori_t, ori_h, ori_w = shape
  ref_t, ref_h, ref_w = ref_feat_shape
  offset_t, offset_h, offset_w = offset_thw

  if time_positions is None:
    time_range = torch.arange(ori_t, device=device, dtype=dtype) + offset_t
  else:
    if time_positions.numel() != ori_t:
      raise ValueError(f"received {time_positions.numel()} time positions for {ori_t} time steps")
    time_range = time_positions.to(device=device, dtype=dtype) + offset_t
  height_range = torch.arange(ori_h, device=device, dtype=dtype) + offset_h
  width_range = torch.arange(ori_w, device=device, dtype=dtype) + offset_w
  time_grid, height_grid, width_grid = torch.meshgrid(time_range,
                                                      height_range,
                                                      width_range,
                                                      indexing="ij")
  coords = torch.stack((time_grid, height_grid, width_grid), dim=-1).reshape(-1, 3)
  metadata = torch.tensor((ori_t, ori_h, ori_w, ref_t, ref_h, ref_w), device=device, dtype=dtype)
  return torch.cat((coords, metadata.expand(coords.shape[0], -1)), dim=-1)


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
  """Build the checkpoint's FP32 diffusion-time channel embedding.

  :param dim: Embedding width.
  :param position: Normalized timestep per packed token.
  :returns: ``[tokens, dim]`` embedding.
  """

  position = position.to(torch.float32) * 1000.0
  half = dim // 2
  frequencies = torch.exp(-math.log(10000) *
                          torch.arange(0, half, dtype=torch.float32, device=position.device) / half)
  arguments = position[:, None].float() * frequencies[None]
  embedding = torch.cat((torch.cos(arguments), torch.sin(arguments)), dim=-1)
  if dim % 2:
    embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
  return embedding


def _patchify_3d(x: torch.Tensor, t_patch_size: int, patch_size: int) -> torch.Tensor:
  """Non-overlapping 3-D unfold with the released channel-major order.

  :param x: ``[B, C, T, H, W]`` latents.
  :param t_patch_size: Temporal patch size.
  :param patch_size: Spatial patch size.
  :returns: ``[B, tokens, C * pT * pH * pW]`` patches.
  """

  kernel = (t_patch_size, patch_size, patch_size)
  if x.ndim != 5:
    raise ValueError(f"expected [B,C,T,H,W], got {tuple(x.shape)}")
  if not all(size >= kernel_size for size, kernel_size in zip(x.shape[2:], kernel, strict=True)):
    return torch.empty(x.shape[0],
                       0,
                       x.shape[1] * math.prod(kernel),
                       device=x.device,
                       dtype=x.dtype)
  patches = x.unfold(2, t_patch_size, t_patch_size)
  patches = patches.unfold(3, patch_size, patch_size)
  patches = patches.unfold(4, patch_size, patch_size)
  patches = patches.permute(0, 2, 3, 4, 1, 5, 6, 7).contiguous()
  return patches.reshape(x.shape[0], -1, x.shape[1] * math.prod(kernel))


@dataclasses.dataclass
class SingleData:
  """Packed metadata for one sample.

  :param video_x_t: ``[tokens, video_channels]`` video tokens.
  :param audio_x_t: ``[tokens, audio_channels]`` audio tokens.
  :param audio_feat_len: Number of valid audio tokens.
  :param txt_feat: ``[tokens, text_channels]`` text features.
  :param txt_feat_len: Number of valid text tokens.
  :param t: Latent temporal length.
  :param h: Latent height.
  :param w: Latent width.
  :param patch_size: Spatial patch size.
  :param t_patch_size: Temporal patch size.
  :param spatial_rope_interpolation: Coordinate interpolation mode.
  :param diffusion_t: Optional diffusion-time token value.
  :param per_token_video_t: Optional per-token video timestep.
  :param per_token_audio_t: Optional per-token audio timestep.
  :param time_channel_dim: Width of the time channel.
  :param vae_first_latent_is_image: Whether the first latent frame is an image.
  :param video_fps: Internal video frame rate.
  :param time_pos_fps: Temporal position scale.
  :param ref_image_feats: Optional reference-image token blocks.
  :param ref_image_feat_lens: Optional reference-image latent grids.
  :param ref_image_special_tokens: Optional reference-image markers.
  """

  video_x_t: torch.Tensor
  audio_x_t: torch.Tensor
  audio_feat_len: int
  txt_feat: torch.Tensor
  txt_feat_len: int
  t: int
  h: int
  w: int
  patch_size: int
  t_patch_size: int
  spatial_rope_interpolation: Literal["inter", "extra"]
  diffusion_t: torch.Tensor = None
  per_token_video_t: torch.Tensor = None
  per_token_audio_t: torch.Tensor = None
  time_channel_dim: int = 0
  vae_first_latent_is_image: bool = True
  video_fps: float = 25.0
  time_pos_fps: float = 3.125
  ref_image_feats: list = None
  ref_image_feat_lens: list = None
  ref_image_special_tokens: list = None

  def __post_init__(self) -> None:
    self.video_token_num = self.video_x_t.shape[0]
    self.origin_audio_feat_len = self.audio_x_t.shape[0]
    self.audio_x_t = self.audio_x_t[:self.audio_feat_len]
    self.txt_feat = self.txt_feat[:self.txt_feat_len]
    if self.per_token_audio_t is not None:
      self.per_token_audio_t = self.per_token_audio_t[:self.audio_feat_len]

    self.ref_image_feats = self.ref_image_feats or []
    self.ref_image_feat_lens = self.ref_image_feat_lens or []
    self.ref_image_special_tokens = self.ref_image_special_tokens or []
    if not (len(self.ref_image_feats) == len(self.ref_image_feat_lens) == len(
        self.ref_image_special_tokens)):
      raise ValueError(
        "reference-image features, lengths, and special tokens must have equal counts")
    self.ref_image_token_nums = [math.prod(feat_len) for feat_len in self.ref_image_feat_lens]
    self.ref_image_feats = [
      feature[:token_num]
      for feature, token_num in zip(self.ref_image_feats, self.ref_image_token_nums, strict=True)
    ]
    self.num_ref_images = len(self.ref_image_feats)
    self.total_ref_image_feat_len = sum(self.ref_image_token_nums)
    self.video_channel = self.video_x_t.shape[-1]
    self.audio_channel = self.audio_x_t.shape[-1]

  @property
  def device(self) -> torch.device:
    """Compute device of the packed tensors."""

    return self.video_x_t.device

  @property
  def default_dtype(self) -> torch.dtype:
    """Dtype of the packed tensors."""

    return self.video_x_t.dtype

  @property
  def add_time_token(self) -> bool:
    """Whether one diffusion-time token is appended."""

    return self.diffusion_t is not None

  @property
  def total_token_num(self) -> int:
    """Total packed token count of this sample."""

    total = self.video_token_num + self.audio_feat_len + self.txt_feat_len
    total += self.total_ref_image_feat_len + self.num_ref_images
    return total + int(self.add_time_token)

  @property
  def feat_to_cat(self) -> list:
    """Token blocks in released packing order."""

    tensors = [self.video_x_t, self.audio_x_t, self.txt_feat]
    for index in range(self.num_ref_images):
      tensors.append(self.ref_image_special_tokens[index])
      tensors.append(self.ref_image_feats[index])
    if self.add_time_token:
      tensors.append(
        self.diffusion_t.to(device=self.device, dtype=self.default_dtype).reshape(1, 1))
    return tensors

  @property
  def token_sequence(self) -> torch.Tensor:
    """``[tokens, max_channels]`` packed token sequence."""

    return _pad_cat(self.feat_to_cat, self.device, self.default_dtype)

  @property
  def modality_map_seqlens(self) -> tuple:
    """Per-segment ``(lengths, modalities)`` of the packed sequence."""

    seqlens = [self.video_token_num, self.audio_feat_len, self.txt_feat_len]
    modalities = [Modality.VIDEO, Modality.AUDIO, Modality.TEXT]
    for index in range(self.num_ref_images):
      seqlens.extend((1, self.ref_image_token_nums[index]))
      modalities.extend((Modality.TEXT, Modality.VIDEO))
    if self.add_time_token:
      seqlens.append(1)
      modalities.append(Modality.TIME)
    return seqlens, modalities

  @property
  def modality_mapping(self) -> torch.Tensor:
    """``[tokens]`` modality id per packed token."""

    seqlens, modalities = self.modality_map_seqlens
    return _segment_paint(modalities,
                          seqlens,
                          dtype=torch.int32,
                          device=self.device,
                          output_size=self.total_token_num)

  def _default_coords(self, shape, ref_feat_shape, offset_thw=(0, 0, 0), time_positions=None):
    return get_coords(
      shape,
      ref_feat_shape,
      offset_thw,
      device=self.device,
      dtype=self.default_dtype,
      time_positions=time_positions,
    )

  @property
  def coords_to_cat(self) -> list:
    """Coordinate blocks in released packing order."""

    t_steps = self.t // self.t_patch_size
    h_steps = self.h // self.patch_size
    w_steps = self.w // self.patch_size
    if self.spatial_rope_interpolation == "inter":
      video_h_ref, video_w_ref = 32, 32
    else:
      video_h_ref, video_w_ref = h_steps, w_steps

    video_coords = self._default_coords((t_steps, h_steps, w_steps),
                                        (t_steps, video_h_ref, video_w_ref))
    magic_audio_ref_t = (self.audio_feat_len - 1) // 8 + 1
    audio_coords = self._default_coords((self.audio_feat_len, 1, 1),
                                        (magic_audio_ref_t // self.t_patch_size, 1, 1))
    coords = [
      video_coords,
      audio_coords,
      self._default_coords((self.txt_feat_len, 1, 1), (1, 1, 1),
                           offset_thw=(-self.txt_feat_len, 0, 0)),
    ]

    for index in range(self.num_ref_images):
      token_len = self.ref_image_token_nums[index]
      if len(self.ref_image_feat_lens[index]) >= 2:
        image_h, image_w = self.ref_image_feat_lens[index][:2]
      else:
        image_h = image_w = math.ceil(math.sqrt(token_len))
      # Keep the image context after the generated clip with a one-step gap.
      time_offset = t_steps + 2 + index
      coords.append(
        torch.tensor([[time_offset, -1, -1, 1, image_h, image_w, 1, image_h, image_w]],
                     device=self.device,
                     dtype=self.default_dtype))
      coords.append(
        self._default_coords((1, image_h, image_w), (1, image_h, image_w),
                             offset_thw=(time_offset, 0, 0))[:token_len])

    if self.add_time_token:
      coords.append(self._default_coords((1, 1, 1), (1, 1, 1))[:1])
    return coords

  @property
  def coords_mapping(self) -> torch.Tensor:
    """``[tokens, 9]`` coordinate rows."""

    return torch.cat(self.coords_to_cat, dim=0)

  @property
  def time_token_sequence(self) -> torch.Tensor:
    """``[tokens, time_channel_dim]`` per-token time channel."""

    if self.time_channel_dim == 0:
      return torch.empty(self.total_token_num, 0, device=self.device)
    if self.per_token_video_t is None or self.per_token_audio_t is None:
      raise ValueError(
        "per-token video and audio timesteps are required when time_channel_dim is non-zero")
    parts = [
      self.per_token_video_t.squeeze(-1),
      self.per_token_audio_t.squeeze(-1),
      torch.zeros(self.txt_feat_len, device=self.device),
    ]
    for index in range(self.num_ref_images):
      parts.append(torch.zeros(1, device=self.device))
      parts.append(torch.zeros(self.ref_image_token_nums[index], device=self.device))
    if self.add_time_token:
      parts.append(self.diffusion_t.reshape(1).to(self.device))
    raw_t = torch.cat(parts, dim=0)
    if self.time_channel_dim == 1:
      return raw_t.unsqueeze(-1)
    return sinusoidal_embedding_1d(self.time_channel_dim, raw_t)


@dataclasses.dataclass
class SimplePackedData:
  """A concatenation of independently addressable packed samples.

  :param items: One :class:`SingleData` per packed sample (conditional, then unconditional).
  """

  items: list

  def __post_init__(self) -> None:
    if not self.items:
      raise ValueError("SimplePackedData must contain at least one item")
    self._total_token_num_list = [item.total_token_num for item in self.items]
    self._total_token_num_sum = sum(self._total_token_num_list)
    self._total_token_num_max = max(self._total_token_num_list)
    self._total_token_cu_seqlens = _seqlens_to_cu_seqlens(self._total_token_num_list)

  @property
  def device(self) -> torch.device:
    """Compute device of the packed tensors."""

    return self.items[0].device

  @property
  def default_dtype(self) -> torch.dtype:
    """Dtype of the packed tensors."""

    return self.items[0].default_dtype

  @property
  def token_sequence(self) -> torch.Tensor:
    """``[tokens, max_channels]`` packed sequence of all samples."""

    tensors = list(chain.from_iterable(item.feat_to_cat for item in self.items))
    return _pad_cat(tensors, self.device, self.default_dtype)

  @property
  def modality_mapping(self) -> torch.Tensor:
    """``[tokens]`` modality id of all samples."""

    seqlens = []
    modalities = []
    for item in self.items:
      item_seqlens, item_modalities = item.modality_map_seqlens
      seqlens.extend(item_seqlens)
      modalities.extend(item_modalities)
    return _segment_paint(modalities,
                          seqlens,
                          dtype=torch.int32,
                          device=self.device,
                          output_size=self.total_token_num)

  @property
  def coords_mapping(self) -> torch.Tensor:
    """``[tokens, 9]`` coordinates of all samples."""

    coords = list(chain.from_iterable(item.coords_to_cat for item in self.items))
    return torch.cat(coords, dim=0)

  @property
  def time_token_sequence(self) -> torch.Tensor:
    """``[tokens, time_channel_dim]`` time channel of all samples."""

    return torch.cat([item.time_token_sequence for item in self.items], dim=0)

  @property
  def total_token_num(self) -> int:
    """Total packed token count."""

    return self._total_token_num_sum

  @property
  def cu_seqlen(self) -> torch.Tensor:
    """Cumulative per-sample lengths."""

    return self._total_token_cu_seqlens.clone()

  @property
  def max_seqlen(self) -> int:
    """Longest packed sample."""

    return self._total_token_num_max

  def __getitem__(self, index: int) -> SingleData:
    return self.items[index]

  def depack_token_sequence(self, token_sequence: torch.Tensor) -> tuple:
    """Split the transformer output back into dense video and audio tensors.

    :param token_sequence: ``[tokens, channels]`` transformer output.
    :returns: ``(video [B,C,T,H,W], audio [B,L,C])``.
    """

    if token_sequence.shape[0] != self.total_token_num:
      raise ValueError(
        f"model returned {token_sequence.shape[0]} tokens, expected {self.total_token_num}")
    videos = []
    audios = []
    token_slices = torch.split(token_sequence, self._total_token_num_list, dim=0)
    for item, token_slice in zip(self.items, token_slices, strict=True):
      video_flat = token_slice[:item.video_token_num, :item.video_channel]
      output_channels = item.video_channel // (item.t_patch_size * item.patch_size *
                                               item.patch_size)
      video = rearrange(
        video_flat,
        "(T H W) (pT pH pW C) -> C (T pT) (H pH) (W pW)",
        T=item.t // item.t_patch_size,
        H=item.h // item.patch_size,
        W=item.w // item.patch_size,
        pT=item.t_patch_size,
        pH=item.patch_size,
        pW=item.patch_size,
        C=output_channels,
      ).contiguous()
      audio = torch.zeros(
        item.origin_audio_feat_len,
        item.audio_channel,
        device=token_sequence.device,
        dtype=token_sequence.dtype,
      )
      audio[:item.audio_feat_len] = token_slice[item.video_token_num:item.video_token_num +
                                                item.audio_feat_len, :item.audio_channel]
      videos.append(video)
      audios.append(audio)
    return torch.stack(videos, dim=0), torch.stack(audios, dim=0)


@dataclasses.dataclass(frozen=True)
class PackedModelInput:
  """Request-owned packed transformer arguments and output layout.

  :param token_sequence: ``[tokens, channels]`` packed input.
  :param coords_mapping: ``[tokens, 9]`` coordinates.
  :param modality_mapping: ``[tokens]`` modality ids.
  :param varlen_handler: Packed-sequence metadata.
  :param time_token_sequence: ``[tokens, time_channel_dim]`` time channel.
  :param output_layout: Layout used to depack the model output.
  """

  token_sequence: torch.Tensor
  coords_mapping: torch.Tensor
  modality_mapping: torch.Tensor
  varlen_handler: VarlenHandler
  time_token_sequence: torch.Tensor
  output_layout: SimplePackedData

  @property
  def model_args(self) -> tuple:
    """Positional arguments of :meth:`Magi2PreviewTransformer.forward`."""

    return (
      self.token_sequence,
      self.coords_mapping,
      self.modality_mapping,
      self.varlen_handler,
      self.time_token_sequence,
    )


class Magi2DataProxy:
  """Convert dense modality tensors to and from the transformer's packed ABI.

  :param config: Packing configuration.
  """

  def __init__(self, config: Magi2PreviewDataProxyConfig = None) -> None:
    self.config = config or Magi2PreviewDataProxyConfig()
    self.patch_size = self.config.patch_size
    self.t_patch_size = self.config.t_patch_size

  def img2tokens(self, x_t: torch.Tensor) -> torch.Tensor:
    """Patchify latents.

    :param x_t: ``[B, C, T, H, W]`` latents.
    :returns: ``[B, tokens, C]`` patches.
    """

    return _patchify_3d(x_t, self.t_patch_size, self.patch_size)

  @staticmethod
  def _validate_batch(data: ModelInput) -> None:
    """Validate one request's dense tensors.

    :param data: Request tensors.
    :raises ValueError: When a shape or a reference-image field is inconsistent.
    """

    if data.x_t.ndim != 5:
      raise ValueError(f"x_t must be [B,C,T,H,W], got {tuple(data.x_t.shape)}")
    if data.audio_x_t.ndim != 3 or data.txt_feat.ndim != 3:
      raise ValueError("audio_x_t and txt_feat must be [B,L,C]")
    batch_size = data.x_t.shape[0]
    if data.audio_x_t.shape[0] != batch_size or data.txt_feat.shape[0] != batch_size:
      raise ValueError("video, audio, and text batch sizes must match")

    ref_values = (data.ref_image_feat, data.ref_image_feat_len,
                  data.ref_image_special_token_embedding)
    if any(value is not None for value in ref_values) and not all(value is not None
                                                                  for value in ref_values):
      raise ValueError(
        "reference-image features, lengths, and special tokens must be provided together")
    if data.ref_image_feat is not None:
      if data.ref_image_feat.ndim != 6:
        raise ValueError("ref_image_feat must be [B,M,C,T,H,W]")
      if data.ref_image_feat_len.ndim != 3 or data.ref_image_feat_len.shape[-1] < 2:
        raise ValueError("ref_image_feat_len must be [B,M,2]")
      if data.ref_image_special_token_embedding.ndim != 3:
        raise ValueError("ref_image_special_token_embedding must be [B,M,D]")
      expected = data.ref_image_feat.shape[:2]
      if data.ref_image_feat_len.shape[:2] != expected:
        raise ValueError("reference-image feature and length batch/image counts differ")
      if data.ref_image_special_token_embedding.shape[:2] != expected:
        raise ValueError("reference-image feature and special-token batch/image counts differ")

  def process_input(self, data: ModelInput) -> PackedModelInput:
    """Pack one request into the transformer ABI.

    :param data: Dense request tensors.
    :returns: The packed input.
    """

    self._validate_batch(data)
    batch_size, _, video_t, video_h, video_w = data.x_t.shape
    video_tokens = self.img2tokens(data.x_t)
    audio_tokens = data.audio_x_t.contiguous()
    text_tokens = data.txt_feat.contiguous()

    per_token_video_tokens = None
    per_token_audio = None
    if self.config.time_channel_dim > 0 and data.per_token_video_t is not None:
      per_token_video_tokens = self.img2tokens(data.per_token_video_t)[:, :, :1]
      per_token_audio = data.per_token_audio_t

    has_ref_image = data.ref_image_feat is not None
    num_ref_images = data.ref_image_feat.shape[1] if has_ref_image else 0
    ref_device, ref_dtype = text_tokens.device, text_tokens.dtype

    items = []
    for batch_index in range(batch_size):
      ref_image_feats = []
      ref_image_feat_lens = []
      ref_image_special_tokens = []
      for image_index in range(num_ref_images):
        feature = self.img2tokens(data.ref_image_feat[batch_index,
                                                      image_index].unsqueeze(0)).squeeze(0)
        ref_image_feats.append(feature)
        ref_image_feat_lens.append(_len_to_list(data.ref_image_feat_len[batch_index, image_index]))
        ref_image_special_tokens.append(
          data.ref_image_special_token_embedding[batch_index,
                                                 image_index].to(device=ref_device,
                                                                 dtype=ref_dtype).unsqueeze(0))

      if isinstance(data.t, torch.Tensor) and data.t.ndim == 0:
        diffusion_t = data.t
      else:
        diffusion_t = data.t[batch_index]
      items.append(
        SingleData(
          video_x_t=video_tokens[batch_index],
          audio_x_t=audio_tokens[batch_index],
          audio_feat_len=_length_at(data.audio_feat_len, batch_index),
          txt_feat=text_tokens[batch_index],
          txt_feat_len=_length_at(data.txt_feat_len, batch_index),
          t=video_t,
          h=video_h,
          w=video_w,
          patch_size=self.patch_size,
          t_patch_size=self.t_patch_size,
          spatial_rope_interpolation=self.config.spatial_rope_interpolation,
          diffusion_t=diffusion_t if self.config.add_time_token else None,
          per_token_video_t=(per_token_video_tokens[batch_index]
                             if per_token_video_tokens is not None else None),
          per_token_audio_t=(per_token_audio[batch_index] if per_token_audio is not None else None),
          time_channel_dim=self.config.time_channel_dim,
          time_pos_fps=self.config.time_pos_fps,
          vae_first_latent_is_image=self.config.vae_first_latent_is_image,
          video_fps=self.config.video_fps,
          ref_image_feats=ref_image_feats,
          ref_image_feat_lens=ref_image_feat_lens,
          ref_image_special_tokens=ref_image_special_tokens,
        ))

    packed = SimplePackedData(items)
    cu_seqlens = packed.cu_seqlen.to(device=data.x_t.device, dtype=torch.int32)
    varlen_handler = VarlenHandler(
      cu_seqlens_q=cu_seqlens,
      cu_seqlens_k=cu_seqlens,
      max_seqlen_q=packed.max_seqlen,
      max_seqlen_k=packed.max_seqlen,
    )
    return PackedModelInput(
      token_sequence=packed.token_sequence,
      coords_mapping=packed.coords_mapping,
      modality_mapping=packed.modality_mapping,
      varlen_handler=varlen_handler,
      time_token_sequence=packed.time_token_sequence,
      output_layout=packed,
    )

  @staticmethod
  def process_output(x: torch.Tensor, output_layout: SimplePackedData) -> tuple:
    """Depack the transformer output.

    :param x: ``[tokens, channels]`` transformer output.
    :param output_layout: Layout produced by :meth:`process_input`.
    :returns: ``(video, audio)``.
    """

    return output_layout.depack_token_sequence(x)
