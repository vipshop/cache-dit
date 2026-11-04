"""Modality grouping, grouped projections and the MAGI-2 Fourier position embed.

Copied from ``vllm-omni @ 7f217a1`` ``vllm_omni/diffusion/models/magi2/common_layers.py``
(Apache-2.0, Copyright (c) 2026 SandAI). Intentional differences:

* ``swiglu7`` moved to :mod:`.layers` (one portable implementation, no fused CustomOp);
* the process group comes from :mod:`.parallel` instead of vLLM's parallel state;
* cache-dit style (2-space indent, Sphinx docstrings).

The grouped linear is the checkpoint's native layout: one flattened weight
``[num_experts * out_features, in_features]`` where a multimodal layer keeps one block per modality.
Tensor parallelism shards it in place, so the same module covers TP and non-TP runs.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .parallel import Magi2ParallelGroup, get_magi2_tp_group

__all__ = [
  "ElementWiseFourierEmbed",
  "Magi2GroupedLinear",
  "ModalityDispatcher",
  "make_grouped_linear",
]


class ModalityDispatcher:
  """Precomputed modality grouping and inverse permutation.

  The packed token sequence is laid out video -> audio -> text (plus reference-image segments), so
  the per-modality projections need a stable grouping and its inverse permutation.

  :param modality_mapping: ``[tokens]`` modality id per packed token.
  :param num_modalities: Number of modality groups.
  """

  def __init__(self, modality_mapping: torch.Tensor, num_modalities: int) -> None:
    if modality_mapping.ndim != 1:
      raise ValueError("modality mapping must be a one-dimensional tensor")
    self.modality_mapping = modality_mapping
    self.num_modalities = num_modalities
    self.permute_mapping = torch.argsort(modality_mapping, stable=True)
    self.inv_permute_mapping = torch.argsort(self.permute_mapping)
    self.permuted_modality_mapping = modality_mapping.index_select(0, self.permute_mapping)
    self.group_size = torch.bincount(self.permuted_modality_mapping.long(),
                                     minlength=num_modalities).to(torch.int32)
    self.group_size_cpu = [int(value) for value in self.group_size.cpu().tolist()]
    self.cu_group_sizes = F.pad(torch.cumsum(self.group_size, dim=0), (1, 0))

  @staticmethod
  def _mapping_on(tensor: torch.Tensor, mapping: torch.Tensor) -> torch.Tensor:
    """Return ``mapping`` on the tensor's device.

    Under layerwise offload the streaming hook may migrate the activations to the compute device while
    the dispatcher, built from the request tensors, still holds host indices.

    :param tensor: Tensor the mapping indexes.
    :param mapping: Index tensor.
    :returns: ``mapping`` on ``tensor.device``.
    """

    return mapping if mapping.device == tensor.device else mapping.to(tensor.device)

  def dispatch(self, tensor: torch.Tensor) -> list:
    """Split ``tensor`` into per-modality chunks.

    :param tensor: ``[tokens, ...]`` tensor in packed order.
    :returns: One chunk per modality.
    """

    return list(torch.split(tensor, self.group_size_cpu, dim=0))

  @staticmethod
  def undispatch(*groups: torch.Tensor) -> torch.Tensor:
    """Concatenate per-modality chunks.

    :param groups: One chunk per modality.
    :returns: The packed tensor.
    """

    return torch.cat(groups, dim=0)

  def permute(self, tensor: torch.Tensor) -> torch.Tensor:
    """Reorder packed tokens into modality-major order.

    :param tensor: ``[tokens, ...]`` tensor.
    :returns: The permuted tensor.
    """

    return tensor.index_select(0, self._mapping_on(tensor, self.permute_mapping))

  def inverse_permute(self, tensor: torch.Tensor) -> torch.Tensor:
    """Undo :meth:`permute`.

    :param tensor: Modality-major ``[tokens, ...]`` tensor.
    :returns: The packed-order tensor.
    """

    return tensor.index_select(0, self._mapping_on(tensor, self.inv_permute_mapping))


class Magi2GroupedLinear(nn.Module):
  """Modality-grouped linear with the released flattened weight layout.

  :param in_features: Input width.
  :param out_features: Output width of one group.
  :param num_experts: Number of groups (modalities); ``1`` is a plain linear.
  :param bias: Whether to allocate a bias.
  :param dtype: Parameter dtype.
  :param device: Parameter device.
  :param parallel_mode: ``column``, ``row``, or ``None`` for a replicated projection.
  :param qkv_splits: Output segment sizes when one projection packs Q/K/V.
  :param tp_group: Parameter-parallel group; defaults to the installed group.
  """

  def __init__(
    self,
    in_features: int,
    out_features: int,
    *,
    num_experts: int = 1,
    bias: bool = False,
    dtype=None,
    device=None,
    parallel_mode=None,
    qkv_splits=None,
    tp_group: Magi2ParallelGroup = None,
  ) -> None:
    super().__init__()
    if parallel_mode not in (None, "column", "row"):
      raise ValueError(f"unknown MAGI-2 linear parallel mode {parallel_mode!r}")
    if qkv_splits is not None and parallel_mode != "column":
      raise ValueError("segmented QKV slicing requires column parallelism")
    if qkv_splits is not None and sum(qkv_splits) != out_features:
      raise ValueError("QKV splits must sum to out_features")
    self.in_features = in_features
    self.out_features = out_features
    self.num_experts = num_experts
    self.parallel_mode = parallel_mode
    self.qkv_splits = qkv_splits
    self.tp_group = tp_group or get_magi2_tp_group()
    self.local_in_features = in_features
    self.local_out_features = out_features
    if parallel_mode == "column" and self.tp_group.world_size > 1:
      split_dims = qkv_splits or (out_features, )
      if any(size % self.tp_group.world_size for size in split_dims):
        raise ValueError(f"column-parallel output dimensions {split_dims} must divide "
                         f"TP={self.tp_group.world_size}")
      self.local_out_features = sum(size // self.tp_group.world_size for size in split_dims)
    elif parallel_mode == "row" and self.tp_group.world_size > 1:
      if in_features % self.tp_group.world_size:
        raise ValueError(
          f"row-parallel input {in_features} must divide TP={self.tp_group.world_size}")
      self.local_in_features = in_features // self.tp_group.world_size

    self.weight = nn.Parameter(
      torch.empty(
        num_experts * self.local_out_features,
        self.local_in_features,
        dtype=dtype,
        device=device,
      ))
    if bias:
      bias_features = self.local_out_features if parallel_mode == "column" else out_features
      self.bias = nn.Parameter(torch.empty(num_experts * bias_features, dtype=dtype, device=device))
    else:
      self.register_parameter("bias", None)
    if self.tp_group.world_size > 1 and parallel_mode is not None:
      self.weight.checkpoint_weight_transform = self.shard_checkpoint_weight
      if self.bias is not None and parallel_mode == "column":
        self.bias.checkpoint_weight_transform = self.shard_checkpoint_bias

  def _column_slices(self):
    """Return the ``(start, stop)`` output range of every segment on this rank.

    :returns: One tuple per QKV segment (or a single tuple for a plain projection).
    """

    splits = self.qkv_splits or (self.out_features, )
    offsets = []
    base = 0
    for size in splits:
      local = size // self.tp_group.world_size
      start = base + self.tp_group.rank * local
      offsets.append((start, start + local))
      base += size
    return tuple(offsets)

  def shard_checkpoint_weight(self, checkpoint_tensor: torch.Tensor) -> torch.Tensor:
    """Convert the released grouped weight into this rank's shard.

    :param checkpoint_tensor: ``[num_experts * out_features, in_features]`` checkpoint weight.
    :returns: The local shard.
    """

    if tuple(checkpoint_tensor.shape) == tuple(self.weight.shape):
      return checkpoint_tensor
    expected = (self.num_experts * self.out_features, self.in_features)
    if tuple(checkpoint_tensor.shape) != expected:
      raise ValueError(f"grouped linear checkpoint has shape {tuple(checkpoint_tensor.shape)}, "
                       f"expected {expected}")
    grouped = checkpoint_tensor.view(self.num_experts, self.out_features, self.in_features)
    if self.parallel_mode == "column":
      shards = [grouped[:, start:end] for start, end in self._column_slices()]
      return torch.cat(shards, dim=1).reshape(self.num_experts * self.local_out_features,
                                              self.in_features)
    if self.parallel_mode == "row":
      start = self.tp_group.rank * self.local_in_features
      return grouped[:, :, start:start + self.local_in_features].reshape(
        self.num_experts * self.out_features, self.local_in_features)
    return checkpoint_tensor

  def shard_checkpoint_bias(self, checkpoint_tensor: torch.Tensor) -> torch.Tensor:
    """Convert the released grouped bias into this rank's shard.

    :param checkpoint_tensor: ``[num_experts * out_features]`` checkpoint bias.
    :returns: The local shard.
    """

    if self.bias is None:
      raise ValueError("cannot load bias into a bias-free grouped linear")
    if tuple(checkpoint_tensor.shape) == tuple(self.bias.shape):
      return checkpoint_tensor
    expected = (self.num_experts * self.out_features, )
    if tuple(checkpoint_tensor.shape) != expected:
      raise ValueError(
        f"grouped linear bias has shape {tuple(checkpoint_tensor.shape)}, expected {expected}")
    grouped = checkpoint_tensor.view(self.num_experts, self.out_features)
    shards = [grouped[:, start:end] for start, end in self._column_slices()]
    return torch.cat(shards, dim=1).reshape(-1)

  def shard_for_tp(self, tp_group: Magi2ParallelGroup) -> None:
    """Re-shard a replicated weight in place for tensor parallelism.

    The pipeline loads the released checkpoint on CPU first (the released topology requires layer
    offload anyway), so tensor parallelism shards the already-loaded weight instead of re-reading
    it.

    :param tp_group: Parameter-parallel group to shard for.
    """

    if tp_group.world_size <= 1:
      return
    if self.parallel_mode is None:
      self.tp_group = tp_group
      return
    if self.tp_group.world_size > 1:
      raise RuntimeError("grouped linear is already sharded; re-sharding needs a fresh module")

    self.tp_group = tp_group
    grouped = self.weight.data.view(self.num_experts, self.out_features, self.in_features)
    if self.parallel_mode == "column":
      splits = self.qkv_splits or (self.out_features, )
      if any(size % tp_group.world_size for size in splits):
        raise ValueError(
          f"column-parallel output dimensions {splits} must divide TP={tp_group.world_size}")
      ranges = []
      base = 0
      for size in splits:
        local = size // tp_group.world_size
        start = base + tp_group.rank * local
        ranges.append((start, start + local))
        base += size
      local_out = sum(end - start for start, end in ranges)
      sharded = torch.cat([grouped[:, start:end] for start, end in ranges], dim=1)
      sharded = sharded.reshape(self.num_experts * local_out, self.in_features).contiguous()
    else:
      if self.in_features % tp_group.world_size:
        raise ValueError(
          f"row-parallel input {self.in_features} must divide TP={tp_group.world_size}")
      local_in = self.in_features // tp_group.world_size
      start = tp_group.rank * local_in
      sharded = grouped[:, :, start:start + local_in].reshape(self.num_experts * self.out_features,
                                                              local_in).contiguous()
      self.local_in_features = local_in

    if self.parallel_mode == "column":
      self.local_out_features = sharded.shape[0] // self.num_experts
    self.weight = nn.Parameter(sharded, requires_grad=self.weight.requires_grad)
    self.weight.checkpoint_weight_transform = self.shard_checkpoint_weight

    if self.bias is not None:
      if self.parallel_mode == "column":
        grouped_bias = self.bias.data.view(self.num_experts, self.out_features)
        sharded_bias = torch.cat([grouped_bias[:, start:end] for start, end in ranges], dim=1)
        self.bias = nn.Parameter(sharded_bias.reshape(-1).contiguous(),
                                 requires_grad=self.bias.requires_grad)
        self.bias.checkpoint_weight_transform = self.shard_checkpoint_bias
      else:
        self.bias = nn.Parameter(self.bias.data.clone(), requires_grad=self.bias.requires_grad)

  def forward(self,
              tensor: torch.Tensor,
              modality_dispatcher: ModalityDispatcher = None) -> torch.Tensor:
    """Apply the (optionally grouped and sharded) projection.

    :param tensor: ``[tokens, in_features]`` input.
    :param modality_dispatcher: Required when ``num_experts > 1``.
    :returns: ``[tokens, out_features]`` output (local width under column parallelism).
    """

    weight = self.weight.view(self.num_experts, self.local_out_features, self.local_in_features)
    bias_features = self.local_out_features if self.parallel_mode == "column" else self.out_features
    bias = self.bias.view(self.num_experts, bias_features) if self.bias is not None else None
    linear_bias = None if self.parallel_mode == "row" else bias
    if self.num_experts == 1:
      output = F.linear(tensor, weight[0], None if linear_bias is None else linear_bias[0])
    else:
      if modality_dispatcher is None:
        raise ValueError("modality_dispatcher is required for grouped linear")
      if modality_dispatcher.num_modalities != self.num_experts:
        raise ValueError("grouped linear expert count does not match modality dispatcher")
      inputs = modality_dispatcher.dispatch(tensor)
      outputs = [
        F.linear(part, weight[index], None if linear_bias is None else linear_bias[index])
        for index, part in enumerate(inputs)
      ]
      output = torch.cat(outputs, dim=0)

    if self.parallel_mode == "row" and self.tp_group.world_size > 1:
      dist.all_reduce(output, group=self.tp_group.group)
    if self.parallel_mode == "row" and bias is not None:
      if self.num_experts == 1:
        output = output + bias[0]
      else:
        outputs = [
          part + bias[index] for index, part in enumerate(modality_dispatcher.dispatch(output))
        ]
        output = torch.cat(outputs, dim=0)
    return output


def make_grouped_linear(
  in_features: int,
  out_features: int,
  *,
  num_experts: int = 1,
  bias: bool = False,
  dtype=None,
  parallel_mode=None,
  qkv_splits=None,
) -> Magi2GroupedLinear:
  """Build a :class:`Magi2GroupedLinear`.

  :param in_features: Input width.
  :param out_features: Output width of one group.
  :param num_experts: Number of modality groups.
  :param bias: Whether to allocate a bias.
  :param dtype: Parameter dtype.
  :param parallel_mode: ``column``, ``row``, or ``None``.
  :param qkv_splits: Output segment sizes when one projection packs Q/K/V.
  :returns: The constructed module.
  """

  return Magi2GroupedLinear(
    in_features,
    out_features,
    num_experts=num_experts,
    bias=bias,
    dtype=dtype,
    parallel_mode=parallel_mode,
    qkv_splits=qkv_splits,
  )


def _frequency_bands(num_bands: int, *, temperature: float, device=None) -> torch.Tensor:
  """Build the geometric frequency bands of the Fourier embed.

  :param num_bands: Number of bands per axis.
  :param temperature: Geometric base.
  :param device: Tensor device.
  :returns: ``[num_bands]`` frequencies.
  """

  exponent = torch.arange(num_bands, dtype=torch.float32, device=device) / num_bands
  return 1.0 / temperature ** exponent


class ElementWiseFourierEmbed(nn.Module):
  """Nine-coordinate Fourier embedding used as MAGI's element-wise RoPE.

  :param dim: Rotary width (``head_dim``).
  :param temperature: Geometric base of the frequency bands.
  :param learnable: Keep the bands trainable (the release keeps them frozen).
  :param device: Parameter device.
  :param dtype: Band dtype.
  """

  def __init__(
    self,
    dim: int,
    *,
    temperature: float = 10000.0,
    learnable: bool = False,
    device=None,
    dtype: torch.dtype = torch.float32,
  ) -> None:
    super().__init__()
    self.dim = dim
    self.temperature = temperature
    bands = _frequency_bands(dim // 8, temperature=temperature, device=device).to(dtype)
    if learnable:
      self.bands = nn.Parameter(bands)
    else:
      self.bands = nn.Parameter(bands, requires_grad=False)

  def forward(self, coords: torch.Tensor) -> torch.Tensor:
    """Project nine-dimensional coordinates into the rotary table.

    :param coords: ``[tokens, 9]`` coordinates ``(t,h,w,T,H,W,ref_T,ref_H,ref_W)``.
    :returns: ``[tokens, dim]`` rotary table holding ``sin`` then ``cos``.
    """

    if coords.ndim != 2 or coords.shape[-1] != 9:
      raise ValueError("MAGI coordinates must have shape [tokens,9]")
    xyz = coords[:, :3]
    sizes = coords[:, 3:6]
    references = coords[:, 6:9]
    scales = (references - 1) / (sizes - 1)
    scales = torch.where((references == 1) & (sizes == 1), torch.ones_like(scales), scales)
    if not torch.isfinite(scales).all():
      raise ValueError("invalid MAGI coordinate scale")
    centers = (sizes - 1) / 2
    centers = centers.clone()
    centers[:, 0] = 0
    projection = (xyz - centers).unsqueeze(-1) * scales.unsqueeze(-1) * self.bands
    return torch.cat((projection.sin(), projection.cos()), dim=1).flatten(1)
