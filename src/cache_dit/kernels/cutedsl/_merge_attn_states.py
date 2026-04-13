from __future__ import annotations

import os
from typing import Any
from typing import Callable
from typing import Tuple

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

_COMPILED_FUSED_MERGE_ATTN_STATES: dict[
  tuple[torch.dtype, torch.dtype, int, int, int, str],
  Callable[..., None],
] = {}
_COPY_BITS = 128
_CTA_THREADS = 128
_WARP_SIZE = 32
_SUPPORTED_HEAD_SIZES = (32, 64, 128)


def _derive_vector_config(head_size: int, element_bits: int) -> tuple[int, int]:
  """Derive the fixed-width copy layout from head size and element width.

  :param head_size: Attention head dimension `D`.
  :param element_bits: Element width in bits for the value tensors.
  :returns: Tuple `(elems_per_thread, row_threads)`.
  """

  assert head_size in _SUPPORTED_HEAD_SIZES, (
    f"CuTe DSL fused_merge_attn_states only supports D in {_SUPPORTED_HEAD_SIZES}, got {head_size}."
  )
  assert element_bits > 0, f"Expected positive element_bits, got {element_bits}."
  assert _COPY_BITS % element_bits == 0, (
    f"The fixed {_COPY_BITS}-bit copy path requires element_bits to divide {_COPY_BITS}, "
    f"got {element_bits}.")

  elems_per_thread = _COPY_BITS // element_bits
  assert head_size % elems_per_thread == 0, (
    "The fixed 128-bit copy path requires each row to be evenly covered by "
    f"whole vectors, but got D={head_size} and element_bits={element_bits}.")

  row_threads = head_size // elems_per_thread
  assert row_threads <= _WARP_SIZE, (
    f"Expected row_threads <= {_WARP_SIZE} for D={head_size}, got {row_threads}.")
  assert _CTA_THREADS % row_threads == 0, (
    f"Expected row_threads={row_threads} to divide CTA size {_CTA_THREADS} exactly.")
  return elems_per_thread, row_threads


class _MergeAttnStatesProgram:
  """Top-level CuTe DSL program object for fused attention-state merging.

  The object owns all static launch configuration so its `@cute.jit` and
  `@cute.kernel` methods can remain at module scope while still specializing on
  Python integers during compilation.

  :param head_size: Attention head dimension `D`.
  :param element_bits: Element width in bits for the value tensors.
  """

  def __init__(
    self,
    head_size: int,
    element_bits: int,
  ) -> None:
    self.elems_per_thread, self.row_threads = _derive_vector_config(
      head_size=head_size,
      element_bits=element_bits,
    )
    self.rows_per_cta = _CTA_THREADS // self.row_threads

  @cute.kernel
  def _merge_attn_states_kernel(
    self,
    out: cute.Tensor,
    lse: cute.Tensor,
    prev_out: cute.Tensor,
    prev_lse: cute.Tensor,
    suff_out: cute.Tensor,
    suff_lse: cute.Tensor,
    tiled_copy_load: cute.TiledCopy,
    tiled_copy_store: cute.TiledCopy,
  ):
    """Merge one or more rows of attention state using vectorized global copies.

    Each thread recomputes the scalar merge weight for its logical row so the hot path stays free of
    warp shuffle coordination.

    :param out: Output tensor `[rows, D]`.
    :param lse: Output LSE tensor `[rows]`.
    :param prev_out: Previous output tensor `[rows, D]`.
    :param prev_lse: Previous LSE tensor `[rows]`.
    :param suff_out: Suffix output tensor `[rows, D]`.
    :param suff_lse: Suffix LSE tensor `[rows]`.
    :param tiled_copy_load: Tiled vectorized gmem load descriptor.
    :param tiled_copy_store: Tiled vectorized gmem store descriptor.
    """

    tidx, _, _ = cute.arch.thread_idx()
    block_row_idx, _, _ = cute.arch.block_idx()
    row_in_block = tidx // self.row_threads
    lane_in_row = tidx % self.row_threads
    row_idx = block_row_idx * self.rows_per_cta + row_in_block

    thr_copy_load = tiled_copy_load.get_slice(lane_in_row)
    thr_copy_store = tiled_copy_store.get_slice(lane_in_row)
    one = cutlass.Float32(1.0)
    neg_inf = cutlass.Float32(-float("inf"))
    zero = cutlass.Float32(0.0)
    merge_weight = zero

    if row_idx < out.shape[0]:
      prev_lse_value = cutlass.Float32(prev_lse[row_idx])
      suff_lse_value = cutlass.Float32(suff_lse[row_idx])
      if prev_lse_value == cutlass.Float32.inf:
        prev_lse_value = neg_inf
      if suff_lse_value == cutlass.Float32.inf:
        suff_lse_value = neg_inf

      merge_weight = one / (one + cute.exp(prev_lse_value - suff_lse_value))
      if lane_in_row == 0:
        logsigmoid = one / (one + cute.exp(suff_lse_value - prev_lse_value))
        lse[row_idx] = (prev_lse_value - cute.log(logsigmoid)).to(lse.element_type)

    if row_idx < out.shape[0]:
      g_out = cute.local_tile(out, tiler=(1, out.shape[1]), coord=(row_idx, 0))[0, None]
      g_prev_out = cute.local_tile(prev_out, tiler=(1, prev_out.shape[1]), coord=(row_idx, 0))[0,
                                                                                               None]
      g_suff_out = cute.local_tile(suff_out, tiler=(1, suff_out.shape[1]), coord=(row_idx, 0))[0,
                                                                                               None]

      t_out_g = thr_copy_store.partition_S(g_out)
      t_prev_g = thr_copy_load.partition_S(g_prev_out)
      t_suff_g = thr_copy_load.partition_S(g_suff_out)

      t_prev_r = cute.make_fragment_like(t_prev_g)
      t_prev_r.fill(0)
      t_suff_r = cute.make_fragment_like(t_suff_g)
      t_suff_r.fill(0)
      t_out_r = cute.make_fragment_like(t_out_g)

      for i in range(cute.size(t_prev_r, mode=[1])):
        cute.autovec_copy(t_prev_g[None, i], t_prev_r[None, i])
        cute.autovec_copy(t_suff_g[None, i], t_suff_r[None, i])

      prev_values = t_prev_r.load().to(cutlass.Float32)
      suff_values = t_suff_r.load().to(cutlass.Float32)
      merged_values = prev_values - merge_weight * (prev_values - suff_values)
      t_out_r.store(merged_values.to(t_out_r.element_type))

      for i in range(cute.size(t_out_r, mode=[1])):
        cute.autovec_copy(t_out_r[None, i], t_out_g[None, i])

  @cute.jit
  def _merge_attn_states(
    self,
    out: cute.Tensor,
    lse: cute.Tensor,
    prev_out: cute.Tensor,
    prev_lse: cute.Tensor,
    suff_out: cute.Tensor,
    suff_lse: cute.Tensor,
  ) -> None:
    """Launch the fused merge kernel with object-owned static configuration.

    :param out: Output tensor `[rows, D]`.
    :param lse: Output LSE tensor `[rows]`.
    :param prev_out: Previous output tensor `[rows, D]`.
    :param prev_lse: Previous LSE tensor `[rows]`.
    :param suff_out: Suffix output tensor `[rows, D]`.
    :param suff_lse: Suffix LSE tensor `[rows]`.
    """

    copy_atom_load = cute.make_copy_atom(
      cute.nvgpu.CopyUniversalOp(),
      prev_out.element_type,
      num_bits_per_copy=_COPY_BITS,
    )
    copy_atom_store = cute.make_copy_atom(
      cute.nvgpu.CopyUniversalOp(),
      out.element_type,
      num_bits_per_copy=_COPY_BITS,
    )
    thread_layout = cute.make_layout(self.row_threads)
    value_layout = cute.make_layout(self.elems_per_thread)
    tiled_copy_load = cute.make_tiled_copy_tv(
      copy_atom_load,
      thread_layout,
      value_layout,
    )
    tiled_copy_store = cute.make_tiled_copy_tv(
      copy_atom_store,
      thread_layout,
      value_layout,
    )
    block_threads = _CTA_THREADS

    self._merge_attn_states_kernel(
      out,
      lse,
      prev_out,
      prev_lse,
      suff_out,
      suff_lse,
      tiled_copy_load,
      tiled_copy_store,
    ).launch(
      grid=[(out.shape[0] + self.rows_per_cta - 1) // self.rows_per_cta, 1, 1],
      block=[block_threads, 1, 1],
    )

  __call__ = _merge_attn_states


def _detect_cutedsl_arch() -> str:
  """Infer the CuTe DSL target architecture from the active CUDA device.

  :returns: SM architecture string such as `sm_89`.
  """

  major, minor = torch.cuda.get_device_capability()
  suffix = "a" if major >= 9 else ""
  return f"sm_{major}{minor}{suffix}"


if torch.cuda.is_available():
  os.environ.setdefault("CUTE_DSL_ARCH", _detect_cutedsl_arch())


def _wrap_tvm_ffi_tensor(tensor: torch.Tensor) -> Any:
  """Convert a torch tensor into a static-layout CuTe tensor.

  :param tensor: Torch tensor to expose to CuTe DSL.
  :returns: CuTe runtime tensor.
  """

  return from_dlpack(
    tensor,
    assumed_align=16,
    enable_tvm_ffi=True,
  )


def _compile_fused_merge_attn_states(
  out_rows: torch.Tensor,
  lse_rows: torch.Tensor,
  prev_out_rows: torch.Tensor,
  prev_lse_rows: torch.Tensor,
  suff_out_rows: torch.Tensor,
  suff_lse_rows: torch.Tensor,
) -> Callable[..., None]:
  """Compile and cache the CuTe DSL launcher for a concrete tensor signature.

  :param out_rows: Output tensor sample `[rows, D]`.
  :param lse_rows: LSE tensor sample `[rows]`.
  :param prev_out_rows: Previous output sample `[rows, D]`.
  :param prev_lse_rows: Previous LSE sample `[rows]`.
  :param suff_out_rows: Suffix output sample `[rows, D]`.
  :param suff_lse_rows: Suffix LSE sample `[rows]`.
  :returns: Compiled CuTe DSL callable that accepts `torch.Tensor` directly.
  """

  arch = os.environ.get("CUTE_DSL_ARCH", _detect_cutedsl_arch())
  cache_key = (
    out_rows.dtype,
    lse_rows.dtype,
    out_rows.shape[0],
    out_rows.shape[1],
    out_rows.device.index or torch.cuda.current_device(),
    arch,
  )
  compiled = _COMPILED_FUSED_MERGE_ATTN_STATES.get(cache_key)
  if compiled is not None:
    return compiled

  sample_out = _wrap_tvm_ffi_tensor(out_rows)
  sample_lse = _wrap_tvm_ffi_tensor(lse_rows)
  sample_prev_out = _wrap_tvm_ffi_tensor(prev_out_rows)
  sample_prev_lse = _wrap_tvm_ffi_tensor(prev_lse_rows)
  sample_suff_out = _wrap_tvm_ffi_tensor(suff_out_rows)
  sample_suff_lse = _wrap_tvm_ffi_tensor(suff_lse_rows)

  launcher = _MergeAttnStatesProgram(
    head_size=out_rows.shape[1],
    element_bits=out_rows.element_size() * 8,
  )
  launcher._merge_attn_states_kernel.set_name_prefix("cache_dit_cutedsl_fused_merge_attn_states")

  compiled = cute.compile(
    launcher,
    sample_out,
    sample_lse,
    sample_prev_out,
    sample_prev_lse,
    sample_suff_out,
    sample_suff_lse,
    options="--enable-tvm-ffi",
  )
  _COMPILED_FUSED_MERGE_ATTN_STATES[cache_key] = compiled
  return compiled


def _flatten_attn_state_rows(tensor: torch.Tensor, head_size: int) -> torch.Tensor:
  """Collapse `[B, N, H, D]` into contiguous `[B * N * H, D]` rows.

  :param tensor: Attention output tensor.
  :param head_size: Attention head dimension `D`.
  :returns: Row-major contiguous tensor.
  """

  return tensor.flatten(0, 1).contiguous().view(-1, head_size)


def _flatten_lse_rows(tensor: torch.Tensor) -> torch.Tensor:
  """Collapse `[B, N, H, 1]` into contiguous `[B * N * H]` rows.

  :param tensor: Attention log-sum-exp tensor.
  :returns: Flat contiguous tensor.
  """

  return tensor.flatten(0, 1).squeeze(-1).contiguous().view(-1)


def fused_merge_attn_states(
  prev_out: torch.Tensor,
  prev_lse: torch.Tensor,
  suff_out: torch.Tensor,
  suff_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Merge two attention-state tensors with a cached CuTe DSL kernel.

  :param prev_out: Previous attention output `[B, N, H, D]`.
  :param prev_lse: Previous attention LSE `[B, N, H, 1]`.
  :param suff_out: Suffix attention output `[B, N, H, D]`.
  :param suff_lse: Suffix attention LSE `[B, N, H, 1]`.
  :returns: Tuple `(out, lse)` with the same shapes as the Triton operator.
  :raises ValueError: If any input tensor is not CUDA resident.
  """

  if any(tensor.device.type != "cuda" for tensor in (prev_out, prev_lse, suff_out, suff_lse)):
    raise ValueError("CuTe DSL fused_merge_attn_states only supports CUDA tensors.")
  assert prev_out.dtype == suff_out.dtype, (
    f"Expected prev_out and suff_out to share dtype, got {prev_out.dtype} and {suff_out.dtype}.")

  _, _, _, head_size = suff_out.shape
  _derive_vector_config(
    head_size=head_size,
    element_bits=suff_out.element_size() * 8,
  )

  prev_out_rows = _flatten_attn_state_rows(prev_out, head_size)
  suff_out_rows = _flatten_attn_state_rows(suff_out, head_size)
  prev_lse_rows = _flatten_lse_rows(prev_lse)
  suff_lse_rows = _flatten_lse_rows(suff_lse)

  out_rows = torch.empty_like(suff_out_rows)
  lse_rows = torch.empty_like(suff_lse_rows)

  compiled = _compile_fused_merge_attn_states(
    out_rows=out_rows,
    lse_rows=lse_rows,
    prev_out_rows=prev_out_rows,
    prev_lse_rows=prev_lse_rows,
    suff_out_rows=suff_out_rows,
    suff_lse_rows=suff_lse_rows,
  )
  compiled(
    out_rows,
    lse_rows,
    prev_out_rows,
    prev_lse_rows,
    suff_out_rows,
    suff_lse_rows,
  )

  batch, seq_len, num_heads, _ = suff_out.shape
  out = out_rows.view(batch, seq_len, num_heads, head_size)
  lse = lse_rows.view(batch, seq_len, num_heads, 1)
  return out, lse
