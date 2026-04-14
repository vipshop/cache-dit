from __future__ import annotations

import os
from typing import Any
from typing import Callable

import cutlass
import cutlass.cute as cute
import torch
import torch.nn.functional as F
from cutlass.cute import make_layout
from cutlass.cute import make_tensor
from cutlass.cute.nvgpu import warp
from cutlass.cute import recast_ptr
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import BFloat16
from cutlass.cute.typing import Int32

from .gemm_utils import h2div_bf16x2_f32
from .gemm_utils import h2div_f16x2_f32
from .gemm_utils import quantize_f32x8_to_int4_word_signed
from .gemm_utils import rcp_approx_f32
from .gemm_utils import require_int4_runtime

_BLOCK_M = 256
_BLOCK_N = 128
_WARP_K = 64
_WARP_N = 128
_THREADS_PER_ROW = 8
_ROWS_PER_CTA = 32
_CTA_THREADS = _THREADS_PER_ROW * _ROWS_PER_CTA
_INT4_MAX = 7.0
_MMA_TILE_M = 16
_MMA_TILE_N = 16
_MMA_TILE_K = 16

_COMPILED_SVDQ_QACT_INT4: dict[tuple[torch.dtype, int, int, int, str], Callable[..., None]] = {}
_COMPILED_SVDQ_LORA_DOWN: dict[tuple[torch.dtype, int, int, int, int, str], Callable[...,
                                                                                     None]] = {}


def _detect_cutedsl_arch() -> str:
  major, minor = torch.cuda.get_device_capability()
  suffix = "a" if major >= 9 else ""
  return f"sm_{major}{minor}{suffix}"


if torch.cuda.is_available():
  os.environ.setdefault("CUTE_DSL_ARCH", _detect_cutedsl_arch())


def _wrap_tensor(tensor: torch.Tensor) -> Any:
  return from_dlpack(
    tensor,
    assumed_align=16,
    enable_tvm_ffi=True,
  )


def _ceil_div(value: int, divisor: int) -> int:
  return (value + divisor - 1) // divisor


def _prepare_runtime_activation(input: torch.Tensor, fuse_glu: bool) -> torch.Tensor:
  if not fuse_glu:
    return input
  if input.shape[1] % 2 != 0:
    raise ValueError(f"Expected an even channel count for fuse_glu=True, got {input.shape[1]}.")
  half_channels = input.shape[1] // 2
  return input[:, :half_channels] * F.silu(input[:, half_channels:])


def _packed_smooth_index(channel_idx: int) -> int:
  block_128 = (channel_idx // _WARP_N) * _WARP_N
  within_128 = channel_idx % _WARP_N
  block_16 = within_128 // 16
  within_16 = within_128 % 16
  return (block_128 + block_16 * 16 + ((within_16 % 8) // 2) * 4 + (within_16 // 8) * 2 +
          (within_16 % 2))


def _packed_lora_down_linear_index(channel_idx: int, rank_idx: int, rank_tiles: int) -> int:
  channel_pack = channel_idx // 16
  channel_inner = channel_idx % 16
  rank_pack = rank_idx // 16
  rank_inner = rank_idx % 16

  n_pack = rank_inner // 8
  n_lane = rank_inner % 8

  channel_pair = channel_inner // 2
  reg_k = channel_inner % 2
  k_pack = channel_pair // 4
  k_lane = channel_pair % 4

  return ((((((
    (channel_pack * rank_tiles + rank_pack) * 8 + n_lane) * 4 + k_lane) * 2 + n_pack) * 2 + k_pack)
           * 2 + reg_k))


def _make_lora_tiled_mma(element_type):
  mma_op = warp.MmaF16BF16Op(element_type, cutlass.Float32, (_MMA_TILE_M, 8, _MMA_TILE_K))
  return cute.make_tiled_mma(
    mma_op,
    (1, 1, 1),
    permutation_mnk=(_MMA_TILE_M, _MMA_TILE_N, _MMA_TILE_K),
  )


def _partition_fragment_abc(thr_mma: cute.ThrMma, sA: cute.Tensor, sB: cute.Tensor):
  acc = cute.make_rmem_tensor(thr_mma.partition_shape_C((_MMA_TILE_M, _MMA_TILE_N)),
                              cutlass.Float32)
  tCsA = thr_mma.partition_A(sA)
  tCsB = thr_mma.partition_B(sB)
  tCrA = thr_mma.make_fragment_A(tCsA)
  tCrB = thr_mma.make_fragment_B(tCsB)
  return acc, tCsA, tCsB, tCrA, tCrB


class _SVDQQuantizeInt4Program:

  def __init__(self, channels: int) -> None:
    self.channels = channels
    self.groups = channels // _WARP_K

  @cute.kernel
  def _kernel(self, qout: cute.Tensor, ascales: cute.Tensor, x: cute.Tensor, smooth: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    block_row_idx, group_idx, _ = cute.arch.block_idx()
    row_in_cta = tidx // _THREADS_PER_ROW
    lane_in_row = tidx % _THREADS_PER_ROW
    global_row = block_row_idx * _ROWS_PER_CTA + row_in_cta
    col_base = group_idx * _WARP_K + lane_in_row * 8

    qout_i32 = make_tensor(
      recast_ptr(qout.iterator, dtype=Int32),
      make_layout((qout.shape[0] * qout.shape[1] // 4, ), stride=(1, )),
    )

    smem = cutlass.utils.SmemAllocator()
    scale_smem = smem.allocate_tensor(x.element_type,
                                      make_layout((_ROWS_PER_CTA, )),
                                      byte_alignment=16)
    qpack_smem = smem.allocate_tensor(Int32,
                                      make_layout((_ROWS_PER_CTA, _THREADS_PER_ROW)),
                                      byte_alignment=16)

    local_values: list[cutlass.Float32] = []
    local_absmax = cutlass.Float32(0.0)
    for pair_idx in cutlass.range_constexpr(4):
      logical_col_0 = col_base + pair_idx * 2
      logical_col_1 = logical_col_0 + 1
      packed_smooth_idx_0 = _packed_smooth_index(logical_col_0)
      packed_smooth_idx_1 = _packed_smooth_index(logical_col_1)

      input_0 = cutlass.Float32(0.0)
      input_1 = cutlass.Float32(0.0)
      smooth_0 = cutlass.Float32(1.0)
      smooth_1 = cutlass.Float32(1.0)
      if global_row < x.shape[0]:
        if logical_col_0 < x.shape[1]:
          input_0 = cutlass.Float32(x[global_row, logical_col_0])
          smooth_0 = cutlass.Float32(smooth[packed_smooth_idx_0])
        if logical_col_1 < x.shape[1]:
          input_1 = cutlass.Float32(x[global_row, logical_col_1])
          smooth_1 = cutlass.Float32(smooth[packed_smooth_idx_1])

      if cutlass.const_expr(x.element_type == BFloat16):
        value_0, value_1 = h2div_bf16x2_f32(input_0, input_1, smooth_0, smooth_1)
      else:
        value_0, value_1 = h2div_f16x2_f32(input_0, input_1, smooth_0, smooth_1)

      local_values.append(value_0)
      local_values.append(value_1)
      local_absmax = cute.arch.fmax(local_absmax,
                                    cute.arch.fmax(value_0, value_0 * cutlass.Float32(-1.0)))
      local_absmax = cute.arch.fmax(local_absmax,
                                    cute.arch.fmax(value_1, value_1 * cutlass.Float32(-1.0)))

    row_absmax = cute.arch.warp_reduction(local_absmax,
                                          cute.arch.fmax,
                                          threads_in_group=_THREADS_PER_ROW)
    scale = row_absmax / cutlass.Float32(_INT4_MAX)
    inv_scale = cutlass.Float32(0.0)
    if scale > cutlass.Float32(0.0):
      inv_scale = rcp_approx_f32(scale)

    qpack_smem[row_in_cta, lane_in_row] = quantize_f32x8_to_int4_word_signed(
      local_values[0] * inv_scale,
      local_values[1] * inv_scale,
      local_values[2] * inv_scale,
      local_values[3] * inv_scale,
      local_values[4] * inv_scale,
      local_values[5] * inv_scale,
      local_values[6] * inv_scale,
      local_values[7] * inv_scale,
    )
    if lane_in_row == 0:
      scale_smem[row_in_cta] = scale.to(scale_smem.element_type)

    cute.arch.barrier()

    if tidx < _ROWS_PER_CTA:
      packed_pos = tidx
      logical_row = (packed_pos // 16) * 16 + (packed_pos % 16) // 2 + (packed_pos % 2) * 8
      ascales[group_idx, block_row_idx * _ROWS_PER_CTA + packed_pos] = scale_smem[logical_row]

    if row_in_cta < 16:
      if lane_in_row < 4:
        tile_idx = row_in_cta // 8
        row_quad = row_in_cta % 8
        top_row = tile_idx * 16 + row_quad
        bottom_row = top_row + 8
        base_word = (((group_idx *
                       (qout.shape[0] // _ROWS_PER_CTA) + block_row_idx) * 2 + tile_idx) * 8 +
                     row_quad) * 4 + lane_in_row
        base = base_word * 4
        qout_i32[base + 0] = qpack_smem[top_row, lane_in_row]
        qout_i32[base + 1] = qpack_smem[bottom_row, lane_in_row]
        qout_i32[base + 2] = qpack_smem[top_row, lane_in_row + 4]
        qout_i32[base + 3] = qpack_smem[bottom_row, lane_in_row + 4]

  @cute.jit
  def __call__(self, qout: cute.Tensor, ascales: cute.Tensor, x: cute.Tensor,
               smooth: cute.Tensor) -> None:
    self._kernel(qout, ascales, x, smooth).launch(
      grid=[qout.shape[0] // _ROWS_PER_CTA, self.groups, 1],
      block=[_CTA_THREADS, 1, 1],
    )


class _SVDQLoraDownProgram:

  def __init__(self, channels: int, rank: int) -> None:
    self.channels = channels
    self.rank = rank
    self.rows_per_warp = 32
    self.rank_tile = 16
    self.rank_tiles = rank // self.rank_tile
    self.warps_per_block_m = _BLOCK_M // self.rows_per_warp
    self.cta_threads = 32

  @cute.kernel
  def _kernel(self, out: cute.Tensor, x: cute.Tensor, lora_down: cute.Tensor):
    lane_id, _, _ = cute.arch.thread_idx()
    warp_tile_idx, rank_tile_idx, _ = cute.arch.block_idx()

    out_linear = make_tensor(
      recast_ptr(out.iterator, dtype=out.element_type),
      make_layout((out.shape[0] * out.shape[1], ), stride=(1, )),
    )

    block_m = warp_tile_idx // self.warps_per_block_m
    warp_idx = warp_tile_idx % self.warps_per_block_m
    tiled_mma = _make_lora_tiled_mma(x.element_type)

    atom_copy_s2r_A = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), x.element_type)
    atom_copy_s2r_B = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), lora_down.element_type)
    tiled_copy_s2r_A = cute.make_tiled_copy_A(atom_copy_s2r_A, tiled_mma)
    tiled_copy_s2r_B = cute.make_tiled_copy_B(atom_copy_s2r_B, tiled_mma)

    thr_mma = tiled_mma.get_slice(lane_id)
    thr_copy_ldmatrix_A = tiled_copy_s2r_A.get_slice(lane_id)
    thr_copy_ldmatrix_B = tiled_copy_s2r_B.get_slice(lane_id)

    smem = cutlass.utils.SmemAllocator()
    a_top_smem = smem.allocate_tensor(
      x.element_type,
      make_layout((_MMA_TILE_M, _MMA_TILE_K), stride=(_MMA_TILE_K, 1)),
      byte_alignment=16,
    )
    a_bottom_smem = smem.allocate_tensor(
      x.element_type,
      make_layout((_MMA_TILE_M, _MMA_TILE_K), stride=(_MMA_TILE_K, 1)),
      byte_alignment=16,
    )
    b_smem = smem.allocate_tensor(
      lora_down.element_type,
      make_layout((_MMA_TILE_N, _MMA_TILE_K), stride=(_MMA_TILE_K, 1)),
      byte_alignment=16,
    )
    out_store_smem = smem.allocate_tensor(
      out.element_type,
      make_layout((2, _MMA_TILE_M * _MMA_TILE_N), stride=(_MMA_TILE_M * _MMA_TILE_N, 1)),
      byte_alignment=16,
    )

    acc_top, _, _, tCrA_top, tCrB = _partition_fragment_abc(thr_mma, a_top_smem, b_smem)
    acc_bottom, _, _, tCrA_bottom, _ = _partition_fragment_abc(thr_mma, a_bottom_smem, b_smem)
    acc_top.fill(0.0)
    acc_bottom.fill(0.0)

    tCsA_top_copy_view = thr_copy_ldmatrix_A.partition_S(a_top_smem)
    tCsA_bottom_copy_view = thr_copy_ldmatrix_A.partition_S(a_bottom_smem)
    tCrA_top_copy_view = thr_copy_ldmatrix_A.retile(tCrA_top)
    tCrA_bottom_copy_view = thr_copy_ldmatrix_A.retile(tCrA_bottom)

    tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(b_smem)
    tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)

    base_row = warp_tile_idx * self.rows_per_warp
    rank_base = rank_tile_idx * self.rank_tile

    for channel_tile_idx in cutlass.range_constexpr(_ceil_div(self.channels, _MMA_TILE_K)):
      channel_base = channel_tile_idx * _MMA_TILE_K

      for linear_idx in cutlass.range_constexpr(_MMA_TILE_M * _MMA_TILE_K // 32):
        tile_linear = linear_idx * 32 + lane_id
        row_idx = tile_linear // _MMA_TILE_K
        col_idx = tile_linear % _MMA_TILE_K
        channel_idx = channel_base + col_idx
        top_row = base_row + row_idx
        bottom_row = top_row + _MMA_TILE_M

        top_value = x.element_type(0)
        bottom_value = x.element_type(0)
        if top_row < x.shape[0] and channel_idx < self.channels:
          top_value = x[top_row, channel_idx]
        if bottom_row < x.shape[0] and channel_idx < self.channels:
          bottom_value = x[bottom_row, channel_idx]

        a_top_smem[row_idx, col_idx] = top_value
        a_bottom_smem[row_idx, col_idx] = bottom_value

      for linear_idx in cutlass.range_constexpr(_MMA_TILE_N * _MMA_TILE_K // 32):
        tile_linear = linear_idx * 32 + lane_id
        rank_local_idx = tile_linear // _MMA_TILE_K
        channel_local_idx = tile_linear % _MMA_TILE_K
        channel_idx = channel_base + channel_local_idx
        rank_idx = rank_base + rank_local_idx

        weight_value = lora_down.element_type(0)
        if channel_idx < self.channels and rank_idx < self.rank:
          packed_idx = _packed_lora_down_linear_index(channel_idx, rank_idx, self.rank_tiles)
          weight_value = lora_down[packed_idx // self.rank, packed_idx % self.rank]

        b_smem[rank_local_idx, channel_local_idx] = weight_value

      cute.arch.barrier()

      cute.copy(
        tiled_copy_s2r_A,
        tCsA_top_copy_view[None, None, 0],
        tCrA_top_copy_view[None, None, 0],
      )
      cute.copy(
        tiled_copy_s2r_A,
        tCsA_bottom_copy_view[None, None, 0],
        tCrA_bottom_copy_view[None, None, 0],
      )
      cute.copy(
        tiled_copy_s2r_B,
        tCsB_copy_view[None, None, 0],
        tCrB_copy_view[None, None, 0],
      )

      cute.gemm(tiled_mma, acc_top, tCrA_top[None, None, 0], tCrB[None, None, 0], acc_top)
      cute.gemm(tiled_mma, acc_bottom, tCrA_bottom[None, None, 0], tCrB[None, None, 0], acc_bottom)

      cute.arch.barrier()

    acc_top_vals = acc_top.load()
    acc_bottom_vals = acc_bottom.load()

    for m_tile in cutlass.range_constexpr(2):
      acc = acc_top_vals if m_tile == 0 else acc_bottom_vals
      smem_tile = out_store_smem[m_tile, None]

      for frag_idx in cutlass.range_constexpr(8):
        smem_tile[frag_idx * 32 + lane_id] = acc[frag_idx]

    cute.arch.barrier()

    for m_tile in cutlass.range_constexpr(2):
      smem_tile = out_store_smem[m_tile, None]

      packed_tile_base = ((
        (block_m * self.rank_tiles + rank_tile_idx) * self.warps_per_block_m + warp_idx) * 2 +
                          m_tile) * 8

      gmem_tile = cute.domain_offset((packed_tile_base * 32, ), out_linear)
      for vec_group in cutlass.range_constexpr(2):
        vec_base = vec_group * 128 + lane_id * 4
        vec_coord = vec_base // 4
        smem_vec = cute.local_tile(smem_tile, (4, ), (vec_coord, ))
        gmem_vec = cute.local_tile(gmem_tile, (4, ), (vec_coord, ))
        cute.autovec_copy(smem_vec, gmem_vec)

  @cute.jit
  def __call__(self, out: cute.Tensor, x: cute.Tensor, lora_down: cute.Tensor) -> None:
    self._kernel(out, x, lora_down).launch(
      grid=[out.shape[0] // self.rows_per_warp, self.rank_tiles, 1],
      block=[self.cta_threads, 1, 1],
    )


def _compile_svdq_qact_int4(qout: torch.Tensor, ascales: torch.Tensor, x: torch.Tensor,
                            smooth: torch.Tensor) -> Callable[..., None]:
  arch = os.environ.get("CUTE_DSL_ARCH", _detect_cutedsl_arch())
  cache_key = (x.dtype, x.shape[0], x.shape[1], x.device.index or torch.cuda.current_device(), arch)
  compiled = _COMPILED_SVDQ_QACT_INT4.get(cache_key)
  if compiled is not None:
    return compiled

  launcher = _SVDQQuantizeInt4Program(channels=x.shape[1])
  launcher._kernel.set_name_prefix("cache_dit_cutedsl_svdq_quantize_int4")
  compiled = cute.compile(
    launcher,
    _wrap_tensor(qout),
    _wrap_tensor(ascales),
    _wrap_tensor(x),
    _wrap_tensor(smooth),
    options="--enable-tvm-ffi",
  )
  _COMPILED_SVDQ_QACT_INT4[cache_key] = compiled
  return compiled


def _compile_svdq_lora_down(out: torch.Tensor, x: torch.Tensor,
                            lora_down: torch.Tensor) -> Callable[..., None]:
  arch = os.environ.get("CUTE_DSL_ARCH", _detect_cutedsl_arch())
  cache_key = (
    x.dtype,
    x.shape[0],
    x.shape[1],
    lora_down.shape[1],
    x.device.index or torch.cuda.current_device(),
    arch,
  )
  compiled = _COMPILED_SVDQ_LORA_DOWN.get(cache_key)
  if compiled is not None:
    return compiled

  launcher = _SVDQLoraDownProgram(channels=x.shape[1], rank=lora_down.shape[1])
  launcher._kernel.set_name_prefix("cache_dit_cutedsl_svdq_lora_down")
  compiled = cute.compile(
    launcher,
    _wrap_tensor(out),
    _wrap_tensor(x),
    _wrap_tensor(lora_down),
    options="--enable-tvm-ffi",
  )
  _COMPILED_SVDQ_LORA_DOWN[cache_key] = compiled
  return compiled


def svdq_quantize_w4a4_act_fuse_lora(
  input: torch.Tensor,
  lora_down: torch.Tensor | None = None,
  smooth: torch.Tensor | None = None,
  fuse_glu: bool = False,
  fp4: bool = False,
  pad_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  require_int4_runtime(fp4, "svdq_quantize_w4a4_act_fuse_lora")
  if fuse_glu:
    raise NotImplementedError(
      "svdq_quantize_w4a4_act_fuse_lora v3 CuTe DSL path currently targets fuse_glu=False only.")
  if input.ndim != 2:
    raise ValueError(f"Expected input with shape [M, K], got {tuple(input.shape)}.")
  if pad_size <= 0 or pad_size % _BLOCK_M != 0:
    raise ValueError(f"pad_size must be a positive multiple of {_BLOCK_M}, got {pad_size}.")

  activation = _prepare_runtime_activation(input=input, fuse_glu=fuse_glu)
  actual_m, actual_n = activation.shape
  if actual_n % _WARP_K != 0:
    raise ValueError(f"Expected channels divisible by {_WARP_K}, got {actual_n}.")
  if smooth is not None and smooth.shape != (actual_n, ):
    raise ValueError(f"Expected smooth with shape {(actual_n, )}, got {tuple(smooth.shape)}.")
  if lora_down is not None and lora_down.shape[0] != actual_n:
    raise ValueError(
      f"Expected lora_down shape [K, R] with K={actual_n}, got {tuple(lora_down.shape)}.")

  padded_m = _ceil_div(actual_m, pad_size) * pad_size
  padded_n = _ceil_div(actual_n, _BLOCK_N) * _BLOCK_N
  padded_activation = activation.new_zeros((padded_m, padded_n))
  padded_activation[:actual_m, :actual_n] = activation

  smooth_runtime = torch.ones((padded_n, ), dtype=input.dtype, device=input.device)
  if smooth is not None:
    smooth_runtime[:actual_n] = smooth.to(dtype=input.dtype, device=input.device)

  qact = torch.empty((padded_m, padded_n // 2), dtype=torch.uint8, device=input.device)
  ascales = torch.empty((padded_n // _WARP_K, padded_m), dtype=input.dtype, device=input.device)
  quantize_launcher = _compile_svdq_qact_int4(qout=qact,
                                              ascales=ascales,
                                              x=padded_activation,
                                              smooth=smooth_runtime)
  quantize_launcher(qact, ascales, padded_activation, smooth_runtime)

  lora_rank = 0 if lora_down is None else lora_down.shape[1]
  lora_act_out = torch.zeros((padded_m, lora_rank), dtype=torch.float32, device=input.device)
  if lora_rank > 0:
    # The current CuTe warp-MMA LoRA-down path is stable on fp16 operands;
    # keeping the packed float32 output preserves the public contract.
    lora_kernel_dtype = torch.float16
    lora_input = torch.zeros((padded_m, actual_n), dtype=lora_kernel_dtype, device=input.device)
    lora_input[:actual_m] = activation.to(dtype=lora_kernel_dtype)
    lora_down_runtime = lora_down.to(dtype=lora_kernel_dtype)
    lora_launcher = _compile_svdq_lora_down(out=lora_act_out,
                                            x=lora_input,
                                            lora_down=lora_down_runtime)
    lora_launcher(lora_act_out, lora_input, lora_down_runtime)
  return qact, ascales, lora_act_out


__all__ = ["svdq_quantize_w4a4_act_fuse_lora"]
