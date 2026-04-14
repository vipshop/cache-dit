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
from .gemm_utils import load_pred_v4_b32
from .gemm_utils import quantize_f32x8_to_int4_word_signed
from .gemm_utils import reduce_add_f32
from .gemm_utils import rcp_approx_f32
from .gemm_utils import require_int4_runtime
from .gemm_utils import store_global_cg_v4_b32
from .gemm_utils import store_shared_v4_b32

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
_I32_WORDS_PER_VEC = 4
_PACKED_HALFS_PER_I32 = 2
_PACKED_HALFS_PER_VEC = _I32_WORDS_PER_VEC * _PACKED_HALFS_PER_I32
_WARP_THREADS = 32
_ROWS_PER_WARP = 32
_WARP_ROW_GROUPS = _WARP_THREADS // _THREADS_PER_ROW
_ROW_BATCHES_PER_WARP = _ROWS_PER_WARP // _WARP_ROW_GROUPS
_WARPS_PER_BLOCK_M = _BLOCK_M // _ROWS_PER_WARP
_GROUPS_PER_BLOCK_N = _BLOCK_N // _WARP_K
_CHANNEL_TILES_PER_BLOCK_N = _BLOCK_N // _MMA_TILE_K

_COMPILED_SVDQ_FUSED: dict[tuple[torch.dtype, int, int, int, int, str], Callable[..., None]] = {}


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


class _SVDQFusedQuantizeLoraProgram:
  """Single-launch CuTe DSL runtime kernel matching CUDA's fused tile flow.

  One CTA covers ``BLOCK_M x BLOCK_N``. Each warp owns one ``32 x 128``
  activation tile, writes the quantized runtime tensors, and accumulates the
  LoRA-down partial sums for the same tile before moving on.
  """

  def __init__(self, channels: int, rank: int) -> None:
    self.channels = channels
    self.rank = rank
    self.groups = channels // _WARP_K
    self.rank_tile = _MMA_TILE_N
    self.rank_tiles = rank // self.rank_tile
    self.rows_per_warp = _ROWS_PER_WARP
    self.warps_per_block_m = _WARPS_PER_BLOCK_M
    self.groups_per_block_n = _GROUPS_PER_BLOCK_N
    self.channel_tiles_per_block_n = _CHANNEL_TILES_PER_BLOCK_N
    self.cta_threads = _CTA_THREADS

  @cute.kernel
  def _kernel(self, qout: cute.Tensor, ascales: cute.Tensor, out: cute.Tensor, x: cute.Tensor,
              smooth: cute.Tensor, lora_down: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    block_m_idx, block_n_idx, _ = cute.arch.block_idx()
    warp_id = tidx // _WARP_THREADS
    lane_id = tidx % _WARP_THREADS
    subgroup_id = lane_id // _THREADS_PER_ROW
    lane_in_row = lane_id % _THREADS_PER_ROW
    block_row_idx = block_m_idx * self.warps_per_block_m + warp_id
    base_row = block_row_idx * _ROWS_PER_CTA
    channel_block_base = block_n_idx * _BLOCK_N

    qout_i32_ptr = recast_ptr(qout.iterator, dtype=Int32)
    x_i32_ptr = recast_ptr(x.iterator, dtype=Int32)
    lora_down_i32_ptr = recast_ptr(lora_down.iterator, dtype=Int32)
    out_ptr = recast_ptr(out.iterator, dtype=out.element_type)

    tiled_mma = _make_lora_tiled_mma(x.element_type)
    atom_copy_s2r_A = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), x.element_type)
    atom_copy_s2r_B = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), lora_down.element_type)
    tiled_copy_s2r_A = cute.make_tiled_copy_A(atom_copy_s2r_A, tiled_mma)
    tiled_copy_s2r_B = cute.make_tiled_copy_B(atom_copy_s2r_B, tiled_mma)

    thr_mma = tiled_mma.get_slice(lane_id)
    thr_copy_ldmatrix_A = tiled_copy_s2r_A.get_slice(lane_id)
    thr_copy_ldmatrix_B = tiled_copy_s2r_B.get_slice(lane_id)

    smem = cutlass.utils.SmemAllocator()
    scale_smem = smem.allocate_tensor(
      x.element_type,
      make_layout((self.warps_per_block_m, self.groups_per_block_n, _ROWS_PER_CTA),
                  stride=(self.groups_per_block_n * _ROWS_PER_CTA, _ROWS_PER_CTA, 1)),
      byte_alignment=16,
    )
    qpack_smem = smem.allocate_tensor(
      Int32,
      make_layout(
        (self.warps_per_block_m, _ROWS_PER_CTA, self.groups_per_block_n * _THREADS_PER_ROW),
        stride=(_ROWS_PER_CTA * self.groups_per_block_n * _THREADS_PER_ROW,
                self.groups_per_block_n * _THREADS_PER_ROW, 1)),
      byte_alignment=16,
    )
    a_top_smem = smem.allocate_tensor(
      x.element_type,
      make_layout((self.warps_per_block_m, _MMA_TILE_M, _MMA_TILE_K),
                  stride=(_MMA_TILE_M * _MMA_TILE_K, _MMA_TILE_K, 1)),
      byte_alignment=16,
    )
    a_bottom_smem = smem.allocate_tensor(
      x.element_type,
      make_layout((self.warps_per_block_m, _MMA_TILE_M, _MMA_TILE_K),
                  stride=(_MMA_TILE_M * _MMA_TILE_K, _MMA_TILE_K, 1)),
      byte_alignment=16,
    )
    b_smem = smem.allocate_tensor(
      lora_down.element_type,
      make_layout((self.warps_per_block_m, _MMA_TILE_N, _MMA_TILE_K),
                  stride=(_MMA_TILE_N * _MMA_TILE_K, _MMA_TILE_K, 1)),
      byte_alignment=16,
    )
    out_store_smem = smem.allocate_tensor(
      out.element_type,
      make_layout((self.warps_per_block_m, 2, _MMA_TILE_M * _MMA_TILE_N),
                  stride=(2 * _MMA_TILE_M * _MMA_TILE_N, _MMA_TILE_M * _MMA_TILE_N, 1)),
      byte_alignment=16,
    )

    warp_scale_smem = scale_smem[warp_id, None, None]
    warp_qpack_smem = qpack_smem[warp_id, None, None]
    warp_a_top_smem = a_top_smem[warp_id, None, None]
    warp_a_bottom_smem = a_bottom_smem[warp_id, None, None]
    warp_b_smem = b_smem[warp_id, None, None]
    warp_out_store_smem = out_store_smem[warp_id, None, None]

    a_top_smem_i32_ptr = recast_ptr(warp_a_top_smem.iterator, dtype=Int32)
    a_bottom_smem_i32_ptr = recast_ptr(warp_a_bottom_smem.iterator, dtype=Int32)
    b_smem_i32 = make_tensor(
      recast_ptr(warp_b_smem.iterator, dtype=Int32),
      make_layout((_MMA_TILE_N, _MMA_TILE_K // _PACKED_HALFS_PER_I32),
                  stride=(_MMA_TILE_K // _PACKED_HALFS_PER_I32, 1)),
    )

    acc_top, _, _, tCrA_top, tCrB = _partition_fragment_abc(thr_mma, warp_a_top_smem, warp_b_smem)
    acc_bottom, _, _, tCrA_bottom, _ = _partition_fragment_abc(thr_mma, warp_a_bottom_smem,
                                                               warp_b_smem)
    tCsA_top_copy_view = thr_copy_ldmatrix_A.partition_S(warp_a_top_smem)
    tCsA_bottom_copy_view = thr_copy_ldmatrix_A.partition_S(warp_a_bottom_smem)
    tCrA_top_copy_view = thr_copy_ldmatrix_A.retile(tCrA_top)
    tCrA_bottom_copy_view = thr_copy_ldmatrix_A.retile(tCrA_bottom)
    tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(warp_b_smem)
    tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)

    for group_in_block in cutlass.range_constexpr(self.groups_per_block_n):
      group_idx = block_n_idx * self.groups_per_block_n + group_in_block
      group_col_base = channel_block_base + group_in_block * _WARP_K

      for row_batch in cutlass.range_constexpr(_ROW_BATCHES_PER_WARP):
        row_local = row_batch * _WARP_ROW_GROUPS + subgroup_id
        global_row = base_row + row_local
        col_base = group_col_base + lane_in_row * 8

        local_values: list[cutlass.Float32] = []
        local_absmax = cutlass.Float32(0.0)
        for pair_idx in cutlass.range_constexpr(4):
          logical_col_0 = col_base + pair_idx * 2
          logical_col_1 = logical_col_0 + 1
          packed_smooth_idx_0 = _packed_smooth_index(logical_col_0)
          packed_smooth_idx_1 = _packed_smooth_index(logical_col_1)

          input_0 = cutlass.Float32(x[global_row, logical_col_0])
          input_1 = cutlass.Float32(x[global_row, logical_col_1])
          smooth_0 = cutlass.Float32(smooth[packed_smooth_idx_0])
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

        warp_qpack_smem[row_local, group_in_block * _THREADS_PER_ROW +
                        lane_in_row] = (quantize_f32x8_to_int4_word_signed(
                          local_values[0] * inv_scale,
                          local_values[1] * inv_scale,
                          local_values[2] * inv_scale,
                          local_values[3] * inv_scale,
                          local_values[4] * inv_scale,
                          local_values[5] * inv_scale,
                          local_values[6] * inv_scale,
                          local_values[7] * inv_scale,
                        ))
        if lane_in_row == 0:
          warp_scale_smem[group_in_block, row_local] = scale.to(warp_scale_smem.element_type)

    cute.arch.barrier()

    packed_pos = lane_id
    logical_row = (packed_pos // 16) * 16 + (packed_pos % 16) // 2 + (packed_pos % 2) * 8
    for group_in_block in cutlass.range_constexpr(self.groups_per_block_n):
      group_idx = block_n_idx * self.groups_per_block_n + group_in_block
      ascales[group_idx, base_row + packed_pos] = warp_scale_smem[group_in_block, logical_row]

    for group_in_block in cutlass.range_constexpr(self.groups_per_block_n):
      group_idx = block_n_idx * self.groups_per_block_n + group_in_block
      lane_base = group_in_block * _THREADS_PER_ROW
      for store_iter in cutlass.range_constexpr(2):
        store_thread = store_iter * _WARP_THREADS + lane_id
        row_store = store_thread // 4
        store_lane = store_thread % 4
        tile_idx = row_store // 8
        row_quad = row_store % 8
        top_row = tile_idx * 16 + row_quad
        bottom_row = top_row + 8
        base_word = (((group_idx *
                       (qout.shape[0] // _ROWS_PER_CTA) + block_row_idx) * 2 + tile_idx) * 8 +
                     row_quad) * 4 + store_lane
        base = base_word * 4
        store_global_cg_v4_b32(
          qout_i32_ptr + base,
          warp_qpack_smem[top_row, lane_base + store_lane],
          warp_qpack_smem[bottom_row, lane_base + store_lane],
          warp_qpack_smem[top_row, lane_base + store_lane + 4],
          warp_qpack_smem[bottom_row, lane_base + store_lane + 4],
        )

    if cutlass.const_expr(self.rank_tiles > 0):
      x_row_stride_i32 = x.stride[0] // _PACKED_HALFS_PER_I32
      a_smem_row_stride_i32 = _MMA_TILE_K // _PACKED_HALFS_PER_I32
      packed_lora_tile_i32 = _MMA_TILE_N * _MMA_TILE_K // _PACKED_HALFS_PER_I32

      for rank_tile_idx in cutlass.range_constexpr(self.rank_tiles):
        acc_top.fill(0.0)
        acc_bottom.fill(0.0)

        for channel_tile_idx in cutlass.range_constexpr(self.channel_tiles_per_block_n):
          channel_tile_global = block_n_idx * self.channel_tiles_per_block_n + channel_tile_idx
          channel_base = channel_tile_global * _MMA_TILE_K
          a_row_idx = lane_id // 2
          a_col_word = (lane_id % 2) * _I32_WORDS_PER_VEC
          top_row = base_row + a_row_idx
          bottom_row = top_row + _MMA_TILE_M
          channel_base_i32 = channel_base // _PACKED_HALFS_PER_I32

          a_top_0, a_top_1, a_top_2, a_top_3 = load_pred_v4_b32(
            x_i32_ptr + top_row * x_row_stride_i32 + channel_base_i32 + a_col_word,
            1,
          )
          a_bottom_0, a_bottom_1, a_bottom_2, a_bottom_3 = load_pred_v4_b32(
            x_i32_ptr + bottom_row * x_row_stride_i32 + channel_base_i32 + a_col_word,
            1,
          )
          store_shared_v4_b32(
            a_top_smem_i32_ptr + a_row_idx * a_smem_row_stride_i32 + a_col_word,
            a_top_0,
            a_top_1,
            a_top_2,
            a_top_3,
          )
          store_shared_v4_b32(
            a_bottom_smem_i32_ptr + a_row_idx * a_smem_row_stride_i32 + a_col_word,
            a_bottom_0,
            a_bottom_1,
            a_bottom_2,
            a_bottom_3,
          )

          n_lane = lane_id % 8
          k_lane = lane_id // 8
          lora_tile_base_i32 = (channel_tile_global * self.rank_tiles +
                                rank_tile_idx) * packed_lora_tile_i32
          lora_lane_base_i32 = (n_lane * 4 + k_lane) * _I32_WORDS_PER_VEC
          b0, b1, b2, b3 = load_pred_v4_b32(
            lora_down_i32_ptr + lora_tile_base_i32 + lora_lane_base_i32,
            1,
          )
          b_smem_i32[n_lane, k_lane] = b0
          b_smem_i32[n_lane, k_lane + 4] = b1
          b_smem_i32[n_lane + 8, k_lane] = b2
          b_smem_i32[n_lane + 8, k_lane + 4] = b3

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
          cute.gemm(tiled_mma, acc_bottom, tCrA_bottom[None, None, 0], tCrB[None, None, 0],
                    acc_bottom)

          cute.arch.barrier()

        acc_top_vals = acc_top.load()
        acc_bottom_vals = acc_bottom.load()

        for m_tile in cutlass.range_constexpr(2):
          acc_vals = acc_top_vals if m_tile == 0 else acc_bottom_vals
          smem_tile = warp_out_store_smem[m_tile, None]
          for frag_idx in cutlass.range_constexpr(8):
            smem_tile[frag_idx * _WARP_THREADS + lane_id] = acc_vals[frag_idx]

        cute.arch.barrier()

        for m_tile in cutlass.range_constexpr(2):
          smem_tile = warp_out_store_smem[m_tile, None]
          linear_tile_base = ((
            (block_m_idx * self.rank_tiles + rank_tile_idx) * self.warps_per_block_m + warp_id) * 2
                              + m_tile) * 256
          for vec_group in cutlass.range_constexpr(2):
            vec_base = vec_group * 128 + lane_id * 4
            for elem_idx in cutlass.range_constexpr(4):
              reduce_add_f32(out_ptr + linear_tile_base + vec_base + elem_idx,
                             smem_tile[vec_base + elem_idx])

        cute.arch.barrier()

  @cute.jit
  def __call__(self, qout: cute.Tensor, ascales: cute.Tensor, out: cute.Tensor, x: cute.Tensor,
               smooth: cute.Tensor, lora_down: cute.Tensor) -> None:
    self._kernel(qout, ascales, out, x, smooth, lora_down).launch(
      grid=[qout.shape[0] // _BLOCK_M, qout.shape[1] * 2 // _BLOCK_N, 1],
      block=[self.cta_threads, 1, 1],
    )


def _compile_act_fused_lora(qout: torch.Tensor, ascales: torch.Tensor, lora_out: torch.Tensor,
                            x: torch.Tensor, smooth: torch.Tensor,
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
  compiled = _COMPILED_SVDQ_FUSED.get(cache_key)
  if compiled is not None:
    return compiled

  launcher = _SVDQFusedQuantizeLoraProgram(channels=x.shape[1], rank=lora_down.shape[1])
  launcher._kernel.set_name_prefix("cache_dit_cutedsl_svdq_quantize_fuse_lora")
  compiled = cute.compile(
    launcher,
    _wrap_tensor(qout),
    _wrap_tensor(ascales),
    _wrap_tensor(lora_out),
    _wrap_tensor(x),
    _wrap_tensor(smooth),
    _wrap_tensor(lora_down),
    options="--enable-tvm-ffi",
  )
  _COMPILED_SVDQ_FUSED[cache_key] = compiled
  return compiled


def svdq_quantize_w4a4_act_fuse_lora(
  input: torch.Tensor,
  lora_down: torch.Tensor | None = None,
  smooth: torch.Tensor | None = None,
  fuse_glu: bool = False,
  fp4: bool = False,
  pad_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Run the CuTe DSL fused quantize plus LoRA-down runtime kernel.

  :param input: Activation matrix with shape ``[M, K]``.
  :param lora_down: Optional LoRA-down matrix with shape ``[K, R]``.
  :param smooth: Optional per-channel smooth factors with shape ``[K]``.
  :param fuse_glu: Whether to fuse the GLU activation path.
  :param fp4: Whether to use FP4 quantization.
  :param pad_size: Padding granularity for the runtime row tile.
  :returns:
      Tuple ``(qact, ascales, lora_act_out)`` with the same contract as the
      CUDA runtime kernel.
  """

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
  if lora_down is not None and lora_down.shape[1] % _MMA_TILE_N != 0:
    raise ValueError(
      f"Expected lora_down rank divisible by {_MMA_TILE_N}, got {lora_down.shape[1]}.")

  padded_m = _ceil_div(actual_m, pad_size) * pad_size
  padded_n = _ceil_div(actual_n, _BLOCK_N) * _BLOCK_N
  padded_activation = activation.new_zeros((padded_m, padded_n))
  padded_activation[:actual_m, :actual_n] = activation

  smooth_runtime = torch.ones((padded_n, ), dtype=input.dtype, device=input.device)
  if smooth is not None:
    smooth_runtime[:actual_n] = smooth.to(dtype=input.dtype, device=input.device)

  qact = torch.empty((padded_m, padded_n // 2), dtype=torch.uint8, device=input.device)
  ascales = torch.empty((padded_n // _WARP_K, padded_m), dtype=input.dtype, device=input.device)

  lora_rank = 0 if lora_down is None else lora_down.shape[1]
  lora_act_out = torch.zeros((padded_m, lora_rank), dtype=torch.float32, device=input.device)
  lora_down_runtime = torch.zeros((padded_n, lora_rank), dtype=input.dtype, device=input.device)
  if lora_rank > 0:
    lora_down_runtime[:actual_n] = lora_down.to(dtype=input.dtype, device=input.device)

  fused_launcher = _compile_act_fused_lora(qout=qact,
                                           ascales=ascales,
                                           lora_out=lora_act_out,
                                           x=padded_activation,
                                           smooth=smooth_runtime,
                                           lora_down=lora_down_runtime)
  fused_launcher(qact, ascales, lora_act_out, padded_activation, smooth_runtime, lora_down_runtime)
  return qact, ascales, lora_act_out


__all__ = ["svdq_quantize_w4a4_act_fuse_lora"]
