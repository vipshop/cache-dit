from __future__ import annotations

import os
from typing import Any
from typing import Callable

import cutlass
import cutlass.cute as cute
import torch
import torch.nn.functional as F
from cutlass.cute import make_layout
from cutlass.cute import recast_ptr
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import BFloat16
from cutlass.cute.typing import Int32

from .gemm_utils import bf16x2_bits_to_f32x2
from .gemm_utils import f16x2_bits_to_f32x2
from .gemm_utils import h2div_bf16x2_f32
from .gemm_utils import h2div_f16x2_f32
from .gemm_utils import load_pred_v2_b32
from .gemm_utils import load_pred_v4_b32
from .gemm_utils import quantize_f32x8_to_int4_word_signed
from .gemm_utils import reduce_add_f32
from .gemm_utils import rcp_approx_f32
from .gemm_utils import require_int4_runtime
from .gemm_utils import store_global_cg_v4_b32
from .mma import extract_f32_fragment_values
from .mma import make_f32_accumulator_fragment
from .mma import mma_f16_or_bf16_f32
from .mma import packed_half_fragment_from_i32_words

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


def _unpack_runtime_half_pair_bits(bits, element_type) -> tuple[cutlass.Float32, cutlass.Float32]:
  """Decode one packed runtime half or bf16 pair into float values."""

  if cutlass.const_expr(element_type == BFloat16):
    return bf16x2_bits_to_f32x2(bits)
  return f16x2_bits_to_f32x2(bits)


def _select_runtime_word(even_word: Int32, odd_word: Int32, lane_is_odd: Int32) -> Int32:
  """Select one packed 32-bit word for the current lane without Python branching."""

  return even_word + lane_is_odd * (odd_word - even_word)


def _load_runtime_quant_input_words(x_i32_ptr, row_idx, row_stride_words: int, group_word_base: int,
                                    lane_in_row) -> tuple[Int32, Int32, Int32, Int32]:
  """Load one 8-element activation pack as four contiguous half or bf16 pairs."""

  return load_pred_v4_b32(
    x_i32_ptr + row_idx * row_stride_words + group_word_base + lane_in_row * 4,
    1,
  )


def _load_runtime_quant_smooth_words(smooth_i32_ptr, block_word_base: int, group_in_block, lane_id,
                                     lane_in_row) -> tuple[Int32, Int32, Int32, Int32]:
  """Load the four smooth-factor half2 pairs needed by one quantization lane.

  The runtime smooth tensor already follows the packed CUDA layout. Within one
  8-lane row group, each 2-lane pair cooperatively issues two `v4.b32` global
  loads and then distributes the required packed half2 words with warp shuffles.
  """

  lane_pair = lane_in_row // 2
  lane_is_odd = Int32(lane_in_row % 2)
  even_lane_pred = Int32(1) - lane_is_odd
  row_group_base_lane = lane_id - lane_in_row
  source_lane = row_group_base_lane + lane_pair * 2
  group_word_offset = group_in_block * (_WARP_K // _PACKED_HALFS_PER_I32)
  pair_word_base = block_word_base + group_word_offset + lane_pair * 8

  low_0, low_1, low_2, low_3 = load_pred_v4_b32(smooth_i32_ptr + pair_word_base, even_lane_pred)
  high_0, high_1, high_2, high_3 = load_pred_v4_b32(smooth_i32_ptr + pair_word_base + 4,
                                                    even_lane_pred)

  low_0 = cute.arch.shuffle_sync(low_0, source_lane)
  low_1 = cute.arch.shuffle_sync(low_1, source_lane)
  low_2 = cute.arch.shuffle_sync(low_2, source_lane)
  low_3 = cute.arch.shuffle_sync(low_3, source_lane)
  high_0 = cute.arch.shuffle_sync(high_0, source_lane)
  high_1 = cute.arch.shuffle_sync(high_1, source_lane)
  high_2 = cute.arch.shuffle_sync(high_2, source_lane)
  high_3 = cute.arch.shuffle_sync(high_3, source_lane)

  return (
    _select_runtime_word(low_0, low_1, lane_is_odd),
    _select_runtime_word(low_2, low_3, lane_is_odd),
    _select_runtime_word(high_0, high_1, lane_is_odd),
    _select_runtime_word(high_2, high_3, lane_is_odd),
  )


def _load_runtime_mma_pair_words(x_i32_ptr, row_idx, row_stride_words: int, channel_word_base: int,
                                 lane_id, lane_in_group) -> tuple[Int32, Int32]:
  """Load one MMA row fragment via vectorized global loads plus warp shuffles.

  Each 4-lane row group cooperatively fetches two contiguous `v2.b32` segments
  from global memory. The even lane in each 2-lane pair performs the vector
  load, then both lanes consume the packed half pair they need via warp
  shuffles. This keeps the CUDA lane-to-fragment mapping unchanged while
  avoiding the previous scalar `x[row, col]` gathers.
  """

  pair_group = lane_in_group // 2
  lane_is_odd = Int32(lane_in_group % 2)
  even_lane_pred = Int32(1) - lane_is_odd
  row_group_base_lane = lane_id - lane_in_group
  source_lane = row_group_base_lane + pair_group * 2
  row_word_base = row_idx * row_stride_words + channel_word_base + pair_group * 2

  low_word_0, low_word_1 = load_pred_v2_b32(x_i32_ptr + row_word_base, even_lane_pred)
  high_word_0, high_word_1 = load_pred_v2_b32(x_i32_ptr + row_word_base + 4, even_lane_pred)

  low_word_0 = cute.arch.shuffle_sync(low_word_0, source_lane)
  low_word_1 = cute.arch.shuffle_sync(low_word_1, source_lane)
  high_word_0 = cute.arch.shuffle_sync(high_word_0, source_lane)
  high_word_1 = cute.arch.shuffle_sync(high_word_1, source_lane)

  low_word = low_word_0 + lane_is_odd * (low_word_1 - low_word_0)
  high_word = high_word_0 + lane_is_odd * (high_word_1 - high_word_0)
  return low_word, high_word


class _SVDQFusedQuantizeLoraProgram:
  """Single-launch CuTe DSL runtime kernel matching CUDA's fused tile flow.

  One CTA covers ``BLOCK_M x BLOCK_N``. Each warp owns one ``32 x 128``
  activation tile, writes the quantized runtime tensors, and accumulates the
  LoRA-down partial sums for the same tile before moving on.

  NOTE: This implementation is slower than the existing CUDA kernel, so it is not
  currently wired up to the public interface. The main goal of this scaffold is to
  stabilize the call path and provide a reference implementation for future optimizations.
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
    smooth_i32_ptr = recast_ptr(smooth.iterator, dtype=Int32)
    lora_down_i32_ptr = recast_ptr(lora_down.iterator, dtype=Int32)
    out_ptr = recast_ptr(out.iterator, dtype=out.element_type)
    x_row_stride_words = self.channels // _PACKED_HALFS_PER_I32
    smooth_block_stride_words = _BLOCK_N // _PACKED_HALFS_PER_I32

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

    warp_scale_smem = scale_smem[warp_id, None, None]
    warp_qpack_smem = qpack_smem[warp_id, None, None]

    for group_in_block in cutlass.range_constexpr(self.groups_per_block_n):
      group_idx = block_n_idx * self.groups_per_block_n + group_in_block
      group_col_base = channel_block_base + group_in_block * _WARP_K
      group_word_base = group_col_base // _PACKED_HALFS_PER_I32
      smooth_words = _load_runtime_quant_smooth_words(
        smooth_i32_ptr,
        block_n_idx * smooth_block_stride_words,
        group_in_block,
        lane_id,
        lane_in_row,
      )

      for row_batch in cutlass.range_constexpr(_ROW_BATCHES_PER_WARP):
        row_local = row_batch * _WARP_ROW_GROUPS + subgroup_id
        global_row = base_row + row_local
        input_words = _load_runtime_quant_input_words(
          x_i32_ptr,
          global_row,
          x_row_stride_words,
          group_word_base,
          lane_in_row,
        )

        local_values: list[cutlass.Float32] = []
        local_absmax = cutlass.Float32(0.0)
        for pair_idx in cutlass.range_constexpr(4):
          input_0, input_1 = _unpack_runtime_half_pair_bits(input_words[pair_idx], x.element_type)
          smooth_0, smooth_1 = _unpack_runtime_half_pair_bits(smooth_words[pair_idx],
                                                              smooth.element_type)

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

    # Match CUDA quantize_w4a4_warp: each warp only consumes its private smem
    # slice, so a warp-scope sync is sufficient here.
    cute.arch.sync_warp()

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
      for rank_tile_idx in cutlass.range_constexpr(self.rank_tiles):
        acc_top = make_f32_accumulator_fragment()
        acc_bottom = make_f32_accumulator_fragment()

        for channel_tile_idx in cutlass.range_constexpr(self.channel_tiles_per_block_n):
          channel_tile_global = block_n_idx * self.channel_tiles_per_block_n + channel_tile_idx
          channel_base = channel_tile_global * _MMA_TILE_K
          channel_word_base = channel_base // _PACKED_HALFS_PER_I32
          lane_group = lane_id // 4
          lane_in_group = lane_id % 4
          top_row_0 = base_row + lane_group
          top_row_1 = top_row_0 + 8
          bottom_row_0 = top_row_0 + _MMA_TILE_M
          bottom_row_1 = bottom_row_0 + 8

          a_top_0, a_top_2 = _load_runtime_mma_pair_words(x_i32_ptr, top_row_0, x_row_stride_words,
                                                          channel_word_base, lane_id, lane_in_group)
          a_top_1, a_top_3 = _load_runtime_mma_pair_words(x_i32_ptr, top_row_1, x_row_stride_words,
                                                          channel_word_base, lane_id, lane_in_group)
          a_bottom_0, a_bottom_2 = _load_runtime_mma_pair_words(x_i32_ptr, bottom_row_0,
                                                                x_row_stride_words,
                                                                channel_word_base, lane_id,
                                                                lane_in_group)
          a_bottom_1, a_bottom_3 = _load_runtime_mma_pair_words(x_i32_ptr, bottom_row_1,
                                                                x_row_stride_words,
                                                                channel_word_base, lane_id,
                                                                lane_in_group)

          a_top_frag = packed_half_fragment_from_i32_words(a_top_0, a_top_1, a_top_2, a_top_3,
                                                           x.element_type)
          a_bottom_frag = packed_half_fragment_from_i32_words(a_bottom_0, a_bottom_1, a_bottom_2,
                                                              a_bottom_3, x.element_type)

          lora_frag_base_i32 = ((
            (channel_tile_global * self.rank_tiles + rank_tile_idx) * _WARP_THREADS) +
                                lane_id) * _I32_WORDS_PER_VEC
          b0, b1, b2, b3 = load_pred_v4_b32(
            lora_down_i32_ptr + lora_frag_base_i32,
            1,
          )
          b_frag = packed_half_fragment_from_i32_words(b0, b1, b2, b3, lora_down.element_type)

          acc_top = mma_f16_or_bf16_f32(a_top_frag, b_frag, acc_top)
          acc_bottom = mma_f16_or_bf16_f32(a_bottom_frag, b_frag, acc_bottom)

        acc_top_vals = extract_f32_fragment_values(acc_top)
        acc_bottom_vals = extract_f32_fragment_values(acc_bottom)

        for m_tile in cutlass.range_constexpr(2):
          acc_vals = acc_top_vals if m_tile == 0 else acc_bottom_vals
          linear_tile_base = ((
            (block_m_idx * self.rank_tiles + rank_tile_idx) * self.warps_per_block_m + warp_id) * 2
                              + m_tile) * 256
          for frag_idx in cutlass.range_constexpr(8):
            reduce_add_f32(out_ptr + linear_tile_base + frag_idx * _WARP_THREADS + lane_id,
                           acc_vals[frag_idx])

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
