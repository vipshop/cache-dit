from __future__ import annotations

import operator
import os
from typing import Any
from typing import Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass._mlir.dialects import vector
from cutlass.cute import make_layout
from cutlass.cute import make_tensor
from cutlass.cute import recast_ptr
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.tensor import TensorSSA
# Ada/sm89 only exposes e4m3x2 <-> f16x2 PTX conversion instructions here.
# The bf16x2 variants are PTX 9.1 / sm100+ features, so the internal bridge stays Float16.
from cutlass.cute.typing import BFloat16
from cutlass.cute.typing import Float16
from cutlass.cute.typing import Int32
from cutlass.cute.typing import Uint8
from cutlass.cutlass_dsl import T
from cutlass.cutlass_dsl import dsl_user_op

_COPY_BITS = 128
_CTA_THREADS = 128
_WARP_SIZE = 32
_FP8_MAX = 448.0
_PER_TOKEN_EPS = 1e-4
_QKV_EPS = 1e-6

_COMPILED_PER_TOKEN_QUANT: dict[tuple[int, int, int, int, int, str], Callable[..., None]] = {}
_COMPILED_PER_TOKEN_DEQUANT: dict[tuple[int, int, int, int, int, str], Callable[..., None]] = {}
_COMPILED_QKV_QUANT: dict[tuple[int, int, int, int, int, int, str], Callable[..., None]] = {}
_COMPILED_QKV_DEQUANT: dict[tuple[int, int, int, int, int, int, str], Callable[..., None]] = {}


def _detect_cutedsl_arch() -> str:
  """Infer the CuTe DSL target architecture from the active CUDA device.

  :returns: SM architecture string such as `sm_89`.
  """

  major, minor = torch.cuda.get_device_capability()
  suffix = "a" if major >= 9 else ""
  return f"sm_{major}{minor}{suffix}"


if torch.cuda.is_available():
  os.environ.setdefault("CUTE_DSL_ARCH", _detect_cutedsl_arch())


def _derive_vector_config(head_size: int) -> tuple[int, int]:
  """Derive the fixed-width vector layout for FP8 communication rows.

  :param head_size: Attention head dimension `D`.
  :returns: Tuple `(elems_per_thread, row_threads)`.
  """

  elems_per_thread = _COPY_BITS // 8
  assert head_size % elems_per_thread == 0, (
    "The CuTe DSL FP8 communication path requires D to be divisible by 16, "
    f"but got D={head_size}.")

  row_threads = head_size // elems_per_thread
  assert row_threads <= _WARP_SIZE, (
    f"Expected row_threads <= {_WARP_SIZE}, but got {row_threads} for D={head_size}.")
  assert _CTA_THREADS % row_threads == 0, (
    f"Expected row_threads={row_threads} to divide CTA size {_CTA_THREADS} exactly.")
  return elems_per_thread, row_threads


def _derive_qkv_cta_threads(row_threads: int, rows_per_slice: int) -> int:
  """Choose a CTA size for QKV kernels based on active rows in one `(p, s)` slice."""

  active_threads = max(row_threads, row_threads * rows_per_slice)
  cta_threads = ((active_threads + _WARP_SIZE - 1) // _WARP_SIZE) * _WARP_SIZE
  cta_threads = min(_CTA_THREADS, max(_WARP_SIZE, cta_threads))
  assert cta_threads % row_threads == 0, (
    f"Expected qkv CTA threads {cta_threads} to divide row_threads={row_threads}.")
  return cta_threads


def _require_fp8_comm_arch(device: torch.device) -> None:
  """Validate that the active device supports FP8 PTX conversion instructions.

  The inline PTX path used by these kernels relies on Ada-or-newer FP8 conversion instructions.

  :param device: CUDA device hosting the tensors.
  :raises RuntimeError: If the active CUDA device is older than SM89.
  """

  major, minor = torch.cuda.get_device_capability(device)
  if (major, minor) < (8, 9):
    raise RuntimeError("CuTe DSL FP8 communication kernels require Ada-class GPUs (sm89+) for the "
                       "inline PTX FP8 conversion path.")


def _wrap_tensor(tensor: torch.Tensor) -> Any:
  """Convert a torch tensor into a CuTe runtime tensor.

  :param tensor: Torch tensor to expose to CuTe DSL.
  :returns: CuTe runtime tensor.
  """

  return from_dlpack(
    tensor,
    assumed_align=16,
    enable_tvm_ffi=True,
  )


@dsl_user_op
def _cvt_fp8x16_to_f16x16(src: TensorSSA, *, loc=None, ip=None) -> TensorSSA:
  """Convert 16 packed FP8 E4M3FN bytes into 16 float16 values via inline PTX.

  The public FP8 communication APIs are BF16-facing, but the Ada/sm89 inline PTX
  path only exposes `cvt.rn.f16x2.e4m3x2`. The BF16-native `e4m3x2 <-> bf16x2`
  instructions are PTX 9.1 / sm100-family features and are not available on the
  current target, so this helper must use `Float16` as the internal conversion type.
  """

  src_i16x8 = vector.bitcast(T.vector(8, T.i16()), src.maybe_downcast())
  pair_bits = [
    vector.extract(
      src_i16x8,
      dynamic_position=[],
      static_position=[idx],
      loc=loc,
      ip=ip,
    ) for idx in range(8)
  ]
  pair_f16x2_struct = llvm.inline_asm(
    llvm.StructType.get_literal([T.i32()] * 8),
    pair_bits,
    """{
      cvt.rn.f16x2.e4m3x2 $0, $8;
      cvt.rn.f16x2.e4m3x2 $1, $9;
      cvt.rn.f16x2.e4m3x2 $2, $10;
      cvt.rn.f16x2.e4m3x2 $3, $11;
      cvt.rn.f16x2.e4m3x2 $4, $12;
      cvt.rn.f16x2.e4m3x2 $5, $13;
      cvt.rn.f16x2.e4m3x2 $6, $14;
      cvt.rn.f16x2.e4m3x2 $7, $15;
    }""",
    "=r,=r,=r,=r,=r,=r,=r,=r,h,h,h,h,h,h,h,h",
    has_side_effects=False,
    is_align_stack=False,
    asm_dialect=llvm.AsmDialect.AD_ATT,
  )
  pair_f16x2 = [llvm.extractvalue(T.i32(), pair_f16x2_struct, [idx]) for idx in range(8)]

  vec_i32x8 = vector.from_elements(T.vector(8, T.i32()), pair_f16x2, loc=loc, ip=ip)
  vec_f16x16 = llvm.bitcast(T.vector(16, T.f16()), vec_i32x8, loc=loc, ip=ip)
  return TensorSSA(vec_f16x16, src.shape, Float16)


@dsl_user_op
def _cvt_f16x16_to_fp8x16(src: TensorSSA, *, loc=None, ip=None) -> TensorSSA:
  """Convert 16 float16 values into 16 packed FP8 E4M3FN bytes via inline PTX.

  This stays on `Float16` intentionally: Ada/sm89 supports
  `cvt.rn.satfinite.e4m3x2.f16x2`, but not the PTX 9.1 `bf16x2` variant.
  """

  src_i32x8 = vector.bitcast(T.vector(8, T.i32()), src.maybe_downcast())
  pair_bits = [
    vector.extract(
      src_i32x8,
      dynamic_position=[],
      static_position=[idx],
      loc=loc,
      ip=ip,
    ) for idx in range(8)
  ]
  pair_fp8x2_struct = llvm.inline_asm(
    llvm.StructType.get_literal([T.i16()] * 8),
    pair_bits,
    """{
      cvt.rn.satfinite.e4m3x2.f16x2 $0, $8;
      cvt.rn.satfinite.e4m3x2.f16x2 $1, $9;
      cvt.rn.satfinite.e4m3x2.f16x2 $2, $10;
      cvt.rn.satfinite.e4m3x2.f16x2 $3, $11;
      cvt.rn.satfinite.e4m3x2.f16x2 $4, $12;
      cvt.rn.satfinite.e4m3x2.f16x2 $5, $13;
      cvt.rn.satfinite.e4m3x2.f16x2 $6, $14;
      cvt.rn.satfinite.e4m3x2.f16x2 $7, $15;
    }""",
    "=h,=h,=h,=h,=h,=h,=h,=h,r,r,r,r,r,r,r,r",
    has_side_effects=False,
    is_align_stack=False,
    asm_dialect=llvm.AsmDialect.AD_ATT,
  )
  pair_fp8x2 = [llvm.extractvalue(T.i16(), pair_fp8x2_struct, [idx]) for idx in range(8)]

  vec_i16x8 = vector.from_elements(T.vector(8, T.i16()), pair_fp8x2, loc=loc, ip=ip)
  vec_i8x16 = llvm.bitcast(T.vector(16, T.i8()), vec_i16x8, loc=loc, ip=ip)
  return TensorSSA(vec_i8x16, src.shape, Uint8)


@dsl_user_op
def _bitcast_fp8x16_to_i32x4(src: TensorSSA, *, loc=None, ip=None) -> TensorSSA:
  """Bitcast 16 FP8 bytes into 4 packed int32 words."""

  vec_i32x4 = llvm.bitcast(T.vector(4, T.i32()), src.maybe_downcast(), loc=loc, ip=ip)
  return TensorSSA(vec_i32x4, (4, ), Int32)


@dsl_user_op
def _bitcast_i32x4_to_fp8x16(src: TensorSSA, *, loc=None, ip=None) -> TensorSSA:
  """Bitcast 4 packed int32 words into 16 FP8 bytes."""

  vec_i8x16 = llvm.bitcast(T.vector(16, T.i8()), src.maybe_downcast(), loc=loc, ip=ip)
  return TensorSSA(vec_i8x16, (16, ), Uint8)


def _copy_partitions(src: cute.Tensor, dst: cute.Tensor) -> None:
  """Copy all vector partitions from a tensor slice into a matching fragment."""

  for idx in range(cute.size(dst, mode=[1])):
    cute.autovec_copy(src[None, idx], dst[None, idx])


class _PerTokenQuantProgram:
  """CuTe DSL row-wise per-token FP8 quantization launcher.

  :param head_size: Attention head dimension `D`.
  """

  def __init__(self, head_size: int) -> None:
    self.elems_per_thread, self.row_threads = _derive_vector_config(head_size)
    self.rows_per_cta = _CTA_THREADS // self.row_threads
    self.eps = cutlass.Float32(_PER_TOKEN_EPS)
    self.fp8_max = cutlass.Float32(_FP8_MAX)

  @cute.kernel
  def _kernel(
    self,
    out: cute.Tensor,
    x: cute.Tensor,
    tiled_copy_load: cute.TiledCopy,
    tiled_copy_store: cute.TiledCopy,
  ):
    tidx, _, _ = cute.arch.thread_idx()
    block_row_idx, _, _ = cute.arch.block_idx()
    row_in_block = tidx // self.row_threads
    lane_in_row = tidx % self.row_threads
    row_idx = block_row_idx * self.rows_per_cta + row_in_block

    q_out = make_tensor(
      recast_ptr(out.iterator, dtype=Uint8),
      make_layout(
        (out.shape[0], self.elems_per_thread * self.row_threads),
        stride=(out.stride[0], 1),
      ),
    )
    s_out = make_tensor(
      recast_ptr(out.iterator, dtype=BFloat16) + self.elems_per_thread * self.row_threads // 2,
      make_layout(
        (out.shape[0], ),
        stride=(out.stride[0] // 2, ),
      ),
    )

    if row_idx < q_out.shape[0]:
      thr_copy_load = tiled_copy_load.get_slice(lane_in_row)
      thr_copy_store = tiled_copy_store.get_slice(lane_in_row)

      g_x = x[row_idx, None]
      g_q = q_out[row_idx, None]

      t_x_g = thr_copy_load.partition_S(g_x)
      t_q_g = thr_copy_store.partition_D(g_q)

      t_x_r = cute.make_fragment_like(t_x_g)
      t_x_r.fill(0)
      _copy_partitions(t_x_g, t_x_r)

      x_values = t_x_r.load().reshape((self.elems_per_thread, )).to(cutlass.Float32)
      neg_values = x_values.apply_op(operator.mul, cutlass.Float32(-1.0))
      local_max = x_values.reduce(cute.ReductionOp.MAX, self.eps, 0)
      local_neg_max = neg_values.reduce(cute.ReductionOp.MAX, self.eps, 0)
      local_absmax = cute.arch.fmax(local_max, local_neg_max)
      row_absmax = cute.arch.warp_reduction(
        local_absmax,
        cute.arch.fmax,
        threads_in_group=self.row_threads,
      )

      scale = cute.arch.fmax(row_absmax / self.fp8_max, self.eps)
      if lane_in_row == 0:
        s_out[row_idx] = scale.to(s_out.element_type)

      q_values = _cvt_f16x16_to_fp8x16((x_values / scale).to(Float16))
      t_q_g.store(q_values.reshape(t_q_g.shape))

  @cute.jit
  def __call__(
    self,
    out: cute.Tensor,
    x: cute.Tensor,
  ) -> None:
    copy_atom_load = cute.make_copy_atom(
      cute.nvgpu.CopyUniversalOp(),
      x.element_type,
      num_bits_per_copy=_COPY_BITS,
    )
    copy_atom_store = cute.make_copy_atom(
      cute.nvgpu.CopyUniversalOp(),
      Uint8,
      num_bits_per_copy=_COPY_BITS,
    )
    thread_layout = cute.make_layout(self.row_threads)
    value_layout = cute.make_layout(self.elems_per_thread)
    tiled_copy_load = cute.make_tiled_copy_tv(copy_atom_load, thread_layout, value_layout)
    tiled_copy_store = cute.make_tiled_copy_tv(copy_atom_store, thread_layout, value_layout)

    self._kernel(
      out,
      x,
      tiled_copy_load,
      tiled_copy_store,
    ).launch(
      grid=[(out.shape[0] + self.rows_per_cta - 1) // self.rows_per_cta, 1, 1],
      block=[_CTA_THREADS, 1, 1],
    )


class _PerTokenDequantProgram:
  """CuTe DSL row-wise per-token FP8 dequantization launcher.

  :param head_size: Attention head dimension `D`.
  """

  def __init__(self, head_size: int) -> None:
    self.elems_per_thread, self.row_threads = _derive_vector_config(head_size)
    self.rows_per_cta = _CTA_THREADS // self.row_threads

  @cute.kernel
  def _kernel(
    self,
    out: cute.Tensor,
    x: cute.Tensor,
    tiled_copy_load: cute.TiledCopy,
    tiled_copy_store: cute.TiledCopy,
  ):
    tidx, _, _ = cute.arch.thread_idx()
    block_row_idx, _, _ = cute.arch.block_idx()
    row_in_block = tidx // self.row_threads
    lane_in_row = tidx % self.row_threads
    row_idx = block_row_idx * self.rows_per_cta + row_in_block

    q = make_tensor(
      recast_ptr(x.iterator, dtype=Uint8),
      make_layout(
        (x.shape[0], self.elems_per_thread * self.row_threads),
        stride=(x.stride[0], 1),
      ),
    )
    scale = make_tensor(
      recast_ptr(x.iterator, dtype=BFloat16) + self.elems_per_thread * self.row_threads // 2,
      make_layout(
        (x.shape[0], ),
        stride=(x.stride[0] // 2, ),
      ),
    )

    if row_idx < out.shape[0]:
      thr_copy_load = tiled_copy_load.get_slice(lane_in_row)
      thr_copy_store = tiled_copy_store.get_slice(lane_in_row)

      g_q = q[row_idx, None]
      g_out = out[row_idx, None]

      t_q_g = thr_copy_load.partition_S(g_q)
      t_out_g = thr_copy_store.partition_D(g_out)

      t_q_r = cute.make_fragment_like(t_q_g)
      t_q_r.fill(0)
      _copy_partitions(t_q_g, t_q_r)

      scale_value = cutlass.Float32(0.0)
      if lane_in_row == 0:
        scale_value = cutlass.Float32(scale[row_idx])
      warp_lane = cute.arch.lane_idx()
      scale_value = cute.arch.shuffle_sync(scale_value, warp_lane - lane_in_row)

      q_values = _cvt_fp8x16_to_f16x16(t_q_r.load().reshape((self.elems_per_thread, )))
      out_values = q_values.to(cutlass.Float32) * scale_value

      t_out_g.store(out_values.to(t_out_g.element_type).reshape(t_out_g.shape))

  @cute.jit
  def __call__(
    self,
    out: cute.Tensor,
    x: cute.Tensor,
  ) -> None:
    copy_atom_load = cute.make_copy_atom(
      cute.nvgpu.CopyUniversalOp(),
      Uint8,
      num_bits_per_copy=_COPY_BITS,
    )
    copy_atom_store = cute.make_copy_atom(
      cute.nvgpu.CopyUniversalOp(),
      out.element_type,
      num_bits_per_copy=_COPY_BITS,
    )
    thread_layout = cute.make_layout(self.row_threads)
    value_layout = cute.make_layout(self.elems_per_thread)
    tiled_copy_load = cute.make_tiled_copy_tv(copy_atom_load, thread_layout, value_layout)
    tiled_copy_store = cute.make_tiled_copy_tv(copy_atom_store, thread_layout, value_layout)

    self._kernel(
      out,
      x,
      tiled_copy_load,
      tiled_copy_store,
    ).launch(
      grid=[(out.shape[0] + self.rows_per_cta - 1) // self.rows_per_cta, 1, 1],
      block=[_CTA_THREADS, 1, 1],
    )


class _QkvQuantProgram:
  """CuTe DSL QKV permute + FP8 quantization launcher.

  :param head_size: Attention head dimension `D`.
  """

  def __init__(self, head_size: int, cta_threads: int) -> None:
    self.elems_per_thread, self.row_threads = _derive_vector_config(head_size)
    self.fp8_words_per_thread = self.elems_per_thread // 4
    self.payload_words = head_size // 4
    self.cta_threads = cta_threads
    self.rows_per_cta = self.cta_threads // self.row_threads
    self.eps = cutlass.Float32(_QKV_EPS)
    self.fp8_max = cutlass.Float32(_FP8_MAX)

  @cute.kernel
  def _kernel(
    self,
    out: cute.Tensor,
    x: cute.Tensor,
    tiled_copy_load: cute.TiledCopy,
    tiled_copy_store: cute.TiledCopy,
  ):
    tidx, _, _ = cute.arch.thread_idx()
    block_row_idx, block_seq_idx, block_part_batch_idx = cute.arch.block_idx()
    row_in_block = tidx // self.row_threads
    lane_in_row = tidx % self.row_threads
    row_idx = block_row_idx * self.rows_per_cta + row_in_block

    q_out = make_tensor(
      recast_ptr(out.iterator, dtype=Int32),
      make_layout(
        (out.shape[0], out.shape[1], out.shape[2], out.shape[3], self.payload_words),
        stride=(
          out.stride[0] // 4,
          out.stride[1] // 4,
          out.stride[2] // 4,
          out.stride[3] // 4,
          1,
        ),
      ),
    )
    s_out = make_tensor(
      recast_ptr(out.iterator, dtype=cutlass.Float32) + self.payload_words,
      make_layout(
        (out.shape[0], out.shape[1], out.shape[2], out.shape[3]),
        stride=(
          out.stride[0] // 4,
          out.stride[1] // 4,
          out.stride[2] // 4,
          out.stride[3] // 4,
        ),
      ),
    )

    if row_idx < q_out.shape[3]:
      n_id = row_idx
      b_id = block_part_batch_idx % q_out.shape[2]
      s_id = block_seq_idx
      p_id = block_part_batch_idx // q_out.shape[2]

      thr_copy_load = tiled_copy_load.get_slice(lane_in_row)
      thr_copy_store = tiled_copy_store.get_slice(lane_in_row)

      g_x = x[p_id, s_id, b_id, n_id, None]
      g_q = q_out[p_id, s_id, b_id, n_id, None]

      t_x_g = thr_copy_load.partition_S(g_x)
      t_q_g = thr_copy_store.partition_D(g_q)

      t_x_r = cute.make_fragment_like(t_x_g)
      t_x_r.fill(0)
      _copy_partitions(t_x_g, t_x_r)

      x_values = t_x_r.load().reshape((self.elems_per_thread, )).to(cutlass.Float32)
      neg_values = x_values.apply_op(operator.mul, cutlass.Float32(-1.0))
      local_max = x_values.reduce(cute.ReductionOp.MAX, self.eps, 0)
      local_neg_max = neg_values.reduce(cute.ReductionOp.MAX, self.eps, 0)
      local_absmax = cute.arch.fmax(local_max, local_neg_max)
      row_absmax = cute.arch.warp_reduction(
        local_absmax,
        cute.arch.fmax,
        threads_in_group=self.row_threads,
      )
      scale = cute.arch.fmax(row_absmax / self.fp8_max, self.eps)

      if lane_in_row == 0:
        s_out[p_id, s_id, b_id, n_id] = scale

      # Materialize the reciprocal once so the vector scale path can reuse it
      # across the 16 payload values instead of lowering each divide separately.
      inv_scale = cutlass.Float32(1.0) / scale
      scaled_values = x_values.apply_op(operator.mul, inv_scale)
      q_values = _bitcast_fp8x16_to_i32x4(_cvt_f16x16_to_fp8x16(scaled_values.to(Float16)))
      # The public QKV layout uses a D+4 row stride, so many rows are not 16-byte aligned.
      # Keep the safe tiled store path instead of forcing a single 128-bit global store.
      t_q_g.store(q_values.reshape(t_q_g.shape))

  @cute.jit
  def __call__(
    self,
    out: cute.Tensor,
    x: cute.Tensor,
  ) -> None:
    copy_atom_load = cute.make_copy_atom(
      cute.nvgpu.CopyUniversalOp(),
      x.element_type,
      num_bits_per_copy=_COPY_BITS,
    )
    copy_atom_store = cute.make_copy_atom(
      cute.nvgpu.CopyUniversalOp(),
      Int32,
      num_bits_per_copy=_COPY_BITS,
    )
    thread_layout = cute.make_layout(self.row_threads)
    x_value_layout = cute.make_layout(self.elems_per_thread)
    q_value_layout = cute.make_layout(self.fp8_words_per_thread)
    tiled_copy_load = cute.make_tiled_copy_tv(copy_atom_load, thread_layout, x_value_layout)
    tiled_copy_store = cute.make_tiled_copy_tv(copy_atom_store, thread_layout, q_value_layout)

    self._kernel(
      out,
      x,
      tiled_copy_load,
      tiled_copy_store,
    ).launch(
      grid=[
        (out.shape[3] + self.rows_per_cta - 1) // self.rows_per_cta,
        out.shape[1],
        out.shape[0] * out.shape[2],
      ],
      block=[self.cta_threads, 1, 1],
    )


class _QkvDequantProgram:
  """CuTe DSL QKV FP8 dequantization + output permutation launcher.

  :param head_size: Attention head dimension `D`.
  """

  def __init__(self, head_size: int, cta_threads: int) -> None:
    self.elems_per_thread, self.row_threads = _derive_vector_config(head_size)
    self.head_size = head_size
    self.cta_threads = cta_threads
    self.rows_per_cta = self.cta_threads // self.row_threads

  @cute.kernel
  def _kernel(
    self,
    out: cute.Tensor,
    quant_x: cute.Tensor,
    tiled_copy_load: cute.TiledCopy,
    tiled_copy_store: cute.TiledCopy,
  ):
    tidx, _, _ = cute.arch.thread_idx()
    block_row_idx, block_seq_idx, _ = cute.arch.block_idx()
    row_in_block = tidx // self.row_threads
    lane_in_row = tidx % self.row_threads
    row_idx = block_row_idx * self.rows_per_cta + row_in_block

    q = make_tensor(
      recast_ptr(quant_x.iterator, dtype=Uint8),
      make_layout(
        (quant_x.shape[0], quant_x.shape[1], quant_x.shape[2], self.head_size),
        stride=(
          quant_x.stride[0],
          quant_x.stride[1],
          quant_x.stride[2],
          1,
        ),
      ),
    )
    scale = make_tensor(
      recast_ptr(quant_x.iterator, dtype=cutlass.Float32) + self.head_size // 4,
      make_layout(
        (quant_x.shape[0], quant_x.shape[1], quant_x.shape[2]),
        stride=(
          quant_x.stride[0] // 4,
          quant_x.stride[1] // 4,
          quant_x.stride[2] // 4,
        ),
      ),
    )

    total_rows = out.shape[1] * out.shape[2]
    if row_idx < total_rows:
      n_id = row_idx % out.shape[2]
      b_id = row_idx // out.shape[2]
      s_id = block_seq_idx

      thr_copy_load = tiled_copy_load.get_slice(lane_in_row)
      thr_copy_store = tiled_copy_store.get_slice(lane_in_row)

      g_q = q[s_id, b_id, n_id, None]
      g_out = out[s_id, b_id, n_id, None]

      t_q_g = thr_copy_load.partition_S(g_q)
      t_out_g = thr_copy_store.partition_D(g_out)

      t_q_r = cute.make_fragment_like(t_q_g)
      t_q_r.fill(0)
      _copy_partitions(t_q_g, t_q_r)

      scale_value = cutlass.Float32(0.0)
      if lane_in_row == 0:
        scale_value = scale[s_id, b_id, n_id]
      warp_lane = cute.arch.lane_idx()
      scale_value = cute.arch.shuffle_sync(scale_value, warp_lane - lane_in_row)

      q_values = _cvt_fp8x16_to_f16x16(t_q_r.load().reshape((self.elems_per_thread, )))
      out_values = q_values.to(cutlass.Float32) * scale_value

      t_out_g.store(out_values.to(t_out_g.element_type).reshape(t_out_g.shape))

  @cute.jit
  def __call__(
    self,
    out: cute.Tensor,
    quant_x: cute.Tensor,
  ) -> None:
    copy_atom_load = cute.make_copy_atom(
      cute.nvgpu.CopyUniversalOp(),
      Uint8,
      num_bits_per_copy=_COPY_BITS,
    )
    copy_atom_store = cute.make_copy_atom(
      cute.nvgpu.CopyUniversalOp(),
      out.element_type,
      num_bits_per_copy=_COPY_BITS,
    )
    thread_layout = cute.make_layout(self.row_threads)
    q_value_layout = cute.make_layout(self.elems_per_thread)
    out_value_layout = cute.make_layout(self.elems_per_thread)
    tiled_copy_load = cute.make_tiled_copy_tv(copy_atom_load, thread_layout, q_value_layout)
    tiled_copy_store = cute.make_tiled_copy_tv(copy_atom_store, thread_layout, out_value_layout)

    self._kernel(
      out,
      quant_x,
      tiled_copy_load,
      tiled_copy_store,
    ).launch(
      grid=[
        (out.shape[1] * out.shape[2] + self.rows_per_cta - 1) // self.rows_per_cta,
        out.shape[0],
        1,
      ],
      block=[self.cta_threads, 1, 1],
    )


def _compile_per_token_quant(
  y_rows: torch.Tensor,
  x_rows: torch.Tensor,
) -> Callable[..., None]:
  """Compile and cache the CuTe DSL per-token quantization launcher."""

  arch = os.environ.get("CUTE_DSL_ARCH", _detect_cutedsl_arch())
  cache_key = (
    y_rows.shape[0],
    y_rows.shape[1],
    y_rows.stride(0),
    x_rows.stride(0),
    arch,
  )
  compiled = _COMPILED_PER_TOKEN_QUANT.get(cache_key)
  if compiled is not None:
    return compiled

  launcher = _PerTokenQuantProgram(head_size=y_rows.shape[1] - 2)
  launcher._kernel.set_name_prefix("cache_dit_cutedsl_fp8_comm_per_token_quant")
  compiled = cute.compile(
    launcher,
    _wrap_tensor(y_rows.view(torch.uint8)),
    _wrap_tensor(x_rows),
    options="--enable-tvm-ffi",
  )
  _COMPILED_PER_TOKEN_QUANT[cache_key] = compiled
  return compiled


def _compile_per_token_dequant(
  out_rows: torch.Tensor,
  x_rows: torch.Tensor,
) -> Callable[..., None]:
  """Compile and cache the CuTe DSL per-token dequantization launcher."""

  arch = os.environ.get("CUTE_DSL_ARCH", _detect_cutedsl_arch())
  cache_key = (
    out_rows.shape[0],
    out_rows.shape[1],
    out_rows.stride(0),
    x_rows.stride(0),
    arch,
  )
  compiled = _COMPILED_PER_TOKEN_DEQUANT.get(cache_key)
  if compiled is not None:
    return compiled

  launcher = _PerTokenDequantProgram(head_size=out_rows.shape[1])
  launcher._kernel.set_name_prefix("cache_dit_cutedsl_fp8_comm_per_token_dequant")
  compiled = cute.compile(
    launcher,
    _wrap_tensor(out_rows),
    _wrap_tensor(x_rows.view(torch.uint8)),
    options="--enable-tvm-ffi",
  )
  _COMPILED_PER_TOKEN_DEQUANT[cache_key] = compiled
  return compiled


def _compile_qkv_quant(
  quant_x: torch.Tensor,
  x_perm: torch.Tensor,
) -> Callable[..., None]:
  """Compile and cache the CuTe DSL QKV permute + quant launcher."""

  arch = os.environ.get("CUTE_DSL_ARCH", _detect_cutedsl_arch())
  cache_key = (
    quant_x.shape[0],
    quant_x.shape[1],
    quant_x.shape[2],
    quant_x.shape[3],
    quant_x.shape[4],
    _derive_qkv_cta_threads(_derive_vector_config(quant_x.shape[4] - 4)[1], quant_x.shape[3]),
    arch,
  )
  compiled = _COMPILED_QKV_QUANT.get(cache_key)
  if compiled is not None:
    return compiled

  cta_threads = _derive_qkv_cta_threads(
    _derive_vector_config(quant_x.shape[4] - 4)[1],
    quant_x.shape[3],
  )
  launcher = _QkvQuantProgram(head_size=quant_x.shape[4] - 4, cta_threads=cta_threads)
  launcher._kernel.set_name_prefix("cache_dit_cutedsl_fp8_comm_qkv_permute_quant")
  compiled = cute.compile(
    launcher,
    _wrap_tensor(quant_x.view(torch.uint8)),
    _wrap_tensor(x_perm),
    options="--enable-tvm-ffi",
  )
  _COMPILED_QKV_QUANT[cache_key] = compiled
  return compiled


def _compile_qkv_dequant(
  out_perm: torch.Tensor,
  quant_x: torch.Tensor,
) -> Callable[..., None]:
  """Compile and cache the CuTe DSL QKV dequant + permute launcher."""

  arch = os.environ.get("CUTE_DSL_ARCH", _detect_cutedsl_arch())
  cache_key = (
    out_perm.shape[0],
    out_perm.shape[1],
    out_perm.shape[2],
    out_perm.shape[3],
    quant_x.shape[3],
    _derive_qkv_cta_threads(
      _derive_vector_config(out_perm.shape[3])[1], out_perm.shape[1] * out_perm.shape[2]),
    arch,
  )
  compiled = _COMPILED_QKV_DEQUANT.get(cache_key)
  if compiled is not None:
    return compiled

  cta_threads = _derive_qkv_cta_threads(
    _derive_vector_config(out_perm.shape[3])[1],
    out_perm.shape[1] * out_perm.shape[2],
  )
  launcher = _QkvDequantProgram(head_size=out_perm.shape[3], cta_threads=cta_threads)
  launcher._kernel.set_name_prefix("cache_dit_cutedsl_fp8_comm_qkv_permute_dequant")
  compiled = cute.compile(
    launcher,
    _wrap_tensor(out_perm),
    _wrap_tensor(quant_x.view(torch.uint8)),
    options="--enable-tvm-ffi",
  )
  _COMPILED_QKV_DEQUANT[cache_key] = compiled
  return compiled


def fp8_comm_per_token_quant(x: torch.Tensor) -> torch.Tensor:
  """Quantize a tensor into the FP8 communication layout with CuTe DSL.

  :param x: Input tensor `[..., D]` in BF16.
  :returns: Quantized tensor `[..., D + 2]` in the mixed FP8+BF16 storage layout.
  """

  if x.device.type != "cuda":
    raise ValueError("CuTe DSL fp8_comm_per_token_quant only supports CUDA tensors.")
  _require_fp8_comm_arch(x.device)
  assert x.dtype == torch.bfloat16, f"expected bfloat16 but got {x.dtype}"

  *shape, head_size = x.shape
  _derive_vector_config(head_size)

  x_rows = x.reshape(-1, head_size).contiguous()
  y_rows = torch.empty((x_rows.shape[0], head_size + 2), dtype=torch.float8_e4m3fn, device=x.device)

  compiled = _compile_per_token_quant(
    y_rows=y_rows,
    x_rows=x_rows,
  )
  compiled(
    y_rows.view(torch.uint8),
    x_rows,
  )
  return y_rows.reshape(*shape, head_size + 2)


def fp8_comm_per_token_dequant(x: torch.Tensor) -> torch.Tensor:
  """Dequantize a per-token FP8 communication tensor with CuTe DSL.

  :param x: Input tensor `[..., D + 2]` in the mixed FP8+BF16 layout.
  :returns: Dequantized BF16 tensor `[..., D]`.
  """

  if x.device.type != "cuda":
    raise ValueError("CuTe DSL fp8_comm_per_token_dequant only supports CUDA tensors.")
  _require_fp8_comm_arch(x.device)
  assert x.dtype == torch.float8_e4m3fn, f"expected float8_e4m3fn but got {x.dtype}"

  *shape, packed_head_size = x.shape
  head_size = packed_head_size - 2
  _derive_vector_config(head_size)

  x_rows = x.reshape(-1, packed_head_size).contiguous()
  out_rows = torch.empty((x_rows.shape[0], head_size), dtype=torch.bfloat16, device=x.device)

  compiled = _compile_per_token_dequant(
    out_rows=out_rows,
    x_rows=x_rows,
  )
  compiled(
    out_rows,
    x_rows.view(torch.uint8),
  )
  return out_rows.reshape(*shape, head_size)


def fp8_comm_qkv_permute_quant(x: torch.Tensor) -> torch.Tensor:
  """Permute QKV heads and quantize them into the FP8 communication layout.

  :param x: Input tensor `[B, S, P, N, D]` in BF16.
  :returns: Quantized tensor `[P, S, B, N, D + 4]`.
  """

  if x.device.type != "cuda":
    raise ValueError("CuTe DSL fp8_comm_qkv_permute_quant only supports CUDA tensors.")
  _require_fp8_comm_arch(x.device)
  assert x.dtype == torch.bfloat16, f"expected bfloat16 but got {x.dtype}"

  batch, seq_len, partitions, num_heads, head_size = x.shape
  _derive_vector_config(head_size)

  quant_x = torch.empty(
    (partitions, seq_len, batch, num_heads, head_size + 4),
    dtype=torch.float8_e4m3fn,
    device=x.device,
  )
  x_perm = x.permute(2, 1, 0, 3, 4)

  compiled = _compile_qkv_quant(
    quant_x=quant_x,
    x_perm=x_perm,
  )
  compiled(
    quant_x.view(torch.uint8),
    x_perm,
  )
  return quant_x


def fp8_comm_qkv_permute_dequant(quant_x: torch.Tensor) -> torch.Tensor:
  """Dequantize and inverse-permute QKV communication tensors with CuTe DSL.

  :param quant_x: Input tensor `[S, B, N, D + 4]` in the mixed FP8+FP32 layout.
  :returns: Output BF16 tensor `[B, S, N, D]`.
  """

  if quant_x.device.type != "cuda":
    raise ValueError("CuTe DSL fp8_comm_qkv_permute_dequant only supports CUDA tensors.")
  _require_fp8_comm_arch(quant_x.device)
  assert quant_x.dtype == torch.float8_e4m3fn, f"expected float8_e4m3fn but got {quant_x.dtype}"

  seq_len, batch, num_heads, packed_head_size = quant_x.shape
  head_size = packed_head_size - 4
  _derive_vector_config(head_size)

  out_perm = torch.empty((seq_len, batch, num_heads, head_size),
                         dtype=torch.bfloat16,
                         device=quant_x.device)

  compiled = _compile_qkv_dequant(
    out_perm=out_perm,
    quant_x=quant_x,
  )
  compiled(
    out_perm,
    quant_x.view(torch.uint8),
  )
  return out_perm.permute(1, 0, 2, 3)


__all__ = [
  "fp8_comm_per_token_quant",
  "fp8_comm_per_token_dequant",
  "fp8_comm_qkv_permute_quant",
  "fp8_comm_qkv_permute_dequant",
]
