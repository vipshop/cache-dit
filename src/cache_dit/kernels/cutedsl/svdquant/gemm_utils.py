"""Shared low-level SVDQ CuTe DSL helpers.

This module mirrors the reusable helper surface in `csrc/kernels/svdq/gemm_utils.cuh` so the Python-
side CuTe kernels can rely on one centralized PTX/NVVM helper layer instead of open-coding inline
assembly in individual kernel files.
"""

from __future__ import annotations

import os

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith
from cutlass._mlir.dialects import llvm
from cutlass._mlir.dialects import vector
from cutlass.cute.tensor import TensorSSA
from cutlass.cute.typing import BFloat16
from cutlass.cute.typing import Float16
from cutlass.cute.typing import Float32
from cutlass.cute.typing import Int32
from cutlass.cutlass_dsl import T
from cutlass.cutlass_dsl import dsl_user_op

_STRICT_ENV = "CACHE_DIT_SVDQ_V3_STRICT"
_SUPPORTED_STAGES = (1, 2, 3)
_LOG2_E = 1.442695041
_GELU_TANH_SCALE = 0.79788456
_GELU_TANH_CUBIC = 0.044715
_LOP3_XOR_LUT = 0x6A


def _make_pair_tensor(v0, v1, element_type, *, loc=None, ip=None) -> TensorSSA:
  raw = vector.from_elements(
    ir.VectorType.get([2], element_type.mlir_type),
    [
      element_type(v0).ir_value(loc=loc, ip=ip),
      element_type(v1).ir_value(loc=loc, ip=ip),
    ],
    loc=loc,
    ip=ip,
  )
  return TensorSSA(raw, (2, ), element_type)


def _make_f32x2_tensor(v0, v1, *, loc=None, ip=None) -> TensorSSA:
  return _make_pair_tensor(v0, v1, Float32, loc=loc, ip=ip)


def _extract_f32x2_pair(src: TensorSSA,
                        *,
                        loc=None,
                        ip=None) -> tuple[cutlass.Float32, cutlass.Float32]:
  raw = src.maybe_downcast()
  return (
    cutlass.Float32(vector.extract(raw, dynamic_position=[], static_position=[0], loc=loc, ip=ip)),
    cutlass.Float32(vector.extract(raw, dynamic_position=[], static_position=[1], loc=loc, ip=ip)),
  )


def _pair_tensor_to_i32_bits(src: TensorSSA, *, loc=None, ip=None) -> Int32:
  return Int32(llvm.bitcast(T.i32(), src.maybe_downcast(), loc=loc, ip=ip))


def _pair_bits_to_tensor(bits, element_type, *, loc=None, ip=None) -> TensorSSA:
  raw = llvm.bitcast(
    ir.VectorType.get([2], element_type.mlir_type),
    Int32(bits).ir_value(loc=loc, ip=ip),
    loc=loc,
    ip=ip,
  )
  return TensorSSA(raw, (2, ), element_type)


def _pair_bits_to_f32_tuple(bits,
                            element_type,
                            *,
                            loc=None,
                            ip=None) -> tuple[cutlass.Float32, cutlass.Float32]:
  return _extract_f32x2_pair(_pair_bits_to_tensor(bits, element_type, loc=loc, ip=ip).to(Float32),
                             loc=loc,
                             ip=ip)


def _smem_ptr_i32(ptr: cute.Pointer, *, loc=None, ip=None):
  return ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)


def _gmem_ptr_i64(ptr: cute.Pointer, *, loc=None, ip=None):
  return ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)


def _extract_i32_struct(result, count: int, *, loc=None, ip=None) -> tuple[Int32, ...]:
  return tuple(
    Int32(llvm.extractvalue(T.i32(), result, [idx], loc=loc, ip=ip)) for idx in range(count))


@dsl_user_op
def float2_to_f16x2_bits(v0, v1, *, loc=None, ip=None) -> Int32:
  """Pack two float values into one `f16x2` register container."""

  return _pair_tensor_to_i32_bits(_make_f32x2_tensor(v0, v1, loc=loc, ip=ip).to(Float16),
                                  loc=loc,
                                  ip=ip)


@dsl_user_op
def float2_to_bf16x2_bits(v0, v1, *, loc=None, ip=None) -> Int32:
  """Pack two float values into one `bf16x2` register container."""

  packed = cute.arch.cvt_f32x2_bf16x2(_make_f32x2_tensor(v0, v1, loc=loc, ip=ip).maybe_downcast(),
                                      loc=loc,
                                      ip=ip)
  return Int32(llvm.bitcast(T.i32(), packed, loc=loc, ip=ip))


@dsl_user_op
def f16x2_bits_to_f32x2(bits, *, loc=None, ip=None) -> tuple[cutlass.Float32, cutlass.Float32]:
  """Convert one packed `f16x2` register back to two float values."""

  return _pair_bits_to_f32_tuple(bits, Float16, loc=loc, ip=ip)


@dsl_user_op
def bf16x2_bits_to_f32x2(bits, *, loc=None, ip=None) -> tuple[cutlass.Float32, cutlass.Float32]:
  """Convert one packed `bf16x2` register back to two float values."""

  return _pair_bits_to_f32_tuple(bits, BFloat16, loc=loc, ip=ip)


@dsl_user_op
def round_f32x2_to_f16x2_f32(v0,
                             v1,
                             *,
                             loc=None,
                             ip=None) -> tuple[cutlass.Float32, cutlass.Float32]:
  """Round a float pair through CUDA `f16x2` storage and back to float.

  :param v0: First float value.
  :param v1: Second float value.
  :returns: Two float values after `f16x2` rounding.
  """

  return f16x2_bits_to_f32x2(float2_to_f16x2_bits(v0, v1, loc=loc, ip=ip), loc=loc, ip=ip)


@dsl_user_op
def round_f32x2_to_bf16x2_f32(v0,
                              v1,
                              *,
                              loc=None,
                              ip=None) -> tuple[cutlass.Float32, cutlass.Float32]:
  """Round a float pair through CUDA `bf16x2` storage and back to float.

  :param v0: First float value.
  :param v1: Second float value.
  :returns: Two float values after `bf16x2` rounding.
  """

  return bf16x2_bits_to_f32x2(float2_to_bf16x2_bits(v0, v1, loc=loc, ip=ip), loc=loc, ip=ip)


@dsl_user_op
def load_shared_v2_b32(ptr: cute.Pointer, *, loc=None, ip=None) -> tuple[Int32, Int32]:
  """Mirror `load<true, uint2>` from `gemm_utils.cuh`."""

  result = llvm.inline_asm(
    llvm.StructType.get_literal([T.i32(), T.i32()]),
    [_smem_ptr_i32(ptr, loc=loc, ip=ip)],
    "ld.shared.v2.b32 {$0, $1}, [$2];",
    "=r,=r,r",
    has_side_effects=True,
    is_align_stack=False,
    asm_dialect=llvm.AsmDialect.AD_ATT,
  )
  return _extract_i32_struct(result, 2, loc=loc, ip=ip)


@dsl_user_op
def load_shared_v4_b32(ptr: cute.Pointer,
                       *,
                       loc=None,
                       ip=None) -> tuple[Int32, Int32, Int32, Int32]:
  """Mirror `load<true, uint4>` from `gemm_utils.cuh`."""

  result = llvm.inline_asm(
    llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
    [_smem_ptr_i32(ptr, loc=loc, ip=ip)],
    "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
    "=r,=r,=r,=r,r",
    has_side_effects=True,
    is_align_stack=False,
    asm_dialect=llvm.AsmDialect.AD_ATT,
  )
  return _extract_i32_struct(result, 4, loc=loc, ip=ip)


@dsl_user_op
def load_pred_b32(ptr: cute.Pointer, pred, *, loc=None, ip=None) -> Int32:
  """Mirror `load_pred<T>` for 32-bit values."""

  return Int32(
    llvm.inline_asm(
      T.i32(),
      [
        _gmem_ptr_i64(ptr, loc=loc, ip=ip),
        Int32(pred).ir_value(loc=loc, ip=ip),
      ],
      """{
        .reg .pred loadpred;
        setp.ne.b32 loadpred, $2, 0;
        @loadpred ld.global.nc.b32 $0, [$1];
      }""",
      "=r,l,r",
      has_side_effects=True,
      is_align_stack=False,
      asm_dialect=llvm.AsmDialect.AD_ATT,
    ))


@dsl_user_op
def load_pred_v2_b32(ptr: cute.Pointer, pred, *, loc=None, ip=None) -> tuple[Int32, Int32]:
  """Mirror `load_pred<T>` for 64-bit vector values."""

  result = llvm.inline_asm(
    llvm.StructType.get_literal([T.i32(), T.i32()]),
    [
      _gmem_ptr_i64(ptr, loc=loc, ip=ip),
      Int32(pred).ir_value(loc=loc, ip=ip),
    ],
    """{
      .reg .pred loadpred;
      setp.ne.b32 loadpred, $3, 0;
      @loadpred ld.global.nc.v2.b32 {$0, $1}, [$2];
    }""",
    "=r,=r,l,r",
    has_side_effects=True,
    is_align_stack=False,
    asm_dialect=llvm.AsmDialect.AD_ATT,
  )
  return _extract_i32_struct(result, 2, loc=loc, ip=ip)


@dsl_user_op
def load_pred_v4_b32(ptr: cute.Pointer,
                     pred,
                     *,
                     loc=None,
                     ip=None) -> tuple[Int32, Int32, Int32, Int32]:
  """Mirror `load_pred<T>` for 128-bit vector values."""

  result = llvm.inline_asm(
    llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
    [
      _gmem_ptr_i64(ptr, loc=loc, ip=ip),
      Int32(pred).ir_value(loc=loc, ip=ip),
    ],
    """{
      .reg .pred loadpred;
      setp.ne.b32 loadpred, $5, 0;
      @loadpred ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];
    }""",
    "=r,=r,=r,=r,l,r",
    has_side_effects=True,
    is_align_stack=False,
    asm_dialect=llvm.AsmDialect.AD_ATT,
  )
  return _extract_i32_struct(result, 4, loc=loc, ip=ip)


@dsl_user_op
def store_shared_v2_b32(ptr: cute.Pointer, v0, v1, *, loc=None, ip=None) -> None:
  """Mirror `store<true, uint2>` from `gemm_utils.cuh`."""

  llvm.inline_asm(
    None,
    [
      _smem_ptr_i32(ptr, loc=loc, ip=ip),
      Int32(v0).ir_value(loc=loc, ip=ip),
      Int32(v1).ir_value(loc=loc, ip=ip),
    ],
    "st.shared.v2.b32 [$0], {$1, $2};",
    "r,r,r",
    has_side_effects=True,
    is_align_stack=False,
    asm_dialect=llvm.AsmDialect.AD_ATT,
  )


@dsl_user_op
def store_shared_v4_b32(ptr: cute.Pointer, v0, v1, v2, v3, *, loc=None, ip=None) -> None:
  """Mirror `store<true, uint4>` from `gemm_utils.cuh`."""

  llvm.inline_asm(
    None,
    [
      _smem_ptr_i32(ptr, loc=loc, ip=ip),
      Int32(v0).ir_value(loc=loc, ip=ip),
      Int32(v1).ir_value(loc=loc, ip=ip),
      Int32(v2).ir_value(loc=loc, ip=ip),
      Int32(v3).ir_value(loc=loc, ip=ip),
    ],
    "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
    "r,r,r,r,r",
    has_side_effects=True,
    is_align_stack=False,
    asm_dialect=llvm.AsmDialect.AD_ATT,
  )


@dsl_user_op
def store_global_cg_b32(ptr: cute.Pointer, value, *, loc=None, ip=None) -> None:
  """Store one 32-bit value via `st.global.cg.b32`."""

  llvm.inline_asm(
    None,
    [
      _gmem_ptr_i64(ptr, loc=loc, ip=ip),
      Int32(value).ir_value(loc=loc, ip=ip),
    ],
    "st.global.cg.b32 [$0], $1;",
    "l,r",
    has_side_effects=True,
    is_align_stack=False,
    asm_dialect=llvm.AsmDialect.AD_ATT,
  )


@dsl_user_op
def store_global_cg_v2_b32(ptr: cute.Pointer, v0, v1, *, loc=None, ip=None) -> None:
  """Store one 64-bit vector via `st.global.cg.v2.b32`."""

  llvm.inline_asm(
    None,
    [
      _gmem_ptr_i64(ptr, loc=loc, ip=ip),
      Int32(v0).ir_value(loc=loc, ip=ip),
      Int32(v1).ir_value(loc=loc, ip=ip),
    ],
    "st.global.cg.v2.b32 [$0], {$1, $2};",
    "l,r,r",
    has_side_effects=True,
    is_align_stack=False,
    asm_dialect=llvm.AsmDialect.AD_ATT,
  )


@dsl_user_op
def store_global_cg_v4_b32(ptr: cute.Pointer, v0, v1, v2, v3, *, loc=None, ip=None) -> None:
  """Store one 128-bit vector via `st.global.cg.v4.b32`."""

  llvm.inline_asm(
    None,
    [
      _gmem_ptr_i64(ptr, loc=loc, ip=ip),
      Int32(v0).ir_value(loc=loc, ip=ip),
      Int32(v1).ir_value(loc=loc, ip=ip),
      Int32(v2).ir_value(loc=loc, ip=ip),
      Int32(v3).ir_value(loc=loc, ip=ip),
    ],
    "st.global.cg.v4.b32 [$0], {$1, $2, $3, $4};",
    "l,r,r,r,r",
    has_side_effects=True,
    is_align_stack=False,
    asm_dialect=llvm.AsmDialect.AD_ATT,
  )


@dsl_user_op
def store_pred_b32(ptr: cute.Pointer, value, pred, *, loc=None, ip=None) -> None:
  """Mirror `store_pred<T>` for 32-bit values."""

  llvm.inline_asm(
    None,
    [
      Int32(pred).ir_value(loc=loc, ip=ip),
      _gmem_ptr_i64(ptr, loc=loc, ip=ip),
      Int32(value).ir_value(loc=loc, ip=ip),
    ],
    """{
      .reg .pred storepred;
      setp.ne.b32 storepred, $0, 0;
      @storepred st.global.cg.b32 [$1], $2;
    }""",
    "r,l,r",
    has_side_effects=True,
    is_align_stack=False,
    asm_dialect=llvm.AsmDialect.AD_ATT,
  )


@dsl_user_op
def store_pred_v2_b32(ptr: cute.Pointer, v0, v1, pred, *, loc=None, ip=None) -> None:
  """Mirror `store_pred<T>` for 64-bit vector values."""

  llvm.inline_asm(
    None,
    [
      Int32(pred).ir_value(loc=loc, ip=ip),
      _gmem_ptr_i64(ptr, loc=loc, ip=ip),
      Int32(v0).ir_value(loc=loc, ip=ip),
      Int32(v1).ir_value(loc=loc, ip=ip),
    ],
    """{
      .reg .pred storepred;
      setp.ne.b32 storepred, $0, 0;
      @storepred st.global.cg.v2.b32 [$1], {$2, $3};
    }""",
    "r,l,r,r",
    has_side_effects=True,
    is_align_stack=False,
    asm_dialect=llvm.AsmDialect.AD_ATT,
  )


@dsl_user_op
def store_pred_v4_b32(ptr: cute.Pointer, v0, v1, v2, v3, pred, *, loc=None, ip=None) -> None:
  """Mirror `store_pred<T>` for 128-bit vector values."""

  llvm.inline_asm(
    None,
    [
      Int32(pred).ir_value(loc=loc, ip=ip),
      _gmem_ptr_i64(ptr, loc=loc, ip=ip),
      Int32(v0).ir_value(loc=loc, ip=ip),
      Int32(v1).ir_value(loc=loc, ip=ip),
      Int32(v2).ir_value(loc=loc, ip=ip),
      Int32(v3).ir_value(loc=loc, ip=ip),
    ],
    """{
      .reg .pred storepred;
      setp.ne.b32 storepred, $0, 0;
      @storepred st.global.cg.v4.b32 [$1], {$2, $3, $4, $5};
    }""",
    "r,l,r,r,r,r",
    has_side_effects=True,
    is_align_stack=False,
    asm_dialect=llvm.AsmDialect.AD_ATT,
  )


@dsl_user_op
def cp_async_ca_4(shared_dst: cute.Pointer,
                  global_src: cute.Pointer,
                  pred,
                  *,
                  loc=None,
                  ip=None) -> None:
  """Mirror `cp_async_ca<T>` for 4-byte copies."""

  llvm.inline_asm(
    None,
    [
      Int32(pred).ir_value(loc=loc, ip=ip),
      _smem_ptr_i32(shared_dst, loc=loc, ip=ip),
      _gmem_ptr_i64(global_src, loc=loc, ip=ip),
    ],
    """{
      .reg .pred p;
      setp.ne.b32 p, $0, 0;
      @p cp.async.ca.shared.global [$1], [$2], 4;
      @!p cp.async.ca.shared.global [$1], [$2], 4, 0;
    }""",
    "r,r,l",
    has_side_effects=True,
    is_align_stack=False,
    asm_dialect=llvm.AsmDialect.AD_ATT,
  )


@dsl_user_op
def cp_async_ca_8(shared_dst: cute.Pointer,
                  global_src: cute.Pointer,
                  pred,
                  *,
                  loc=None,
                  ip=None) -> None:
  """Mirror `cp_async_ca<T>` for 8-byte copies."""

  llvm.inline_asm(
    None,
    [
      Int32(pred).ir_value(loc=loc, ip=ip),
      _smem_ptr_i32(shared_dst, loc=loc, ip=ip),
      _gmem_ptr_i64(global_src, loc=loc, ip=ip),
    ],
    """{
      .reg .pred p;
      setp.ne.b32 p, $0, 0;
      @p cp.async.ca.shared.global [$1], [$2], 8;
      @!p cp.async.ca.shared.global [$1], [$2], 8, 0;
    }""",
    "r,r,l",
    has_side_effects=True,
    is_align_stack=False,
    asm_dialect=llvm.AsmDialect.AD_ATT,
  )


@dsl_user_op
def cp_async_ca_16(shared_dst: cute.Pointer,
                   global_src: cute.Pointer,
                   pred,
                   *,
                   loc=None,
                   ip=None) -> None:
  """Mirror `cp_async_ca<T>` for 16-byte copies."""

  llvm.inline_asm(
    None,
    [
      Int32(pred).ir_value(loc=loc, ip=ip),
      _smem_ptr_i32(shared_dst, loc=loc, ip=ip),
      _gmem_ptr_i64(global_src, loc=loc, ip=ip),
    ],
    """{
      .reg .pred p;
      setp.ne.b32 p, $0, 0;
      @p cp.async.ca.shared.global [$1], [$2], 16;
      @!p cp.async.ca.shared.global [$1], [$2], 16, 0;
    }""",
    "r,r,l",
    has_side_effects=True,
    is_align_stack=False,
    asm_dialect=llvm.AsmDialect.AD_ATT,
  )


@dsl_user_op
def cp_async_commit(*, loc=None, ip=None) -> None:
  """Commit the current non-bulk `cp.async` group."""

  cute.arch.cp_async_commit_group(loc=loc, ip=ip)


@dsl_user_op
def cp_async_wait(n: int, *, loc=None, ip=None) -> None:
  """Wait until at most `n` `cp.async` groups remain pending."""

  cute.arch.cp_async_wait_group(n, loc=loc, ip=ip)


@dsl_user_op
def ldmatrix_x4_m8n8_shared_b16(ptr: cute.Pointer,
                                *,
                                loc=None,
                                ip=None) -> tuple[Int32, Int32, Int32, Int32]:
  """Mirror `ldmatrix.sync.aligned.x4.m8n8.shared.b16`."""

  result = llvm.inline_asm(
    llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
    [_smem_ptr_i32(ptr, loc=loc, ip=ip)],
    "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {$0, $1, $2, $3}, [$4];",
    "=r,=r,=r,=r,r",
    has_side_effects=True,
    is_align_stack=False,
    asm_dialect=llvm.AsmDialect.AD_ATT,
  )
  return _extract_i32_struct(result, 4, loc=loc, ip=ip)


@dsl_user_op
def movmatrix_m8n8_trans_b16(x, *, loc=None, ip=None) -> Int32:
  """Mirror `movmatrix.sync.aligned.m8n8.trans.b16`."""

  return Int32(
    llvm.inline_asm(
      T.i32(),
      [Int32(x).ir_value(loc=loc, ip=ip)],
      "movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;",
      "=r,r",
      has_side_effects=False,
      is_align_stack=False,
      asm_dialect=llvm.AsmDialect.AD_ATT,
    ))


@dsl_user_op
def quantize_float2_int4_signed(v0, v1, *, loc=None, ip=None) -> Int32:
  """Pack two float values into one signed INT4 byte.

  This mirrors `quantize_float2<4, false>` in `gemm_utils.cuh`.
  """

  return Int32(
    llvm.inline_asm(
      T.i32(),
      [
        cutlass.Float32(v0).ir_value(loc=loc, ip=ip),
        cutlass.Float32(v1).ir_value(loc=loc, ip=ip),
      ],
      """{
        .reg .s32 a0, a1;
        cvt.rni.s32.f32 a0, $1;
        cvt.rni.s32.f32 a1, $2;
        cvt.pack.sat.s4.s32.b32 $0, a1, a0, 0;
      }""",
      "=r,f,f",
      has_side_effects=False,
      is_align_stack=False,
      asm_dialect=llvm.AsmDialect.AD_ATT,
    ))


@dsl_user_op
def quantize_float2_int4_unsigned(v0, v1, *, loc=None, ip=None) -> Int32:
  """Pack two float values into one unsigned INT4 byte.

  This mirrors `quantize_float2<4, true>` in `gemm_utils.cuh`.
  """

  return Int32(
    llvm.inline_asm(
      T.i32(),
      [
        cutlass.Float32(v0).ir_value(loc=loc, ip=ip),
        cutlass.Float32(v1).ir_value(loc=loc, ip=ip),
      ],
      """{
        .reg .s32 a0, a1;
        cvt.rni.s32.f32 a0, $1;
        cvt.rni.s32.f32 a1, $2;
        cvt.pack.sat.u4.s32.b32 $0, a1, a0, 0;
      }""",
      "=r,f,f",
      has_side_effects=False,
      is_align_stack=False,
      asm_dialect=llvm.AsmDialect.AD_ATT,
    ))


@dsl_user_op
def quantize_float2_int8_signed(v0, v1, *, loc=None, ip=None) -> Int32:
  """Pack two float values into one signed INT8 pair container."""

  return Int32(
    llvm.inline_asm(
      T.i32(),
      [
        cutlass.Float32(v0).ir_value(loc=loc, ip=ip),
        cutlass.Float32(v1).ir_value(loc=loc, ip=ip),
      ],
      """{
        .reg .s32 a0, a1;
        cvt.rni.s32.f32 a0, $1;
        cvt.rni.s32.f32 a1, $2;
        cvt.pack.sat.s8.s32.b32 $0, a1, a0, 0;
      }""",
      "=r,f,f",
      has_side_effects=False,
      is_align_stack=False,
      asm_dialect=llvm.AsmDialect.AD_ATT,
    ))


@dsl_user_op
def quantize_float2_fp4(v0, v1, *, loc=None, ip=None) -> Int32:
  """Pack two float values into one `e2m1x2` byte container."""

  return Int32(
    llvm.inline_asm(
      T.i32(),
      [
        cutlass.Float32(v1).ir_value(loc=loc, ip=ip),
        cutlass.Float32(v0).ir_value(loc=loc, ip=ip),
      ],
      """{
        .reg .b8 tmp;
        cvt.rn.satfinite.e2m1x2.f32 tmp, $1, $2;
        cvt.u32.u8 $0, tmp;
      }""",
      "=r,f,f",
      has_side_effects=False,
      is_align_stack=False,
      asm_dialect=llvm.AsmDialect.AD_ATT,
    ))


@dsl_user_op
def quantize_float4_fp8(v0, v1, v2, v3, *, loc=None, ip=None) -> Int32:
  """Pack four float values into one `e4m3x4` word."""

  return Int32(
    llvm.inline_asm(
      T.i32(),
      [
        cutlass.Float32(v0).ir_value(loc=loc, ip=ip),
        cutlass.Float32(v1).ir_value(loc=loc, ip=ip),
        cutlass.Float32(v2).ir_value(loc=loc, ip=ip),
        cutlass.Float32(v3).ir_value(loc=loc, ip=ip),
      ],
      """{
        .reg .b16 lo, hi;
        cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;
        cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;
        mov.b32 $0, {lo, hi};
      }""",
      "=r,f,f,f,f",
      has_side_effects=False,
      is_align_stack=False,
      asm_dialect=llvm.AsmDialect.AD_ATT,
    ))


@dsl_user_op
def pack_int4_pairs_to_word(p0: Int32,
                            p1: Int32,
                            p2: Int32,
                            p3: Int32,
                            *,
                            loc=None,
                            ip=None) -> Int32:
  """Pack four INT4 bytes into one 32-bit word."""

  return Int32(
    llvm.inline_asm(
      T.i32(),
      [
        p0.ir_value(loc=loc, ip=ip),
        p1.ir_value(loc=loc, ip=ip),
        p2.ir_value(loc=loc, ip=ip),
        p3.ir_value(loc=loc, ip=ip),
      ],
      """{
        .reg .b32 q1, q2, q3, t0, t1;
        shl.b32 q1, $2, 8;
        shl.b32 q2, $3, 16;
        shl.b32 q3, $4, 24;
        or.b32 t0, $1, q1;
        or.b32 t1, q2, q3;
        or.b32 $0, t0, t1;
      }""",
      "=r,r,r,r,r",
      has_side_effects=False,
      is_align_stack=False,
      asm_dialect=llvm.AsmDialect.AD_ATT,
    ))


@dsl_user_op
def quantize_f32x8_to_int4_word_signed(v0,
                                       v1,
                                       v2,
                                       v3,
                                       v4,
                                       v5,
                                       v6,
                                       v7,
                                       *,
                                       loc=None,
                                       ip=None) -> Int32:
  """Pack eight float values into one signed INT4 word."""

  return pack_int4_pairs_to_word(
    quantize_float2_int4_signed(v0, v1, loc=loc, ip=ip),
    quantize_float2_int4_signed(v2, v3, loc=loc, ip=ip),
    quantize_float2_int4_signed(v4, v5, loc=loc, ip=ip),
    quantize_float2_int4_signed(v6, v7, loc=loc, ip=ip),
    loc=loc,
    ip=ip,
  )


@dsl_user_op
def quantize_f32x8_to_int4_word_unsigned(v0,
                                         v1,
                                         v2,
                                         v3,
                                         v4,
                                         v5,
                                         v6,
                                         v7,
                                         *,
                                         loc=None,
                                         ip=None) -> Int32:
  """Pack eight float values into one unsigned INT4 word."""

  return pack_int4_pairs_to_word(
    quantize_float2_int4_unsigned(v0, v1, loc=loc, ip=ip),
    quantize_float2_int4_unsigned(v2, v3, loc=loc, ip=ip),
    quantize_float2_int4_unsigned(v4, v5, loc=loc, ip=ip),
    quantize_float2_int4_unsigned(v6, v7, loc=loc, ip=ip),
    loc=loc,
    ip=ip,
  )


@dsl_user_op
def tanh_approx_f32(x, *, loc=None, ip=None):
  """Mirror `cuda_tanhf` from `gemm_utils.cuh`."""

  return cutlass.Float32(
    llvm.inline_asm(
      T.f32(),
      [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
      "tanh.approx.f32 $0, $1;",
      "=f,f",
      has_side_effects=False,
      is_align_stack=False,
      asm_dialect=llvm.AsmDialect.AD_ATT,
    ))


@dsl_user_op
def rcp_approx_f32(x, *, loc=None, ip=None):
  """Mirror `cuda_frcp` from `gemm_utils.cuh`."""

  return cutlass.Float32(
    llvm.inline_asm(
      T.f32(),
      [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
      "rcp.approx.ftz.f32 $0, $1;",
      "=f,f",
      has_side_effects=False,
      is_align_stack=False,
      asm_dialect=llvm.AsmDialect.AD_ATT,
    ))


@dsl_user_op
def fdivide_approx_f32(a, b, *, loc=None, ip=None):
  """Approximate float divide helper used to model CUDA fast-divide paths."""

  return cutlass.Float32(
    llvm.inline_asm(
      T.f32(),
      [
        cutlass.Float32(a).ir_value(loc=loc, ip=ip),
        cutlass.Float32(b).ir_value(loc=loc, ip=ip),
      ],
      "div.approx.ftz.f32 $0, $1, $2;",
      "=f,f,f",
      has_side_effects=False,
      is_align_stack=False,
      asm_dialect=llvm.AsmDialect.AD_ATT,
    ))


@dsl_user_op
def fdivide_like_cuda_f32(a, b, *, loc=None, ip=None):
  """Current PTX-level approximation of the CUDA `__fdividef` path."""

  return fdivide_approx_f32(a, b, loc=loc, ip=ip)


@dsl_user_op
def rsqrt_approx_f32(x, *, loc=None, ip=None):
  """Mirror `cuda_frsqrt` from `gemm_utils.cuh`."""

  return cutlass.Float32(
    llvm.inline_asm(
      T.f32(),
      [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
      "rsqrt.approx.ftz.f32 $0, $1;",
      "=f,f",
      has_side_effects=False,
      is_align_stack=False,
      asm_dialect=llvm.AsmDialect.AD_ATT,
    ))


@dsl_user_op
def sin_approx_f32(x, *, loc=None, ip=None):
  """Mirror `cuda_sin` from `gemm_utils.cuh`."""

  return cutlass.Float32(
    llvm.inline_asm(
      T.f32(),
      [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
      "sin.approx.ftz.f32 $0, $1;",
      "=f,f",
      has_side_effects=False,
      is_align_stack=False,
      asm_dialect=llvm.AsmDialect.AD_ATT,
    ))


@dsl_user_op
def cos_approx_f32(x, *, loc=None, ip=None):
  """Mirror `cuda_cos` from `gemm_utils.cuh`."""

  return cutlass.Float32(
    llvm.inline_asm(
      T.f32(),
      [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
      "cos.approx.ftz.f32 $0, $1;",
      "=f,f",
      has_side_effects=False,
      is_align_stack=False,
      asm_dialect=llvm.AsmDialect.AD_ATT,
    ))


@dsl_user_op
def exp2_approx_f32(x, *, loc=None, ip=None):
  """Mirror `cuda_exp2` from `gemm_utils.cuh`."""

  return cutlass.Float32(
    llvm.inline_asm(
      T.f32(),
      [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
      "ex2.approx.ftz.f32 $0, $1;",
      "=f,f",
      has_side_effects=False,
      is_align_stack=False,
      asm_dialect=llvm.AsmDialect.AD_ATT,
    ))


@dsl_user_op
def sigmoid_approx_f32(x, use_tanh: bool = False, *, loc=None, ip=None):
  """Mirror `cuda_sigmoidf` from `gemm_utils.cuh`.

  :param x: Input scalar.
  :param use_tanh: Whether to force the tanh-based branch.
  :returns: Approximate sigmoid value.
  """

  xf = cutlass.Float32(x)
  if cutlass.const_expr(use_tanh):
    return cutlass.Float32(0.5) + cutlass.Float32(0.5) * tanh_approx_f32(
      cutlass.Float32(0.5) * xf, loc=loc, ip=ip)
  exp_term = exp2_approx_f32(cutlass.Float32(-_LOG2_E) * xf, loc=loc, ip=ip)
  return rcp_approx_f32(exp_term + cutlass.Float32(1.0), loc=loc, ip=ip)


@dsl_user_op
def gelu_f32(x, *, loc=None, ip=None):
  """Mirror the tanh-approx GELU helper in `gemm_utils.cuh`."""

  xf = cutlass.Float32(x)
  x3f = xf * xf * xf
  inner = cutlass.Float32(_GELU_TANH_SCALE) * (xf + cutlass.Float32(_GELU_TANH_CUBIC) * x3f)
  scale = cutlass.Float32(0.5) + cutlass.Float32(0.5) * tanh_approx_f32(inner, loc=loc, ip=ip)
  return xf * scale


@dsl_user_op
def silu_f32(x, use_tanh: bool = False, *, loc=None, ip=None):
  """Mirror the scalar `silu` helper in `gemm_utils.cuh`."""

  xf = cutlass.Float32(x)
  return xf * sigmoid_approx_f32(xf, use_tanh=use_tanh, loc=loc, ip=ip)


@dsl_user_op
def h2div_f16x2_f32(a0,
                    a1,
                    b0,
                    b1,
                    *,
                    loc=None,
                    ip=None) -> tuple[cutlass.Float32, cutlass.Float32]:
  """Mirror `h2div(half2, half2)` and return rounded float values."""

  return round_f32x2_to_f16x2_f32(
    fdivide_like_cuda_f32(a0, b0, loc=loc, ip=ip),
    fdivide_like_cuda_f32(a1, b1, loc=loc, ip=ip),
    loc=loc,
    ip=ip,
  )


@dsl_user_op
def h2div_bf16x2_f32(a0,
                     a1,
                     b0,
                     b1,
                     *,
                     loc=None,
                     ip=None) -> tuple[cutlass.Float32, cutlass.Float32]:
  """Mirror `h2div(__nv_bfloat162, __nv_bfloat162)` and return rounded float values."""

  return round_f32x2_to_bf16x2_f32(
    fdivide_like_cuda_f32(a0, b0, loc=loc, ip=ip),
    fdivide_like_cuda_f32(a1, b1, loc=loc, ip=ip),
    loc=loc,
    ip=ip,
  )


@dsl_user_op
def gelu_f16x2_f32(v0, v1, *, loc=None, ip=None) -> tuple[cutlass.Float32, cutlass.Float32]:
  """Mirror `gelu_half2<half2>` and return rounded float values."""

  return round_f32x2_to_f16x2_f32(gelu_f32(v0, loc=loc, ip=ip),
                                  gelu_f32(v1, loc=loc, ip=ip),
                                  loc=loc,
                                  ip=ip)


@dsl_user_op
def gelu_bf16x2_f32(v0, v1, *, loc=None, ip=None) -> tuple[cutlass.Float32, cutlass.Float32]:
  """Mirror `gelu_half2<__nv_bfloat162>` and return rounded float values."""

  return round_f32x2_to_bf16x2_f32(gelu_f32(v0, loc=loc, ip=ip),
                                   gelu_f32(v1, loc=loc, ip=ip),
                                   loc=loc,
                                   ip=ip)


@dsl_user_op
def reduce_add_f32(ptr: cute.Pointer, value, *, loc=None, ip=None) -> None:
  """Mirror `red.relaxed.gpu.global.add.f32`."""

  llvm.inline_asm(
    None,
    [
      _gmem_ptr_i64(ptr, loc=loc, ip=ip),
      cutlass.Float32(value).ir_value(loc=loc, ip=ip),
    ],
    "red.relaxed.gpu.global.add.f32 [$0], $1;",
    "l,f",
    has_side_effects=True,
    is_align_stack=False,
    asm_dialect=llvm.AsmDialect.AD_ATT,
  )


@dsl_user_op
def reduce_add_f32_pred(ptr: cute.Pointer, value, pred, *, loc=None, ip=None) -> None:
  """Mirror predicated `red.relaxed.gpu.global.add.f32`."""

  llvm.inline_asm(
    None,
    [
      Int32(pred).ir_value(loc=loc, ip=ip),
      _gmem_ptr_i64(ptr, loc=loc, ip=ip),
      cutlass.Float32(value).ir_value(loc=loc, ip=ip),
    ],
    """{
      .reg .pred storepred;
      setp.ne.b32 storepred, $0, 0;
      @storepred red.relaxed.gpu.global.add.f32 [$1], $2;
    }""",
    "r,l,f",
    has_side_effects=True,
    is_align_stack=False,
    asm_dialect=llvm.AsmDialect.AD_ATT,
  )


@dsl_user_op
def prmt_b32(src, src_reg_shifted, prmt_indices, *, loc=None, ip=None) -> Int32:
  """Mirror the register-level `prmt.b32` helper from `gemm_utils.cuh`."""

  return Int32(
    llvm.inline_asm(
      T.i32(),
      [
        Int32(src).ir_value(loc=loc, ip=ip),
        Int32(src_reg_shifted).ir_value(loc=loc, ip=ip),
        Int32(prmt_indices).ir_value(loc=loc, ip=ip),
      ],
      "prmt.b32 $0, $1, $2, $3;",
      "=r,r,r,r",
      has_side_effects=False,
      is_align_stack=False,
      asm_dialect=llvm.AsmDialect.AD_ATT,
    ))


@dsl_user_op
def int2float_fast(value, *, loc=None, ip=None):
  """Mirror the `int2float_fast` bit-hack from `gemm_utils.cuh`."""

  fval = llvm.inline_asm(
    T.f32(),
    [Int32(value).ir_value(loc=loc, ip=ip)],
    f"lop3.b32 $0, $1, 0x7fffff, 0x4b400000, {_LOP3_XOR_LUT};",
    "=f,r",
    has_side_effects=False,
    is_align_stack=False,
    asm_dialect=llvm.AsmDialect.AD_ATT,
  )
  return cutlass.Float32(fval) - cutlass.Float32(12582912.0)


def _half2_add_bits(a_bits: Int32, b_bits: Int32, *, loc=None, ip=None) -> Int32:
  return Int32(
    llvm.inline_asm(
      T.i32(),
      [
        a_bits.ir_value(loc=loc, ip=ip),
        b_bits.ir_value(loc=loc, ip=ip),
      ],
      "add.f16x2 $0, $1, $2;",
      "=r,r,r",
      has_side_effects=False,
      is_align_stack=False,
      asm_dialect=llvm.AsmDialect.AD_ATT,
    ))


@dsl_user_op
def int2half2_fast_8192(x, y, *, loc=None, ip=None) -> TensorSSA:
  """Mirror `int2half2_fast_8192` from `gemm_utils.cuh`."""

  packed = prmt_b32(x, y, Int32(0x5410), loc=loc, ip=ip)
  shifted = Int32(
    llvm.lshr(packed.ir_value(loc=loc, ip=ip),
              arith.constant(T.i32(), 4, loc=loc, ip=ip),
              loc=loc,
              ip=ip))
  hbits = Int32(
    llvm.inline_asm(
      T.i32(),
      [shifted.ir_value(loc=loc, ip=ip)],
      f"lop3.b32 $0, $1, 0x03ff03ff, 0x76007600, {_LOP3_XOR_LUT};",
      "=r,r",
      has_side_effects=False,
      is_align_stack=False,
      asm_dialect=llvm.AsmDialect.AD_ATT,
    ))
  bias = float2_to_f16x2_bits(-24576.0, -24576.0, loc=loc, ip=ip)
  return _pair_bits_to_tensor(_half2_add_bits(hbits, bias, loc=loc, ip=ip), Float16, loc=loc, ip=ip)


@dsl_user_op
def int2half2_fast_4096_rn(x, y, *, loc=None, ip=None) -> TensorSSA:
  """Mirror `int2half2_fast_4096_rn` from `gemm_utils.cuh`."""

  x_scaled = Int32(x) * Int32(8192) + Int32(32768)
  y_scaled = Int32(y) * Int32(8192) + Int32(32768)
  packed = prmt_b32(x_scaled, y_scaled, Int32(0x7632), loc=loc, ip=ip)
  hbits = Int32(
    llvm.inline_asm(
      T.i32(),
      [packed.ir_value(loc=loc, ip=ip)],
      f"lop3.b32 $0, $1, 0x03ff03ff, 0x72007200, {_LOP3_XOR_LUT};",
      "=r,r",
      has_side_effects=False,
      is_align_stack=False,
      asm_dialect=llvm.AsmDialect.AD_ATT,
    ))
  bias = float2_to_f16x2_bits(-12288.0, -12288.0, loc=loc, ip=ip)
  return _pair_bits_to_tensor(_half2_add_bits(hbits, bias, loc=loc, ip=ip), Float16, loc=loc, ip=ip)


@dsl_user_op
def int2half2_fast_512(x, y, *, loc=None, ip=None) -> TensorSSA:
  """Mirror `int2half2_fast_512` from `gemm_utils.cuh`."""

  packed = prmt_b32(x, y, Int32(0x5410), loc=loc, ip=ip)
  hbits = Int32(
    llvm.inline_asm(
      T.i32(),
      [packed.ir_value(loc=loc, ip=ip)],
      f"lop3.b32 $0, $1, 0x03ff03ff, 0x66006600, {_LOP3_XOR_LUT};",
      "=r,r",
      has_side_effects=False,
      is_align_stack=False,
      asm_dialect=llvm.AsmDialect.AD_ATT,
    ))
  bias = float2_to_f16x2_bits(-1536.0, -1536.0, loc=loc, ip=ip)
  return _pair_bits_to_tensor(_half2_add_bits(hbits, bias, loc=loc, ip=ip), Float16, loc=loc, ip=ip)


def normalize_runtime_stage(stage: int | None, op_name: str) -> int:
  """Validate the stage values shared by the SVDQ v3 wrappers.

  :param stage: Optional requested pipeline stage count.
  :param op_name: Public operator name used for validation errors.
  :returns: Validated stage count.
  :raises ValueError: If the stage is outside the compiled range.
  """

  normalized_stage = 1 if stage is None else int(stage)
  if normalized_stage not in _SUPPORTED_STAGES:
    raise ValueError(f"{op_name} stage must be one of {_SUPPORTED_STAGES}, got {normalized_stage}.")
  return normalized_stage


def require_int4_runtime(fp4: bool, op_name: str) -> None:
  """Reject FP4/NVFP4 while the v3 path only covers the INT4 milestone.

  :param fp4: Whether the caller requested the FP4 path.
  :param op_name: Public operator name used for errors.
  :raises NotImplementedError: If the FP4 path is requested.
  """

  if fp4:
    raise NotImplementedError(f"{op_name} v3 currently supports only the INT4 runtime path.")


def should_allow_cuda_fallback() -> bool:
  """Return whether the temporary v3 compatibility fallback is enabled.

  Setting `CACHE_DIT_SVDQ_V3_STRICT=1` is useful while bringing up the first real CuTe DSL kernels
  because it exposes any accidental fallback usage immediately.
  """

  return os.environ.get(_STRICT_ENV, "0") != "1"


__all__ = [
  "bf16x2_bits_to_f32x2",
  "cos_approx_f32",
  "cp_async_ca_16",
  "cp_async_ca_4",
  "cp_async_ca_8",
  "cp_async_commit",
  "cp_async_wait",
  "exp2_approx_f32",
  "f16x2_bits_to_f32x2",
  "fdivide_approx_f32",
  "fdivide_like_cuda_f32",
  "float2_to_bf16x2_bits",
  "float2_to_f16x2_bits",
  "gelu_bf16x2_f32",
  "gelu_f16x2_f32",
  "gelu_f32",
  "h2div_bf16x2_f32",
  "h2div_f16x2_f32",
  "int2float_fast",
  "int2half2_fast_4096_rn",
  "int2half2_fast_512",
  "int2half2_fast_8192",
  "ldmatrix_x4_m8n8_shared_b16",
  "load_pred_b32",
  "load_pred_v2_b32",
  "load_pred_v4_b32",
  "load_shared_v2_b32",
  "load_shared_v4_b32",
  "movmatrix_m8n8_trans_b16",
  "normalize_runtime_stage",
  "pack_int4_pairs_to_word",
  "prmt_b32",
  "quantize_f32x8_to_int4_word_signed",
  "quantize_f32x8_to_int4_word_unsigned",
  "quantize_float2_fp4",
  "quantize_float2_int4_signed",
  "quantize_float2_int4_unsigned",
  "quantize_float2_int8_signed",
  "quantize_float4_fp8",
  "rcp_approx_f32",
  "reduce_add_f32",
  "reduce_add_f32_pred",
  "require_int4_runtime",
  "round_f32x2_to_bf16x2_f32",
  "round_f32x2_to_f16x2_f32",
  "rsqrt_approx_f32",
  "should_allow_cuda_fallback",
  "sigmoid_approx_f32",
  "silu_f32",
  "sin_approx_f32",
  "store_global_cg_b32",
  "store_global_cg_v2_b32",
  "store_global_cg_v4_b32",
  "store_pred_b32",
  "store_pred_v2_b32",
  "store_pred_v4_b32",
  "store_shared_v2_b32",
  "store_shared_v4_b32",
  "tanh_approx_f32",
]
