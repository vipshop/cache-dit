"""Inline-PTX warp MMA wrappers used by the SVDQ CuTe DSL kernels.

This module centralizes the register-level PTX fragments that mirror the CUDA
helpers in `csrc/kernels/svdq/mma.cuh`. The current focus is the warp-level
`m16n8k16` f16/bf16 -> f32 path used by the LoRA epilogues, while the INT4
opcode constants stay available for the later GEMM mainloop rewrite.
"""

from __future__ import annotations

from typing import Sequence

from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass._mlir.dialects import vector
from cutlass.cute.tensor import TensorSSA
from cutlass.cute.typing import BFloat16
from cutlass.cute.typing import Float16
from cutlass.cute.typing import Float32
from cutlass.cute.typing import Int32
from cutlass.cutlass_dsl import T
from cutlass.cutlass_dsl import dsl_user_op

_SIGNED_INT4_MMA = "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32"
_UNSIGNED_ACT_INT4_MMA = "mma.sync.aligned.m16n8k64.row.col.s32.u4.s4.s32"
_F16_MMA_F32_OPCODE = "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
_BF16_MMA_F32_OPCODE = "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"


def int4_mma_opcode(act_unsigned: bool) -> str:
  """Return the PTX opcode for the signed or unsigned-activation INT4 MMA."""

  return _UNSIGNED_ACT_INT4_MMA if act_unsigned else _SIGNED_INT4_MMA


def _to_tensor_ssa(src):
  if hasattr(src, "maybe_downcast"):
    return src
  if hasattr(src, "load"):
    return src.load()
  raise TypeError(f"Expected a CuTe fragment tensor or TensorSSA, got {type(src).__name__}.")


def _shape_num_elements(shape) -> int:
  if isinstance(shape, tuple):
    total = 1
    for dim in shape:
      total *= _shape_num_elements(dim)
    return total
  return int(shape)


def _tensor_num_elements(src: TensorSSA) -> int:
  return _shape_num_elements(_to_tensor_ssa(src).shape)


def _extract_tensor_values(src: TensorSSA, count: int, element_type) -> list:
  raw_vec = _to_tensor_ssa(src).maybe_downcast()
  return [
    element_type(vector.extract(raw_vec, dynamic_position=[], static_position=[idx]), )
    for idx in range(count)
  ]


def _make_tensor(values: Sequence, shape: tuple[int, ...], element_type) -> TensorSSA:
  ir_values = [value.ir_value() for value in values]
  vec_ty = ir.VectorType.get([len(values)], ir_values[0].type)
  raw_vec = vector.from_elements(vec_ty, ir_values)
  return TensorSSA(raw_vec, shape, element_type)


def _slice_tensor(src: TensorSSA, start: int, size: int) -> TensorSSA:
  tensor_ssa = _to_tensor_ssa(src)
  values = _extract_tensor_values(tensor_ssa, _tensor_num_elements(tensor_ssa),
                                  tensor_ssa.element_type)
  sliced = values[start:start + size]
  return _make_tensor(sliced, (size, ), tensor_ssa.element_type)


def _bitcast_half_fragment_to_i32(src: TensorSSA, words: int) -> TensorSSA:
  raw_vec = llvm.bitcast(T.vector(words, T.i32()), _to_tensor_ssa(src).maybe_downcast())
  return TensorSSA(raw_vec, (words, ), Int32)


def _extract_bitcast_words(src, words: int) -> list[Int32]:
  return _extract_tensor_values(_bitcast_half_fragment_to_i32(src, words), words, Int32)


def _concat_tensors(lhs: TensorSSA, rhs: TensorSSA, element_type) -> TensorSSA:
  lhs_vals = _extract_tensor_values(lhs, lhs.shape[0], element_type)
  rhs_vals = _extract_tensor_values(rhs, rhs.shape[0], element_type)
  return _make_tensor(lhs_vals + rhs_vals, (lhs.shape[0] + rhs.shape[0], ), element_type)


def packed_half_fragment_from_i32_words(w0, w1, w2, w3, element_type) -> TensorSSA:
  """Build one packed half/bf16 fragment from four 32-bit register words.

  This is the inverse of the internal fragment bitcasts used by the MMA
  wrappers. It lets CuTe kernels load one 128-bit vector from global memory,
  keep the data in registers, and hand the packed fragment directly to
  `mma_f16xf16_f32` / `mma_bf16xbf16_f32`.

  :param w0: First 32-bit word.
  :param w1: Second 32-bit word.
  :param w2: Third 32-bit word.
  :param w3: Fourth 32-bit word.
  :param element_type: `Float16` or `BFloat16`.
  :returns: Packed fragment tensor with shape `(8,)`.
  """

  if element_type not in (Float16, BFloat16):
    raise TypeError("packed_half_fragment_from_i32_words expects Float16 or BFloat16, "
                    f"got {element_type}.")

  word_values = [
    Int32(w0).ir_value(),
    Int32(w1).ir_value(),
    Int32(w2).ir_value(),
    Int32(w3).ir_value()
  ]
  word_vec = vector.from_elements(ir.VectorType.get([4], word_values[0].type), word_values)
  raw_vec = llvm.bitcast(ir.VectorType.get([8], element_type.mlir_type), word_vec)
  return TensorSSA(raw_vec, (8, ), element_type)


def make_f32_accumulator_fragment(value: float = 0.0) -> TensorSSA:
  """Create one zero-initialized packed fp32 accumulator fragment."""

  values = [Float32(value) for _ in range(8)]
  return _make_tensor(values, (8, ), Float32)


def extract_f32_fragment_values(src: TensorSSA) -> list[Float32]:
  """Return the eight fp32 scalars stored in one packed accumulator fragment."""

  return _extract_tensor_values(_to_tensor_ssa(src), 8, Float32)


@dsl_user_op
def _mma_m16n8k16_f32f16f16f32_regs(
  a0: Int32,
  a1: Int32,
  a2: Int32,
  a3: Int32,
  b0: Int32,
  b1: Int32,
  c0: Float32,
  c1: Float32,
  c2: Float32,
  c3: Float32,
  *,
  loc=None,
  ip=None,
) -> tuple[Float32, Float32, Float32, Float32]:
  result = llvm.inline_asm(
    llvm.StructType.get_literal([T.f32()] * 4),
    [
      a0.ir_value(loc=loc, ip=ip),
      a1.ir_value(loc=loc, ip=ip),
      a2.ir_value(loc=loc, ip=ip),
      a3.ir_value(loc=loc, ip=ip),
      b0.ir_value(loc=loc, ip=ip),
      b1.ir_value(loc=loc, ip=ip),
      c0.ir_value(loc=loc, ip=ip),
      c1.ir_value(loc=loc, ip=ip),
      c2.ir_value(loc=loc, ip=ip),
      c3.ir_value(loc=loc, ip=ip),
    ],
    f"""{{
      {_F16_MMA_F32_OPCODE} {{$0, $1, $2, $3}},
      {{$4, $5, $6, $7}},
      {{$8, $9}},
      {{$10, $11, $12, $13}};
    }}""",
    "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
    has_side_effects=False,
    is_align_stack=False,
    asm_dialect=llvm.AsmDialect.AD_ATT,
  )
  return (
    Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
    Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
    Float32(llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip)),
    Float32(llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip)),
  )


@dsl_user_op
def _mma_m16n8k16_f32bf16bf16f32_regs(
  a0: Int32,
  a1: Int32,
  a2: Int32,
  a3: Int32,
  b0: Int32,
  b1: Int32,
  c0: Float32,
  c1: Float32,
  c2: Float32,
  c3: Float32,
  *,
  loc=None,
  ip=None,
) -> tuple[Float32, Float32, Float32, Float32]:
  result = llvm.inline_asm(
    llvm.StructType.get_literal([T.f32()] * 4),
    [
      a0.ir_value(loc=loc, ip=ip),
      a1.ir_value(loc=loc, ip=ip),
      a2.ir_value(loc=loc, ip=ip),
      a3.ir_value(loc=loc, ip=ip),
      b0.ir_value(loc=loc, ip=ip),
      b1.ir_value(loc=loc, ip=ip),
      c0.ir_value(loc=loc, ip=ip),
      c1.ir_value(loc=loc, ip=ip),
      c2.ir_value(loc=loc, ip=ip),
      c3.ir_value(loc=loc, ip=ip),
    ],
    f"""{{
      {_BF16_MMA_F32_OPCODE} {{$0, $1, $2, $3}},
      {{$4, $5, $6, $7}},
      {{$8, $9}},
      {{$10, $11, $12, $13}};
    }}""",
    "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
    has_side_effects=False,
    is_align_stack=False,
    asm_dialect=llvm.AsmDialect.AD_ATT,
  )
  return (
    Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
    Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
    Float32(llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip)),
    Float32(llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip)),
  )


def mma_m16n8k16_f32f16f16f32(a: TensorSSA, b: TensorSSA, c: TensorSSA) -> TensorSSA:
  """Run one `m16n8k16` f16 MMA step and return 4 f32 accumulators.

  :param a: Packed A fragment with shape `(8,)` and `Float16` element type.
  :param b: Packed B fragment with shape `(4,)` and `Float16` element type.
  :param c: Accumulator fragment with shape `(4,)` and `Float32` element type.
  :returns: Updated accumulator fragment with shape `(4,)`.
  """

  a = _to_tensor_ssa(a)
  b = _to_tensor_ssa(b)
  c = _to_tensor_ssa(c)
  a_words = _extract_tensor_values(_bitcast_half_fragment_to_i32(a, 4), 4, Int32)
  b_words = _extract_tensor_values(_bitcast_half_fragment_to_i32(b, 2), 2, Int32)
  c_vals = _extract_tensor_values(c, 4, Float32)
  result_vals = _mma_m16n8k16_f32f16f16f32_regs(*a_words, *b_words, *c_vals)
  return _make_tensor(result_vals, (4, ), Float32)


def mma_m16n8k16_f32bf16bf16f32(a: TensorSSA, b: TensorSSA, c: TensorSSA) -> TensorSSA:
  """Run one `m16n8k16` bf16 MMA step and return 4 f32 accumulators.

  :param a: Packed A fragment with shape `(8,)` and `BFloat16` element type.
  :param b: Packed B fragment with shape `(4,)` and `BFloat16` element type.
  :param c: Accumulator fragment with shape `(4,)` and `Float32` element type.
  :returns: Updated accumulator fragment with shape `(4,)`.
  """

  a = _to_tensor_ssa(a)
  b = _to_tensor_ssa(b)
  c = _to_tensor_ssa(c)
  a_words = _extract_tensor_values(_bitcast_half_fragment_to_i32(a, 4), 4, Int32)
  b_words = _extract_tensor_values(_bitcast_half_fragment_to_i32(b, 2), 2, Int32)
  c_vals = _extract_tensor_values(c, 4, Float32)
  result_vals = _mma_m16n8k16_f32bf16bf16f32_regs(*a_words, *b_words, *c_vals)
  return _make_tensor(result_vals, (4, ), Float32)


def mma_f16xf16_f32(a: TensorSSA, b: TensorSSA, c: TensorSSA) -> TensorSSA:
  """Run the packed `mma_f16xf16_f32` helper used by the CUDA LoRA epilogues.

  This mirrors `mma_f16xf16_f32` in `csrc/kernels/svdq/mma.cuh`: one packed
  `A` fragment and one packed `B` fragment each hold the two `m16n8k16` calls
  needed to cover a logical `m16n16k16` tile.

  :param a: Packed A fragment with shape `(8,)` and `Float16` element type.
  :param b: Packed B fragment with shape `(8,)` and `Float16` element type.
  :param c: Packed accumulator fragment with shape `(8,)` and `Float32` element type.
  :returns: Updated packed accumulator fragment with shape `(8,)`.
  """

  a = _to_tensor_ssa(a)
  b = _to_tensor_ssa(b)
  c = _to_tensor_ssa(c)
  a_words = _extract_bitcast_words(a, 4)
  b_words = _extract_bitcast_words(b, 4)
  c_vals = _extract_tensor_values(c, 8, Float32)
  c_lo = _mma_m16n8k16_f32f16f16f32_regs(*a_words, b_words[0], b_words[1], *c_vals[:4])
  c_hi = _mma_m16n8k16_f32f16f16f32_regs(*a_words, b_words[2], b_words[3], *c_vals[4:])
  return _make_tensor(list(c_lo) + list(c_hi), (8, ), Float32)


def mma_bf16xbf16_f32(a: TensorSSA, b: TensorSSA, c: TensorSSA) -> TensorSSA:
  """Run the packed `mma_f16xf16_f32` contract for bf16 fragments.

  :param a: Packed A fragment with shape `(8,)` and `BFloat16` element type.
  :param b: Packed B fragment with shape `(8,)` and `BFloat16` element type.
  :param c: Packed accumulator fragment with shape `(8,)` and `Float32` element type.
  :returns: Updated packed accumulator fragment with shape `(8,)`.
  """

  a = _to_tensor_ssa(a)
  b = _to_tensor_ssa(b)
  c = _to_tensor_ssa(c)
  a_words = _extract_bitcast_words(a, 4)
  b_words = _extract_bitcast_words(b, 4)
  c_vals = _extract_tensor_values(c, 8, Float32)
  c_lo = _mma_m16n8k16_f32bf16bf16f32_regs(*a_words, b_words[0], b_words[1], *c_vals[:4])
  c_hi = _mma_m16n8k16_f32bf16bf16f32_regs(*a_words, b_words[2], b_words[3], *c_vals[4:])
  return _make_tensor(list(c_lo) + list(c_hi), (8, ), Float32)


def mma_f16_or_bf16_f32(a: TensorSSA, b: TensorSSA, c: TensorSSA) -> TensorSSA:
  """Dispatch to the f16 or bf16 packed MMA helper based on fragment dtype.

  :param a: Packed A fragment with shape `(8,)`.
  :param b: Packed B fragment with shape `(8,)`.
  :param c: Packed accumulator fragment with shape `(8,)` and `Float32` type.
  :returns: Updated packed accumulator fragment with shape `(8,)`.
  """

  if a.element_type == Float16 and b.element_type == Float16:
    return mma_f16xf16_f32(a, b, c)
  if a.element_type == BFloat16 and b.element_type == BFloat16:
    return mma_bf16xbf16_f32(a, b, c)
  raise TypeError("mma_f16_or_bf16_f32 expects matching Float16 or BFloat16 fragments, "
                  f"got {a.element_type} and {b.element_type}.")


__all__ = [
  "extract_f32_fragment_values",
  "int4_mma_opcode",
  "make_f32_accumulator_fragment",
  "mma_m16n8k16_f32f16f16f32",
  "mma_m16n8k16_f32bf16bf16f32",
  "mma_f16xf16_f32",
  "mma_bf16xbf16_f32",
  "mma_f16_or_bf16_f32",
  "packed_half_fragment_from_i32_words",
]
