import torch
from typing import Tuple

from ..cuda._svdquant import _decode_svdq_output_dtype
from ._float8_comm import fp8_comm_per_token_dequant as _cutedsl_fp8_comm_per_token_dequant
from ._float8_comm import fp8_comm_per_token_quant as _cutedsl_fp8_comm_per_token_quant
from ._float8_comm import fp8_comm_qkv_permute_dequant as _cutedsl_fp8_comm_qkv_permute_dequant
from ._float8_comm import fp8_comm_qkv_permute_quant as _cutedsl_fp8_comm_qkv_permute_quant
from ._merge_attn_states import fused_merge_attn_states as _cutedsl_fused_merge_attn_states
from .svdquant import svdq_gemm_w4a4_v2 as _cutedsl_svdq_gemm_w4a4_v2
from .svdquant import svdq_quantize_w4a4_act_fuse_lora as _cutedsl_svdq_quantize_w4a4_act_fuse_lora
from .svdquant.gemm_utils import normalize_runtime_stage as _normalize_cutedsl_svdq_stage

# FP8 related ops
torch.library.define(
  "cache_dit_cutedsl_ops::fp8_comm_per_token_quant",
  "(Tensor x) -> Tensor",
)
torch.library.define(
  "cache_dit_cutedsl_ops::fp8_comm_per_token_dequant",
  "(Tensor x) -> Tensor",
)
torch.library.define(
  "cache_dit_cutedsl_ops::fp8_comm_qkv_permute_quant",
  "(Tensor x) -> Tensor",
)
torch.library.define(
  "cache_dit_cutedsl_ops::fp8_comm_qkv_permute_dequant",
  "(Tensor quant_x) -> Tensor",
)

# Attention related ops
torch.library.define(
  "cache_dit_cutedsl_ops::fused_merge_attn_states",
  "(Tensor prev_out, Tensor prev_lse, Tensor suff_out, Tensor suff_lse) "
  "-> (Tensor out, Tensor lse)",
)
torch.library.define(
  "cache_dit_cutedsl_ops::svdq_quantize_w4a4_act_fuse_lora_v3",
  "(Tensor input, Tensor? lora_down, Tensor? smooth, bool fuse_glu, bool fp4, int pad_size) "
  "-> (Tensor output, Tensor oscales, Tensor lora_act_out)",
)
torch.library.define(
  "cache_dit_cutedsl_ops::svdq_gemm_w4a4_v2_v3",
  "(Tensor act, Tensor wgt, Tensor ascales, Tensor wscales, Tensor? lora_act_in, Tensor? lora_up, Tensor? bias, bool fp4, float alpha, Tensor? wcscales, bool act_unsigned, int out_dtype_id, int stage) -> Tensor",
)


@torch.library.impl("cache_dit_cutedsl_ops::fp8_comm_per_token_quant", "CUDA")
def _fp8_comm_per_token_quant(x: torch.Tensor) -> torch.Tensor:
  return _cutedsl_fp8_comm_per_token_quant(x)


@torch.library.register_fake("cache_dit_cutedsl_ops::fp8_comm_per_token_quant")
def _fake_fp8_comm_per_token_quant(x: torch.Tensor) -> torch.Tensor:
  assert x.dtype == torch.bfloat16
  *shape, head_size = x.shape
  return x.new_empty((*shape, head_size + 2), dtype=torch.float8_e4m3fn)


@torch.library.impl("cache_dit_cutedsl_ops::fp8_comm_per_token_dequant", "CUDA")
def _fp8_comm_per_token_dequant(x: torch.Tensor) -> torch.Tensor:
  return _cutedsl_fp8_comm_per_token_dequant(x)


@torch.library.register_fake("cache_dit_cutedsl_ops::fp8_comm_per_token_dequant")
def _fake_fp8_comm_per_token_dequant(x: torch.Tensor) -> torch.Tensor:
  assert x.dtype == torch.float8_e4m3fn
  *shape, packed_head_size = x.shape
  return x.new_empty((*shape, packed_head_size - 2), dtype=torch.bfloat16)


@torch.library.impl("cache_dit_cutedsl_ops::fp8_comm_qkv_permute_quant", "CUDA")
def _fp8_comm_qkv_permute_quant(x: torch.Tensor) -> torch.Tensor:
  return _cutedsl_fp8_comm_qkv_permute_quant(x)


@torch.library.register_fake("cache_dit_cutedsl_ops::fp8_comm_qkv_permute_quant")
def _fake_fp8_comm_qkv_permute_quant(x: torch.Tensor) -> torch.Tensor:
  batch, seq_len, partitions, num_heads, head_size = x.shape
  return x.new_empty((partitions, seq_len, batch, num_heads, head_size + 4),
                     dtype=torch.float8_e4m3fn)


@torch.library.impl("cache_dit_cutedsl_ops::fp8_comm_qkv_permute_dequant", "CUDA")
def _fp8_comm_qkv_permute_dequant(quant_x: torch.Tensor) -> torch.Tensor:
  return _cutedsl_fp8_comm_qkv_permute_dequant(quant_x)


@torch.library.register_fake("cache_dit_cutedsl_ops::fp8_comm_qkv_permute_dequant")
def _fake_fp8_comm_qkv_permute_dequant(quant_x: torch.Tensor) -> torch.Tensor:
  seq_len, batch, num_heads, packed_head_size = quant_x.shape
  return quant_x.new_empty((batch, seq_len, num_heads, packed_head_size - 4), dtype=torch.bfloat16)


@torch.library.impl("cache_dit_cutedsl_ops::fused_merge_attn_states", "CUDA")
def _fused_merge_attn_states(
  prev_out: torch.Tensor,
  prev_lse: torch.Tensor,
  suff_out: torch.Tensor,
  suff_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
  return _cutedsl_fused_merge_attn_states(
    prev_out=prev_out,
    prev_lse=prev_lse,
    suff_out=suff_out,
    suff_lse=suff_lse,
  )


@torch.library.register_fake("cache_dit_cutedsl_ops::fused_merge_attn_states")
def _fake_fused_merge_attn_states(
  prev_out: torch.Tensor,
  prev_lse: torch.Tensor,
  suff_out: torch.Tensor,
  suff_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
  batch, seq_len, num_heads, head_size = suff_out.shape

  suff_out = suff_out.flatten(0, 1).contiguous().view(-1, head_size)
  suff_lse = suff_lse.flatten(0, 1).squeeze(-1).contiguous().view(-1)
  out = suff_out.new_empty(suff_out.shape, dtype=suff_out.dtype)
  lse = suff_lse.new_empty(suff_lse.shape, dtype=suff_lse.dtype)
  out = out.view(batch, seq_len, num_heads, head_size)
  lse = lse.view(batch, seq_len, num_heads, 1)
  return out, lse


@torch.library.impl("cache_dit_cutedsl_ops::svdq_quantize_w4a4_act_fuse_lora_v3", "CUDA")
def _svdq_quantize_w4a4_act_fuse_lora_v3(
  input: torch.Tensor,
  lora_down: torch.Tensor | None,
  smooth: torch.Tensor | None,
  fuse_glu: bool,
  fp4: bool,
  pad_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  return _cutedsl_svdq_quantize_w4a4_act_fuse_lora(
    input=input,
    lora_down=lora_down,
    smooth=smooth,
    fuse_glu=fuse_glu,
    fp4=fp4,
    pad_size=pad_size,
  )


@torch.library.register_fake("cache_dit_cutedsl_ops::svdq_quantize_w4a4_act_fuse_lora_v3")
def _fake_svdq_quantize_w4a4_act_fuse_lora_v3(
  input: torch.Tensor,
  lora_down: torch.Tensor | None,
  smooth: torch.Tensor | None,
  fuse_glu: bool,
  fp4: bool,
  pad_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  del smooth, fuse_glu
  batch_size, channels = input.shape
  batch_size_pad = ((batch_size + pad_size - 1) // pad_size) * pad_size
  rank = 0 if lora_down is None else lora_down.shape[1]
  output = input.new_empty((batch_size_pad, channels // 2), dtype=torch.uint8)
  oscales = input.new_empty(
    (channels // (16 if fp4 else 64), batch_size_pad),
    dtype=torch.float8_e4m3fn if fp4 else input.dtype,
  )
  lora_act_out = input.new_empty((batch_size_pad, rank), dtype=torch.float32)
  return output, oscales, lora_act_out


@torch.library.impl("cache_dit_cutedsl_ops::svdq_gemm_w4a4_v2_v3", "CUDA")
def _svdq_gemm_w4a4_v2_v3(
  act: torch.Tensor,
  wgt: torch.Tensor,
  ascales: torch.Tensor,
  wscales: torch.Tensor,
  lora_act_in: torch.Tensor | None,
  lora_up: torch.Tensor | None,
  bias: torch.Tensor | None,
  fp4: bool,
  alpha: float,
  wcscales: torch.Tensor | None,
  act_unsigned: bool,
  out_dtype_id: int,
  stage: int,
) -> torch.Tensor:
  return _cutedsl_svdq_gemm_w4a4_v2(
    act=act,
    wgt=wgt,
    ascales=ascales,
    wscales=wscales,
    lora_act_in=lora_act_in,
    lora_up=lora_up,
    bias=bias,
    fp4=fp4,
    alpha=alpha,
    wcscales=wcscales,
    act_unsigned=act_unsigned,
    output_dtype=_decode_svdq_output_dtype(out_dtype_id),
    stage=stage,
  )


@torch.library.register_fake("cache_dit_cutedsl_ops::svdq_gemm_w4a4_v2_v3")
def _fake_svdq_gemm_w4a4_v2_v3(
  act: torch.Tensor,
  wgt: torch.Tensor,
  ascales: torch.Tensor,
  wscales: torch.Tensor,
  lora_act_in: torch.Tensor | None,
  lora_up: torch.Tensor | None,
  bias: torch.Tensor | None,
  fp4: bool,
  alpha: float,
  wcscales: torch.Tensor | None,
  act_unsigned: bool,
  out_dtype_id: int,
  stage: int,
) -> torch.Tensor:
  del ascales, wscales, lora_act_in, lora_up, bias, fp4, alpha, wcscales, act_unsigned
  _normalize_cutedsl_svdq_stage(stage, "svdq_gemm_w4a4_v2_v3")
  output_dtype = _decode_svdq_output_dtype(out_dtype_id)
  return act.new_empty((act.shape[0], wgt.shape[0]), dtype=output_dtype)


def fused_merge_attn_states(
  prev_out: torch.Tensor,
  prev_lse: torch.Tensor,
  suff_out: torch.Tensor,
  suff_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Dispatch the CuTe DSL fused attention-state merge operator.

  :param prev_out: Previous attention output `[B, N, H, D]`.
  :param prev_lse: Previous attention LSE `[B, N, H, 1]`.
  :param suff_out: Suffix attention output `[B, N, H, D]`.
  :param suff_lse: Suffix attention LSE `[B, N, H, 1]`.
  :returns: Tuple `(out, lse)`.
  """

  return torch.ops.cache_dit_cutedsl_ops.fused_merge_attn_states(
    prev_out,
    prev_lse,
    suff_out,
    suff_lse,
  )


def fp8_comm_per_token_quant(x: torch.Tensor) -> torch.Tensor:
  return torch.ops.cache_dit_cutedsl_ops.fp8_comm_per_token_quant(x)


def fp8_comm_per_token_dequant(x: torch.Tensor) -> torch.Tensor:
  return torch.ops.cache_dit_cutedsl_ops.fp8_comm_per_token_dequant(x)


def fp8_comm_qkv_permute_quant(x: torch.Tensor) -> torch.Tensor:
  return torch.ops.cache_dit_cutedsl_ops.fp8_comm_qkv_permute_quant(x)


def fp8_comm_qkv_permute_dequant(quant_x: torch.Tensor) -> torch.Tensor:
  return torch.ops.cache_dit_cutedsl_ops.fp8_comm_qkv_permute_dequant(quant_x)


def svdq_quantize_w4a4_act_fuse_lora_v3(
  input: torch.Tensor,
  lora_down: torch.Tensor | None = None,
  smooth: torch.Tensor | None = None,
  fuse_glu: bool = False,
  fp4: bool = False,
  pad_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  return torch.ops.cache_dit_cutedsl_ops.svdq_quantize_w4a4_act_fuse_lora_v3(
    input,
    lora_down,
    smooth,
    fuse_glu,
    fp4,
    pad_size,
  )


def svdq_gemm_w4a4_v2_v3(
  act: torch.Tensor,
  wgt: torch.Tensor,
  ascales: torch.Tensor,
  wscales: torch.Tensor,
  lora_act_in: torch.Tensor | None,
  lora_up: torch.Tensor | None,
  bias: torch.Tensor | None,
  fp4: bool,
  alpha: float | None,
  wcscales: torch.Tensor | None,
  act_unsigned: bool,
  output_dtype: torch.dtype | None = None,
  stage: int = 1,
) -> torch.Tensor:
  if alpha is None:
    alpha = 1.0
  if output_dtype is None:
    output_dtype = lora_up.dtype if lora_up is not None else (
      bias.dtype if bias is not None else wscales.dtype)
  out_dtype_id = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}.get(output_dtype)
  if out_dtype_id is None:
    raise ValueError(f"Unsupported SVDQ output dtype for v3: {output_dtype}.")
  normalized_stage = _normalize_cutedsl_svdq_stage(stage, "svdq_gemm_w4a4_v2_v3")
  return torch.ops.cache_dit_cutedsl_ops.svdq_gemm_w4a4_v2_v3(
    act,
    wgt,
    ascales,
    wscales,
    lora_act_in,
    lora_up,
    bias,
    fp4,
    alpha,
    wcscales,
    act_unsigned,
    out_dtype_id,
    normalized_stage,
  )


__all__ = [
  "fp8_comm_per_token_quant",
  "fp8_comm_per_token_dequant",
  "fp8_comm_qkv_permute_quant",
  "fp8_comm_qkv_permute_dequant",
  "fused_merge_attn_states",
  "svdq_quantize_w4a4_act_fuse_lora_v3",
  "svdq_gemm_w4a4_v2_v3",
]
