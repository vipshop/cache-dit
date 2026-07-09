from typing import Optional

import torch

from .register import (
  _AttnBackend,
  _AttnBackendRegistry,
  _context_parallel_attention,
  _ContextParallelConfig,
)

try:
  from sageattn3 import sageattn3_blackwell
except Exception:
  sageattn3_blackwell = None


def _sage3_attention_forward_op(
  ctx: torch.autograd.function.FunctionCtx,
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  attn_mask: Optional[torch.Tensor] = None,
  dropout_p: float = 0.0,
  is_causal: bool = False,
  scale: Optional[float] = None,
  enable_gqa: bool = False,
  return_lse: bool = False,
  _save_ctx: bool = True,
  _cp_config: Optional["_ContextParallelConfig"] = None,
):
  if attn_mask is not None:
    raise ValueError("`attn_mask` is not yet supported for SageAttention-3.")
  if dropout_p > 0.0:
    raise ValueError("`dropout_p` is not yet supported for SageAttention-3.")
  if enable_gqa:
    raise ValueError("`enable_gqa` is not yet supported for SageAttention-3.")
  if return_lse:
    raise ValueError("`return_lse` is not supported for SageAttention-3.")
  if sageattn3_blackwell is None:
    raise RuntimeError("SageAttention-3 backend is not available. "
                       "Please install `sageattn3` to use it.")

  out = sageattn3_blackwell(
    q=query.clone(),
    k=key.clone(),
    v=value.clone(),
    is_causal=is_causal,
  )
  return out


def _sage3_attention_backward_op(
  ctx: torch.autograd.function.FunctionCtx,
  grad_out: torch.Tensor,
  *args,
):
  raise NotImplementedError("Backward pass is not implemented for SageAttention-3.")


@_AttnBackendRegistry.register(
  _AttnBackend.SAGE3,
  constraints=[],
  supports_context_parallel=True,
)
def _sage3_attention(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  attn_mask: Optional[torch.Tensor] = None,
  dropout_p: float = 0.0,
  is_causal: bool = False,
  scale: Optional[float] = None,
  enable_gqa: bool = False,
  return_lse: bool = False,
  cp_gqa_strategy: Optional[str] = None,
  _cp_config: Optional["_ContextParallelConfig"] = None,
) -> torch.Tensor:
  """SageAttention-3 Blackwell FP4 backend.

  .. warning::
     **Significant precision limitation** — not recommended for general use
     as a drop-in attention replacement, especially at low sequence lengths
     (N < 4096) or when Q and K come from independent projection matrices.

     Root cause analysis of ``sageattn3_blackwell`` internal data flow::

         Q_fp4 x K_fp4  ──MMA──→  S (f32)         # OK: float32 accumulator
           + delta_s (f32)   →  S (f32)           # OK: exact delta correction
           → softmax         →  P (f32)           # OK: fine-grained attn weights
                                ╔══════════════╗
                                ║ quantize()   ║  # BUG: P (f32, [0,1]) → E2M1
                                ║ P → FP4      ║  # Only {0, 0.5, 1} usable in [0,1]
                                ╚══════╤═══════╝
         P_fp4 x V_fp4  ──MMA──→  O (f32)         # BROKEN: PV based on destroyed P

     The E2M1 FP4 format only provides 2 non-zero levels ({0.5, 1}) in the
     [0, 1] softmax range, effectively collapsing continuous attention
     weights into 3 buckets: {ignore, partial, full}.  PV MMA operates on
     this coarsely quantized P, producing output uncorrelated with standard
     attention (cos_sim ≈ 0.03 vs SDPA).  FP8 (E4M3) P achieves cos_sim ≈
     0.996 in simulation — a 30x precision improvement.

     The SAGE3 README recommends alternating SageAttention2++ and
     SageAttention3 across layers/timesteps, suggesting the authors are
     aware of this limitation.  For cache-dit the preferred path is
     ``--attn sage`` (SAGE2, lossless) rather than ``--attn sage3`` until
     the upstream kernel is fixed to keep P at FP16 or FP8 precision.

  .. warning::
     **Fixing this requires a kernel-level refactor, not a simple patch.**

     The entire kernel data pipeline is deeply coupled around FP4 (E2M1)
     block-scaled MMA.  Upgrading P/V from FP4 to FP8 (E4M3) touches the
     full chain from Python quantization down to PTX MMA instructions.
     Estimated scope: **12-15 files, 1-2 weeks**.

     Full change chain (Python → CUDA → PTX)::

         Python: v → scale_and_quant_fp4_transpose(v) → v_fp4 (uint8, D×N/2)
                                                       → sfv (fp8_e4m3, D×N/16)
         Python: v → scale_and_quant_fp8_transpose(v) → v_fp8 (uint8, D×N/4)  ← new kernel
                                                       → sfv (not needed)

         api.cu:  params.v_ptr, params.sfv_ptr         → delete sfv_ptr
                  params.v_row_stride                  → update (FP8 stride vs FP4)

         mainloop_tma_ws.h:
                  SmemLayoutVt    ← E2M1 smem selector → E4M3 smem selector
                  SmemLayoutSFVt  ← block-scale SF     → delete
                  TMA load V      ← E2M1 element type  → E4M3 element type
                  TMA load SFV    ← block-scale SF     → delete
                  copy_v_block()  ← LDSM for E2M1      → LDSM for E4M3
                  tOrVt           ← FP4 reg tensor     → FP8 reg tensor
                  tOrSFVt         ← FP4 scale factors  → delete
                  (P side also:)
                  quantize()      ← P(f32)→E2M1+UE4M3  → P(f32)→E4M3 (no SF)
                  tOrP            ← FP4 reg tensor     → FP8 reg tensor
                  tOrSFP          ← FP4 scale factors  → delete
                  TiledMmaPV      ← SM120 block-scaled FP4 → SM80 FP8 MMA

         kernel_traits.h:
                  LayoutP         ← FP4 register layout → FP8 register layout
                  LayoutSFP       ← SF layout           → delete
                  SmemLayoutSFV*  ← SF deduced layouts  → delete
                  NumSFPV         ← kBlockN/16          → delete

         blockscaled_layout.h:
                  SmemLayoutAtomSFV/Vt ← deduced       → delete

         softmax_fused.h:
                  AbsMaxP         ← per-block tracking  → delete (FP8 no block-scale)
                  fp8_scalexfp4_* ← SF scale constants  → delete
                  fp4_scale_log2  ← E2M1 offset         → delete

         params.h:
                  sfv_ptr, sfv_*_stride                 → delete

         launch.h / static_switch.h:
                  run_mha_fwd_ template args            → remove SFV params

         fp4_quantization_4d.cu:
                  +scaled_fp8_quant_trans kernel        → cvt e4m3, transpose, pack

         api.py:
                  scale_and_quant_fp4_transpose         → scale_and_quant_fp8_transpose
                  blockscaled_fp4_attn                  → remove sfv arg
  """
  if attn_mask is not None:
    raise ValueError("`attn_mask` is not yet supported for SageAttention-3.")
  if dropout_p > 0.0:
    raise ValueError("`dropout_p` is not yet supported for SageAttention-3.")
  if enable_gqa:
    raise ValueError("`enable_gqa` is not yet supported for SageAttention-3.")
  if return_lse:
    raise ValueError("`return_lse` is not supported for SageAttention-3.")

  if _cp_config is None:
    if sageattn3_blackwell is None:
      raise RuntimeError("SageAttention-3 backend is not available. "
                         "Please install `sageattn3` to use it.")
    out = sageattn3_blackwell(
      q=query.clone(),
      k=key.clone(),
      v=value.clone(),
      is_causal=is_causal,
    )
  else:
    out = _context_parallel_attention(
      query,
      key,
      value,
      None,
      0.0,
      is_causal,
      scale,
      False,
      return_lse,
      cp_gqa_strategy,
      forward_op=_sage3_attention_forward_op,
      backward_op=_sage3_attention_backward_op,
      _cp_config=_cp_config,
    )

  return out
