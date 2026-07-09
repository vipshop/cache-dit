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

  # SageAttention-3 required [B,H,N,D] tensor layout for Q/K/V.
  out = sageattn3_blackwell(
    q=query.transpose(1, 2).contiguous(),  # [B,N,H,D] -> [B,H,N,D]
    k=key.transpose(1, 2).contiguous(),
    v=value.transpose(1, 2).contiguous(),
    is_causal=is_causal,
  )
  out = out.transpose(1, 2).contiguous()
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
                                ║ quantize()   ║  # NOTE: P (f32, [0,1]) → E2M1
                                ║ P → FP4      ║  # Only {0, 0.5, 1} usable in [0,1]
                                ╚══════╤═══════╝
         P_fp4 x V_fp4  ──MMA──→  O (f32)         # NOTE: PV based on FP4 P

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
    # SageAttention-3 required [B,H,N,D] tensor layout for Q/K/V.
    out = sageattn3_blackwell(
      q=query.transpose(1, 2).contiguous(),  # [B,N,H,D] -> [B,H,N,D]
      k=key.transpose(1, 2).contiguous(),
      v=value.transpose(1, 2).contiguous(),
      is_causal=is_causal,
    )
    out = out.transpose(1, 2).contiguous()
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
