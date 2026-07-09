from typing import Optional

import torch
import torch.nn.functional as F

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

_SAGE3_NS = "cache_dit_sage3_ops"
_SAGE3_OP_NAME = f"{_SAGE3_NS}::sage3_attention"
_SAGE3_OP_DEFINED = False


def _ensure_sage3_op_defined() -> None:
  """Define the torch.library custom op for SageAttention-3 (idempotent).

  Wrapping ``sageattn3_blackwell`` as a custom :mod:`torch.library` op
  prevents :mod:`torch.compile` / inductor from tracing into the upstream
  Triton kernels (``group_mean_kernel``) and attempting to recompile them
  under dynamo, which fails because inductor imposes additional constraints
  (e.g. ``tl.arange`` range must be a power of 2) that the upstream kernel
  does not satisfy.
  """

  global _SAGE3_OP_DEFINED
  if _SAGE3_OP_DEFINED:
    return
  _SAGE3_OP_DEFINED = True

  torch.library.define(
    _SAGE3_OP_NAME,
    "(Tensor q, Tensor k, Tensor v, bool is_causal) -> Tensor",
  )

  @torch.library.impl(_SAGE3_OP_NAME, "CUDA")
  def _sage3_attention_op_impl(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                               is_causal: bool) -> torch.Tensor:
    # sageattn3_blackwell expects [B, H, N, D]; diffusers convention is [B, N, H, D].
    B, N, H, D = q.shape

    # SageAttention 3 CUDA kernels only support head_dim 64 or 128 (see
    # DISPATCH_HEAD_DIM in dispatch_utils.h), and the Triton
    # group_mean_kernel requires tl.arange(0, D) with D a power of 2.
    # For unsupported head dims, pad with zeros to the nearest supported
    # size following the minimal-pad principle:
    #   D ≤ 64  → pad to 64
    #   D ≤ 128 → pad to 128  (e.g. 120 in Boogu-Image)
    #   D > 128 → fall back to SDPA (no larger supported kernel)
    # Mathematically:
    #   - QK^T dot products unchanged (extra dims contribute 0)
    #   - softmax_scale changes 1/sqrt(D)→1/sqrt(D_pad), at most ~3%
    #     for typical gaps, negligible vs FP4 (E2M1) quantization error
    #   - V padded dims are 0 → output slice back to D is exact.
    _SUPPORTED_HEAD_DIMS = (64, 128)
    if D not in _SUPPORTED_HEAD_DIMS:
      if D <= 64:
        D_pad = 64
      elif D <= 128:
        D_pad = 128
      else:
        return F.scaled_dot_product_attention(
          q,
          k,
          v,
          is_causal=is_causal,
        ).contiguous()
      pad_len = D_pad - D
      q = F.pad(q, (0, pad_len))
      k = F.pad(k, (0, pad_len))
      v = F.pad(v, (0, pad_len))
    else:
      D_pad = D

    out = sageattn3_blackwell(
      q=q.transpose(1, 2).contiguous(),
      k=k.transpose(1, 2).contiguous(),
      v=v.transpose(1, 2).contiguous(),
      is_causal=is_causal,
      per_block_mean=True,
    )
    out = out.transpose(1, 2).contiguous()

    if D != D_pad:
      out = out[:, :, :, :D].contiguous()
    return out

  @torch.library.register_fake(_SAGE3_OP_NAME)
  def _sage3_attention_op_fake(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                               is_causal: bool) -> torch.Tensor:
    return torch.empty_like(q)


def _sage3_attention_op(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                        is_causal: bool) -> torch.Tensor:
  """Call the SageAttention-3 custom op, auto-defining it on first use.

  :param query: ``[B, N, H, D]`` (diffusers / NHD convention).
  :param key: ``[B, N_kv, H, D]``.
  :param value: ``[B, N_kv, H, D]``.
  :param is_causal: Whether to apply causal masking.
  :returns: ``[B, N, H, D]``.
  """

  if sageattn3_blackwell is None:
    raise RuntimeError("SageAttention-3 backend is not available. "
                       "Please install `sageattn3` to use it.")
  _ensure_sage3_op_defined()
  return torch.ops.cache_dit_sage3_ops.sage3_attention(
    query,
    key,
    value,
    is_causal,
  )


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

  # SageAttention-3 required [B,H,N,D] tensor layout for Q/K/V; wrapped as
  # a torch.library custom op so torch.compile treats it as a black box.
  out = _sage3_attention_op(query, key, value, is_causal)
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
    # Wrapped as torch.library custom op for torch.compile compatibility.
    out = _sage3_attention_op(query, key, value, is_causal)
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
