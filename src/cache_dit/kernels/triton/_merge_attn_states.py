import triton
import triton.language as tl


@triton.jit
def _fused_merge_attn_states_kernel(
  out_ptr: tl.tensor,  # [B*N, H, D],
  lse_ptr: tl.tensor,  # [B*N, H],
  prev_out_ptr: tl.tensor,  # [B*N, H, D],
  prev_lse_ptr: tl.tensor,  # [B*N, H],
  suff_out_ptr: tl.tensor,  # [B*N, H, D],
  suff_lse_ptr: tl.tensor,  # [B*N, H],
  HEAD_SIZE: tl.constexpr,
  PADDED_HEAD_SIZE: tl.constexpr,
):
  token_idx = tl.program_id(0)
  head_idx = tl.program_id(1)
  num_heads = tl.num_programs(1)

  # NOTE(DefTruth): Use float32 for numerical stability
  prev_lse = tl.load(prev_lse_ptr + token_idx * num_heads + head_idx).to(tl.float32)
  suff_lse = tl.load(suff_lse_ptr + token_idx * num_heads + head_idx).to(tl.float32)
  prev_lse = float("-inf") if prev_lse == float("inf") else prev_lse
  suff_lse = float("-inf") if suff_lse == float("inf") else suff_lse

  head_arange = tl.arange(0, PADDED_HEAD_SIZE)
  head_mask = head_arange < HEAD_SIZE
  prev_out = tl.load(
    prev_out_ptr + token_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE + head_arange,
    mask=head_mask,
  ).to(tl.float32)
  suff_out = tl.load(
    suff_out_ptr + token_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE + head_arange,
    mask=head_mask,
  ).to(tl.float32)

  # compute: out = prev_out - F.sigmoid(lse - prev_lse) * (prev_out - out)
  out = prev_out - tl.sigmoid(suff_lse - prev_lse) * (prev_out - suff_out)
  out = out.to(out_ptr.dtype.element_ty)
  tl.store(
    out_ptr + token_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE + head_arange,
    out,
    mask=head_mask,
  )

  # compute: lse = prev_lse - F.logsigmoid(prev_lse - lse)
  lse = prev_lse - tl.log(tl.sigmoid(prev_lse - suff_lse))  # type: tl.tensor
  lse = lse.to(lse_ptr.dtype.element_ty)
  tl.store(lse_ptr + token_idx * num_heads + head_idx, lse)
