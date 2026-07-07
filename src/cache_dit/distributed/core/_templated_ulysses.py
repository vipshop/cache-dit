from typing import Optional

import torch
import torch.distributed as dist

from ._distributed_primitives import _All2AllComm, _gather_size
from ._modeling_parallel import _ContextParallelConfig

__all__ = [
  "UlyssesAttention",
]


def _all_gather_tensor_dim(
  x: torch.Tensor,
  dim: int,
  group: dist.ProcessGroup,
) -> torch.Tensor:
  x = x.contiguous()
  shape = list(x.shape)
  dim_sizes = _gather_size(shape[dim], group)
  gathered = []
  for dim_size in dim_sizes:
    current_shape = list(shape)
    current_shape[dim] = dim_size
    gathered.append(torch.empty(current_shape, device=x.device, dtype=x.dtype))
  dist.all_gather(gathered, x, group=group)
  return torch.cat(gathered, dim=dim).contiguous()


def _local_head_indices(
  num_heads: int,
  group: dist.ProcessGroup,
  device: torch.device,
) -> torch.Tensor:
  rank = dist.get_rank(group=group)
  world_size = dist.get_world_size(group=group)
  local_heads = (num_heads + world_size - 1) // world_size
  start = rank * local_heads
  end = min(start + local_heads, num_heads)
  return torch.arange(start, end, device=device, dtype=torch.long)


def _repeat_replicated_kv_for_local_q(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  num_q_heads: int,
  group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor]:
  num_kv_heads = key.shape[2]
  if num_kv_heads == query.shape[2]:
    return key, value
  if num_q_heads % num_kv_heads != 0:
    raise ValueError(f"GQA requires num_q_heads to be divisible by num_kv_heads, got "
                     f"{num_q_heads} and {num_kv_heads}.")

  q_head_indices = _local_head_indices(num_q_heads, group, key.device)
  if q_head_indices.numel() != query.shape[2]:
    raise ValueError(f"Local Q head count mismatch after Ulysses all-to-all: expected "
                     f"{q_head_indices.numel()}, got {query.shape[2]}.")

  repeat_factor = num_q_heads // num_kv_heads
  kv_head_indices = torch.div(q_head_indices, repeat_factor, rounding_mode="floor")
  key = key.index_select(2, kv_head_indices).contiguous()
  value = value.index_select(2, kv_head_indices).contiguous()
  return key, value


class UlyssesAttention(torch.autograd.Function):
  """Ulysses attention with cache-dit's async all-to-all kernels."""

  @staticmethod
  def forward(
    ctx: torch.autograd.function.FunctionCtx,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    dropout_p: float,
    is_causal: bool,
    scale: Optional[float],
    enable_gqa: bool,
    return_lse: bool,
    cp_gqa_strategy: Optional[str],
    forward_op,
    backward_op,
    _cp_config: Optional["_ContextParallelConfig"] = None,
  ):
    if _cp_config is None:
      raise ValueError("Context parallel config must be provided for Ulysses attention.")

    ctx.forward_op = forward_op
    ctx.backward_op = backward_op
    ctx._cp_config = _cp_config

    comm = _All2AllComm(_cp_config)
    num_q_heads = query.shape[2]

    # Keep K/LSE on the non-fp8 path for better numerical stability.
    query_wait = comm.send_q(query)
    query = query_wait.wait()  # type: torch.Tensor

    if cp_gqa_strategy == "replicate_kv_sequence":
      key = _all_gather_tensor_dim(key, dim=1, group=comm.group)
      value = _all_gather_tensor_dim(value, dim=1, group=comm.group)
      key, value = _repeat_replicated_kv_for_local_q(query, key, value, num_q_heads, comm.group)
      enable_gqa = False
    else:
      key_wait = comm.send_k(key)
      value_wait = comm.send_v(value)
      key = key_wait.wait()  # type: torch.Tensor
      value = value_wait.wait()  # type: torch.Tensor

    out = forward_op(
      ctx,
      query,
      key,
      value,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      enable_gqa,
      return_lse,
      _save_ctx=False,
      _cp_config=_cp_config,
    )
    if return_lse:
      out, lse, *_ = out

    out_wait = comm.send_o(out)

    if return_lse:
      lse = lse.unsqueeze(-1)  # (B, S_Q_GLOBAL, H_LOCAL, D=1)
      lse_wait = comm.send_lse(lse)
      out = out_wait.wait()  # type: torch.Tensor
      lse = lse_wait.wait()  # type: torch.Tensor
      lse = lse.squeeze(-1).contiguous()  # (B, S_Q_LOCAL, H_GLOBAL)
    else:
      out = out_wait.wait()  # type: torch.Tensor
      lse = None

    return (out, lse) if return_lse else out

  @staticmethod
  def backward(
    ctx: torch.autograd.function.FunctionCtx,
    grad_out: torch.Tensor,
    *args,
  ):
    raise NotImplementedError(
      "Backward pass for Ulysses Attention in cache-dit is not implemented yet.")
