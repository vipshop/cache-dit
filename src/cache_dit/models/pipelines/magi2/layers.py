"""Checkpoint-compatible MAGI-2 Preview layers (portable PyTorch paths only).

Adapted from ``vllm-omni @ 7f217a1``:

* ``vllm_omni/diffusion/models/magi2/layers/native/{normalization,rope,mhc,mh_moe,attention_ops}.py``
* ``vllm_omni/diffusion/layers/{swiglu7,mhc}.py``

Apache-2.0, Copyright (c) 2025-2026 SandAI. Intentional differences:

* no Triton, no ``CustomOp`` dispatch layer, no NPU/MUSA variants: cache-dit keeps one portable
  PyTorch implementation per op (see the repository rule that forbids Triton operators here);
* ``swiglu7`` and the mHC leaves are plain functions instead of ``CustomOp`` subclasses;
* the multi-head MoE keeps the upstream reference loop as the correctness oracle and adds a padded
  grouped path (``grouped_mh_moe_forward``) so the 3072 flat experts of a real layer do not cost
  3072 Python-level matmuls per call;
* packed attention prefers ``flash_attn_varlen_func`` with the released learned-sink correction and
  falls back to the chunked reference when the fast kernel cannot run.
"""

import dataclasses
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .parallel import Magi2ParallelGroup, ep_dispatch, ep_undispatch, get_magi2_ep_group

__all__ = [
  "MHCHandler",
  "Magi2MultiHeadMoE",
  "Magi2MultiHeadMoEConfig",
  "MultiModalityRMSNorm",
  "apply_rotary_emb",
  "compute_topk_probs_and_indices",
  "correct_out_lse_with_sink",
  "global_sort_routes",
  "grouped_mh_moe_forward",
  "rotate_half",
  "sinkhorn_knopp",
  "swiglu7",
  "swiglu7_pair",
  "torch_mh_moe_forward",
  "torch_varlen_attention_with_sink",
  "varlen_attention_with_sink",
]

try:  # pragma: no cover - exercised on GPU hosts only
  from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func
except Exception:  # noqa: BLE001 - the reference kernel is the documented fallback
  _flash_attn_varlen_func = None


def swiglu7(x: torch.Tensor, alpha: float = 1.702, limit: float = 7.0) -> torch.Tensor:
  """Clamped SwiGLU7 over packed ``[..., gate_0, up_0, gate_1, up_1, ...]`` columns.

  :param x: Tensor whose last dimension interleaves gate and up channels.
  :param alpha: Sigmoid slope of the released activation.
  :param limit: Gate upper clamp and linear two-sided clamp.
  :returns: Tensor whose last dimension is halved, in the input dtype.
  """

  out_dtype = x.dtype
  values = x.to(torch.float32)
  gate, linear = values[..., ::2], values[..., 1::2]
  gate = gate.clamp(max=limit)
  linear = linear.clamp(min=-limit, max=limit)
  return (gate * torch.sigmoid(alpha * gate) * (linear + 1.0)).to(out_dtype)


def swiglu7_pair(gate: torch.Tensor, up: torch.Tensor, limit: float = 7.0) -> torch.Tensor:
  """Clamped SwiGLU7 from separately projected gate/up tensors.

  :param gate: Gate projection output.
  :param up: Up projection output.
  :param limit: Gate upper clamp and up two-sided clamp.
  :returns: The activated tensor in the gate dtype.
  """

  dtype = gate.dtype
  gate = gate.float().clamp(max=limit)
  up = up.float().clamp(min=-limit, max=limit)
  return (gate * torch.sigmoid(1.702 * gate) * (up + 1.0)).to(dtype)


class MultiModalityRMSNorm(nn.Module):
  """RMSNorm with independent modality and mHC-stream scales.

  The released weight is a ``+1`` offset scale (``weight`` initialized to zeros), and a multimodal
  layer carries one weight per modality.

  :param dim: Normalized feature width.
  :param eps: Numerical epsilon.
  :param num_modality: Number of per-modality weight groups.
  :param num_patterns: Number of pattern groups inside one modality weight.
  :param out_dtype: Forced output dtype; ``None`` keeps the input dtype.
  :param device: Parameter device.
  """

  def __init__(
    self,
    dim: int,
    *,
    eps: float = 1e-6,
    num_modality: int = 1,
    num_patterns: int = 1,
    out_dtype=None,
    device=None,
  ) -> None:
    super().__init__()
    self.dim = dim
    self.eps = eps
    self.num_modality = num_modality
    self.num_patterns = num_patterns
    self.out_dtype = out_dtype
    self.weight = nn.Parameter(
      torch.zeros(num_patterns * dim * num_modality, dtype=torch.float32, device=device))

  def forward(self, tensor: torch.Tensor, modality_dispatcher=None) -> torch.Tensor:
    """Normalize ``tensor`` over its trailing features.

    :param tensor: ``[tokens, dim]`` or ``[tokens, patterns, dim]`` tensor.
    :param modality_dispatcher: Required when ``num_modality > 1``.
    :returns: The normalized tensor.
    """

    original_dtype = tensor.dtype
    split = tensor.shape[-1] != self.dim
    if split and (tensor.ndim < 2 or tensor.shape[-2] * tensor.shape[-1] != self.dim
                  or self.num_patterns != 1):
      raise ValueError(f"cannot normalize a {tuple(tensor.shape)} tensor over {self.dim} features")
    feature_shape = tuple(tensor.shape[-2:]) if split else (self.num_patterns, self.dim)
    normalized = tensor.float()
    reduce_dims = (-2, -1) if split else -1
    normalized = normalized * torch.rsqrt(normalized.square().mean(dim=reduce_dims, keepdim=True) +
                                          self.eps)
    if self.num_modality == 1:
      weight = self.weight.view(feature_shape) + 1.0
      result = normalized * weight
    else:
      if modality_dispatcher is None:
        raise ValueError("modality_dispatcher is required for multimodal RMSNorm")
      inputs = modality_dispatcher.dispatch(normalized)
      weights = self.weight.view(self.num_modality, *feature_shape)
      result = modality_dispatcher.undispatch(*(part * (weights[index] + 1.0)
                                                for index, part in enumerate(inputs)))
    return result.to(self.out_dtype or original_dtype)


def rotate_half(x: torch.Tensor, interleaved: bool = False) -> torch.Tensor:
  """Rotate the trailing rotary features by half a period.

  :param x: Tensor whose last dimension is the rotary width.
  :param interleaved: Use the even/odd pairing instead of the first/second half pairing.
  :returns: The rotated tensor.
  """

  if not interleaved:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)
  x1, x2 = x[..., ::2], x[..., 1::2]
  return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_emb(x: torch.Tensor,
                     cos: torch.Tensor,
                     sin: torch.Tensor,
                     *,
                     interleaved: bool = False):
  """Apply the released element-wise RoPE layout to ``[..., heads, head_dim]``.

  :param x: Query or key tensor.
  :param cos: Cosine table broadcastable over the head axis.
  :param sin: Sine table broadcastable over the head axis.
  :param interleaved: Use the even/odd pairing instead of the first/second half pairing.
  :returns: The rotated tensor.
  """

  rotary_dim = cos.shape[-1] * 2
  if rotary_dim > x.shape[-1]:
    raise ValueError(f"RoPE dimension {rotary_dim} exceeds head dimension {x.shape[-1]}")
  if interleaved:
    cos = cos.unsqueeze(-2).repeat_interleave(2, dim=-1)
    sin = sin.unsqueeze(-2).repeat_interleave(2, dim=-1)
  else:
    cos = torch.cat((cos, cos), dim=-1).unsqueeze(-2)
    sin = torch.cat((sin, sin), dim=-1).unsqueeze(-2)
  rotated = x[..., :rotary_dim] * cos + rotate_half(x[..., :rotary_dim], interleaved) * sin
  return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)


def sinkhorn_knopp(matrix_logits: torch.Tensor, iterations: int, epsilon: float) -> torch.Tensor:
  """Normalize a logit matrix into a doubly stochastic one.

  :param matrix_logits: ``[tokens, streams, streams]`` logits.
  :param iterations: Sinkhorn-Knopp iterations.
  :param epsilon: Numerical epsilon added to both normalizations.
  :returns: The doubly stochastic matrix.
  """

  matrix = torch.exp(matrix_logits - matrix_logits.amax(dim=(-2, -1), keepdim=True))
  for _ in range(iterations):
    matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + epsilon)
    matrix = matrix / (matrix.sum(dim=-1, keepdim=True) + epsilon)
  return matrix


def _mhc_post_residual(
  post_logits: torch.Tensor,
  residual_logits: torch.Tensor,
  alpha_post: torch.Tensor,
  bias_post: torch.Tensor,
  alpha_residual: torch.Tensor,
  bias_residual: torch.Tensor,
  *,
  scale: float,
  iterations: int,
  epsilon: float,
  out_dtype: torch.dtype,
):
  """Prepare the mHC post coefficients and residual matrix in FP32.

  :param post_logits: ``[tokens, streams]`` post logits.
  :param residual_logits: ``[tokens, streams, streams]`` residual logits.
  :param alpha_post: Scalar gate of the post logits.
  :param bias_post: ``[streams]`` post bias.
  :param alpha_residual: Scalar gate of the residual logits.
  :param bias_residual: ``[streams, streams]`` residual bias.
  :param scale: ``1 / sqrt(streams * hidden)`` normalization of the logits.
  :param iterations: Sinkhorn-Knopp iterations.
  :param epsilon: Sinkhorn epsilon.
  :param out_dtype: Output dtype of both tensors.
  :returns: ``(post_coefficients, residual_matrix)``.
  """

  post = 2.0 * torch.sigmoid(alpha_post * scale * post_logits + bias_post.unsqueeze(0))
  residual = sinkhorn_knopp(
    alpha_residual * scale * residual_logits.float() + bias_residual.unsqueeze(0).float(),
    iterations,
    epsilon,
  )
  return post.to(out_dtype), residual.to(out_dtype)


class MHCHandler:
  """Four-stream manifold-constrained hyper-connection math.

  :param num_streams: Number of residual streams.
  :param hidden_size: Hidden width of one stream.
  :param sinkhorn_iterations: Sinkhorn-Knopp iterations of the residual matrix.
  :param sinkhorn_epsilon: Sinkhorn epsilon.
  :param dtype: Accumulation dtype of the projections.
  """

  def __init__(
    self,
    num_streams: int,
    hidden_size: int,
    *,
    sinkhorn_iterations: int = 20,
    sinkhorn_epsilon: float = 1e-12,
    dtype: torch.dtype = torch.float32,
  ) -> None:
    self.num_streams = num_streams
    self.hidden_size = hidden_size
    self.sinkhorn_iterations = sinkhorn_iterations
    self.sinkhorn_epsilon = sinkhorn_epsilon
    self.dtype = dtype
    self.matmul_scale = 1.0 / math.sqrt(float(num_streams * hidden_size))

  def flatten(self, tensor: torch.Tensor) -> torch.Tensor:
    """Flatten ``[tokens, streams, hidden]`` to ``[tokens, streams * hidden]``.

    :param tensor: Stream-stacked hidden states.
    :returns: The flattened tensor.
    """

    self._check_multi(tensor)
    return tensor.reshape(tensor.shape[0], -1)

  def compute_logits(self, flattened: torch.Tensor, norm, phi_fused: torch.Tensor):
    """Compute the pre/post/residual logits of one branch.

    :param flattened: ``[tokens, streams * hidden]`` stream stack.
    :param norm: Callable normalizing the flattened stack.
    :param phi_fused: ``[streams * hidden, streams + streams + streams**2]`` projection.
    :returns: ``(pre_logits, post_logits, residual_logits)``.
    """

    if flattened.ndim != 2 or flattened.shape[-1] != self.num_streams * self.hidden_size:
      raise ValueError("invalid flattened mHC shape")
    fused = norm(flattened).to(self.dtype) @ phi_fused
    pre, post, residual = torch.split(
      fused,
      (self.num_streams, self.num_streams, self.num_streams ** 2),
      dim=-1,
    )
    return pre, post, residual.view(-1, self.num_streams, self.num_streams)

  def apply_pre(self, streams: torch.Tensor, alpha_bias_logits, *, out_dtype=None) -> torch.Tensor:
    """Mix the streams into one branch input.

    :param streams: ``[tokens, streams, hidden]`` stream stack.
    :param alpha_bias_logits: ``(alpha, bias, pre_logits)``.
    :param out_dtype: Output dtype; defaults to the stream dtype.
    :returns: ``[tokens, hidden]`` branch input.
    """

    self._check_multi(streams)
    alpha, bias, logits = alpha_bias_logits
    coefficients = torch.sigmoid(alpha * self.matmul_scale * logits + bias.unsqueeze(0))
    coefficients = coefficients.to(out_dtype or streams.dtype)
    return torch.einsum("tn,tnc->tc", coefficients, streams)

  def compute_post_residual(self, post, residual, *, out_dtype: torch.dtype):
    """Prepare the post coefficients and residual matrix.

    :param post: ``(alpha_post, bias_post, post_logits)``.
    :param residual: ``(alpha_residual, bias_residual, residual_logits)``.
    :param out_dtype: Output dtype of both tensors.
    :returns: ``(post_coefficients, residual_matrix)``.
    """

    alpha_post, bias_post, post_logits = post
    alpha_residual, bias_residual, residual_logits = residual
    return _mhc_post_residual(
      post_logits,
      residual_logits,
      alpha_post,
      bias_post,
      alpha_residual,
      bias_residual,
      scale=self.matmul_scale,
      iterations=self.sinkhorn_iterations,
      epsilon=self.sinkhorn_epsilon,
      out_dtype=out_dtype,
    )

  def hyper_connect(
    self,
    residual_streams: torch.Tensor,
    branch_output: torch.Tensor,
    post_coefficients: torch.Tensor,
    residual_matrix: torch.Tensor,
  ) -> torch.Tensor:
    """Mix the residual streams and add the gated branch output.

    :param residual_streams: ``[tokens, streams, hidden]`` residual stack.
    :param branch_output: ``[tokens, hidden]`` branch output.
    :param post_coefficients: ``[tokens, streams]`` branch gate.
    :param residual_matrix: ``[tokens, streams, streams]`` stream mixing matrix.
    :returns: The updated ``[tokens, streams, hidden]`` stream stack.
    """

    self._check_multi(residual_streams)
    if branch_output.ndim != 2 or branch_output.shape[-1] != self.hidden_size:
      raise ValueError("invalid mHC branch-output shape")
    branch = torch.einsum("tn,tc->tnc", post_coefficients, branch_output)
    mixed = torch.einsum("tij,tjc->tic", residual_matrix, residual_streams)
    return mixed + branch

  def _check_multi(self, tensor: torch.Tensor) -> None:
    expected = (self.num_streams, self.hidden_size)
    if tensor.ndim != 3 or tensor.shape[1:] != expected:
      raise ValueError(
        f"expected mHC tensor [tokens,{expected[0]},{expected[1]}], got {tuple(tensor.shape)}")


def compute_topk_probs_and_indices(
  router_logits: torch.Tensor,
  top_k: int,
  *,
  score_func: str = "sigmoid",
  expert_bias: torch.Tensor = None,
  route_norm: bool = True,
  norm_eps: float = 1e-12,
):
  """Route independently for every ``[head, token]`` pair.

  The auxiliary-free bias affects expert selection but deliberately does not affect the returned
  routing probability, matching the released training recipe.

  :param router_logits: ``[heads, tokens, experts]`` logits.
  :param top_k: Experts kept per token and head.
  :param score_func: ``sigmoid`` or ``softmax``.
  :param expert_bias: Optional ``[heads, experts]`` selection bias.
  :param route_norm: L1-normalize the kept probabilities.
  :param norm_eps: Normalization epsilon.
  :returns: ``(topk_probs, topk_indices)``, both ``[heads, tokens, top_k]``.
  """

  if router_logits.ndim != 3:
    raise ValueError("router_logits must be [heads,tokens,experts]")
  if not 0 < top_k <= router_logits.shape[-1]:
    raise ValueError("top_k must be in [1, num_experts]")
  if score_func == "sigmoid":
    router_scores = torch.sigmoid(router_logits)
  elif score_func == "softmax":
    router_scores = torch.softmax(router_logits, dim=-1)
  else:
    raise ValueError(f"unsupported routing score function {score_func!r}")
  selection_scores = router_scores
  if expert_bias is not None:
    selection_scores = selection_scores + expert_bias.view(router_logits.shape[0], 1, -1)
  topk_indices = torch.topk(selection_scores, top_k, dim=-1).indices
  topk_probs = router_scores.gather(-1, topk_indices)
  if route_norm:
    topk_probs = F.normalize(topk_probs, p=1, dim=-1, eps=norm_eps)
  return topk_probs, topk_indices


def global_sort_routes(topk_probs: torch.Tensor, topk_indices: torch.Tensor, num_experts: int):
  """Convert per-head routes into a stable flattened-expert CSR layout.

  :param topk_probs: ``[heads, tokens, top_k]`` routing probabilities.
  :param topk_indices: ``[heads, tokens, top_k]`` expert indices.
  :param num_experts: Experts per head.
  :returns: ``(gather_ids, sorted_probs, expert_offsets)``.
  """

  if topk_probs.shape != topk_indices.shape or topk_indices.ndim != 3:
    raise ValueError("top-k probabilities and indices must have the same [H,S,K] shape")
  heads, sequence, top_k = topk_indices.shape
  device = topk_indices.device
  head_offset = torch.arange(heads, device=device).view(heads, 1, 1) * num_experts
  flattened_experts = (topk_indices + head_offset).reshape(-1)
  flat_probs = topk_probs.reshape(-1)
  flat_tokens = torch.arange(sequence, device=device).view(1, sequence,
                                                           1).expand(heads, sequence,
                                                                     top_k).reshape(-1)
  order = flattened_experts.argsort(stable=True)
  gather_ids = flat_tokens[order].to(torch.int32)
  sorted_probs = flat_probs[order].float()
  counts = torch.bincount(flattened_experts, minlength=heads * num_experts)
  offsets = torch.zeros(heads * num_experts + 1, device=device, dtype=torch.long)
  offsets[1:] = counts.cumsum(0)
  return gather_ids, sorted_probs, offsets


def torch_mh_moe_forward(
  x: torch.Tensor,
  gather_ids: torch.Tensor,
  probs: torch.Tensor,
  expert_offsets: torch.Tensor,
  w_gate: torch.Tensor,
  w_up: torch.Tensor,
  w_down: torch.Tensor,
) -> torch.Tensor:
  """Reference multi-head MoE forward: one matmul triple per flat expert.

  :param x: ``[tokens, heads, head_dim]`` MoE input.
  :param gather_ids: Token id of every routed (expert, token) pair, sorted by flat expert.
  :param probs: Routing probability of every routed pair.
  :param expert_offsets: CSR offsets over flat experts.
  :param w_gate: ``[flat_experts, head_dim, expert_inter]`` gate weights.
  :param w_up: ``[flat_experts, head_dim, expert_inter]`` up weights.
  :param w_down: ``[flat_experts, expert_inter, head_dim]`` down weights.
  :returns: ``[tokens, heads, head_dim]`` MoE output.
  """

  if x.ndim != 3:
    raise ValueError("multi-head MoE input must be [tokens,heads,head_dim]")
  output = torch.zeros_like(x)
  experts_per_head = (expert_offsets.numel() - 1) // x.shape[1]
  for flat_expert in range(expert_offsets.numel() - 1):
    begin = int(expert_offsets[flat_expert].item())
    end = int(expert_offsets[flat_expert + 1].item())
    if begin == end:
      continue
    head = flat_expert // experts_per_head
    token_ids = gather_ids[begin:end].long()
    expert_input = x.index_select(0, token_ids)[:, head]
    gate = expert_input @ w_gate[flat_expert]
    up = expert_input @ w_up[flat_expert]
    hidden = swiglu7_pair(gate, up)
    expert_output = hidden @ w_down[flat_expert]
    expert_output = expert_output * probs[begin:end, None].to(expert_output.dtype)
    output[:, head].index_add_(0, token_ids, expert_output)
  return output


def grouped_mh_moe_forward(
  x: torch.Tensor,
  gather_ids: torch.Tensor,
  probs: torch.Tensor,
  expert_offsets: torch.Tensor,
  w_gate: torch.Tensor,
  w_up: torch.Tensor,
  w_down: torch.Tensor,
  *,
  token_budget: int = 2048,
) -> torch.Tensor:
  """Padded grouped multi-head MoE forward.

  Groups consecutive flat experts until ``token_budget`` routed tokens are reached, pads each group
  to its longest expert block and evaluates the three expert projections as batched matmuls. This is
  the numerical equivalent of :func:`torch_mh_moe_forward` with a small constant number of kernel
  launches instead of one triple per expert.

  :param x: ``[tokens, heads, head_dim]`` MoE input.
  :param gather_ids: Token id of every routed pair, sorted by flat expert.
  :param probs: Routing probability of every routed pair.
  :param expert_offsets: CSR offsets over flat experts.
  :param w_gate: ``[flat_experts, head_dim, expert_inter]`` gate weights.
  :param w_up: ``[flat_experts, head_dim, expert_inter]`` up weights.
  :param w_down: ``[flat_experts, expert_inter, head_dim]`` down weights.
  :param token_budget: Routed tokens per padded group.
  :returns: ``[tokens, heads, head_dim]`` MoE output.
  """

  if x.ndim != 3:
    raise ValueError("multi-head MoE input must be [tokens,heads,head_dim]")
  tokens, heads, head_dim = x.shape
  offsets = expert_offsets.tolist()
  flat_experts = len(offsets) - 1
  experts_per_head = flat_experts // heads
  output = torch.zeros_like(x)
  if tokens == 0:
    return output
  flat_x = x.reshape(tokens, heads * head_dim)

  start = 0
  while start < flat_experts:
    stop = start
    routed = 0
    widest = 0
    while stop < flat_experts and (stop == start or routed +
                                   (offsets[stop + 1] - offsets[stop]) <= token_budget):
      count = offsets[stop + 1] - offsets[stop]
      routed += count
      widest = max(widest, count)
      stop += 1
    if routed == 0:
      start = stop
      continue
    group = stop - start
    # [group, widest] gathered token ids, zero-padded; -1 marks padding.
    ids = torch.full((group, widest), -1, dtype=torch.long, device=x.device)
    weights = torch.zeros((group, widest), dtype=torch.float32, device=x.device)
    for index in range(group):
      begin, end = offsets[start + index], offsets[start + index + 1]
      count = end - begin
      if count:
        ids[index, :count] = gather_ids[begin:end].long()
        weights[index, :count] = probs[begin:end]
    head_ids = torch.arange(start, stop, device=x.device) // experts_per_head
    gathered = flat_x.index_select(0,
                                   ids.clamp_min(0).reshape(-1)).view(group, widest, heads,
                                                                      head_dim)
    gathered = gathered.gather(2,
                               head_ids.view(group, 1, 1, 1).expand(group, widest, 1,
                                                                    head_dim)).squeeze(2)
    gate = torch.bmm(gathered, w_gate[start:stop])
    up = torch.bmm(gathered, w_up[start:stop])
    hidden = swiglu7_pair(gate, up)
    expert_output = torch.bmm(hidden, w_down[start:stop])
    expert_output = expert_output * weights.unsqueeze(-1).to(expert_output.dtype)
    for index in range(group):
      count = offsets[start + index + 1] - offsets[start + index]
      if not count:
        continue
      head = int(head_ids[index].item())
      output[:, head].index_add_(0, ids[index, :count], expert_output[index, :count])
    start = stop
  return output


@dataclasses.dataclass(frozen=True)
class Magi2MultiHeadMoEConfig:
  """Multi-head MoE settings consumed by :class:`Magi2MultiHeadMoE`.

  :param hidden_size: Model hidden width.
  :param num_heads: MoE heads.
  :param num_experts: Experts per head.
  :param top_k: Activated experts per token and head.
  :param expert_intermediate_size: Hidden width of one expert.
  :param params_dtype: Expert weight dtype.
  :param score_func: Routing score function.
  :param route_norm: L1-normalize the kept probabilities.
  :param route_scale: Constant scale applied to the routing probabilities.
  """

  hidden_size: int
  num_heads: int
  num_experts: int
  top_k: int
  expert_intermediate_size: int
  params_dtype: torch.dtype
  score_func: str = "sigmoid"
  route_norm: bool = True
  route_scale: float = 1.0


class Magi2MultiHeadMoE(nn.Module):
  """Checkpoint-compatible head-routed experts.

  The released checkpoint stores one flattened ``(head, expert)`` row block per parameter, so the
  head axis is the sharding axis: with a parameter-parallel group each rank owns a contiguous head
  slice and no token dispatch happens.

  :param config: MoE settings.
  :param ep_group: Group sharding the MoE head axis; defaults to the installed group.
  """

  _EP_SHARDED_PARAMETER_NAMES = frozenset(
    {"gate", "W_gate", "W_up", "W_down", "router.expert_bias", "router.expert_bias_ema"})

  def __init__(self,
               config: Magi2MultiHeadMoEConfig,
               *,
               ep_group: Magi2ParallelGroup = None) -> None:
    super().__init__()
    if config.hidden_size % config.num_heads:
      raise ValueError("hidden_size must be divisible by the number of MoE heads")
    self.config = config
    self.num_heads = config.num_heads
    self.num_experts = config.num_experts
    self.top_k = config.top_k
    self.d_head = config.hidden_size // config.num_heads
    self.d_expert = config.expert_intermediate_size
    self.ep_group = ep_group or get_magi2_ep_group()
    self.padded_num_heads = math.ceil(
      self.num_heads / self.ep_group.world_size) * self.ep_group.world_size
    self.local_num_heads = self.padded_num_heads // self.ep_group.world_size
    self.local_flatten_num_experts = self.local_num_heads * self.num_experts
    self.ep_pad_heads = self.padded_num_heads - self.num_heads
    self.local_head_start = self.ep_group.rank * self.local_num_heads
    self.has_real_moe_heads = self.local_head_start < self.num_heads

    self.gate = nn.Parameter(
      torch.empty(self.local_flatten_num_experts, self.d_head, dtype=torch.float32))
    self.W_gate = nn.Parameter(
      torch.empty(self.local_flatten_num_experts,
                  self.d_head,
                  self.d_expert,
                  dtype=config.params_dtype))
    self.W_up = nn.Parameter(
      torch.empty(self.local_flatten_num_experts,
                  self.d_head,
                  self.d_expert,
                  dtype=config.params_dtype))
    self.W_down = nn.Parameter(
      torch.empty(self.local_flatten_num_experts,
                  self.d_expert,
                  self.d_head,
                  dtype=config.params_dtype))
    self.router = nn.Module()
    self.router.expert_bias = nn.Parameter(torch.zeros(self.local_flatten_num_experts,
                                                       dtype=torch.float32),
                                           requires_grad=False)
    self.router.expert_bias_ema = nn.Parameter(torch.zeros(self.local_flatten_num_experts,
                                                           dtype=torch.float32),
                                               requires_grad=False)

    for name in self._EP_SHARDED_PARAMETER_NAMES:
      target = self
      parts = name.split(".")
      for part in parts[:-1]:
        target = getattr(target, part)
      parameter = getattr(target, parts[-1])
      parameter.checkpoint_weight_transform = self.ep_slice

  def ep_slice(self, checkpoint_tensor: torch.Tensor) -> torch.Tensor:
    """Slice flattened ``(head, expert)`` checkpoint rows for this rank.

    :param checkpoint_tensor: Full ``[heads * experts, ...]`` checkpoint tensor.
    :returns: This rank's contiguous head slice, zero-padded when the split is uneven.
    """

    if checkpoint_tensor.shape[0] == self.local_flatten_num_experts:
      return checkpoint_tensor
    start = self.local_head_start * self.num_experts
    end = min(start + self.local_flatten_num_experts, checkpoint_tensor.shape[0])
    if start >= checkpoint_tensor.shape[0]:
      return torch.zeros((self.local_flatten_num_experts, *checkpoint_tensor.shape[1:]),
                         dtype=checkpoint_tensor.dtype,
                         device=checkpoint_tensor.device)
    local = checkpoint_tensor[start:end]
    if local.shape[0] < self.local_flatten_num_experts:
      padding = torch.zeros((self.local_flatten_num_experts - local.shape[0], *local.shape[1:]),
                            dtype=local.dtype,
                            device=local.device)
      local = torch.cat((local, padding), dim=0)
    return local

  def shard_for_tp(self, ep_group: Magi2ParallelGroup) -> None:
    """Re-shard the flattened ``(head, expert)`` weights in place.

    :param ep_group: Group sharding the MoE head axis.
    """

    if ep_group.world_size <= 1:
      return
    if self.ep_group.world_size > 1:
      raise RuntimeError("multi-head MoE is already sharded; re-sharding needs a fresh module")

    self.ep_group = ep_group
    self.padded_num_heads = math.ceil(self.num_heads / ep_group.world_size) * ep_group.world_size
    self.local_num_heads = self.padded_num_heads // ep_group.world_size
    self.local_flatten_num_experts = self.local_num_heads * self.num_experts
    self.ep_pad_heads = self.padded_num_heads - self.num_heads
    self.local_head_start = ep_group.rank * self.local_num_heads
    self.has_real_moe_heads = self.local_head_start < self.num_heads

    start = self.local_head_start * self.num_experts
    end = start + self.local_flatten_num_experts
    for name in self._EP_SHARDED_PARAMETER_NAMES:
      target = self
      parts = name.split(".")
      for part in parts[:-1]:
        target = getattr(target, part)
      parameter = getattr(target, parts[-1])
      rows = parameter.data[start:end]
      if rows.shape[0] < self.local_flatten_num_experts:
        padding = torch.zeros((self.local_flatten_num_experts - rows.shape[0], *rows.shape[1:]),
                              dtype=rows.dtype,
                              device=rows.device)
        rows = torch.cat((rows, padding), dim=0)
      replacement = torch.nn.Parameter(rows.contiguous(), requires_grad=parameter.requires_grad)
      replacement.checkpoint_weight_transform = self.ep_slice
      setattr(target, parts[-1], replacement)

  def _route(self, x_heads: torch.Tensor):
    """Route every ``(head, token)`` pair.

    :param x_heads: ``[tokens, local_heads, head_dim]`` MoE input.
    :returns: ``(scaled_probs, indices)``.
    """

    gate = self.gate.view(self.local_num_heads, self.num_experts, self.d_head).float()
    x_route = x_heads.transpose(0, 1).to(torch.float32, memory_format=torch.contiguous_format)
    logits = torch.einsum("hsd,hed->hse", x_route, gate)
    bias = self.router.expert_bias_ema.view(self.local_num_heads, self.num_experts)
    probs, indices = compute_topk_probs_and_indices(
      logits,
      self.top_k,
      score_func=self.config.score_func,
      expert_bias=bias,
      route_norm=self.config.route_norm,
    )
    return probs * self.config.route_scale, indices

  def _local_forward(self, x_heads: torch.Tensor) -> torch.Tensor:
    """Run the local head shard.

    :param x_heads: ``[tokens, local_heads, head_dim]`` input.
    :returns: ``[tokens, local_heads, head_dim]`` output.
    """

    probabilities, indices = self._route(x_heads)
    gather_ids, sorted_probs, offsets = global_sort_routes(probabilities, indices, self.num_experts)
    if x_heads.device.type == "cpu":
      return torch_mh_moe_forward(x_heads, gather_ids, sorted_probs, offsets, self.W_gate,
                                  self.W_up, self.W_down)
    return grouped_mh_moe_forward(x_heads, gather_ids, sorted_probs, offsets, self.W_gate,
                                  self.W_up, self.W_down)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Run the multi-head MoE.

    :param x:``[tokens, local_heads * head_dim]`` (parameter-parallel) or ``[tokens, heads *
        head_dim]`` (sequence-parallel) input.
    :returns: The MoE output in the input layout.
    """

    if self.ep_group.world_size > 1 and self.ep_group.replicated_sequence:
      local_hidden_size = self.local_num_heads * self.d_head
      if x.shape[-1] != local_hidden_size:
        raise ValueError(
          f"TP-local MAGI MoE input has width {x.shape[-1]}, expected {local_hidden_size}")
      local = x.view(-1, self.local_num_heads, self.d_head)
      output = self._local_forward(local) if self.has_real_moe_heads else torch.zeros_like(local)
      return output.reshape(-1, local_hidden_size)

    x_heads = x.view(-1, self.num_heads, self.d_head)
    if self.ep_pad_heads:
      padding = x_heads.new_zeros((x_heads.shape[0], self.ep_pad_heads, self.d_head))
      x_heads = torch.cat((x_heads, padding), dim=1)
    sequence_split_sizes = None
    if self.ep_group.world_size > 1:
      local_size = torch.tensor([x_heads.shape[0]], dtype=torch.int64, device=x_heads.device)
      gathered_sizes = [torch.empty_like(local_size) for _ in range(self.ep_group.world_size)]
      torch.distributed.all_gather(gathered_sizes, local_size, group=self.ep_group.group)
      sequence_split_sizes = [int(size.item()) for size in gathered_sizes]
      x_heads = ep_dispatch(x_heads, self.ep_group, sequence_split_sizes)
    output = self._local_forward(x_heads) if self.has_real_moe_heads else torch.zeros_like(x_heads)
    if self.ep_group.world_size > 1:
      output = ep_undispatch(output, self.ep_group, sequence_split_sizes)
    if self.ep_pad_heads:
      output = output[:, :self.num_heads]
    return output.reshape(-1, self.num_heads * self.d_head)


def correct_out_lse_with_sink(out: torch.Tensor, lse: torch.Tensor, sink: torch.Tensor = None):
  """Add zero-valued attention sinks to an already-computed softmax.

  ``flash_attn`` returns ``out[tokens, heads, dim]`` and ``lse[heads, tokens]``. A MAGI sink holds
  additional logits ``[num_sink, heads]`` whose value vectors are zero, so only the denominator and
  the LSE change.

  :param out: Attention output.
  :param lse: Log-sum-exp of the attention probabilities.
  :param sink: Optional ``[num_sink, heads]`` learned sink logits.
  :returns: ``(corrected_out, corrected_lse)``.
  """

  if sink is None or sink.numel() == 0:
    return out, lse
  if out.ndim != 3 or lse.ndim != 2 or sink.ndim != 2:
    raise ValueError(f"expected out[T,H,D], lse[H,T], sink[num_sink,H], got {tuple(out.shape)}, "
                     f"{tuple(lse.shape)}, {tuple(sink.shape)}")
  old_lse = lse.float().transpose(0, 1)
  sink_lse = torch.logsumexp(sink.float(), dim=0).unsqueeze(0)
  if old_lse.shape[-1] != sink_lse.shape[-1]:
    raise ValueError("attention sink and FlashAttention head counts differ")
  new_lse = torch.logaddexp(old_lse, sink_lse)
  delta = old_lse - new_lse
  delta = torch.where(torch.isfinite(delta), delta, torch.full_like(delta, -torch.inf))
  corrected = out * torch.exp(delta).unsqueeze(-1).to(out.dtype)
  return corrected, new_lse.transpose(0, 1).contiguous()


def _repeat_kv_heads(tensor: torch.Tensor, query_heads: int) -> torch.Tensor:
  """Repeat grouped key/value heads up to the query head count.

  :param tensor: ``[tokens, kv_heads, head_dim]`` tensor.
  :param query_heads: Target head count.
  :returns: The repeated tensor.
  """

  kv_heads = tensor.shape[1]
  if kv_heads == query_heads:
    return tensor
  if query_heads % kv_heads:
    raise ValueError(f"query heads {query_heads} must be divisible by KV heads {kv_heads}")
  return tensor.repeat_interleave(query_heads // kv_heads, dim=1)


def torch_varlen_attention_with_sink(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  *,
  cu_seqlens_q: torch.Tensor,
  cu_seqlens_k: torch.Tensor,
  softcap: float = -1.0,
  sink: torch.Tensor = None,
  query_chunk_size: int = 512,
) -> torch.Tensor:
  """Reference packed attention with optional softcap and learned sinks.

  :param q: ``[tokens, heads, head_dim]`` queries.
  :param k: ``[tokens, kv_heads, head_dim]`` keys.
  :param v: ``[tokens, kv_heads, head_dim]`` values.
  :param cu_seqlens_q: Cumulative query lengths.
  :param cu_seqlens_k: Cumulative key lengths.
  :param softcap: Logit softcap; ``<= 0`` disables it.
  :param sink: Optional ``[num_sink, heads]`` learned sink logits.
  :param query_chunk_size: Query rows per score tile.
  :returns: ``[tokens, heads, head_dim]`` attention output.
  """

  if type(query_chunk_size) is not int or query_chunk_size <= 0:
    raise ValueError("query_chunk_size must be a positive integer")
  if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
    raise ValueError("packed attention expects q/k/v shaped [tokens,heads,dim]")
  if cu_seqlens_q.numel() != cu_seqlens_k.numel():
    raise ValueError("query and key cumulative-length arrays must contain the same batch count")
  output = torch.empty_like(q)
  scale = q.shape[-1] ** -0.5
  for batch_idx in range(cu_seqlens_q.numel() - 1):
    q_start, q_end = (int(value) for value in cu_seqlens_q[batch_idx:batch_idx + 2].tolist())
    k_start, k_end = (int(value) for value in cu_seqlens_k[batch_idx:batch_idx + 2].tolist())
    q_part = q[q_start:q_end].float()
    k_part = _repeat_kv_heads(k[k_start:k_end], q.shape[1]).float()
    v_part = _repeat_kv_heads(v[k_start:k_end], q.shape[1]).float()
    for offset in range(0, max(1, q_part.shape[0]), query_chunk_size):
      q_chunk = q_part[offset:offset + query_chunk_size]
      scores = torch.einsum("qhd,khd->hqk", q_chunk, k_part) * scale
      if softcap > 0:
        scores = softcap * torch.tanh(scores / softcap)
      if sink is not None and sink.numel() > 0:
        sink_scores = sink.float().transpose(0, 1).unsqueeze(1).expand(-1, q_chunk.shape[0], -1)
        probabilities = torch.softmax(torch.cat((scores, sink_scores), dim=-1),
                                      dim=-1)[..., :k_part.shape[0]]
      else:
        probabilities = torch.softmax(scores, dim=-1)
      output[q_start + offset:q_start + offset + q_chunk.shape[0]] = torch.einsum(
        "hqk,khd->qhd", probabilities, v_part).to(output.dtype)
  return output


def varlen_attention_with_sink(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  *,
  cu_seqlens_q: torch.Tensor,
  cu_seqlens_k: torch.Tensor,
  max_seqlen_q: int = None,
  max_seqlen_k: int = None,
  softcap: float = -1.0,
  sink: torch.Tensor = None,
) -> torch.Tensor:
  """Packed attention with the released learned sinks.

  Uses ``flash_attn_varlen_func`` plus the sink LSE correction when available, and the chunked
  reference otherwise. The fallback is numerically the same operation but much slower, so a GPU
  benchmark row must run on the fast path (``cache_dit`` logs which one is active).

  :param q: ``[tokens, heads, head_dim]`` queries.
  :param k: ``[tokens, kv_heads, head_dim]`` keys.
  :param v: ``[tokens, kv_heads, head_dim]`` values.
  :param cu_seqlens_q: Cumulative query lengths.
  :param cu_seqlens_k: Cumulative key lengths.
  :param max_seqlen_q: Longest query segment; derived when omitted.
  :param max_seqlen_k: Longest key segment; derived when omitted.
  :param softcap: Logit softcap; ``<= 0`` disables it.
  :param sink: Optional ``[num_sink, heads]`` learned sink logits.
  :returns: ``[tokens, heads, head_dim]`` attention output.
  """

  if _flash_attn_varlen_func is not None and q.dtype in (torch.float16,
                                                         torch.bfloat16) and q.is_cuda:
    if max_seqlen_q is None:
      max_seqlen_q = int(torch.diff(cu_seqlens_q).max().item()) if cu_seqlens_q.numel() > 1 else 0
    if max_seqlen_k is None:
      max_seqlen_k = int(torch.diff(cu_seqlens_k).max().item()) if cu_seqlens_k.numel() > 1 else 0
    # flash-attn requires the bounds on the compute device; the packing helper already places them
    # there, this only pins it for direct callers that keep them on the host.
    out, lse, _ = _flash_attn_varlen_func(
      q,
      k,
      v,
      cu_seqlens_q.to(device=q.device, dtype=torch.int32),
      cu_seqlens_k.to(device=q.device, dtype=torch.int32),
      max_seqlen_q,
      max_seqlen_k,
      dropout_p=0.0,
      causal=False,
      softcap=float(softcap) if softcap and softcap > 0 else 0.0,
      return_attn_probs=True,
    )
    out, _ = correct_out_lse_with_sink(out, lse, sink)
    return out.to(q.dtype)
  return torch_varlen_attention_with_sink(
    q,
    k,
    v,
    cu_seqlens_q=cu_seqlens_q.cpu(),
    cu_seqlens_k=cu_seqlens_k.cpu(),
    softcap=softcap,
    sink=sink,
  )
