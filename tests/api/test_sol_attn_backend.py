import math

import pytest
import torch
import torch.nn.functional as F

from diffusers.models.modeling_utils import ModelMixin

from cache_dit.attention import set_attn_backend, set_sol_attn_config
from cache_dit.attention.backends import sol_attn as sol_backend
from cache_dit.attention.backends.register import (
  _AttnBackend,
  _AttnBackendRegistry,
  _dispatch_attention_fn,
)


class _DummyDiffusersModule(ModelMixin):

  def __init__(self):
    super().__init__()
    self.backends = []

  def set_attention_backend(self, backend: str) -> None:
    self.backends.append(backend)


def _relative_l2(output: torch.Tensor, reference: torch.Tensor) -> float:
  return ((output.float() - reference.float()).norm() /
          reference.float().norm().clamp_min(1e-12)).item()


def _cosine(output: torch.Tensor, reference: torch.Tensor) -> float:
  return F.cosine_similarity(output.float().flatten(), reference.float().flatten(), dim=0).item()


def _pool_64(x: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
  batch, tokens, heads, dim = x.shape
  blocks = (tokens + 63) // 64
  lengths = [min(64, tokens - block * 64) for block in range(blocks)]
  padded = F.pad(x, (0, 0, 0, 0, 0, blocks * 64 - tokens))
  pooled = padded.view(batch, blocks, 64, heads, dim)
  means = torch.stack(
    [pooled[:, block, :length].float().mean(dim=1) for block, length in enumerate(lengths)], dim=1)
  return means.to(x.dtype), lengths


def _routing_threshold(
  q: torch.Tensor,
  k: torch.Tensor,
  tau: float,
  thresh_type: str,
) -> torch.Tensor:
  batch, tokens, heads, dim = q.shape
  q_blocks, _ = _pool_64(q)
  k_blocks, _ = _pool_64(k)
  blocks = k_blocks.shape[1]
  k_bh = k_blocks.permute(0, 2, 1, 3)
  q_bh = q_blocks.permute(0, 2, 1, 3)
  mean = k_bh.float().mean(dim=2)
  log2_scale = dim ** -0.5 * math.log2(math.e)
  raw_mean = (q_bh.float() * mean[:, :, None, :]).sum(dim=-1)
  if thresh_type == "exact":
    second = torch.matmul(k_bh.transpose(-1, -2), k_bh) / blocks
    projected = torch.matmul(q_bh.float(), second.float())
    raw_second = (projected.float() * q_bh.float()).sum(dim=-1)
    variance = (raw_second - raw_mean.square()).clamp_min(0.0)
  else:
    variance_k = (k_bh.float().square().mean(dim=2) - mean.square()).clamp_min(0.0)
    variance = (q_bh.float().square() * variance_k[:, :, None, :]).sum(dim=-1)
  return raw_mean * log2_scale + tau * (variance * log2_scale ** 2 + 1e-6).sqrt()


def _route_mask(
  q: torch.Tensor,
  k: torch.Tensor,
  tau: float,
  thresh_type: str,
  sink_tokens: int = 0,
) -> torch.Tensor:
  q_blocks, _ = _pool_64(q)
  k_blocks, _ = _pool_64(k)
  scores = torch.einsum("bihd,bjhd->bhij", q_blocks.float(), k_blocks.float())
  scores = scores * (q.shape[-1] ** -0.5 * math.log2(math.e))
  threshold = _routing_threshold(q, k, tau, thresh_type)
  exact = scores > threshold[..., None]
  query_blocks = q_blocks.shape[1]
  key_blocks = k_blocks.shape[1]
  key_block_ids = torch.arange(key_blocks, device=q.device)
  local = (torch.arange(query_blocks, device=q.device)[:, None] - key_block_ids[None, :]).abs() <= 1
  sink = key_block_ids * 64 < sink_tokens
  return exact | local[None, None, :, :] | sink[None, None, None, :]


def _sol_reference(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  tau: float,
  thresh_type: str,
  sink_tokens: int = 0,
) -> torch.Tensor:
  batch, tokens, heads, dim = q.shape
  q_blocks, query_lengths = _pool_64(q)
  k_blocks, key_lengths = _pool_64(k)
  route = _route_mask(q, k, tau, thresh_type, sink_tokens)
  scale = dim ** -0.5 * math.log2(math.e)
  key_blocks = k_blocks.shape[1]
  v_padded = F.pad(v, (0, 0, 0, 0, 0, key_blocks * 64 - tokens))
  v_padded = v_padded.view(batch, key_blocks, 64, heads, dim)
  v_sums = torch.stack([
    v_padded[:, block, :length].float().sum(dim=1).to(v.dtype)
    for block, length in enumerate(key_lengths)
  ],
                       dim=1)
  output = torch.empty(batch, tokens, heads, dim, device=q.device, dtype=q.dtype)
  for b in range(batch):
    for h in range(heads):
      for i, query_length in enumerate(query_lengths):
        q_tile = q[b, i * 64:i * 64 + query_length, h]
        accumulator = torch.zeros(query_length, dim, device=q.device, dtype=torch.float32)
        row_sum = torch.zeros(query_length, device=q.device, dtype=torch.float32)
        row_max = torch.full((query_length, ), -float("inf"), device=q.device)
        for group_start in range(0, key_blocks, 64):
          group_end = min(group_start + 64, key_blocks)
          selected = route[b, h, i, group_start:group_end]
          scores = (q_tile.float() @ k_blocks[b, group_start:group_end, h].float().T) * scale
          approximate_scores = scores.masked_fill(selected[None, :], -float("inf"))
          if not bool(selected.all()):
            new_max = torch.maximum(row_max, approximate_scores.max(dim=1).values)
            alpha = torch.exp2(row_max - new_max)
            probabilities = torch.exp2(approximate_scores - new_max[:, None])
            probabilities = probabilities.masked_fill(selected[None, :], 0.0)
            accumulator = (
              accumulator * alpha[:, None] +
              probabilities.to(q.dtype).float() @ v_sums[b, group_start:group_end, h].float())
            lengths = torch.tensor(key_lengths[group_start:group_end],
                                   device=q.device,
                                   dtype=torch.float32)
            row_sum = row_sum * alpha + (probabilities * lengths[None, :]).sum(dim=1)
            row_max = new_max
          for j in range(group_start, group_end):
            if not route[b, h, i, j]:
              continue
            key_length = key_lengths[j]
            k_tile = k[b, j * 64:j * 64 + key_length, h]
            exact_scores = (q_tile.float() @ k_tile.float().T) * scale
            new_max = torch.maximum(row_max, exact_scores.max(dim=1).values)
            alpha = torch.exp2(row_max - new_max)
            probabilities = torch.exp2(exact_scores - new_max[:, None])
            accumulator = (
              accumulator * alpha[:, None] +
              probabilities.to(q.dtype).float() @ v[b, j * 64:j * 64 + key_length, h].float())
            row_sum = row_sum * alpha + probabilities.sum(dim=1)
            row_max = new_max
        output[b, i * 64:i * 64 + query_length, h] = (accumulator / row_sum[:, None]).to(q.dtype)
  return output


def test_sol_attn_registration_and_full_signature():
  assert _AttnBackend.SOL_ATTN.value == "sol_attn"
  assert _AttnBackendRegistry.get_backend("sol_attn") is not None
  assert _AttnBackendRegistry.is_context_parallel_available("sol_attn")
  required = {
    "attn_mask",
    "dropout_p",
    "is_causal",
    "scale",
    "enable_gqa",
    "return_lse",
    "cp_gqa_strategy",
    "_cp_config",
  }
  assert required <= _AttnBackendRegistry.get_supported_arg_names(_AttnBackend.SOL_ATTN)


def test_sol_attn_diffusers_registration():
  module = _DummyDiffusersModule()
  set_attn_backend(module, "sol_attn")
  assert module.backends == ["sol_attn"]
  from cache_dit.attention import _maybe_register_custom_attn_backends
  _maybe_register_custom_attn_backends()
  from diffusers.models.attention_dispatch import AttentionBackendName, _AttentionBackendRegistry
  assert _AttentionBackendRegistry._backends.get(AttentionBackendName("sol_attn")) is not None


def test_sol_attn_missing_dependency_is_clear(monkeypatch):
  monkeypatch.setattr(sol_backend, "_sol_attn", None)
  monkeypatch.setattr(sol_backend, "_SOL_RUNTIME_VALIDATED", False)
  with pytest.raises(RuntimeError, match="sol_attn.*not installed"):
    sol_backend._validate_sol_runtime()


def test_sol_attn_config_validation():
  with pytest.raises(ValueError, match="finite"):
    set_sol_attn_config(tau=float("nan"))
  with pytest.raises(ValueError, match="diag.*exact"):
    set_sol_attn_config(thresh_type="invalid")
  for invalid_sink in (-1, 1.5, True):
    with pytest.raises(ValueError, match="non-negative integer"):
      set_sol_attn_config(sink_tokens=invalid_sink)


def test_sol_attn_native_fallback_and_strict_mode():
  set_sol_attn_config(strict=False)
  sol_backend._reset_sol_attn_stats()
  q, k, v = (torch.randn(1, 64, 1, 128, dtype=torch.bfloat16) for _ in range(3))
  output = _dispatch_attention_fn(q, k, v, backend="sol_attn")
  expected = F.scaled_dot_product_attention(
    q.permute(0, 2, 1, 3),
    k.permute(0, 2, 1, 3),
    v.permute(0, 2, 1, 3),
  ).permute(0, 2, 1, 3)
  torch.testing.assert_close(output, expected)
  stats = sol_backend._get_sol_attn_stats()
  assert stats["fallback_calls"] == 1
  assert stats["kernel_calls"] == 0
  assert stats["fallback_reasons"]
  set_sol_attn_config(strict=True)
  with pytest.raises(ValueError, match="same CUDA device"):
    _dispatch_attention_fn(q, k, v, backend="sol_attn")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sol_attn_rejects_unsupported_semantics_in_strict_mode():
  set_sol_attn_config(strict=True)
  q = torch.randn(1, 64, 1, 128, device="cuda", dtype=torch.bfloat16)
  with pytest.raises(ValueError, match="attn_mask"):
    _dispatch_attention_fn(q, q, q, backend="sol_attn", attn_mask=torch.zeros(1, 64))
  with pytest.raises(ValueError, match="causal"):
    _dispatch_attention_fn(q, q, q, backend="sol_attn", is_causal=True)
  with pytest.raises(ValueError, match="dropout"):
    _dispatch_attention_fn(q, q, q, backend="sol_attn", dropout_p=0.1)
  with pytest.raises(ValueError, match="GQA"):
    _dispatch_attention_fn(q, q, q, backend="sol_attn", enable_gqa=True)
  with pytest.raises(ValueError, match="return_lse"):
    _dispatch_attention_fn(q, q, q, backend="sol_attn", attention_kwargs={"return_lse": True})
  with pytest.raises(ValueError, match="default scale"):
    _dispatch_attention_fn(q, q, q, backend="sol_attn", scale=1.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("tokens", [1, 63, 64, 65, 127, 128, 129, 1024])
@pytest.mark.parametrize("thresh_type", ["diag", "exact"])
def test_sol_attn_matches_independent_reference(tokens: int, thresh_type: str):
  torch.manual_seed(1234 + tokens)
  set_sol_attn_config(tau=1.0, thresh_type=thresh_type, sink_tokens=0)
  q, k, v = (torch.randn(1, tokens, 2, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
  output = _dispatch_attention_fn(q, k, v, backend="sol_attn")
  reference = _sol_reference(q, k, v, tau=1.0, thresh_type=thresh_type)
  # Long-sequence online reductions use a 4e-3 BF16 gate; short shapes stay at 3e-3.
  tolerance = 4e-3 if tokens >= 1024 else 3e-3
  assert _relative_l2(output, reference) <= tolerance
  assert _cosine(output, reference) >= 0.999


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sol_attn_all_exact_is_close_to_dense():
  torch.manual_seed(4321)
  set_sol_attn_config(tau=-1e9, thresh_type="diag", sink_tokens=0)
  q, k, v = (torch.randn(1, 257, 2, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
  output = _dispatch_attention_fn(q, k, v, backend="sol_attn")
  reference = F.scaled_dot_product_attention(
    q.transpose(1, 2).float(),
    k.transpose(1, 2).float(),
    v.transpose(1, 2).float(),
  ).transpose(1, 2)
  assert _relative_l2(output, reference) <= 3e-3
  assert _cosine(output, reference) >= 0.999


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sol_attn_default_prefix_sink_contract(monkeypatch):
  set_sol_attn_config()
  calls = []

  def _fake_sol_attn(q, k, v, **kwargs):
    calls.append(kwargs)
    return v

  monkeypatch.setattr(sol_backend._sol_attn, "sol_attn", _fake_sol_attn)
  q = torch.randn(1, 1100, 1, 128, device="cuda", dtype=torch.bfloat16)
  _dispatch_attention_fn(q, q, q, backend="sol_attn")
  assert calls[-1]["sink_start"] == 0
  assert calls[-1]["sink_tokens"] == 1024

  short = q[:, :64]
  _dispatch_attention_fn(short, short, short, backend="sol_attn")
  assert calls[-1]["sink_start"] == 0
  assert calls[-1]["sink_tokens"] == 64

  set_sol_attn_config(sink_tokens=0)
  _dispatch_attention_fn(short, short, short, backend="sol_attn")
  assert calls[-1]["sink_start"] is None
  assert calls[-1]["sink_tokens"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sol_attn_default_prefix_sink_matches_reference():
  torch.manual_seed(8765)
  set_sol_attn_config(tau=1e9)
  q, k, v = (torch.randn(1, 1100, 1, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
  output = _dispatch_attention_fn(q, k, v, backend="sol_attn")
  reference = _sol_reference(q, k, v, tau=1e9, thresh_type="diag", sink_tokens=1024)
  assert _relative_l2(output, reference) <= 4e-3
  assert _cosine(output, reference) >= 0.999


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sol_attn_opcheck_and_fullgraph_compile():
  set_sol_attn_config()
  q, k, v = (torch.randn(1, 128, 1, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
  sol_backend._ensure_sol_attn_op_defined()
  scale = q.shape[-1] ** -0.5
  torch.library.opcheck(
    torch.ops.cache_dit_sol_attn_ops.forward,
    (q, k, v, scale, 1.0, 0, 128),
  )
  eager = _dispatch_attention_fn(q, k, v, backend="sol_attn")
  compiled = torch.compile(lambda a, b, c: _dispatch_attention_fn(a, b, c, backend="sol_attn"),
                           fullgraph=True)
  actual = compiled(q, k, v)
  torch.testing.assert_close(actual, eager, rtol=0, atol=0)
  assert actual.is_contiguous()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sol_attn_ring_is_explicitly_rejected():
  from cache_dit.distributed.parallel import _ContextParallelConfig
  set_sol_attn_config()
  q = torch.randn(1, 128, 1, 128, device="cuda", dtype=torch.bfloat16)
  cp_config = _ContextParallelConfig(ring_degree=2, ulysses_degree=1)
  with pytest.raises(NotImplementedError, match="Ring.*pure Ulysses"):
    _dispatch_attention_fn(q, q, q, backend="sol_attn", cp_config=cp_config)
