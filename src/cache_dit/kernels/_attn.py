import importlib
from types import ModuleType

import torch

_EXTENSION_MODULE_NAME = "cache_dit._C_attn"
_cached_extension_module: ModuleType | None = None
_cached_load_error: Exception | None = None


def _load_extension_module() -> ModuleType | None:
  """Import the optional attention CUDA extension at most once.

  :returns: The loaded extension module, or `None` when the import failed and the
    failure was cached.
  """

  global _cached_extension_module, _cached_load_error

  if _cached_extension_module is not None:
    return _cached_extension_module
  if _cached_load_error is not None:
    return None

  try:
    _cached_extension_module = importlib.import_module(_EXTENSION_MODULE_NAME)
  except Exception as exc:  # pragma: no cover - exercised in environments without the extension
    _cached_load_error = exc
    return None

  return _cached_extension_module


def attn_extension_is_available() -> bool:
  """Return whether the optional attention CUDA extension can be imported.

  :returns: `True` when the compiled attention extension module is available.
  """

  return _load_extension_module() is not None


def attn_get_load_error() -> Exception | None:
  """Return the cached import error from the optional attention CUDA extension.

  :returns: The import error raised while loading the extension, or `None` when the
    extension loaded successfully.
  """

  _load_extension_module()
  return _cached_load_error


def _get_required_ops_module() -> ModuleType:
  """Return the extension's `ops` submodule used by the Python wrappers.

  :returns: The `ops` submodule exported by the loaded attention extension.
  :raises RuntimeError: If the extension is unavailable or does not expose `ops`.
  """

  extension_module = _load_extension_module()
  if extension_module is None:
    raise RuntimeError(
      "The optional Cache-DiT attention CUDA extension is not available. Build it with "
      "`CACHE_DIT_BUILD_ATTN=1 python -m pip install -e . --no-deps --no-build-isolation` "
      "after `conda activate cdit`.") from attn_get_load_error()

  ops_module = getattr(extension_module, "ops", None)
  if ops_module is None:
    raise RuntimeError(
      "The loaded Cache-DiT attention CUDA extension does not expose an `ops` submodule.")
  return ops_module


def _call_fused_merge_attn_states(
  prev_out: torch.Tensor,
  prev_lse: torch.Tensor,
  suff_out: torch.Tensor,
  suff_lse: torch.Tensor,
  out: torch.Tensor,
  lse: torch.Tensor,
) -> None:
  """Bridge the attention-state merge operator to the compiled extension.

  :param prev_out: Contiguous 3D previous attention output `[B * N, H, D]`.
  :param prev_lse: Contiguous 2D previous log-sum-exp `[B * N, H]`.
  :param suff_out: Contiguous 3D suffix attention output `[B * N, H, D]`.
  :param suff_lse: Contiguous 2D suffix log-sum-exp `[B * N, H]`.
  :param out: Contiguous 3D merged attention output `[B * N, H, D]`, written in place.
  :param lse: Contiguous 2D merged log-sum-exp `[B * N, H]`, written in place.
  """

  _get_required_ops_module().fused_merge_attn_states(
    prev_out,
    prev_lse,
    suff_out,
    suff_lse,
    out,
    lse,
  )


torch.library.define(
  "cache_dit_cuda_ops::fused_merge_attn_states",
  "(Tensor prev_out, Tensor prev_lse, Tensor suff_out, Tensor suff_lse) "
  "-> (Tensor out, Tensor lse)",
)


@torch.library.impl("cache_dit_cuda_ops::fused_merge_attn_states", "CUDA")
def _merge_attn_states_impl(
  prev_out: torch.Tensor,
  prev_lse: torch.Tensor,
  suff_out: torch.Tensor,
  suff_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  batch, seq_len, num_heads, head_size = suff_out.shape
  # The extension consumes flattened attention states, so the batch and sequence
  # dimensions are merged before the call and restored afterwards.
  prev_out = prev_out.flatten(0, 1).contiguous()
  suff_out = suff_out.flatten(0, 1).contiguous()
  prev_lse = prev_lse.flatten(0, 1).squeeze(-1).contiguous()
  suff_lse = suff_lse.flatten(0, 1).squeeze(-1).contiguous()

  out = torch.empty_like(suff_out)
  lse = torch.empty_like(suff_lse)
  _call_fused_merge_attn_states(prev_out, prev_lse, suff_out, suff_lse, out, lse)
  return out.view(batch, seq_len, num_heads, head_size), lse.view(batch, seq_len, num_heads, 1)


@torch.library.register_fake("cache_dit_cuda_ops::fused_merge_attn_states")
def _fake_merge_attn_states(
  prev_out: torch.Tensor,
  prev_lse: torch.Tensor,
  suff_out: torch.Tensor,
  suff_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  del prev_out, prev_lse
  batch, seq_len, num_heads, head_size = suff_out.shape
  # Exactly replicate the view chain of the real implementation so the abstract
  # tensors keep the same shape and strides under torch.compile.
  suff_out = suff_out.flatten(0, 1).contiguous()
  suff_lse = suff_lse.flatten(0, 1).squeeze(-1).contiguous()
  out = suff_out.new_empty(suff_out.shape, dtype=suff_out.dtype)
  lse = suff_lse.new_empty(suff_lse.shape, dtype=suff_lse.dtype)
  return out.view(batch, seq_len, num_heads, head_size), lse.view(batch, seq_len, num_heads, 1)


def fused_merge_attn_states(
  prev_out: torch.Tensor,
  prev_lse: torch.Tensor,
  suff_out: torch.Tensor,
  suff_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Fuse the attention states of two consecutive attention chunks, e.g. Ring Attention.

  :param prev_out: Previous attention output shaped `[B, N, H, D]`.
  :param prev_lse: Previous log-sum-exp shaped `[B, N, H, 1]`.
  :param suff_out: Suffix attention output shaped `[B, N, H, D]`.
  :param suff_lse: Suffix log-sum-exp shaped `[B, N, H, 1]`.
  :returns: The fused attention output `[B, N, H, D]` and log-sum-exp `[B, N, H, 1]`.
  """

  return torch.ops.cache_dit_cuda_ops.fused_merge_attn_states(
    prev_out,
    prev_lse,
    suff_out,
    suff_lse,
  )


__all__ = [
  "_get_required_ops_module",
  "attn_extension_is_available",
  "attn_get_load_error",
  "fused_merge_attn_states",
]
