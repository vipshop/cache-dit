"""Vendored Boogu-Image model stack.

Upstream Boogu-Image ships its transformer, attention processors and pipeline
inside the third-party `boogu` research package (plus a `trust_remote_code`
shim in the released checkpoints). Those classes are copied here so cache-dit
owns the code it optimizes: the fused kernels, the context/tensor-parallel
bookkeeping and the cache hooks all live inside the model instead of being
monkey-patched from the outside.

The copy tracks `/workspace/dev/vipshop/tmp/Boogu-Image` (the released research
repo) and keeps its class names, block attribute names and `state_dict` keys, so
released checkpoints load with `strict=True`. Intentional differences:

* Triton RMSNorm replaced by `rms_norm.RMSNorm` (pure PyTorch) and the
  `fused_ops` call sites for the `cache_dit._C_models` kernels;
* TeaCache / TaylorSeer removed (cache-dit's own cache owns that role);
* context-parallel support written into the attention processors and the
  double-stream block instead of being patched in;
* the RoPE table budget is floored (`rope._resolve_axes_lens`) so long
  instructions cannot drive `torch.gather` out of bounds.

Import is lazy: `models/pipelines/__init__.py` keeps `__all__ = []`, so
importing `cache_dit` never pulls the model code.
"""

__all__ = [
  "BooguImagePipeline",
  "BooguImageTransformer2DModel",
  "BooguImageScheduler",
]


def __getattr__(name: str):
  if name == "BooguImageTransformer2DModel":
    from .transformer_boogu import BooguImageTransformer2DModel

    return BooguImageTransformer2DModel
  if name == "BooguImagePipeline":
    from .pipeline_boogu import BooguImagePipeline

    return BooguImagePipeline
  if name == "BooguImageScheduler":
    from .scheduler import BooguImageScheduler

    return BooguImageScheduler
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
