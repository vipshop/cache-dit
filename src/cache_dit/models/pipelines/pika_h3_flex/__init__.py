"""Pika H3 Flex modular pipeline (MiniMax-H3 trunk + six-step LoRA adapters).

The released model is the diffusers modular pipeline `diffusers.PikaH3FlexModularPipeline`. A diffusers
build that predates it — one without `modular_pipelines/pika_h3_flex/` and without the `stochastic_sampling`
/ `shift_provided_sigmas` half of `MiniMaxH3Scheduler` — cannot run the checkpoint at all, so this
package carries a copy of the pipeline layer for those builds. `resolve_pika_h3_flex()` picks between
the two: the official implementation whenever it imports, the vendored one otherwise. The vendored
classes are imported lazily, so importing `cache_dit` never pulls the model code in.

The transformer is *not* vendored in either case: the installed `MiniMaxH3Transformer3DModel` is the one
the checkpoint ships, and cache-dit's MiniMax-H3 adapter / CP / TP planners dispatch on that class name.

Source and re-sync:

* upstream is `diffusers@d0b9b1b5a540e5f9df2f8ec236347bf9c5aa7d1f`, directory
  `src/diffusers/modular_pipelines/pika_h3_flex/` (Apache-2.0, Copyright 2026 Mellis, Inc. (Pika), The
  MiniMax and HuggingFace Teams);
* the files were copied verbatim and only their imports rewritten, but **upstream indents with four
  spaces while this repository formats with two** (`yapf`, `.style.yapf: indent_width = 2`). The copy
  is normalized on commit like any other file here, so a plain `diff` against upstream is noisy by
  design;
* to re-sync, run `pre-commit run yapf --files src/cache_dit/models/pipelines/pika_h3_flex/*.py` on
  the incoming copy **before** diffing, otherwise every line shows up as changed. The intentional
  diffs are listed in `modular_pipeline.py`'s module docstring and must be re-applied by hand. Three
  `__init__` docstrings (`before_encoder`, `denoise`, `encoders`) additionally had one wrapped `Args:`
  entry per file un-wrapped onto a single line, because this repository's `pydocstyle` hook reports
  `D417` for a line-broken parameter description;
* the transformer is deliberately **not** copied: `diff -q` against the same upstream commit shows the
  installed `MiniMaxH3Transformer3DModel` is identical, and cache-dit's MiniMax-H3 adapter / CP / TP
  planners dispatch on that class name.
"""

from dataclasses import dataclass
from typing import Callable, Optional

__all__ = [
  "PikaH3FlexBinding",
  "PikaH3FlexBlocks",
  "PikaH3FlexModularPipeline",
  "PikaH3FlexScheduler",
  "resolve_pika_h3_flex",
  "upstream_pika_h3_flex_available",
]

_REFERENCE_CLASSES = (
  "PikaH3FlexAudioReference",
  "PikaH3FlexImageReference",
  "PikaH3FlexReference",
  "PikaH3FlexVideoReference",
)

__all__ += list(_REFERENCE_CLASSES)

# The official copy lives in diffusers. Only the pipeline and its block tree are exported at the top
# level; the reference types stay in their own module, so a caller that builds `--reference-*` inputs has
# to be pointed at the package rather than at `diffusers`.
_UPSTREAM_REFERENCE_PACKAGE = "diffusers.modular_pipelines.pika_h3_flex.references"


@dataclass(frozen=True)
class PikaH3FlexBinding:
  r"""Which copy of the Pika H3 Flex pipeline a run should use.

  :param pipeline_class: The `PikaH3FlexModularPipeline` class to load.
  :param blocks_factory: Callable returning the block tree, or `None` when the pipeline resolves its
    own blocks through the checkpoint's `modular_model_index.json`. The official `PikaH3FlexBlocks` is
    found by name in the `diffusers` namespace (`ModularPipeline._blocks_class_name`), which the
    vendored copy cannot be, so the two have to be selected together.
  :param reference_package: Module the `PikaH3Flex*Reference` classes are imported from. The vendored
    and the official reference types are unrelated classes, and the blocks reject a foreign one.
  :param is_upstream: Whether this binding is the diffusers implementation. `False` means the vendored
    copy is in use, which is the `diffusers < 0.43` fallback.
  """

  pipeline_class: type
  blocks_factory: Optional[Callable[[], object]]
  reference_package: str
  is_upstream: bool


def upstream_pika_h3_flex_available() -> bool:
  r"""Whether the installed diffusers ships the official Pika H3 Flex modular pipeline.

  The classes are reached through `getattr` because `diffusers` is a lazy module: a missing name raises
  `AttributeError` from its `__getattr__`, not `ImportError`.

  :returns: Whether `PikaH3FlexModularPipeline` and `PikaH3FlexBlocks` can be imported.
  """
  import diffusers

  return all(
    getattr(diffusers, name, None) is not None
    for name in ("PikaH3FlexModularPipeline", "PikaH3FlexBlocks"))


def resolve_pika_h3_flex() -> PikaH3FlexBinding:
  r"""Pick the pipeline the installed diffusers can actually run.

  The official pipeline is the released model and the vendored copy is only a stand-in for a
  diffusers that predates it, so the official one wins whenever it imports.

  :returns: The binding to load, with its block factory and reference package.
  """
  if upstream_pika_h3_flex_available():
    import diffusers

    return PikaH3FlexBinding(
      pipeline_class=diffusers.PikaH3FlexModularPipeline,
      blocks_factory=None,
      reference_package=_UPSTREAM_REFERENCE_PACKAGE,
      is_upstream=True,
    )
  from .modular_blocks_pika_h3_flex import PikaH3FlexBlocks
  from .modular_pipeline import PikaH3FlexModularPipeline

  return PikaH3FlexBinding(
    pipeline_class=PikaH3FlexModularPipeline,
    blocks_factory=PikaH3FlexBlocks,
    reference_package=__name__,
    is_upstream=False,
  )


def __getattr__(name):
  if name == "PikaH3FlexModularPipeline":
    from .modular_pipeline import PikaH3FlexModularPipeline

    return PikaH3FlexModularPipeline
  if name == "PikaH3FlexBlocks":
    from .modular_blocks_pika_h3_flex import PikaH3FlexBlocks

    return PikaH3FlexBlocks
  if name == "PikaH3FlexScheduler":
    from .scheduler import PikaH3FlexScheduler

    return PikaH3FlexScheduler
  if name in _REFERENCE_CLASSES:
    from . import references

    return getattr(references, name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
