"""Vendored Pika H3 Flex modular pipeline (MiniMax-H3 trunk + six-step LoRA adapters).

The released checkpoint runs on a diffusers build that ships `modular_pipelines/pika_h3_flex/` and a
`MiniMaxH3Scheduler` with the re-noising update; the diffusers release cache-dit pins ships neither,
and the transformer itself is byte-identical to the installed one. So only the pipeline layer is
vendored here, and the classes are imported lazily: importing `cache_dit` never pulls the model code
in.

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

__all__ = [
  "PikaH3FlexBlocks",
  "PikaH3FlexModularPipeline",
  "PikaH3FlexScheduler",
]

_REFERENCE_CLASSES = (
  "PikaH3FlexAudioReference",
  "PikaH3FlexImageReference",
  "PikaH3FlexReference",
  "PikaH3FlexVideoReference",
)

__all__ += list(_REFERENCE_CLASSES)


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
