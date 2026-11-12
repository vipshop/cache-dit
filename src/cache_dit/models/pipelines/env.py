"""Environment switches for cache-dit's model pipelines.

The module hosts the variables the vendored model pipelines read. Today that is the single
fused-operators switch below; further model-side switches belong here rather than in one variable
per model, so this file is the place to grep for "what can I configure on a model pipeline".

Four model families ship memory-bound CUDA operators that replace expressions the stock diffusers
module already evaluates: :mod:`.flux2_klein`, :mod:`.flux3_klein`, :mod:`.qwen_image21` and
:mod:`.boogu_image`. They are one optional optimisation, not a required path, and one variable
controls all of them: ``CACHE_DIT_FUSED_OPS``.

* unset - off until a cache-dit entry point installs the fused transformer
  (``_Flux2KleinPipeline.from_pretrained``, ``_QwenImage21Pipeline.from_pretrained``, the
  ``flux3_fused`` CLI example, a direct ``upgrade_transformer()``, or - for Boogu-Image, which has
  no separate install step - construction of ``BooguImageTransformer2DModel`` from its own
  pipeline), which enables the family from :func:`install_fused_modules`. A pipeline that only goes
  through stock diffusers therefore never reaches an operator, and needs no configuration.
* ``1`` - on wherever a fused transformer is installed. Only useful when wiring the install by
  hand; the cache-dit entry points already do the equivalent.
* ``0`` - off: every ``apply_*`` runs the literal eager expression, and the attention seams that
  hold a kernel (FLUX.2 and Qwen-Image 2.1) are not installed. FLUX.3 still swaps in its fused
  attention processor, whose operator then takes the eager branch; Boogu-Image keeps its block
  forwards and simply evaluates the pre-fusion chain. This is the switch to use when a kernel is
  suspected of producing the image.

It has to be set before the first fused pipeline is built in the process: the switch is a module
global that only the install writes, so flipping it afterwards changes nothing.

Measured on torch 2.15 (RTX PRO 5000, 1024x1024, 16 steps), the fused arm is **not** faster than
``torch.compile``. Inductor already fuses these pointwise chains at the DRAM ceiling, so the
compiled arm lands at parity (FLUX.3-Klein: 236.4 ms/forward with and without the operators), and
Qwen-Image 2.1 measured -1% on that arm (369.2 against 372.9 ms/step, torch 2.11). The gain only
exists against the eager baseline, where FLUX.3-Klein improves -16.7%. Treat
``CACHE_DIT_FUSED_OPS=1`` as an optional optimisation and keep ``torch.compile`` as the baseline to
measure against, rather than as the baseline to replace.

This is the only fused-ops environment variable. Per-operator selection is a family constant
(``ACTIVE_OPS``, the measured winner set) that a Python assignment narrows for an A/B run, and the
two FLUX.2 sub-features that once had variables of their own (the rotation-table cache and the
joined q/k stream) measured neutral and are code constants too. Nothing about this family is
configured by a model-named variable any more.

This lives next to the families instead of on :class:`cache_dit.envs.ENV` because ``ENV`` snapshots
``os.environ`` at import time and stores booleans as ``bool(int(...))``; neither can express
"unset", which is the state the install-time activation depends on.
"""

import os
from typing import Optional

FUSED_OPS_ENV = "CACHE_DIT_FUSED_OPS"
# Everything else, including the empty string, counts as "on": an explicit falsey value is the only
# way to disable an installed family, so a typo can never silently turn a kernel off.
FALSEY_VALUES = ("0", "false", "off", "no")


def fused_ops_from_env() -> Optional[bool]:
  """Parse ``CACHE_DIT_FUSED_OPS``, reading the environment on every call.

  Not cached: the value is consulted once per family import and once per install, and a snapshot
  would make `monkeypatch.setenv` invisible to the tests that pin this contract.

  :returns: `None` when the variable is unset, else the parsed flag.
  """

  value = os.environ.get(FUSED_OPS_ENV)
  if value is None:
    return None
  return value.strip().lower() not in FALSEY_VALUES


def fused_ops_default() -> bool:
  """Return the value a family starts with at import time.

  Unset resolves to `False`, i.e. "off until installed": that is the only state in which a stock
  diffusers pipeline cannot reach an operator, and it keeps the switch honest in the documentation.

  :returns: Whether the operators start enabled.
  """

  return fused_ops_from_env() is True


def fused_ops_enabled_on_install() -> bool:
  """Return whether installing a fused transformer should enable the family.

  :returns: `False` only when ``CACHE_DIT_FUSED_OPS`` explicitly turns the operators off.
  """

  return fused_ops_from_env() is not False


__all__ = [
  "FALSEY_VALUES",
  "FUSED_OPS_ENV",
  "fused_ops_default",
  "fused_ops_enabled_on_install",
  "fused_ops_from_env",
]
