"""Model pass system for SVDQ fused-MLP integration.

A *pass* is a model-structure-aware transformation that wraps quantized
linear layers with fused runtime paths.  Passes are applied **after**
quantization has replaced ``nn.Linear`` submodules with
``SVDQW4A4Linear`` instances, and they operate purely on the forward
dispatch — weight format and serialization are unchanged.

Pass registry
-------------

| Name                              | Scope                                                  |
|-----------------------------------|--------------------------------------------------------|
| ``fused_gelu_mlp``                | Standard diffusers ``FeedForward`` GELU MLP (dual      |
|                                   | and single stream).                                    |
| ``fused_gelu_proj``               | Single-stream blocks with standalone GELU between two  |
|                                   | SVDQW4A4Linear layers (e.g. FLUX single blocks).       |
| ``fused_swiglu_mlp``              | Gated MLPs whose gate is applied inside the activation |
|                                   | producer (Qwen-Image-2.1 style triple-linear blocks and |
|                                   | diffusers ``FeedForward(activation_fn='swiglu')``).    |

Usage::

    from cache_dit.quantization.svdquant.passes import apply_passes

    apply_passes(model, ["fused_gelu_mlp"])
"""

from __future__ import annotations

import types
from typing import Any

from torch import nn
from torch.nn import functional as F

from ...logger import init_logger
from .dtensor import SVDQW4A4ShardLinear

logger = init_logger(__name__)

_REGISTRY: dict[str, type[BasePass]] = {}

# Default pass list applied when ``fused_mlp`` is enabled.  Callers that
# need a custom subset can override via ``apply_passes(..., pass_names=...)``.
DEFAULT_FUSED_MLP_PASSES: list[str] = [
  "fused_gelu_mlp", "fused_gelu_proj", "fused_swiglu_mlp", "fused_geglu_mlp"
]

_FUSED_N_ALIGNMENT_NATIVE = 128
_FUSED_N_ALIGNMENT_CUTE = 256
_FUSED_K_ALIGNMENT = 128
_PASS_MARKER = "_svdq_fused_mlp_pass"
_TARGET_GROUPS = {
  "fused_gelu_mlp": "gelu_mlp",
  "fused_gelu_proj": "gelu_proj",
  "fused_swiglu_mlp": "swiglu",
  "fused_geglu_mlp": "geglu_mlp",
}


def _is_gelu_module(module: nn.Module) -> bool:
  """Return whether *module* is a plain, non-gated GELU wrapper."""

  return isinstance(module, nn.GELU) or module.__class__.__name__ == "GELU"


def _gelu_approximate(module: nn.Module) -> str:
  return str(getattr(module, "approximate", "none"))


def _fused_n_alignment(layer: nn.Module) -> int:
  """Return the fused epilogue N alignment for a layer's runtime."""

  runtime = str(getattr(layer, "runtime_kernel", "native")).lower()
  return _FUSED_N_ALIGNMENT_CUTE if runtime == "cute" else _FUSED_N_ALIGNMENT_NATIVE


def _inference_guard(parent: nn.Module) -> str | None:
  """Reject training state and any active Dropout in a candidate block."""

  if parent.training:
    return "training"
  if any(isinstance(child, nn.Dropout) and child.p != 0 for child in parent.modules()):
    return "active_dropout"
  return None


def _fused_gelu_geometry(fc1: nn.Module, fc2: nn.Module | None = None) -> str | None:
  """Validate static fused-GELU widths before patching a block."""

  out_padded = int(getattr(fc1, "_padded_out_features", fc1.out_features))
  in_padded = int(getattr(fc1, "_padded_in_features", fc1.in_features))
  # SVDQW4A4Linear constructs padded K via ceil(128); keep the defensive check
  # explicit because fused kernels assume that invariant.
  if in_padded % _FUSED_K_ALIGNMENT != 0:
    return f"fused K={in_padded} is not divisible by {_FUSED_K_ALIGNMENT}"
  n_alignment = _fused_n_alignment(fc1)
  if out_padded % n_alignment != 0:
    return f"fused N={out_padded} is not divisible by {n_alignment}"
  if fc2 is not None:
    next_in = int(getattr(fc2, "_padded_in_features", fc2.in_features))
    if next_in != out_padded:
      return f"fc1 padded N={out_padded} does not match fc2 padded K={next_in}"
  return None


def _fused_swiglu_geometry(up: nn.Module, gate: nn.Module | None, down: nn.Module) -> str | None:
  """Validate the logical producer width required by a fused gated chain."""

  hidden = int(down.in_features)
  # The CuTe producer packs logical K and requires K%64; the native pack
  # requires a BLOCK_N-aligned logical width.
  runtime = str(getattr(down, "runtime_kernel", "native")).lower()
  alignment = 64 if runtime == "cute" else _FUSED_N_ALIGNMENT_NATIVE
  if hidden % alignment != 0:
    return f"fused gated hidden={hidden} is not divisible by {alignment}"

  up_in = int(up.in_features)
  up_out = int(up.out_features)
  expected = hidden if gate is not None else 2 * hidden
  if up_out != expected:
    return f"up N={up_out} does not match gated hidden requirement {expected}"
  if gate is not None:
    if int(gate.in_features) != up_in:
      return f"gate K={gate.in_features} does not match up K={up_in}"
    if int(gate.out_features) != hidden:
      return f"gate N={gate.out_features} does not match down K={hidden}"
  return None


_ACTIVATION_ATTRS = ("activation_fn", "act_fn", "mlp_act", "act", "activation")

# (value/up, gate, down) aliases used by diffusers LLAMA/SwiGLU MLPs.
_GATED_ALIAS_GROUPS = (
  ("proj", "gate_layer", "out"),
  ("up_proj", "gate_proj", "down_proj"),
  ("up", "gate", "down"),
  # Llama/Z-Image compute down(silu(w1(x)) * w3(x)): w3 is the value operand.
  ("w3", "w1", "w2"),
)

_GELU_CHAIN_ALIASES = (
  ("fc1", "fc2"),
  ("linear1", "linear2"),
  ("dense_in", "dense_out"),
)


def _activation_spec(parent: nn.Module) -> tuple[str | None, object | None]:
  """Return the first declared activation attribute, without walking modules."""

  for name in _ACTIVATION_ATTRS:
    if hasattr(parent, name):
      value = getattr(parent, name)
      if value is not None:
        return name, value
  return None, None


def _is_silu_activation(value: object) -> bool:
  """Recognize both ``nn.SiLU`` modules and functional ``F.silu``."""

  if isinstance(value, nn.SiLU):
    return True
  if value is F.silu:
    return True
  return callable(value) and getattr(value, "__name__", None) == "silu"


def _is_standalone_mlp(parent: nn.Module,
                       layers: tuple[nn.Module, ...],
                       activation: object | None = None) -> bool:
  """Require that patching *parent* bypasses only the target MLP operations."""

  allowed = {id(layer) for layer in layers}
  if isinstance(activation, nn.Module):
    allowed.add(id(activation))
  for child in parent.children():
    if id(child) in allowed:
      continue
    if isinstance(child, (nn.Dropout, nn.Identity)):
      continue
    return False
  return True


def _select_named_gated(parent: nn.Module):
  """Select a supported gated MLP by conventional attribute aliases."""

  from .linear import SVDQW4A4Linear

  activation_name, activation = _activation_spec(parent)
  for up_name, gate_name, down_name in _GATED_ALIAS_GROUPS:
    up = getattr(parent, up_name, None)
    gate = getattr(parent, gate_name, None)
    down = getattr(parent, down_name, None)
    if not all(isinstance(layer, SVDQW4A4Linear) for layer in (up, gate, down)):
      continue
    if activation is not None and not _is_silu_activation(activation):
      return None, f"non_silu_activation:{activation_name}"
    if not _is_standalone_mlp(parent, (up, gate, down), activation):
      return None, "non_standalone_mlp"
    return (up, gate, down), None

  if activation is None or not _is_silu_activation(activation):
    return None, None

  # Three same-shape projections with unknown names cannot establish which one
  # is value and which one is gate. Do not guess; report them as unsupported.
  layers = [child for child in parent.children() if isinstance(child, SVDQW4A4Linear)]
  if len(layers) == 3 and _is_standalone_mlp(parent, tuple(layers), activation):
    return None, "unsupported_gated_names"
  return None, None


def _detect_sequential_gelu(parent: nn.Sequential):
  """Detect ``Sequential(Linear, GELU, [Dropout], Linear)``."""

  from .linear import SVDQW4A4Linear

  children = list(parent.children())
  if len(children) not in (3, 4):
    return None
  fc1 = children[0]
  act = children[1]
  fc2 = children[-1]
  if not isinstance(fc1, SVDQW4A4Linear) or not isinstance(fc2, SVDQW4A4Linear):
    return None
  if not _is_gelu_module(act):
    return None
  if len(children) == 4 and not isinstance(children[2], nn.Dropout):
    return None
  if fc1.out_features != fc2.in_features:
    return None
  return {
    "parent": parent,
    "fc1": fc1,
    "fc2": fc2,
    "activation": act,
    "pattern": "sequential",
  }


def _detect_attribute_gelu(parent: nn.Module):
  """Detect an MLP whose fc1/activation/fc2 are direct module attributes."""

  from .linear import SVDQW4A4Linear

  for fc1_name, fc2_name in _GELU_CHAIN_ALIASES:
    fc1 = getattr(parent, fc1_name, None)
    fc2 = getattr(parent, fc2_name, None)
    if not isinstance(fc1, SVDQW4A4Linear) or not isinstance(fc2, SVDQW4A4Linear):
      continue
    activation_name, activation = _activation_spec(parent)
    if not _is_gelu_module(activation):
      continue
    if int(fc1.out_features) != int(fc2.in_features):
      return None, "gelu_chain_width_mismatch"
    if not _is_standalone_mlp(parent, (fc1, fc2), activation):
      return None, "non_standalone_mlp"
    return {
      "parent": parent,
      "fc1": fc1,
      "fc2": fc2,
      "activation": activation,
      "pattern": f"attributes:{fc1_name}/{activation_name}/{fc2_name}",
    }, None
  return None, None


def _gelu_chain_reason(parent: nn.Module, fc1: nn.Module, fc2: nn.Module,
                       activation: nn.Module) -> str | None:
  """Validate a non-gated fused GELU chain independent of its container type."""

  from .dtensor import SVDQW4A4ShardLinear

  if isinstance(fc1, SVDQW4A4ShardLinear) or isinstance(fc2, SVDQW4A4ShardLinear):
    return "tp_sharded"
  guard = _inference_guard(parent)
  if guard is not None:
    return guard
  gelu_modules = [child for child in parent.modules() if _is_gelu_module(child)]
  if len(gelu_modules) != 1 or gelu_modules[0] is not activation:
    return "ambiguous_gelu"
  approximate = _gelu_approximate(activation)
  if approximate != "tanh":
    return f"exact_gelu:{approximate}"
  geometry = _fused_gelu_geometry(fc1, fc2)
  if geometry is not None:
    return geometry
  return None


class BasePass:
  """Abstract base for a model pass.

  A pass detects applicable submodules and patches their forward methods to use fused runtime paths.
  """

  name: str = ""

  def __init__(self) -> None:
    self.skipped_reasons: dict[str, int] = {}

  def _skip(self, reason: str) -> None:
    self.skipped_reasons[reason] = self.skipped_reasons.get(reason, 0) + 1

  def detect(self, module: nn.Module) -> list[dict[str, Any]]:
    """Return target descriptors for applicable submodules.

    Each returned dict must contain at least:

    * ``parent`` — the parent module whose forward should be patched.
    * ``fc1`` — the first ``SVDQW4A4Linear`` (input → hidden).
    * ``fc2`` — the second ``SVDQW4A4Linear`` (hidden → output).

    :param module: Root module to scan.
    :returns: List of target info dicts.
    """
    raise NotImplementedError

  def apply(self, module: nn.Module, targets: list[dict[str, Any]]) -> None:
    """Patch every detected target.

    :param module: Root module (passed for context, not necessarily
        modified directly).
    :param targets: List of target descriptors returned by ``detect``.
    """
    raise NotImplementedError


def register_pass(pass_cls: type[BasePass]) -> type[BasePass]:
  """Register a pass class in the global registry.

  :param pass_cls: A ``BasePass`` subclass with a non-empty ``name``.
  :returns: The same class (enables decorator usage).
  """
  if not pass_cls.name:
    raise ValueError(f"Pass class {pass_cls.__name__} must define a non-empty 'name'.")
  if pass_cls.name in _REGISTRY:
    raise ValueError(f"Pass name {pass_cls.name!r} is already registered.")
  _REGISTRY[pass_cls.name] = pass_cls
  return pass_cls


def get_pass(name: str) -> type[BasePass]:
  """Look up a registered pass class by name.

  :param name: Pass name as registered.
  :returns: The pass class.
  :raises KeyError: If the name is unknown.
  """
  if name not in _REGISTRY:
    raise KeyError(f"Unknown pass {name!r}.  Registered: {sorted(_REGISTRY)}.")
  return _REGISTRY[name]


def apply_passes(module: nn.Module, pass_names: list[str] | None = None) -> dict[str, int]:
  """Run every named pass on *module* in order.

  :param module: Root module to transform.
  :param pass_names: List of registered pass names.  When ``None``
      (the default), ``DEFAULT_FUSED_MLP_PASSES`` is used.
  :returns: Per-pass target counts for this invocation.  Counts are zero on
    an idempotent re-application.
  """

  if pass_names is None:
    pass_names = DEFAULT_FUSED_MLP_PASSES
  if not pass_names:
    return {}

  counts: dict[str, int] = {}
  skipped: dict[str, dict[str, int]] = {}
  for name in pass_names:
    pass_cls = get_pass(name)
    instance = pass_cls()
    targets = instance.detect(module)
    counts[name] = len(targets)
    skipped[name] = dict(instance.skipped_reasons)
    if targets:
      logger.info(
        "✅ Applying %s pass to %d target(s).",
        name,
        len(targets),
      )
      instance.apply(module, targets)
    else:
      logger.debug("Pass %s found no targets.", name)

  total = sum(counts.values())
  existing = getattr(module, "_svdq_fused_mlp_targets", None)
  reason_counts: dict[str, int] = {}
  for reasons in skipped.values():
    for reason, count in reasons.items():
      reason_counts[reason] = reason_counts.get(reason, 0) + count
  details = ", ".join(f"{reason}: {count}"
                      for reason, count in sorted(reason_counts.items())) or "no candidate blocks"

  if total > 0:
    merged_targets = {group: 0 for group in _TARGET_GROUPS.values()}
    merged_targets.update(getattr(module, "_svdq_fused_mlp_targets", {}))
    for name, count in counts.items():
      group = _TARGET_GROUPS.get(name, name)
      merged_targets[group] = merged_targets.get(group, 0) + count

    merged_passes = dict(getattr(module, "_svdq_fused_mlp_passes", {}))
    for name, count in counts.items():
      merged_passes[name] = merged_passes.get(name, 0) + count

    merged_skipped = {
      name: dict(reasons)
      for name, reasons in getattr(module, "_svdq_fused_mlp_skipped", {}).items()
    }
    for name, reasons in skipped.items():
      bucket = merged_skipped.setdefault(name, {})
      for reason, count in reasons.items():
        bucket[reason] = bucket.get(reason, 0) + count

    module._svdq_fused_mlp_targets = merged_targets
    module._svdq_fused_mlp_passes = merged_passes
    module._svdq_fused_mlp_skipped = merged_skipped
    if reason_counts:
      logger.warning(
        "SVDQ fused_mlp skipped %d candidate(s) after checking supported structures: %s.",
        sum(reason_counts.values()),
        details,
      )
  elif not existing or not any(existing.values()):
    module._svdq_fused_mlp_targets = {group: 0 for group in _TARGET_GROUPS.values()}
    for name in counts:
      module._svdq_fused_mlp_targets.setdefault(_TARGET_GROUPS.get(name, name), 0)
    module._svdq_fused_mlp_passes = counts
    merged_skipped = {
      name: dict(reasons)
      for name, reasons in getattr(module, "_svdq_fused_mlp_skipped", {}).items()
    }
    for name, reasons in skipped.items():
      bucket = merged_skipped.setdefault(name, {})
      for reason, count in reasons.items():
        bucket[reason] = bucket.get(reason, 0) + count
    module._svdq_fused_mlp_skipped = merged_skipped
    logger.warning(
      "SVDQ fused_mlp is enabled but no supported MLP targets were found. "
      "Skips: %s.",
      details,
    )
  return counts


@register_pass
class FusedGeluMlpPass(BasePass):
  """Fuse standard diffusers ``FeedForward`` GELU MLP blocks.

  Detects modules that match the pattern::

      FeedForward(
        net = ModuleList([
          GELU(proj=Linear),   # fc1  (plain GELU, NOT GEGLU)
          Dropout,
          Linear,              # fc2
        ])
      )

  After quantization the linear layers become ``SVDQW4A4Linear``
  instances.  This pass patches the ``FeedForward.forward`` method to
  call ``fused_gelu_mlp(fc1, fc2)``, which uses the
  native ``qout`` path to eliminate the intermediate fp16 HBM write between
  the two GEMMs.

  .. note::

     This pass detects the **generic** ``FeedForward`` / ``GELU(proj=…)``
     pattern defined by diffusers' ``attention.py`` (not any specific
     model class).  It works automatically with most diffusers
     transformer families — FLUX, SD3, PixArt, DiT, HunyuanVideo,
     Wan, Cosmos, QwenImage, and many more — without per-model
     configuration.

  .. rubric:: Supported models

  This pass only fuses ``GELU(approximate="tanh")``, i.e. diffusers
  ``activation_fn="gelu-approximate"``.  The exact ``activation_fn="gelu"``
  path is deliberately skipped because the current SVDQ qout epilogue
  hardcodes the tanh approximation; silently substituting it would change
  model numerics.  Models below that use exact GELU are therefore candidates
  for the structural pass but not for the current fused kernel.

  - **FLUX**: ``FluxTransformer2DModel`` (``gelu-approximate``)
  - **SD3**: ``SD3TransformerBlock`` (``gelu-approximate``)
  - **PixArt / DiT**: ``PixArtTransformer2DModel``,
    ``DiTTransformer2DModel`` (default ``gelu-approximate``)
  - **HunyuanVideo**: ``HunyuanVideoTransformerBlock``,
    ``HunyuanVideoSingleTransformerBlock`` (``gelu-approximate``)
  - **HunyuanVideo 1.5**: ``HunyuanVideo15TransformerBlock``,
    ``HunyuanVideo15SingleTransformerBlock`` (``gelu-approximate``)
  - **HunyuanImage**: ``HunyuanImageTransformerBlock``,
    ``HunyuanImageSingleTransformerBlock`` (``gelu-approximate``)
  - **HunyuanDiT 2D**: 2nd variant ``HunyuanDiTBlock``
    (``gelu-approximate``)
  - **Wan**: ``WanTransformerBlock`` (``gelu``),
    ``WanAttentionBlock`` (``gelu-approximate``)
  - **Wan Animate**: ``WanAnimateTransformerBlock`` (``gelu``),
    ``WanAnimateAttentionBlock`` (``gelu-approximate``)
  - **Wan VACE**: ``WanVaceBlock`` (``gelu-approximate``)
  - **Cosmos**: ``CosmosTransformerBlock`` (``gelu``)
  - **SkyReels V2**: ``SkyReelsV2TransformerBlock`` (``gelu``),
    ``SkyReelsV2AttentionBlock`` (``gelu-approximate``)
  - **ChronoEdit**: ``ChronoEditTransformerBlock`` (``gelu``),
    ``ChronoEditAttentionBlock`` (``gelu-approximate``)
  - **AnyFlow**: ``AnyFlowTransformerBlock`` (``gelu``),
    ``AnyFlowAttentionBlock`` (``gelu-approximate``)
  - **AnyFlow FAR**: ``AnyFlowFarTransformerBlock`` (``gelu``),
    ``AnyFlowFarAttentionBlock`` (``gelu-approximate``)
  - **Bria**: ``BriaTransformerModel`` (``gelu-approximate``)
  - **Bria Fibo**: ``BriaFiboTransformerBlock`` (``gelu-approximate``)
  - **GLM Image**: ``GLMImageTransformerModel``
    (``gelu-approximate`` / ``gelu``)
  - **QwenImage**: ``QwenImageTransformerBlock``
    (``gelu-approximate``)
  - **JoyImage**: ``JoyImageTransformerBlock``
    (``gelu-approximate``)
  - **LongCat Image**: ``LongCatImageTransformerBlock``
    (``gelu-approximate``)
  - **CogView3Plus**: ``CogView3PlusTransformerBlock``
    (``gelu-approximate``)
  - **CogView4**: ``CogView4TransformerBlock``
    (``gelu-approximate``)
  - **Chroma**: ``ChromaTransformerBlock`` (``gelu-approximate``)
  - **Helios**: ``HeliosTransformerBlock`` (``gelu-approximate``)
  - **Motif Video**: ``MotifVideoTransformerBlock``
    (``gelu-approximate``)
  - **CogVideoX**: ``CogVideoXTransformerBlock``
    (default ``gelu-approximate``)
  - **EasyAnimate**: ``EasyAnimateTransformerBlock``
    (default ``gelu-approximate``)
  - **LTX**: ``LTXTransformerBlock`` (default ``gelu-approximate``)
  - **LTX2**: ``LTX2TransformerBlock`` (default ``gelu-approximate``)
  - **ConsisID**: ``ConsisIDTransformerBlock``
    (default ``gelu-approximate``)
  - **Allegro**: ``AllegroTransformerBlock``
    (default ``gelu-approximate``)
  - **Prior Transformer**: ``PriorTransformer`` (``gelu``)

  The pass also detects two non-gated layouts:

  - ``nn.Sequential(Linear, GELU, [Dropout], Linear)``;
  - direct ``fc1``/``linear1`` + GELU + ``fc2``/``linear2`` attributes.

  The patched parent must own only the two layers, the activation, and
  Dropout/Identity children; blocks that also own attention, normalization,
  or other operational modules are skipped. Gated activations (SwiGLU,
  GEGLU), packed QKV+MLP blocks, and custom residual/MoE MLPs are not
  targeted by this pass.
  """

  name = "fused_gelu_mlp"

  def detect(self, module: nn.Module) -> list[dict[str, Any]]:
    from .linear import SVDQW4A4Linear

    targets: list[dict[str, Any]] = []
    for parent_name, parent in module.named_modules():
      if hasattr(parent, _PASS_MARKER):
        continue

      target = None
      reason = None
      if parent.__class__.__name__ == "FeedForward":
        net = getattr(parent, "net", None)
        if (not isinstance(net, nn.ModuleList) or len(net) != 3
            or not isinstance(net[1], (nn.Dropout, nn.Identity))):
          continue
        act_mod = net[0]
        fc1 = getattr(act_mod, "proj", None)
        fc2 = net[2]
        if (not _is_gelu_module(act_mod) or not isinstance(fc1, SVDQW4A4Linear)
            or not isinstance(fc2, SVDQW4A4Linear)):
          continue
        target = {
          "parent": parent,
          "fc1": fc1,
          "fc2": fc2,
          "activation": act_mod,
          "pattern": "feedforward",
        }
      elif isinstance(parent, nn.Sequential):
        target = _detect_sequential_gelu(parent)
      else:
        target, reason = _detect_attribute_gelu(parent)

      if target is None:
        if reason is not None:
          self._skip(f"fused_gelu_mlp:{reason}")
        continue

      reason = _gelu_chain_reason(target["parent"], target["fc1"], target["fc2"],
                                  target["activation"])
      if reason is not None:
        self._skip(f"fused_gelu_mlp:{reason}")
        logger.debug("%s is not fused: %s.", parent_name, reason)
        continue

      target["parent_name"] = parent_name
      targets.append(target)

    return targets

  def apply(self, module: nn.Module, targets: list[dict[str, Any]]) -> None:
    from .fused import fused_gelu_mlp

    for target in targets:
      parent = target["parent"]
      fc1 = target["fc1"]
      fc2 = target["fc2"]
      if hasattr(parent, _PASS_MARKER):
        continue

      parent._svdq_fused_mlp_original_forward = parent.forward
      parent._svdq_fused_mlp_pass = self.name

      def _patched_forward(self, hidden_states, *args, _fc1=fc1, _fc2=fc2, **kwargs):
        if args or kwargs:
          return self._svdq_fused_mlp_original_forward(hidden_states, *args, **kwargs)
        return fused_gelu_mlp(hidden_states, _fc1, _fc2)

      parent.forward = types.MethodType(_patched_forward, parent)

      logger.debug(
        "Fused GELU MLP patched: %s (fc1=%s, fc2=%s)",
        target["parent_name"],
        fc1.__class__.__name__,
        fc2.__class__.__name__,
      )


@register_pass
class FusedGeluProjPass(BasePass):
  """Fuse fc1 + GELU in single-stream transformer blocks.

  .. note::

     Like ``FusedGeluMlpPass``, this pass uses **generic structural
     detection** (``SVDQW4A4Linear`` + ``nn.GELU`` + concat-dimension
     arithmetic) rather than class-name matching.  It automatically
     covers all diffusers single-stream blocks that follow the
     ``proj_mlp → GELU → concat → proj_out`` pattern — FLUX, Bria,
     HunyuanVideo, HunyuanImage, Chroma, LongCat Image, Motif Video,
     and any future model with the same structure.

  .. rubric:: Why full fc1+fc2 fusion is impossible here

  Standard double-stream blocks (handled by ``FusedGeluMlpPass``)
  have a purely sequential MLP::

      fc1(x) → GELU → fc2 → output

  The native GEMM can fuse all three steps because fc2's only input
  is the GELU output — the qout path writes quantized activations
  directly from the first GEMM into the second GEMM, skipping HBM.

  Single-stream blocks have a **concat** in the middle::

      fc1(x) ──→ GELU ──→ mlp_out ──┐
                                      ├─→ concat ──→ fc2 ──→ output
      attn(x) ───────────────────────┘

  fc2's input is ``concat(attn_output, mlp_output)``, not just the
  GELU output.  fc2 needs the attention half to compute correctly,
  so the kernel cannot simply chain fc1's GELU output into fc2's
  GEMM — it would be missing half the features.

  **What we CAN fuse**: ``fc1 + GELU`` into a single kernel via
  ``fused_gelu_proj``.  This uses the native GEMM's GELU epilogue
  to write fp16 GELU'd output directly, eliminating one GELU kernel
  launch and one fp16 HBM roundtrip per single block.  The concat
  and fc2 remain unchanged.

  .. rubric:: Why not restructure the block (Nunchaku's approach)

  Nunchaku's ``NunchakuFluxSingleTransformerBlock`` achieves **full**
  fc1+GELU+fc2 fusion by *restructuring* the model at quantization
  time: ``proj_out`` (which takes concatenated input) is split into
  two independent ``SVDQW4A4Linear`` layers via
  ``from_linear(proj_out, in_features=N)`` — one for the attention
  half, one for the MLP half.  The forward then becomes::

      mlp  = fused_gelu_mlp(x, proj_mlp, mlp_fc2)   # full fusion
      attn = attn.to_out(attn_output)
      output = mlp + attn                           # add, not concat

  This is mathematically equivalent to the original but requires
  the PTQ pipeline to split weights and LoRA components at
  quantization time, which affects the serialized checkpoint format
  and SVD decomposition granularity.

  **cache-dit's choice**: ``FusedGeluProjPass`` deliberately does
  **not** restructure the model.  It keeps ``proj_out`` intact and
  only fuses fc1+GELU via monkey-patching.  This is compatible with
  cache-dit's existing SVDQ PTQ / DQ / few-shot workflow without
  any changes to quantization, serialization, or weight loading.
  The trade-off — partial fusion instead of full — is accepted for
  the sake of integration simplicity and zero checkpoint-format
  impact.

  .. rubric:: Detection (generic, not class-name based)

  A parent module is targeted when:

  1. It is **not** a ``FeedForward`` (already handled upstream) and has
     not already installed a fused-MLP pass.
  2. It directly owns exactly one bare ``nn.GELU`` with
     ``approximate="tanh"``. Exact GELU and diffusers GELU wrappers that
     own a projection are rejected.
  3. It is in eval mode with no active Dropout and contains no TP-sharded
     SVDQ layer.
  4. It contains at least two ``SVDQW4A4Linear`` submodules and exactly
     one valid expansion/projection pair:
     ``fc2.in_features == fc1.in_features + fc1.out_features``.
  5. The static fused N/K geometry is valid for ``fc1``'s runtime.

  The fusion is applied by monkey-patching ``fc1.forward`` to include
  GELU and replacing ``gelu_mod.forward`` with an identity pass-through.
  Extra ``fc1`` arguments fall back to the original GEMM and explicitly
  re-apply GELU, so the fallback never drops the activation.

  .. rubric:: Supported models

  Diffusers transformers whose single-stream blocks follow the
  ``proj_mlp → act_mlp(GELU) → concat(attn, mlp) → proj_out``
  pattern and satisfy ``fc2.in_features == fc1.in_features +
  fc1.out_features``:

  - **FLUX**: ``FluxSingleTransformerBlock``
  - **Bria**: ``BriaSingleTransformerBlock``
  - **Bria Fibo**: ``BriaFiboSingleTransformerBlock``
  - **Chroma**: ``ChromaSingleTransformerBlock``
  - **HunyuanVideo**: ``HunyuanVideoSingleTransformerBlock``,
    ``HunyuanVideoTokenReplaceSingleTransformerBlock``
  - **HunyuanImage**: ``HunyuanImageSingleTransformerBlock``
  - **LongCat Image**: ``LongCatImageSingleTransformerBlock``
  - **Motif Video**: ``MotifVideoSingleTransformerBlock``

  Models whose single blocks use **gated** activations (e.g.
  Flux2 with ``Flux2SwiGLU``), ``SiLU``, or standard sequential
  ``FeedForward`` are **not** targeted by this pass.
  """

  name = "fused_gelu_proj"

  def detect(self, module: nn.Module) -> list[dict[str, Any]]:
    from .linear import SVDQW4A4Linear

    targets: list[dict[str, Any]] = []
    for parent_name, parent in module.named_modules():
      if parent.__class__.__name__ == "FeedForward" or hasattr(parent, _PASS_MARKER):
        continue

      gelu_modules: list[nn.Module] = []
      svdq_layers: list[tuple[str, SVDQW4A4Linear]] = []
      for child_name, child in parent.named_children():
        if isinstance(child, nn.GELU):
          gelu_modules.append(child)
        if isinstance(child, SVDQW4A4Linear):
          svdq_layers.append((child_name, child))

      if len(gelu_modules) != 1 or len(svdq_layers) < 2:
        if len(gelu_modules) > 1 and len(svdq_layers) >= 2:
          self._skip("fused_gelu_proj:ambiguous_gelu")
        continue
      gelu_mod = gelu_modules[0]

      guard = _inference_guard(parent)
      if guard is not None:
        self._skip(f"fused_gelu_proj:{guard}")
        continue

      if any(isinstance(layer, SVDQW4A4ShardLinear) for _, layer in svdq_layers):
        self._skip("fused_gelu_proj:tp_sharded")
        logger.debug("Single-block %s uses TP-sharded SVDQ layers — skipping fused GELU proj.",
                     parent_name)
        continue

      # The single-block concat pattern implies
      #   fc2.in_features == fc1.in_features + fc1.out_features
      # where fc1.in_features is the attention stream dimension and
      # fc1.out_features is the MLP hidden dimension.
      pairs = []
      for _name_a, layer_a in svdq_layers:
        for _name_b, layer_b in svdq_layers:
          if layer_a is layer_b:
            continue
          if (layer_a.out_features > layer_a.in_features
              and layer_b.in_features > layer_b.out_features
              and layer_b.in_features == layer_a.in_features + layer_a.out_features):
            pairs.append((layer_a, layer_b))

      if not pairs:
        self._skip("fused_gelu_proj:projection_pair_not_found")
        continue
      if len(pairs) > 1:
        self._skip("fused_gelu_proj:ambiguous_projection_pair")
        logger.debug("Single-block %s has %d candidate projection pairs; skipping fused GELU proj.",
                     parent_name, len(pairs))
        continue
      best_fc1, best_fc2 = pairs[0]

      approximate = _gelu_approximate(gelu_mod)
      if approximate != "tanh":
        self._skip(f"fused_gelu_proj:exact_gelu:{approximate}")
        logger.debug(
          "Single-block %s uses GELU(approximate=%r); only the tanh qout epilogue is fused.",
          parent_name,
          approximate,
        )
        continue

      geometry = _fused_gelu_geometry(best_fc1)
      if geometry is not None:
        self._skip(f"fused_gelu_proj:{geometry}")
        logger.debug("Single-block %s is not fused: %s.", parent_name, geometry)
        continue

      targets.append({
        "parent": parent,
        "fc1": best_fc1,
        "fc2": best_fc2,
        "gelu": gelu_mod,
        "approximate": approximate,
        "parent_name": parent_name,
      })

    return targets

  def apply(self, module: nn.Module, targets: list[dict[str, Any]]) -> None:
    from .fused import fused_gelu_proj

    for target in targets:
      parent = target["parent"]
      fc1 = target["fc1"]
      gelu_mod = target["gelu"]
      approximate = target["approximate"]
      if hasattr(parent, _PASS_MARKER) or hasattr(fc1, _PASS_MARKER):
        continue

      parent._svdq_fused_mlp_pass = self.name
      fc1._svdq_fused_mlp_pass = self.name
      fc1._svdq_fused_mlp_original_forward = fc1.forward
      gelu_mod._svdq_fused_mlp_pass = self.name
      gelu_mod._svdq_fused_mlp_original_forward = gelu_mod.forward

      def _patched_fc1_forward(self, x, *args, _approximate=approximate, **kwargs):
        if args or kwargs:
          linear_out = self._svdq_fused_mlp_original_forward(x, *args, **kwargs)
          return F.gelu(linear_out, approximate=_approximate)
        return fused_gelu_proj(x, self)

      def _identity_forward(self, x, *args, **kwargs):
        return x

      fc1.forward = types.MethodType(_patched_fc1_forward, fc1)
      gelu_mod.forward = types.MethodType(_identity_forward, gelu_mod)

      logger.debug(
        "Fused GELU proj patched: %s (fc1=%s, gelu=%s)",
        target["parent_name"],
        fc1.__class__.__name__,
        gelu_mod.__class__.__name__,
      )


@register_pass
class FusedSwiGluMlpPass(BasePass):
  """Fuse gated MLPs whose activation is ``value * silu(gate)``.

  Three structures are detected after quantization has replaced their projections with
  ``SVDQW4A4Linear``:

  1. **Separate gate projection** — a standalone MLP owning one of these alias groups:
     ``proj/gate_layer/out``, ``up_proj/gate_proj/down_proj``, ``up/gate/down``, or
     ``w3/w1/w2`` (`w3` is the value operand for Llama-style ``down(silu(w1)*w3)``).
     Functional ``F.silu`` and ``nn.SiLU`` are equivalent. The parent may own only the three
     projections, the activation module, and Dropout/Identity; attention/norm-containing blocks
     are skipped even when their projection names match.
  2. **Single gated projection** — a standard three-slot diffusers ``FeedForward`` whose ``net[0]``
     is a ``SwiGLU`` wrapper: one ``proj`` emits ``2 * inner`` columns split into value and gate.
  3. **FLUX.2/3 gate-first FFN** — standalone ``Flux2FeedForward``/``Flux3FeedForward`` modules
     whose packed ``linear_in`` stores ``[gate | value]`` and whose ``act_fn.gate_fn`` is SiLU.
     Packed QKV+MLP blocks are deliberately excluded.

  In every case the gate is applied by the activation producer of ``down`` (see
  ``fused_swiglu_mlp``), which removes the standalone gating kernel and one ``[tokens, hidden]``
  HBM round trip. Geometry uses the producer's logical hidden contract rather than GEMM tile
  widths. Value-first GEGLU is handled by ``FusedGegluMlpPass`` with exact/tanh GELU instead of
  being misclassified as SwiGLU. ``QwenImage21AdaLayerNormContinuous`` and similar modulation
  blocks are never matched because they own at most one quantized projection.
  """

  name = "fused_swiglu_mlp"

  def detect(self, module: nn.Module) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for parent_name, parent in module.named_modules():
      if hasattr(parent, _PASS_MARKER):
        continue
      target, reason = _select_named_gated(parent)
      gate_first = False
      if target is None:
        if reason is not None:
          self._skip(f"fused_swiglu_mlp:{reason}")
          continue
        target = self._detect_gated_projection(parent)
      if target is None:
        target = self._detect_flux_gated_projection(parent)
        gate_first = target is not None
      if target is None:
        continue
      up, gate, down = target

      guard = _inference_guard(parent)
      if guard is not None:
        self._skip(f"fused_swiglu_mlp:{guard}")
        continue

      # The fused chain bypasses the TP collectives inside SVDQW4A4ShardLinear.forward.
      if any(isinstance(layer, SVDQW4A4ShardLinear) for layer in (up, gate, down) if layer):
        self._skip("fused_swiglu_mlp:tp_sharded")
        logger.debug("Gated MLP %s uses TP-sharded SVDQ layers — skipping fused SwiGLU.",
                     parent_name)
        continue

      geometry = _fused_swiglu_geometry(up, gate, down)
      if geometry is not None:
        self._skip(f"fused_swiglu_mlp:{geometry}")
        logger.debug("Gated MLP %s is not fused: %s.", parent_name, geometry)
        continue

      targets.append({
        "parent": parent,
        "up": up,
        "gate": gate,
        "down": down,
        "parent_name": parent_name,
        "gate_first": gate_first,
      })
    return targets

  @staticmethod
  def _detect_gated_projection(
    parent: nn.Module, ) -> tuple[nn.Module, nn.Module | None, nn.Module] | None:
    from .linear import SVDQW4A4Linear

    if parent.__class__.__name__ != "FeedForward":
      return None
    net = getattr(parent, "net", None)
    if (not isinstance(net, nn.ModuleList) or len(net) != 3
        or not isinstance(net[1], (nn.Dropout, nn.Identity))):
      return None
    activation = net[0]
    if activation.__class__.__name__ == "GEGLU":
      logger.debug("FeedForward uses GEGLU; the dedicated GEGLU pass handles it.")
      return None
    if activation.__class__.__name__ != "SwiGLU":
      return None
    up = getattr(activation, "proj", None)
    down = net[2]
    if not isinstance(up, SVDQW4A4Linear) or not isinstance(down, SVDQW4A4Linear):
      return None
    return up, None, down

  @staticmethod
  def _detect_flux_gated_projection(
    parent: nn.Module, ) -> tuple[nn.Module, None, nn.Module] | None:
    """Detect FLUX.2/3's standalone ``[gate | value]`` SwiGLU feed-forward."""

    from .linear import SVDQW4A4Linear

    if parent.__class__.__name__ not in ("Flux2FeedForward", "Flux3FeedForward"):
      return None
    activation = getattr(parent, "act_fn", None)
    if activation.__class__.__name__ not in ("Flux2SwiGLU", "Flux3SwiGLU"):
      return None
    if not _is_silu_activation(getattr(activation, "gate_fn", None)):
      return None
    up = getattr(parent, "linear_in", None)
    down = getattr(parent, "linear_out", None)
    if not isinstance(up, SVDQW4A4Linear) or not isinstance(down, SVDQW4A4Linear):
      return None
    if not _is_standalone_mlp(parent, (up, down), activation):
      return None
    return up, None, down

  def apply(self, module: nn.Module, targets: list[dict[str, Any]]) -> None:
    from .fused import fused_swiglu_mlp

    for target in targets:
      parent = target["parent"]
      up = target["up"]
      gate = target["gate"]
      down = target["down"]
      gate_act = target.get("gate_act", "silu")
      gate_first = target.get("gate_first", False)
      if hasattr(parent, _PASS_MARKER):
        continue

      parent._svdq_fused_mlp_original_forward = parent.forward
      parent._svdq_fused_mlp_pass = self.name

      def _patched_forward(self,
                           hidden_states,
                           *args,
                           _up=up,
                           _gate=gate,
                           _down=down,
                           _gate_act=gate_act,
                           _gate_first=gate_first,
                           **kwargs):
        if args or kwargs:
          return self._svdq_fused_mlp_original_forward(hidden_states, *args, **kwargs)
        return fused_swiglu_mlp(hidden_states,
                                _up,
                                _down,
                                gate=_gate,
                                gate_act=_gate_act,
                                gate_first=_gate_first)

      parent.forward = types.MethodType(_patched_forward, parent)

      logger.debug(
        "Fused gated MLP patched: %s (up=%s, gate=%s, down=%s, act=%s, gate_first=%s)",
        target["parent_name"],
        up.__class__.__name__,
        None if gate is None else gate.__class__.__name__,
        down.__class__.__name__,
        gate_act,
        gate_first,
      )


@register_pass
class FusedGegluMlpPass(FusedSwiGluMlpPass):
  """Fuse diffusers ``FeedForward(GEGLU(...))`` with exact or tanh GELU gating.

  GEGLU is value-first: ``proj(x).chunk(2, -1)`` yields value then gate. The
  producer therefore uses the default column order and only changes the gate
  activation. Exact GELU maps to ``gate_act='gelu'``; tanh maps to
  ``gate_act='gelu_tanh'``. Upstream diffusers GEGLU is exact-only; the tanh
  branch covers custom GEGLU-compatible modules exposing ``approximate='tanh'``.
  The numerical and structural guards are shared
  with the SwiGLU pass, but hits are reported as ``geglu_mlp``.
  """

  name = "fused_geglu_mlp"

  def detect(self, module: nn.Module) -> list[dict[str, Any]]:
    from .linear import SVDQW4A4Linear

    targets: list[dict[str, Any]] = []
    for parent_name, parent in module.named_modules():
      if hasattr(parent, _PASS_MARKER) or parent.__class__.__name__ != "FeedForward":
        continue
      net = getattr(parent, "net", None)
      if (not isinstance(net, nn.ModuleList) or len(net) != 3
          or not isinstance(net[1], (nn.Dropout, nn.Identity))):
        continue
      activation = net[0]
      if activation.__class__.__name__ != "GEGLU":
        continue
      approximate = _gelu_approximate(activation)
      if approximate not in ("none", "tanh"):
        self._skip(f"fused_geglu_mlp:unsupported_gelu_approximate:{approximate}")
        continue
      gate_act = "gelu_tanh" if approximate == "tanh" else "gelu"
      up = getattr(activation, "proj", None)
      down = net[2]
      if not isinstance(up, SVDQW4A4Linear) or not isinstance(down, SVDQW4A4Linear):
        continue

      guard = _inference_guard(parent)
      if guard is not None:
        self._skip(f"fused_geglu_mlp:{guard}")
        continue
      if isinstance(up, SVDQW4A4ShardLinear) or isinstance(down, SVDQW4A4ShardLinear):
        self._skip("fused_geglu_mlp:tp_sharded")
        continue
      geometry = _fused_swiglu_geometry(up, None, down)
      if geometry is not None:
        self._skip(f"fused_geglu_mlp:{geometry}")
        continue
      targets.append({
        "parent": parent,
        "up": up,
        "gate": None,
        "down": down,
        "parent_name": parent_name,
        "gate_act": gate_act,
        "gate_first": False,
      })
    return targets


__all__ = [
  "BasePass",
  "DEFAULT_FUSED_MLP_PASSES",
  "FusedGeluMlpPass",
  "FusedGeluProjPass",
  "FusedGegluMlpPass",
  "FusedSwiGluMlpPass",
  "apply_passes",
  "get_pass",
  "register_pass",
]
