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

Usage::

    from cache_dit.quantization.svdquant.passes import apply_passes

    apply_passes(model, ["fused_gelu_mlp"])
"""

from __future__ import annotations

import types
from typing import Any

from torch import nn

from ...logger import init_logger

logger = init_logger(__name__)

_REGISTRY: dict[str, type[BasePass]] = {}

# Default pass list applied when ``fused_mlp`` is enabled.  Callers that
# need a custom subset can override via ``apply_passes(..., pass_names=...)``.
DEFAULT_FUSED_MLP_PASSES: list[str] = ["fused_gelu_mlp", "fused_gelu_proj"]


class BasePass:
  """Abstract base for a model pass.

  A pass detects applicable submodules and patches their forward methods to use fused runtime paths.
  """

  name: str = ""

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


def apply_passes(module: nn.Module, pass_names: list[str] | None = None) -> None:
  """Run every named pass on *module* in order.

  :param module: Root module to transform.
  :param pass_names: List of registered pass names.  When ``None``
      (the default), ``DEFAULT_FUSED_MLP_PASSES`` is used.
  """
  if pass_names is None:
    pass_names = DEFAULT_FUSED_MLP_PASSES
  if not pass_names:
    return
  for name in pass_names:
    pass_cls = get_pass(name)
    instance = pass_cls()
    targets = instance.detect(module)
    if targets:
      logger.info(
        "✅ Applying %s pass to %d target(s).",
        name,
        len(targets),
      )
      instance.apply(module, targets)
    else:
      logger.debug("Pass %s found no targets.", name)


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
  ``svdq_gemm_w4a4_ext`` qout path to eliminate the intermediate fp16
  HBM write between the two GEMMs.

  .. note::

     This pass detects the **generic** ``FeedForward`` / ``GELU(proj=…)``
     pattern defined by diffusers' ``attention.py`` (not any specific
     model class).  It works automatically with most diffusers
     transformer families — FLUX, SD3, PixArt, DiT, HunyuanVideo,
     Wan, Cosmos, QwenImage, and many more — without per-model
     configuration.

  .. rubric:: Supported models

  This pass targets diffusers transformers that use ``activation_fn``
  set to ``"gelu"`` or ``"gelu-approximate"`` (both resolve to the
  ``GELU`` activation class, not ``GEGLU``).  Models known to match
  this pattern include:

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

  Models that use **gated** activations (``GEGLU``, ``SwiGLU``,
  ``LinearActivation``) or custom ``FeedForward`` classes (e.g.
  Flux2, ErnieImage, LongCat Audio DiT, AuraFlow, OmniGen,
  Kandinsky5, HiDream, Z Image, Lumina2) are **not** targeted by
  this pass.
  """

  name = "fused_gelu_mlp"

  # Activation class names that use plain (non-gated) GELU.
  _PLAIN_GELU_NAMES = frozenset({"GELU"})

  def detect(self, module: nn.Module) -> list[dict[str, Any]]:
    from .linear import SVDQW4A4Linear

    targets: list[dict[str, Any]] = []
    for parent_name, parent in module.named_modules():
      if parent.__class__.__name__ != "FeedForward":
        continue
      net = getattr(parent, "net", None)
      if not isinstance(net, nn.ModuleList) or len(net) < 3:
        continue

      # net[0] must be an activation module wrapping a Linear.
      act_mod = net[0]
      act_cls_name = act_mod.__class__.__name__
      fc1 = getattr(act_mod, "proj", None)
      if not isinstance(fc1, (nn.Linear, SVDQW4A4Linear)):
        continue

      # net[2] must be a Linear / SVDQW4A4Linear.
      fc2 = net[2]
      if not isinstance(fc2, (nn.Linear, SVDQW4A4Linear)):
        continue

      # Both must be SVDQW4A4Linear (pass is applied post-quantize).
      if not isinstance(fc1, SVDQW4A4Linear) or not isinstance(fc2, SVDQW4A4Linear):
        continue

      # Only plain GELU can be fused with the current CUDA kernel.
      # GEGLU (gate * GELU(up)), SwiGLU, etc. need kernel-level
      # support and are left as-is.
      if act_cls_name not in self._PLAIN_GELU_NAMES:
        logger.debug(
          "FeedForward %s uses %s activation — skipping fusion "
          "(only plain GELU is supported).",
          parent_name,
          act_cls_name,
        )
        continue

      targets.append({
        "parent": parent,
        "fc1": fc1,
        "fc2": fc2,
        "parent_name": parent_name,
      })

    return targets

  def apply(self, module: nn.Module, targets: list[dict[str, Any]]) -> None:
    from .fused import fused_gelu_mlp

    for target in targets:
      parent = target["parent"]
      fc1 = target["fc1"]
      fc2 = target["fc2"]

      # qout kernel now uses signed INT4 (FUSE_GELU=true, USE_UNSIGNED=false),
      # matching the non-fused fc2.quantize output exactly.
      # No act_unsigned change needed.

      # Save original forward for potential introspection.
      parent._ff_original_forward = parent.forward

      def _patched_forward(self, hidden_states, _fc1=fc1, _fc2=fc2):
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

  The ext kernel can fuse all three steps because fc2's only input
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
  ``fused_gelu_proj``.  This uses the ext kernel's GELU epilogue
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

  1. It is **not** a ``FeedForward`` (already handled upstream).
  2. It contains at least one ``nn.GELU`` submodule.
  3. It contains at least two ``SVDQW4A4Linear`` submodules.
  4. One ``SVDQW4A4Linear`` has ``out_features > in_features``
     (expansion — fc1) and another has ``in_features >=
     fc1.out_features`` with ``in_features > out_features``
     (projection accepting concat — fc2).

  The fusion is applied by monkey-patching ``fc1.forward`` to include
  GELU and replacing ``gelu_mod.forward`` with an identity pass-through.
  This works for **any** parent class without inspecting its forward
  code.

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
      if parent.__class__.__name__ == "FeedForward":
        continue

      gelu_mod: nn.Module | None = None
      svdq_layers: list[tuple[str, SVDQW4A4Linear]] = []
      for child_name, child in parent.named_children():
        if isinstance(child, nn.GELU):
          gelu_mod = child
        if isinstance(child, SVDQW4A4Linear):
          svdq_layers.append((child_name, child))

      if gelu_mod is None or len(svdq_layers) < 2:
        continue

      # Heuristic: the single-block concat pattern implies
      #   fc2.in_features == fc1.in_features + fc1.out_features
      # where fc1.in_features is the attention stream dimension and
      # fc1.out_features is the MLP hidden dimension.  fc1 must
      # expand (out_features > in_features) and fc2 must project
      # (in_features > out_features).
      best_fc1: SVDQW4A4Linear | None = None
      best_fc2: SVDQW4A4Linear | None = None
      for _name_a, layer_a in svdq_layers:
        for _name_b, layer_b in svdq_layers:
          if layer_a is layer_b:
            continue
          if (layer_a.out_features > layer_a.in_features
              and layer_b.in_features > layer_b.out_features
              and layer_b.in_features == layer_a.in_features + layer_a.out_features):
            if best_fc1 is None or layer_a.out_features > best_fc1.out_features:
              best_fc1 = layer_a
              best_fc2 = layer_b

      if best_fc1 is None or best_fc2 is None:
        continue

      targets.append({
        "parent": parent,
        "fc1": best_fc1,
        "fc2": best_fc2,
        "gelu": gelu_mod,
        "parent_name": parent_name,
      })

    return targets

  def apply(self, module: nn.Module, targets: list[dict[str, Any]]) -> None:
    from .fused import fused_gelu_proj

    for target in targets:
      fc1 = target["fc1"]
      gelu_mod = target["gelu"]

      fc1._saved_forward = fc1.forward

      def _patched_fc1_forward(self, x, output=None):
        return fused_gelu_proj(x, self)

      fc1.forward = types.MethodType(_patched_fc1_forward, fc1)

      gelu_mod._saved_forward = gelu_mod.forward
      gelu_mod.forward = lambda x: x

      logger.debug(
        "Fused GELU proj patched: %s (fc1=%s, gelu=%s)",
        target["parent_name"],
        fc1.__class__.__name__,
        gelu_mod.__class__.__name__,
      )


__all__ = [
  "BasePass",
  "DEFAULT_FUSED_MLP_PASSES",
  "FusedGeluMlpPass",
  "FusedGeluProjPass",
  "apply_passes",
  "get_pass",
  "register_pass",
]
