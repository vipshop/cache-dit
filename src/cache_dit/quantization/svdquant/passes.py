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
| ``diffusers_gelu_feedforward``    | Standard diffusers ``FeedForward`` GELU MLP (dual      |
|                                   | and single stream).                                    |

Usage::

    from cache_dit.quantization.svdquant.passes import apply_passes

    apply_passes(model, ["diffusers_gelu_feedforward"])
"""

from __future__ import annotations

import types
from typing import Any

from torch import nn

from ...logger import init_logger

logger = init_logger(__name__)

_REGISTRY: dict[str, type[BasePass]] = {}


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


def apply_passes(module: nn.Module, pass_names: list[str]) -> None:
  """Run every named pass on *module* in order.

  :param module: Root module to transform.
  :param pass_names: List of registered pass names.
  """
  if not pass_names:
    return
  for name in pass_names:
    pass_cls = get_pass(name)
    instance = pass_cls()
    targets = instance.detect(module)
    if targets:
      logger.info(
        "Applying pass %s to %d target(s).",
        name,
        len(targets),
      )
      instance.apply(module, targets)
    else:
      logger.debug("Pass %s found no targets.", name)


# ---------------------------------------------------------------------------
# DiffusersGeluFeedForwardPass
# ---------------------------------------------------------------------------


@register_pass
class DiffusersGeluFeedForwardPass(BasePass):
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

  name = "diffusers_gelu_feedforward"

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
    from .fused_mlp import fused_gelu_mlp

    logger.info_once(
      "DiffusersGeluFeedForwardPass is active — "
      "%d FeedForward block(s) will be fused.",
      len(targets),
    )

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


__all__ = [
  "BasePass",
  "DiffusersGeluFeedForwardPass",
  "apply_passes",
  "get_pass",
  "register_pass",
]
