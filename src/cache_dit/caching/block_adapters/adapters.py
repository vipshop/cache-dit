import inspect
import torch
from typing import List, Tuple, Union, Optional
from ..forward_pattern import ForwardPattern
from .block_adapters import BlockAdapter
from .block_registers import BlockAdapterRegister
from ...envs import ENV
from ...logger import init_logger
from ...patches import maybe_patch_nucleus_moe_grouped_mm_offsets
from ...utils import declare_repeated_blocks, resolve_transformer

logger = init_logger(__name__)


def _relaxed_assert(
  transformer: torch.nn.Module,
  allow_classes: Optional[Union[
    torch.nn.Module,
    List[torch.nn.Module],
    Tuple[torch.nn.Module],
  ]] = None,
) -> None:
  if allow_classes is not None and not isinstance(allow_classes, (list, tuple)):
    allow_classes = (allow_classes, )
  _imported_module_ = transformer.__module__
  if _imported_module_.startswith("diffusers"):
    # Only apply strict check for Diffusers transformers
    if allow_classes is not None:
      assert isinstance(
        transformer, allow_classes), (f"Transformer class {transformer.__class__.__name__} not in "
                                      f"allowed classes: {[cls.__name__ for cls in allow_classes]}")
    else:
      logger.warning("No allowed classes provided for transformer strict type check "
                     "in BlockAdapter. Skipping strict type check.")
  else:
    # Otherwise, just log a warning and skip strict type check, e.g:
    # sglang/multimodal_gen/runtime/models/dits/flux.py#L411
    logger.warning(f"Transformer class {transformer.__class__.__name__} is from "
                   f"{_imported_module_.split('.')[0]} not diffusers, skipping strict type check "
                   "in BlockAdapter.")


# NOTE: The registry returns the first class-name prefix that matches, so "Flux3" must be
# registered before "Flux" ("Flux3*" class names also start with "Flux").
@BlockAdapterRegister.register("Flux3")
def flux3_adapter(pipe, **kwargs) -> BlockAdapter:
  from diffusers import Flux3Transformer2DModel

  _relaxed_assert(pipe.transformer, Flux3Transformer2DModel)

  # FLUX.3 first runs one per-mode block stack per token mode (text / image / reference image),
  # then a single joint stack over [text, image(+reference)] tokens. The joint stack
  # `single_transformer_blocks` is the one cached: it carries the whole sequence and holds most of
  # the FLOPs, while the per-mode stacks are recomputed every step so that the cached joint
  # residuals stay anchored to the current per-mode features. See `docs/` for the modular pipeline
  # template.
  guider = getattr(pipe, "guider", None)
  has_separate_cfg = bool(guider is not None and getattr(guider, "num_conditions", 1) > 1)

  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.single_transformer_blocks,
    forward_pattern=ForwardPattern.Pattern_1,
    check_forward_pattern=True,
    has_separate_cfg=has_separate_cfg,
    **kwargs,
  )


@BlockAdapterRegister.register("Flux")
def flux_adapter(pipe, **kwargs) -> BlockAdapter:
  from diffusers import FluxTransformer2DModel
  from cache_dit.caching.patch_functors import FluxPatchFunctor

  supported_transformers = (FluxTransformer2DModel, )
  try:
    from diffusers import Flux2Transformer2DModel

    supported_transformers += (Flux2Transformer2DModel, )
  except ImportError:
    Flux2Transformer2DModel = None  # requires diffusers>=0.36.dev0

  _relaxed_assert(pipe.transformer, supported_transformers)

  transformer_cls_name: str = pipe.transformer.__class__.__name__
  if not transformer_cls_name.startswith("Nunchaku") and not transformer_cls_name.startswith(
      "Flux2"):
    # NOTE(DefTruth): Users should never use this variable directly,
    # it is only for developers to control whether to enable dummy
    # blocks, default to enabled.

    if not ENV.CACHE_DIT_FLUX_ENABLE_DUMMY_BLOCKS:
      return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=[
          pipe.transformer.transformer_blocks,
          pipe.transformer.single_transformer_blocks,
        ],
        forward_pattern=[
          ForwardPattern.Pattern_1,
          ForwardPattern.Pattern_1,
        ],
        check_forward_pattern=True,
        **kwargs,
      )
    else:
      return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=(pipe.transformer.transformer_blocks + pipe.transformer.single_transformer_blocks),
        blocks_name="transformer_blocks",
        dummy_blocks_names=["single_transformer_blocks"],
        patch_functor=FluxPatchFunctor(),
        forward_pattern=ForwardPattern.Pattern_1,
        **kwargs,
      )
  else:
    # Case for Flux2Transformer2DModel and NunchakuFluxTransformer2DModel
    return BlockAdapter(
      pipe=pipe,
      transformer=pipe.transformer,
      blocks=[
        pipe.transformer.transformer_blocks,
        pipe.transformer.single_transformer_blocks,
      ],
      forward_pattern=[
        ForwardPattern.Pattern_1,
        ForwardPattern.Pattern_3,
      ],
      check_forward_pattern=True,
      **kwargs,
    )


@BlockAdapterRegister.register("Mochi")
def mochi_adapter(pipe, **kwargs) -> BlockAdapter:
  from diffusers import MochiTransformer3DModel

  _relaxed_assert(pipe.transformer, MochiTransformer3DModel)
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.transformer_blocks,
    forward_pattern=ForwardPattern.Pattern_0,
    check_forward_pattern=True,
    **kwargs,
  )


@BlockAdapterRegister.register("CogVideoX")
def cogvideox_adapter(pipe, **kwargs) -> BlockAdapter:
  from diffusers import CogVideoXTransformer3DModel

  _relaxed_assert(pipe.transformer, CogVideoXTransformer3DModel)
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.transformer_blocks,
    forward_pattern=ForwardPattern.Pattern_0,
    check_forward_pattern=True,
    **kwargs,
  )


@BlockAdapterRegister.register("MiniMaxH3")
def minimax_h3_adapter(pipe, **kwargs) -> BlockAdapter:
  from diffusers import MiniMaxH3Transformer3DModel

  # One checkpoint repository holds both partitions: `transformer` drives the `t2va` / `fl2va`
  # workflows and `transformer_ref` the `ref2va` one, and a workflow loads only its own.
  _, transformer = resolve_transformer(pipe)
  if transformer is None:
    raise ValueError("The MiniMax-H3 pipeline has neither `transformer` nor `transformer_ref`: "
                     "pass `workflow=` to `from_pretrained` / `load_components` so one of the two "
                     "checkpoint partitions is loaded.")
  _relaxed_assert(transformer, MiniMaxH3Transformer3DModel)

  # MiniMax-H3 denoises **one packed sequence** — text, keyframe, audio and video rows — through a
  # single stack of blocks, and the checkpoint is guidance-distilled, so every step is exactly one
  # forward pass. Only the block stack is cached: `token_refiner` runs once per step over the text
  # stream alone and carries no AdaLN parameters, so caching it would only add noise to the
  # residual statistics.
  return BlockAdapter(
    pipe=pipe,
    transformer=transformer,
    blocks=transformer.transformer_blocks,
    forward_pattern=ForwardPattern.Pattern_3,
    check_forward_pattern=True,
    has_separate_cfg=False,
    **kwargs,
  )


@BlockAdapterRegister.register("FastH3")
def fast_h3_adapter(pipe, **kwargs) -> BlockAdapter:
  # FastH3 is MiniMax-H3 with a distilled sigma ladder, a VSA compression gate per block and video
  # sparse attention, so the block contract is unchanged: one packed sequence, one forward per
  # step, no classifier-free guidance. The transformer subclass keeps the parent's module tree,
  # which is why this reuses the MiniMax-H3 adapter instead of duplicating it.
  return minimax_h3_adapter(pipe, **kwargs)


# NOTE: The registry returns the first class-name prefix that matches, so these must be registered
# before "WanAnimate2" and "Wan". Two prefixes on purpose: the model is matched by *pipeline* class
# name, whose upstream name is `WanSpeechToVideoPipeline` and which does not start with "WanS2V";
# "WanS2V" additionally covers the transformer-only path used by the tests.
@BlockAdapterRegister.register("WanSpeechToVideo")
@BlockAdapterRegister.register("WanS2V")
def wan_s2v_adapter(pipe, **kwargs) -> BlockAdapter:
  from ...models.pipelines.wan_s2v.transformer import (
    WanS2VTransformer3DModel,
    WanS2VTransformerBlock,
  )

  # cache-dit owns this transformer, so `_relaxed_assert` would only log a warning for it; matching
  # the generic "Wan" adapter would then silently run with the wrong block contract.
  assert isinstance(pipe.transformer, WanS2VTransformer3DModel), (
    f"WanS2V adapter expects WanS2VTransformer3DModel, got {type(pipe.transformer)}")
  assert isinstance(pipe.transformer.blocks[0], WanS2VTransformerBlock), (
    f"WanS2V adapter expects WanS2VTransformerBlock, got {type(pipe.transformer.blocks[0])}")

  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    # The block list is named `blocks`; blocks take `(hidden_states, encoder_hidden_states, ...)` and
    # return a single tensor (Wan-family Pattern_2). The audio injection and the segment-aware
    # modulation live inside the block, so the loop body has nothing left to do.
    blocks=pipe.transformer.blocks,
    forward_pattern=ForwardPattern.Pattern_2,
    check_forward_pattern=True,
    # `guidance_scale > 1` runs the conditional and the unconditional pass as two sequential
    # transformer calls, same as Wan 2.1/2.2.
    has_separate_cfg=True,
    **kwargs,
  )


# NOTE: The registry returns the first class-name prefix that matches, so "WanAnimate2" must be
# registered before "Wan" ("WanAnimate2*" class names also start with "Wan").
@BlockAdapterRegister.register("WanAnimate2")
def wan_animate_2_adapter(pipe, **kwargs) -> BlockAdapter:
  from diffusers import WanAnimate2Transformer3DModel

  _relaxed_assert(pipe.transformer, WanAnimate2Transformer3DModel)

  from cache_dit.caching.patch_functors import WanAnimate2PatchFunctor

  # Wan-Animate-2 guides with `ClassifierFreeGuidance`: `num_conditions == 2` runs the conditional
  # and unconditional passes as two sequential transformer calls (the base checkpoint uses
  # `guidance_scale=3.0`), while the distilled checkpoint keeps `guidance_scale=1.0` and runs a
  # single call per step. The two cases need different cond/uncond cache bookkeeping.
  guider = getattr(pipe, "guider", None)
  has_separate_cfg = bool(guider is not None and getattr(guider, "num_conditions", 1) > 1)

  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    # The block list is named `blocks` (not `transformer_blocks`), and the blocks return a single
    # tensor with only `hidden_states` in their signature.
    blocks=pipe.transformer.blocks,
    forward_pattern=ForwardPattern.Pattern_3,
    patch_functor=WanAnimate2PatchFunctor(),
    check_forward_pattern=True,
    has_separate_cfg=has_separate_cfg,
    **kwargs,
  )


@BlockAdapterRegister.register("Wan")
def wan_adapter(pipe, **kwargs) -> BlockAdapter:
  from diffusers import (
    WanTransformer3DModel,
    WanVACETransformer3DModel,
  )
  from cache_dit.caching.patch_functors import WanVACEPatchFunctor

  _relaxed_assert(
    pipe.transformer,
    (WanTransformer3DModel, WanVACETransformer3DModel),
  )
  cls_name = pipe.transformer.__class__.__name__  # type: str
  patch_functor = WanVACEPatchFunctor() if cls_name.startswith("WanVACE") else None

  if getattr(pipe, "transformer_2", None):
    _relaxed_assert(
      pipe.transformer_2,
      (WanTransformer3DModel, WanVACETransformer3DModel),
    )
    # Wan 2.2 MoE
    return BlockAdapter(
      pipe=pipe,
      transformer=[
        pipe.transformer,
        pipe.transformer_2,
      ],
      blocks=[
        pipe.transformer.blocks,
        pipe.transformer_2.blocks,
      ],
      forward_pattern=[
        ForwardPattern.Pattern_2,
        ForwardPattern.Pattern_2,
      ],
      patch_functor=patch_functor,
      check_forward_pattern=True,
      has_separate_cfg=True,
      **kwargs,
    )
  else:
    # Wan 2.1 or Transformer only case
    return BlockAdapter(
      pipe=pipe,
      transformer=pipe.transformer,
      blocks=pipe.transformer.blocks,
      forward_pattern=ForwardPattern.Pattern_2,
      patch_functor=patch_functor,
      check_forward_pattern=True,
      has_separate_cfg=True,
      **kwargs,
    )


@BlockAdapterRegister.register("HunyuanVideo")
def hunyuanvideo_adapter(pipe, **kwargs) -> BlockAdapter:
  from diffusers import HunyuanVideoTransformer3DModel

  transformer_cls_name: str = pipe.transformer.__class__.__name__
  supported_transformers = (HunyuanVideoTransformer3DModel, )
  try:
    from diffusers import HunyuanVideo15Transformer3DModel

    supported_transformers += (HunyuanVideo15Transformer3DModel, )
  except ImportError:
    HunyuanVideo15Transformer3DModel = None  # requires diffusers>=0.36.dev0

  _relaxed_assert(pipe.transformer, supported_transformers)

  if transformer_cls_name.startswith("HunyuanVideo15"):
    # HunyuanVideo 1.5, has speparate cfg for conditional and unconditional forward
    # Reference:
    # - https://huggingface.co/hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v/blob/main/guider/guider_config.json#L4
    # - https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/hunyuan_video1_5/pipeline_hunyuan_video1_5.py#L753
    return BlockAdapter(
      pipe=pipe,
      transformer=pipe.transformer,
      blocks=pipe.transformer.transformer_blocks,
      forward_pattern=ForwardPattern.Pattern_0,
      check_forward_pattern=True,
      has_separate_cfg=True,
      **kwargs,
    )
  else:
    # HunyuanVideo 1.0
    return BlockAdapter(
      pipe=pipe,
      transformer=pipe.transformer,
      blocks=[
        pipe.transformer.transformer_blocks,
        pipe.transformer.single_transformer_blocks,
      ],
      forward_pattern=[
        ForwardPattern.Pattern_0,
        ForwardPattern.Pattern_0,
      ],
      check_forward_pattern=True,
      # The type hint in diffusers is wrong
      check_num_outputs=False,
      **kwargs,
    )


@BlockAdapterRegister.register("MotifVideo")
def motif_video_adapter(pipe, **kwargs) -> BlockAdapter:
  from diffusers import MotifVideoTransformer3DModel
  from cache_dit.caching.patch_functors import MotifVideoPatchFunctor

  _relaxed_assert(pipe.transformer, MotifVideoTransformer3DModel)

  # The functor splits the sliced single-stream loops into dedicated sub-module
  # attributes before the BlockAdapter reads them: `blocks=[...]` is evaluated as a
  # constructor argument, i.e. BEFORE `__post_init__` would apply the functor. Planners
  # (CP/TP) apply it themselves as well; guard on `_is_patched` to keep re-entry silent.
  if not getattr(pipe.transformer, "_is_patched", False):
    MotifVideoPatchFunctor().apply(pipe.transformer)

  # Motif-Video guides via `AdaptiveProjectedGuidance`: `num_conditions == 2` runs the
  # conditional and unconditional passes as two sequential transformer calls (the batch
  # is never doubled), so the cond/uncond cache contexts must be tracked separately.
  guider = getattr(pipe, "guider", None)
  has_separate_cfg = bool(guider is not None and getattr(guider, "num_conditions", 1) > 1)
  if guider is not None and (float(getattr(guider, "start", 0.0) or 0.0),
                             float(getattr(guider, "stop", 1.0) or 1.0)) != (0.0, 1.0):
    # `num_conditions` is a dynamic property: a split [start, stop) window would flip the
    # parity of the two-pass bookkeeping mid-run and silently misalign the cache contexts.
    logger.warning("MotifVideo guider start/stop != (0.0, 1.0): num_conditions varies per "
                   "step, separate-CFG cache bookkeeping may be misaligned.")

  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=[
      pipe.transformer.transformer_blocks,
      pipe.transformer._motif_single_encoder_blocks,
      pipe.transformer._motif_single_decoder_blocks,
    ],
    blocks_name=[
      "transformer_blocks",
      "_motif_single_encoder_blocks",
      "_motif_single_decoder_blocks",
    ],
    forward_pattern=[
      ForwardPattern.Pattern_0,
      ForwardPattern.Pattern_0,
      ForwardPattern.Pattern_0,
    ],
    check_forward_pattern=True,
    # The return annotation of MotifVideoSingleTransformerBlock is `torch.Tensor`
    # while it actually returns a (hidden_states, encoder_hidden_states) tuple.
    check_num_outputs=False,
    has_separate_cfg=has_separate_cfg,
    patch_functor=MotifVideoPatchFunctor(),
    **kwargs,
  )


def _default_call_float(pipe, name: str) -> Optional[float]:
  """Return a numeric default of the pipeline's `__call__`, if discoverable.

  :param pipe: Pipeline instance.
  :param name: Name of the `__call__` parameter to look up.
  :returns: The parameter default, or `None` when it is missing or not a plain number.
  """

  try:
    default = inspect.signature(pipe.__call__).parameters[name].default
    if isinstance(default, (int, float)):
      return float(default)
  except (AttributeError, KeyError, TypeError, ValueError):
    pass
  return None


def _qwenimage21_default_cfg_scale(pipe) -> Optional[float]:
  """Return the pipeline's default classifier-free guidance scale, if discoverable.

  :param pipe: Pipeline instance.
  :returns: The default `true_cfg_scale` / `guidance_scale`, or `None` when unknown.
  """

  default = _default_call_float(pipe, "true_cfg_scale")
  if default is not None:
    return default

  guider = getattr(pipe, "guider", None)
  if guider is not None:
    for attr in ("guidance_scale", "true_cfg_scale"):
      value = getattr(guider, attr, None)
      if isinstance(value, (int, float)):
        return float(value)
      config = getattr(guider, "config", None)
      value = getattr(config, attr, None)
      if isinstance(value, (int, float)):
        return float(value)

  return None


# NOTE: The registry returns the first class-name prefix that matches, so "QwenImage21" must be
# registered before "QwenImage" ("QwenImage21*" class names also start with "QwenImage").
@BlockAdapterRegister.register("QwenImage21")
def qwenimage21_adapter(pipe, **kwargs) -> BlockAdapter:
  try:
    from diffusers import QwenImage21Transformer2DModel
  except ImportError:
    QwenImage21Transformer2DModel = None  # requires a diffusers build with Qwen-Image 2.1

  _relaxed_assert(pipe.transformer, QwenImage21Transformer2DModel)

  from cache_dit.caching.patch_functors import QwenImage21PatchFunctor

  cfg_scale = _qwenimage21_default_cfg_scale(pipe)
  # Qwen-Image 2.1 samples without guidance by default, so each step runs the transformer once.
  # Cache-dit would otherwise assume two forwards per step (cond -> uncond) and count every step
  # as half a step. Unknown defaults keep the previous, guidance-enabled behavior.
  has_separate_cfg = True if cfg_scale is None else cfg_scale > 1.0
  if not has_separate_cfg:
    logger.warning(
      f"Qwen-Image 2.1 defaults to true_cfg_scale={cfg_scale}, which runs a single transformer "
      "forward per step; separate CFG bookkeeping is disabled. Set `enable_separate_cfg=True` in "
      "the cache config if you pass `true_cfg_scale > 1` together with a negative prompt.")

  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    # Single-stream blocks; the transformer forward is patched by the functor so that the
    # per-layer prefix KV cache survives the cached-blocks wrapper.
    blocks=pipe.transformer.transformer_blocks,
    forward_pattern=ForwardPattern.Pattern_3,
    patch_functor=QwenImage21PatchFunctor(),
    check_forward_pattern=True,
    has_separate_cfg=has_separate_cfg,
    **kwargs,
  )


@BlockAdapterRegister.register("QwenImage")
def qwenimage_adapter(pipe, **kwargs) -> BlockAdapter:
  try:
    from diffusers import QwenImageTransformer2DModel
  except ImportError:
    QwenImageTransformer2DModel = None  # requires diffusers>=0.35.2

  _relaxed_assert(pipe.transformer, QwenImageTransformer2DModel)

  pipe_cls_name: str = pipe.__class__.__name__
  if pipe_cls_name.startswith("QwenImageControlNet"):
    from cache_dit.caching.patch_functors import (
      QwenImageControlNetPatchFunctor, )

    return BlockAdapter(
      pipe=pipe,
      transformer=pipe.transformer,
      blocks=pipe.transformer.transformer_blocks,
      forward_pattern=ForwardPattern.Pattern_1,
      patch_functor=QwenImageControlNetPatchFunctor(),
      check_forward_pattern=True,
      has_separate_cfg=True,
    )
  else:
    return BlockAdapter(
      pipe=pipe,
      transformer=pipe.transformer,
      blocks=pipe.transformer.transformer_blocks,
      forward_pattern=ForwardPattern.Pattern_1,
      check_forward_pattern=True,
      has_separate_cfg=True,
      **kwargs,
    )


@BlockAdapterRegister.register("Echo")
def echo_adapter(pipe, **kwargs) -> BlockAdapter:
  """Adapter for the Echo (JoyAI-Echo) text-to-video+audio modular pipeline.

  Echo reuses the LTX-2 audiovisual backbone (`LTX2VideoTransformer3DModel`), so it shares the LTX-2
  block contract and the `LTX2PatchFunctor` rewrite. The pipeline itself is guidance-free
  (positive-only conditioning, one transformer forward per stochastic DMD step), hence no separate
  CFG bookkeeping.

  :param pipe: Echo modular pipeline.
  :param kwargs: Extra `BlockAdapter` arguments.
  :returns: The block adapter for Echo.
  """
  try:
    from diffusers import LTX2VideoTransformer3DModel
  except ImportError:
    LTX2VideoTransformer3DModel = None  # requires a diffusers build with the LTX-2 backbone

  from cache_dit.caching.patch_functors import LTX2PatchFunctor

  _relaxed_assert(pipe.transformer, LTX2VideoTransformer3DModel)

  guider = getattr(pipe, "guider", None)
  has_separate_cfg = bool(guider is not None and getattr(guider, "num_conditions", 1) > 1)

  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.transformer_blocks,
    # Same treatment as LTX-2: `audio_hidden_states` plays the `encoder_hidden_states` role of
    # Pattern_0, and the functor rewrites the keyword block call into the positional contract.
    forward_pattern=ForwardPattern.Pattern_0,
    patch_functor=LTX2PatchFunctor(),
    check_forward_pattern=False,
    has_separate_cfg=has_separate_cfg,
    **kwargs,
  )


@BlockAdapterRegister.register("LTX")
def ltxvideo_adapter(pipe, **kwargs) -> BlockAdapter:
  # LTX-1 (LTXVideoTransformer3DModel) and LTX-2 / LTX-2.5 (LTX2VideoTransformer3DModel) share
  # the `transformer_blocks` structure, but differ in block forward IO:
  # - LTX-1 blocks return only `hidden_states`  -> Pattern_2
  # - LTX-2 blocks return `(hidden_states, audio_hidden_states)` -> Pattern_0
  from diffusers import LTXVideoTransformer3DModel

  cls_name: str = pipe.transformer.__class__.__name__
  try:
    from diffusers import LTX2VideoTransformer3DModel
    from cache_dit.caching.patch_functors import LTX2PatchFunctor
  except ImportError:
    LTX2VideoTransformer3DModel = None  # requires newer diffusers
    LTX2PatchFunctor = None

  # LTX-2.5 ships the same `LTX2VideoTransformer3DModel` class, so the class-name prefix check stays
  # for older diffusers while `isinstance` keeps working if the class is ever renamed.
  is_ltx2: bool = cls_name.startswith("LTX2") or (LTX2VideoTransformer3DModel is not None
                                                  and isinstance(pipe.transformer,
                                                                 LTX2VideoTransformer3DModel))
  forward_pattern = ForwardPattern.Pattern_0 if is_ltx2 else ForwardPattern.Pattern_2

  patch_functor = None
  if is_ltx2:
    if LTX2PatchFunctor is None:
      logger.warning("LTX-2 detected but `LTX2PatchFunctor` is unavailable; block calls keep the "
                     "upstream keyword-argument form and caching will not be applied.")
    else:
      patch_functor = LTX2PatchFunctor()

  supported_transformers = (LTXVideoTransformer3DModel, )
  if LTX2VideoTransformer3DModel is not None:
    supported_transformers = supported_transformers + (LTX2VideoTransformer3DModel, )

  _relaxed_assert(pipe.transformer, supported_transformers)

  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.transformer_blocks,
    forward_pattern=forward_pattern,
    patch_functor=patch_functor,
    # Tips: Treat the audio_hidden_states in LTX-2 as encoder_hidden_states in Pattern_0
    # while using cache. This values will not affect the correctness since audio_hidden_states
    # will be cache and update normally.
    check_forward_pattern=False,
    **kwargs,
  )


@BlockAdapterRegister.register("Allegro")
def allegro_adapter(pipe, **kwargs) -> BlockAdapter:
  from diffusers import AllegroTransformer3DModel

  _relaxed_assert(pipe.transformer, AllegroTransformer3DModel)
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.transformer_blocks,
    forward_pattern=ForwardPattern.Pattern_2,
    check_forward_pattern=True,
    **kwargs,
  )


@BlockAdapterRegister.register("CogView3Plus")
def cogview3plus_adapter(pipe, **kwargs) -> BlockAdapter:
  from diffusers import CogView3PlusTransformer2DModel

  _relaxed_assert(pipe.transformer, CogView3PlusTransformer2DModel)
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.transformer_blocks,
    forward_pattern=ForwardPattern.Pattern_0,
    check_forward_pattern=True,
    **kwargs,
  )


@BlockAdapterRegister.register("CogView4")
def cogview4_adapter(pipe, **kwargs) -> BlockAdapter:
  from diffusers import CogView4Transformer2DModel

  _relaxed_assert(pipe.transformer, CogView4Transformer2DModel)
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.transformer_blocks,
    forward_pattern=ForwardPattern.Pattern_0,
    check_forward_pattern=True,
    has_separate_cfg=True,
    **kwargs,
  )


@BlockAdapterRegister.register("Cosmos")
def cosmos_adapter(pipe, **kwargs) -> BlockAdapter:
  try:
    from diffusers import Cosmos3OmniTransformer
  except ImportError:
    Cosmos3OmniTransformer = None

  if Cosmos3OmniTransformer is not None and isinstance(pipe.transformer, Cosmos3OmniTransformer):
    # Cosmos3 Omni (Cosmos3-Nano/Super): MoT dual-stream (und/gen) DiT.
    # Needs a patch functor to reorder the block contract so the cache
    # decision tracks the denoised gen stream (see functor docstring).
    from ..patch_functors import Cosmos3OmniPatchFunctor

    return BlockAdapter(
      pipe=pipe,
      transformer=pipe.transformer,
      blocks=pipe.transformer.layers,
      forward_pattern=ForwardPattern.Pattern_0,
      patch_functor=Cosmos3OmniPatchFunctor(),
      check_forward_pattern=True,
      # Cosmos3OmniPipeline runs cond/uncond as separate transformer calls.
      has_separate_cfg=True,
      **kwargs,
    )

  return _cosmos_predict2_adapter(pipe, has_separate_cfg=True, **kwargs)


def _cosmos_predict2_adapter(
  pipe,
  has_separate_cfg: bool,
  **kwargs,
) -> BlockAdapter:
  """Build the block adapter for the Cosmos Predict2 backbone.

  `CosmosTransformer3DModel` runs a uniform `transformer_blocks` stack with the
  Pattern_2 contract (hidden + encoder states in, hidden states out), shared by
  the Cosmos Predict video pipelines and the Anima text-to-image modular pipeline.

  :param pipe: Pipeline owning a `CosmosTransformer3DModel` transformer.
  :param has_separate_cfg: Whether cond/uncond run as separate forwards.
  :returns: The BlockAdapter for the `transformer_blocks` stack.
  """
  from diffusers import CosmosTransformer3DModel

  _relaxed_assert(pipe.transformer, CosmosTransformer3DModel)
  # diffusers does not declare `_repeated_blocks` for Cosmos, so regional quantization and
  # `compile_repeated_blocks()` cannot discover the denoising stack on their own.
  declare_repeated_blocks(pipe.transformer)
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.transformer_blocks,
    forward_pattern=ForwardPattern.Pattern_2,
    check_forward_pattern=True,
    has_separate_cfg=has_separate_cfg,
    **kwargs,
  )


@BlockAdapterRegister.register("Anima")
def anima_adapter(pipe, **kwargs) -> BlockAdapter:
  """Adapter for the Anima text-to-image modular pipeline.

  Anima reuses the Cosmos Predict2 DiT (`CosmosTransformer3DModel`); the
  model-specific conditioning is folded into `AnimaTextConditioner` before the
  denoise loop, so only the shared Cosmos block contract needs caching. Unlike
  the regular Cosmos pipelines, guidance is driven by the modular guider
  component, so `has_separate_cfg` follows the guider's condition count
  instead of a fixed True (a guidance_scale of 1.0 samples without a negative
  branch and must not halve the step accounting).

  :param pipe: The `AnimaModularPipeline` instance.
  :param kwargs: Additional keyword arguments forwarded to the `BlockAdapter`.
  :returns: The BlockAdapter for the `transformer_blocks` stack.
  """

  guider = getattr(pipe, "guider", None)
  if guider is None:
    logger.warning("Anima adapter found no guider component; assuming no separate CFG.")
  has_separate_cfg = bool(guider is not None and getattr(guider, "num_conditions", 1) > 1)

  return _cosmos_predict2_adapter(pipe, has_separate_cfg, **kwargs)


@BlockAdapterRegister.register("EasyAnimate")
def easyanimate_adapter(pipe, **kwargs) -> BlockAdapter:
  from diffusers import EasyAnimateTransformer3DModel

  _relaxed_assert(pipe.transformer, EasyAnimateTransformer3DModel)
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.transformer_blocks,
    forward_pattern=ForwardPattern.Pattern_0,
    check_forward_pattern=True,
    **kwargs,
  )


@BlockAdapterRegister.register("SkyReelsV2")
def skyreelsv2_adapter(pipe, **kwargs) -> BlockAdapter:
  from diffusers import SkyReelsV2Transformer3DModel

  _relaxed_assert(pipe.transformer, SkyReelsV2Transformer3DModel)
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.blocks,
    # NOTE: Use Pattern_3 instead of Pattern_2 because the
    # encoder_hidden_states will never change in the blocks
    # forward loop.
    forward_pattern=ForwardPattern.Pattern_3,
    check_forward_pattern=True,
    has_separate_cfg=True,
    **kwargs,
  )


@BlockAdapterRegister.register("StableDiffusion3")
def sd3_adapter(pipe, **kwargs) -> BlockAdapter:
  from diffusers import SD3Transformer2DModel

  _relaxed_assert(pipe.transformer, SD3Transformer2DModel)
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.transformer_blocks,
    forward_pattern=ForwardPattern.Pattern_1,
    check_forward_pattern=True,
    **kwargs,
  )


@BlockAdapterRegister.register("ConsisID")
def consisid_adapter(pipe, **kwargs) -> BlockAdapter:
  from diffusers import ConsisIDTransformer3DModel

  _relaxed_assert(pipe.transformer, ConsisIDTransformer3DModel)
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.transformer_blocks,
    forward_pattern=ForwardPattern.Pattern_0,
    check_forward_pattern=True,
    **kwargs,
  )


@BlockAdapterRegister.register("DiT")
def dit_adapter(pipe, **kwargs) -> BlockAdapter:
  from diffusers import DiTTransformer2DModel
  from cache_dit.caching.patch_functors import DiTPatchFunctor

  _relaxed_assert(pipe.transformer, DiTTransformer2DModel)
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.transformer_blocks,
    forward_pattern=ForwardPattern.Pattern_3,
    patch_functor=DiTPatchFunctor(),
    check_forward_pattern=True,
    **kwargs,
  )


@BlockAdapterRegister.register("Amused")
def amused_adapter(pipe, **kwargs) -> BlockAdapter:
  from diffusers import UVit2DModel

  _relaxed_assert(pipe.transformer, UVit2DModel)
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.transformer_layers,
    forward_pattern=ForwardPattern.Pattern_3,
    check_forward_pattern=True,
    **kwargs,
  )


# NOTE: The registry returns the first class-name prefix that matches, so "BriaFibo2" must be
# registered before "BriaFibo" ("BriaFibo2*" class names also start with "BriaFibo").
@BlockAdapterRegister.register("BriaFibo2")
def bria_fibo2_adapter(pipe, **kwargs) -> BlockAdapter:
  try:
    from diffusers import BriaFibo2Transformer2DModel
  except ImportError:
    BriaFibo2Transformer2DModel = None  # requires a diffusers build with Bria Fibo 2.0

  from cache_dit.caching.patch_functors import BriaFibo2PatchFunctor

  _relaxed_assert(pipe.transformer, BriaFibo2Transformer2DModel)

  # Fibo 2.0 runs a real CFG pass (two transformer forwards per step) unless the caller lowers
  # `guidance_scale` to 1, which is what the distilled `Fibo-2.0-edit-turbo` checkpoint does. The
  # pipeline default is authoritative here; a turbo run must set `enable_separate_cfg=False` in the
  # cache config, otherwise cache-dit counts two forwards as one step and almost nothing is cached.
  guidance_scale = _default_call_float(pipe, "guidance_scale")
  has_separate_cfg = True if guidance_scale is None else guidance_scale > 1.0
  if not has_separate_cfg:
    logger.info(
      f"Bria Fibo 2.0 defaults to guidance_scale={guidance_scale}, i.e. a single transformer "
      "forward per step; separate CFG bookkeeping is disabled. Pass "
      "`enable_separate_cfg=True` in the cache config if the run enables guidance.")

  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    # Fibo 2.0 has a single-stream `layers` stack; `noise_refiner` and `perceiver` run before it
    # and stay uncached. The functor distributes the per-injection-layer text bundles.
    blocks=pipe.transformer.layers,
    forward_pattern=ForwardPattern.Pattern_3,
    patch_functor=BriaFibo2PatchFunctor(),
    # Fibo 2.0 blocks take `x` instead of `hidden_states`.
    check_forward_pattern=False,
    has_separate_cfg=has_separate_cfg,
    **kwargs,
  )


@BlockAdapterRegister.register("BriaFibo")
def bria_fibo_adapter(pipe, **kwargs) -> BlockAdapter:
  from diffusers import BriaFiboTransformer2DModel
  from cache_dit.caching.patch_functors import BriaFiboPatchFunctor

  _relaxed_assert(pipe.transformer, BriaFiboTransformer2DModel)
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=[
      pipe.transformer.transformer_blocks,
      pipe.transformer.single_transformer_blocks,
    ],
    forward_pattern=[
      ForwardPattern.Pattern_1,
      ForwardPattern.Pattern_1,
    ],
    check_forward_pattern=False,  # single blocks are patched to Pattern_1 later
    patch_functor=BriaFiboPatchFunctor(),
    **kwargs,
  )


@BlockAdapterRegister.register("Bria")
def bria_adapter(pipe, **kwargs) -> BlockAdapter:
  try:
    from diffusers import BriaTransformer2DModel
  except ImportError:
    BriaTransformer2DModel = None  # requires diffusers>=0.36.dev0

  _relaxed_assert(pipe.transformer, BriaTransformer2DModel)
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=[
      pipe.transformer.transformer_blocks,
      pipe.transformer.single_transformer_blocks,
    ],
    forward_pattern=[
      ForwardPattern.Pattern_0,
      ForwardPattern.Pattern_0,
    ],
    check_forward_pattern=True,
    **kwargs,
  )


@BlockAdapterRegister.register("Lumina")
def lumina2_adapter(pipe, **kwargs) -> BlockAdapter:
  from diffusers import Lumina2Transformer2DModel
  from diffusers import LuminaNextDiT2DModel

  _relaxed_assert(pipe.transformer, (Lumina2Transformer2DModel, LuminaNextDiT2DModel))
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.layers,
    forward_pattern=ForwardPattern.Pattern_3,
    check_forward_pattern=True,
    **kwargs,
  )


@BlockAdapterRegister.register("OmniGen")
def omnigen_adapter(pipe, **kwargs) -> BlockAdapter:
  from diffusers import OmniGenTransformer2DModel

  _relaxed_assert(pipe.transformer, OmniGenTransformer2DModel)
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.layers,
    forward_pattern=ForwardPattern.Pattern_3,
    check_forward_pattern=True,
    **kwargs,
  )


@BlockAdapterRegister.register("PixArt")
def pixart_adapter(pipe, **kwargs) -> BlockAdapter:
  from diffusers import PixArtTransformer2DModel

  _relaxed_assert(pipe.transformer, PixArtTransformer2DModel)
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.transformer_blocks,
    forward_pattern=ForwardPattern.Pattern_3,
    check_forward_pattern=True,
    **kwargs,
  )


# NOTE: `SanaWMTransformer3DModel` and `SanaWMLTX2RefinerTransformer3DModel` share the `SanaWM`
# prefix, and the registry returns the first matching prefix, so a single registration dispatches
# on the block-stack attribute instead of depending on registration order.
@BlockAdapterRegister.register("SanaWM")
def sana_wm_adapter(pipe, **kwargs) -> BlockAdapter:
  """BlockAdapter for SANA-WM (`SanaWMPipeline` and `SanaWMLTX2Refiner`).

  Two transformers share the `SanaWM` prefix and are dispatched here by their block stack:

  * Stage 1 (`SanaWMTransformer3DModel`) caches `blocks` with `Pattern_3`: the block contract is
    `(x, y, t, mask, THW, rotary_emb, block_mask, camera_conditions, ucpe_ray_transforms,
    plucker_emb, frame_valid_mask, chunk_size) -> x`, called positionally, so the cached
    wrapper's `block(hidden_states, *args)` call carries all eleven extra arguments through
    unchanged and no PatchFunctor is needed. Every one of those extras (text embeddings,
    modulation, RoPE, camera raymaps, UCPE transforms, Plücker embeddings) is built once per
    forward *outside* the loop, so a cached step cannot go stale. The parameters are named `x` /
    `y` instead of `hidden_states` / `encoder_hidden_states`, hence the disabled signature check.
  * Stage 2 (`SanaWMLTX2RefinerTransformer3DModel`) caches `transformer_blocks` with `Pattern_3`
    and needs `SanaWMRefinerPatchFunctor`: the collapsed block loop would hand every layer the
    first layer's KV slot, and the two `kv_cache_mode` passes that *write* the sliding-window
    cache must not be skipped.

  CFG runs inside a single forward (the pipeline concatenates `[negative, conditional]` into one
  batch), so `has_separate_cfg` stays `False`.

  :param pipe: `SanaWMPipeline` or `SanaWMLTX2Refiner` instance.
  :param kwargs: Additional keyword arguments forwarded to the `BlockAdapter`.
  :returns: The BlockAdapter covering the stage-1 or stage-2 denoising stack.
  """
  from diffusers import SanaWMTransformer3DModel
  from diffusers.models.transformers.transformer_sana_wm_refiner import (
    SanaWMLTX2RefinerTransformer3DModel, )

  transformer = getattr(pipe, "transformer", None)
  if isinstance(transformer, SanaWMLTX2RefinerTransformer3DModel):
    from ..patch_functors import SanaWMRefinerPatchFunctor

    # The functor only rebinds `forward` (it keeps the `ModuleList` object), so letting
    # `__post_init__` apply it is enough -- `blocks` below is the same list either way.
    patch_functor = SanaWMRefinerPatchFunctor()
    return BlockAdapter(
      pipe=pipe,
      transformer=transformer,
      blocks=transformer.transformer_blocks,
      blocks_name="transformer_blocks",
      forward_pattern=ForwardPattern.Pattern_3,
      patch_functor=patch_functor,
      has_separate_cfg=False,
      **kwargs,
    )

  _relaxed_assert(transformer, SanaWMTransformer3DModel)
  return BlockAdapter(
    pipe=pipe,
    transformer=transformer,
    blocks=transformer.blocks,
    blocks_name="blocks",
    forward_pattern=ForwardPattern.Pattern_3,
    check_forward_pattern=False,
    has_separate_cfg=False,
    **kwargs,
  )


@BlockAdapterRegister.register("Sana")
def sana_adapter(pipe, **kwargs) -> BlockAdapter:
  from diffusers import SanaTransformer2DModel

  _relaxed_assert(pipe.transformer, SanaTransformer2DModel)
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.transformer_blocks,
    forward_pattern=ForwardPattern.Pattern_3,
    check_forward_pattern=True,
    **kwargs,
  )


@BlockAdapterRegister.register("StableAudio3")
def stable_audio_3_adapter(pipe, **kwargs) -> BlockAdapter:
  """Adapter for the Stable Audio 3 text-to-audio / audio-to-audio / inpaint pipelines.

  `StableAudio3DiTBlock.forward(hidden_states, context, global_modulation, rope, context_mask,
  local_seq, num_memory_tokens)` takes `hidden_states` first and returns a single tensor, and the
  transformer calls it with `x` positional plus everything else by keyword, so the cached
  `Pattern_3` wrapper binds `hidden_states` by name and replays the rest untouched: no PatchFunctor
  is needed. Every extra argument is computed once per step outside the block loop (the text/duration
  context projection, the 6-way AdaLN modulation, the full RoPE table and the 64 memory tokens), which
  is what makes block-level caching valid on an audio sequence of only ~75 (1 s) to ~323 (30 s)
  latent frames.

  The registration name must stay ahead of `StableAudio` (Stable Audio 1/2): adapter lookup is a
  first-hit `startswith` over the registry, so `"StableAudio"` would otherwise capture every
  `StableAudio3*` pipeline and fail its own transformer assertion.

  :param pipe: Stable Audio 3 pipeline.
  :param kwargs: Extra `BlockAdapter` arguments.
  :returns: The block adapter for Stable Audio 3.
  """

  from diffusers import StableAudio3DiTModel

  _relaxed_assert(pipe.transformer, StableAudio3DiTModel)

  # All three SA3 pipelines run **one** transformer forward per step even when guidance is on: they
  # concatenate the latents along the batch axis (`torch.cat([latents, latents])`), call the
  # transformer once and split the velocity into `(uncond, text)`. `has_separate_cfg=False` is
  # therefore unconditional — deriving it from `guidance_scale` (the LongCat-AudioDiT rule) would halve
  # the counted steps as soon as the non-distilled `-base` checkpoint is used with guidance > 1.
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.transformer_blocks,
    forward_pattern=ForwardPattern.Pattern_3,
    check_forward_pattern=True,
    has_separate_cfg=False,
    **kwargs,
  )


@BlockAdapterRegister.register("StableAudio")
def stabledudio_adapter(pipe, **kwargs) -> BlockAdapter:
  from diffusers import StableAudioDiTModel

  _relaxed_assert(pipe.transformer, StableAudioDiTModel)
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.transformer_blocks,
    forward_pattern=ForwardPattern.Pattern_3,
    check_forward_pattern=True,
    **kwargs,
  )


@BlockAdapterRegister.register("VisualCloze")
def visualcloze_adapter(pipe, **kwargs) -> BlockAdapter:
  from diffusers import FluxTransformer2DModel

  _relaxed_assert(pipe.transformer, FluxTransformer2DModel)
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=[
      pipe.transformer.transformer_blocks,
      pipe.transformer.single_transformer_blocks,
    ],
    forward_pattern=[
      ForwardPattern.Pattern_1,
      ForwardPattern.Pattern_1,
    ],
    check_forward_pattern=True,
    **kwargs,
  )


@BlockAdapterRegister.register("AuraFlow")
def auraflow_adapter(pipe, **kwargs) -> BlockAdapter:
  from diffusers import AuraFlowTransformer2DModel

  _relaxed_assert(pipe.transformer, AuraFlowTransformer2DModel)
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.single_transformer_blocks,
    forward_pattern=ForwardPattern.Pattern_3,
    check_forward_pattern=True,
    **kwargs,
  )


@BlockAdapterRegister.register("Chroma")
def chroma_adapter(pipe, **kwargs) -> BlockAdapter:
  from diffusers import ChromaTransformer2DModel
  from cache_dit.caching.patch_functors import ChromaPatchFunctor

  _relaxed_assert(pipe.transformer, ChromaTransformer2DModel)
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=[
      pipe.transformer.transformer_blocks,
      pipe.transformer.single_transformer_blocks,
    ],
    forward_pattern=[
      ForwardPattern.Pattern_1,
      ForwardPattern.Pattern_3,
    ],
    patch_functor=ChromaPatchFunctor(),
    check_forward_pattern=True,
    has_separate_cfg=True,
    **kwargs,
  )


@BlockAdapterRegister.register("ShapE")
def shape_adapter(pipe, **kwargs) -> BlockAdapter:
  from diffusers import PriorTransformer

  _relaxed_assert(pipe.prior, PriorTransformer)
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.prior,
    blocks=pipe.prior.transformer_blocks,
    forward_pattern=ForwardPattern.Pattern_3,
    check_forward_pattern=True,
    **kwargs,
  )


# NOTE: The registry returns the first class-name prefix that matches, so "HiDreamO1Image" must be
# registered before "HiDream" (HiDream-I1), which would otherwise capture it.
@BlockAdapterRegister.register("HiDreamO1Image")
def hidream_o1_image_adapter(pipe, **kwargs) -> BlockAdapter:
  """BlockAdapter for HiDream-O1-Image (`HiDreamO1ImagePipeline` /
  `HiDreamO1ImageTransformer2DModel`).

  The transformer is cache-dit's built-in copy (`cache_dit.models.pipelines.hidream_o1_image`), so
  it is not a diffusers module and no strict type check applies. Its denoising stack is
  `transformer.layers` and each block takes `(hidden_states, rope, attention_mask, cp_config)` and
  returns `hidden_states` (Pattern_3).

  Two model specifics matter here:

  * `has_separate_cfg=True`: the pipeline runs one `transformer.forward()` per CFG branch
    (conditional, then the `" "` unconditional branch), so the two branches need independent cache
    contexts and one shared step counter.
  * `check_forward_pattern=True`: the built-in block's first parameter is `hidden_states`, so the
    contract is verifiable, but cache-dit defaults that check off for non-diffusers transformers.

  The block loop calls one block per iteration and carries the timestep through the `<|tms_token|>`
  embedding row before the loop, so no PatchFunctor is needed.

  :param pipe: `HiDreamO1ImagePipeline` instance, or a wrapper exposing only `transformer`.
  :param kwargs: Additional keyword arguments forwarded to the `BlockAdapter`.
  :returns: The BlockAdapter covering the `layers` denoising stack.
  """
  _relaxed_assert(pipe.transformer)
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.layers,
    blocks_name="layers",
    forward_pattern=ForwardPattern.Pattern_3,
    check_forward_pattern=True,
    has_separate_cfg=True,
    **kwargs,
  )


@BlockAdapterRegister.register("HiDream")
def hidream_adapter(pipe, **kwargs) -> BlockAdapter:
  # NOTE: Need to patch Transformer forward to fully support
  # double_stream_blocks and single_stream_blocks, namely, need
  # to remove the logics inside the blocks forward loop:
  # https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_hidream_image.py#L893
  # https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_hidream_image.py#L927
  from diffusers import HiDreamImageTransformer2DModel
  from cache_dit.caching.patch_functors import HiDreamPatchFunctor

  _relaxed_assert(pipe.transformer, HiDreamImageTransformer2DModel)
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=[
      pipe.transformer.double_stream_blocks,
      pipe.transformer.single_stream_blocks,
    ],
    forward_pattern=[
      ForwardPattern.Pattern_0,
      ForwardPattern.Pattern_3,
    ],
    patch_functor=HiDreamPatchFunctor(),
    # NOTE: The type hint in diffusers is wrong
    check_forward_pattern=True,
    check_num_outputs=True,
    **kwargs,
  )


@BlockAdapterRegister.register("HunyuanDiT")
def hunyuandit_adapter(pipe, **kwargs) -> BlockAdapter:
  from diffusers import HunyuanDiT2DModel, HunyuanDiT2DControlNetModel
  from cache_dit.caching.patch_functors import HunyuanDiTPatchFunctor

  _relaxed_assert(
    pipe.transformer,
    (HunyuanDiT2DModel, HunyuanDiT2DControlNetModel),
  )
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.blocks,
    forward_pattern=ForwardPattern.Pattern_3,
    patch_functor=HunyuanDiTPatchFunctor(),
    check_forward_pattern=True,
    **kwargs,
  )


@BlockAdapterRegister.register("HunyuanDiTPAG")
def hunyuanditpag_adapter(pipe, **kwargs) -> BlockAdapter:
  from diffusers import HunyuanDiT2DModel
  from cache_dit.caching.patch_functors import HunyuanDiTPatchFunctor

  _relaxed_assert(pipe.transformer, HunyuanDiT2DModel)
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.blocks,
    forward_pattern=ForwardPattern.Pattern_3,
    patch_functor=HunyuanDiTPatchFunctor(),
    check_forward_pattern=True,
    **kwargs,
  )


@BlockAdapterRegister.register("Kandinsky5")
def kandinsky5_adapter(pipe, **kwargs) -> BlockAdapter:
  try:
    from diffusers import Kandinsky5Transformer3DModel
  except ImportError:
    Kandinsky5Transformer3DModel = None  # requires diffusers>=0.36.dev

  _relaxed_assert(pipe.transformer, Kandinsky5Transformer3DModel)
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.visual_transformer_blocks,
    forward_pattern=ForwardPattern.Pattern_3,  # or Pattern_2
    has_separate_cfg=True,
    check_forward_pattern=False,
    check_num_outputs=False,
    **kwargs,
  )


@BlockAdapterRegister.register("PRX")
def prx_adapter(pipe, **kwargs) -> BlockAdapter:
  try:
    from diffusers import PRXTransformer2DModel
  except ImportError:
    PRXTransformer2DModel = None  # requires diffusers>=0.36.dev0

  _relaxed_assert(pipe.transformer, PRXTransformer2DModel)
  # diffusers does not declare `_repeated_blocks` for PRX, so regional quantization and
  # `compile_repeated_blocks()` cannot discover the denoising stack on their own.
  declare_repeated_blocks(pipe.transformer)
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.blocks,
    forward_pattern=ForwardPattern.Pattern_3,
    check_forward_pattern=True,
    check_num_outputs=False,
    **kwargs,
  )


@BlockAdapterRegister.register("HunyuanImage")
def hunyuan_image_adapter(pipe, **kwargs) -> BlockAdapter:
  try:
    from diffusers import HunyuanImageTransformer2DModel
  except ImportError:
    HunyuanImageTransformer2DModel = None  # requires diffusers>=0.36

  _relaxed_assert(pipe.transformer, HunyuanImageTransformer2DModel)
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=[
      pipe.transformer.transformer_blocks,
      pipe.transformer.single_transformer_blocks,
    ],
    forward_pattern=[
      ForwardPattern.Pattern_0,
      ForwardPattern.Pattern_0,
    ],
    # set `has_separate_cfg` as True to enable separate cfg caching
    # since in hyimage-2.1 the `guider_state` contains 2 input batches.
    # The cfg is `enabled` by default in AdaptiveProjectedMixGuidance.
    has_separate_cfg=True,
    check_forward_pattern=True,
    **kwargs,
  )


@BlockAdapterRegister.register("ChronoEdit")
def chronoedit_adapter(pipe, **kwargs) -> BlockAdapter:
  try:
    from diffusers import ChronoEditTransformer3DModel
  except ImportError:
    ChronoEditTransformer3DModel = None  # requires diffusers>=0.36.dev0

  _relaxed_assert(pipe.transformer, ChronoEditTransformer3DModel)
  # Same as Wan 2.1 adapter
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.blocks,
    forward_pattern=ForwardPattern.Pattern_2,
    check_forward_pattern=True,
    has_separate_cfg=True,
    **kwargs,
  )


@BlockAdapterRegister.register("MingImage")
def ming_image_adapter(pipe, **kwargs) -> BlockAdapter:
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.layers,
    blocks_name="layers",
    forward_pattern=ForwardPattern.Pattern_3,
    # Ming uses Z-Image blocks whose first argument is named `x`.
    check_forward_pattern=False,
    has_separate_cfg=False,
    **kwargs,
  )


@BlockAdapterRegister.register("ZImage")
def zimage_adapter(pipe, **kwargs) -> BlockAdapter:
  from cache_dit.caching.patch_functors import ZImageControlNetPatchFunctor

  try:
    from diffusers import ZImageTransformer2DModel
  except ImportError:
    ZImageTransformer2DModel = None  # requires diffusers>=0.36.dev0

  has_controlnet = hasattr(pipe, "controlnet") and pipe.controlnet is not None
  _relaxed_assert(pipe.transformer, ZImageTransformer2DModel)
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.layers,
    forward_pattern=ForwardPattern.Pattern_3,
    patch_functor=ZImageControlNetPatchFunctor() if has_controlnet else None,
    # ZImage DON'T have 'hidden_states' (use 'x') in its block
    # forward signature. So we disable the forward pattern check here.
    check_forward_pattern=False,
    **kwargs,
  )


@BlockAdapterRegister.register("OvisImage")
def ovis_image_adapter(pipe, **kwargs) -> BlockAdapter:
  try:
    from diffusers import OvisImageTransformer2DModel
  except ImportError:
    OvisImageTransformer2DModel = None  # requires diffusers>=0.36.dev

  _relaxed_assert(pipe.transformer, OvisImageTransformer2DModel)
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=[
      pipe.transformer.transformer_blocks,
      pipe.transformer.single_transformer_blocks,
    ],
    forward_pattern=[
      ForwardPattern.Pattern_1,
      ForwardPattern.Pattern_1,
    ],
    check_forward_pattern=True,
    has_separate_cfg=True,
    **kwargs,
  )


@BlockAdapterRegister.register("LongCatAudioDiT")
def longcat_audio_dit_adapter(pipe, **kwargs) -> BlockAdapter:
  """Adapter for the LongCat-AudioDiT text-to-speech pipeline.

  `AudioDiTBlock.forward` takes `hidden_states` first and returns a single tensor, and the
  transformer calls every block with keyword arguments only, so the cached `Pattern_3` wrapper binds
  `hidden_states` by name and forwards `timestep_embed / cond / mask / cond_mask / rope / cond_rope /
  adaln_global_out` untouched: no PatchFunctor is needed. Everything except `hidden_states` is
  computed once per step before the block loop, which is what makes block-level caching valid on an
  audio sequence of only 58 (5 s) to 351 (30 s) latent frames.

  :param pipe: LongCat-AudioDiT pipeline.
  :param kwargs: Extra `BlockAdapter` arguments.
  :returns: The block adapter for LongCat-AudioDiT.
  """

  try:
    from diffusers import LongCatAudioDiTTransformer
  except ImportError:
    LongCatAudioDiTTransformer = None  # requires a diffusers build with LongCat-AudioDiT

  _relaxed_assert(pipe.transformer, LongCatAudioDiTTransformer)

  # The pipeline always runs the conditional forward and runs the null forward only when
  # `guidance_scale > 1` (default 4.0), so the bookkeeping has to follow the reachable default.
  guidance_scale = _default_call_float(pipe, "guidance_scale")
  has_separate_cfg = True if guidance_scale is None else guidance_scale > 1.0

  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.blocks,
    forward_pattern=ForwardPattern.Pattern_3,
    check_forward_pattern=True,
    has_separate_cfg=has_separate_cfg,
    **kwargs,
  )


@BlockAdapterRegister.register("LongCatImage")
def longcat_image_adapter(pipe, **kwargs) -> BlockAdapter:
  try:
    from diffusers import LongCatImageTransformer2DModel
  except ImportError:
    LongCatImageTransformer2DModel = None  # requires diffusers>=0.36.dev

  _relaxed_assert(pipe.transformer, LongCatImageTransformer2DModel)
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=[
      pipe.transformer.transformer_blocks,
      pipe.transformer.single_transformer_blocks,
    ],
    forward_pattern=[
      ForwardPattern.Pattern_1,
      ForwardPattern.Pattern_1,
    ],
    check_forward_pattern=True,
    has_separate_cfg=True,
    **kwargs,
  )


@BlockAdapterRegister.register("GlmImage")
def glm_image_adapter(pipe, **kwargs) -> BlockAdapter:
  try:
    from diffusers import GlmImageTransformer2DModel
  except ImportError:
    GlmImageTransformer2DModel = None  # requires diffusers>=0.37.dev

  from cache_dit.caching.patch_functors import GlmImagePatchFunctor

  _relaxed_assert(pipe.transformer, GlmImageTransformer2DModel)
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.transformer_blocks,
    forward_pattern=ForwardPattern.Pattern_0,
    patch_functor=GlmImagePatchFunctor(),
    check_forward_pattern=True,
    has_separate_cfg=True,  # default, guidance_scale > 1
    **kwargs,
  )


@BlockAdapterRegister.register("Helios")
def helios_adapter(pipe, **kwargs) -> BlockAdapter:
  try:
    from diffusers import HeliosTransformer3DModel
  except ImportError:
    HeliosTransformer3DModel = None  # requires diffusers>=0.37.dev

  _relaxed_assert(pipe.transformer, HeliosTransformer3DModel)

  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.blocks,
    # (hidden_states, encoder_hidden_states) -> hidden_states
    forward_pattern=ForwardPattern.Pattern_2,
    check_forward_pattern=True,
    has_separate_cfg=True,  # default, guidance_scale > 1
    **kwargs,
  )


@BlockAdapterRegister.register("ErnieImage")
def ernie_image_adapter(pipe, **kwargs) -> BlockAdapter:
  from diffusers import ErnieImageTransformer2DModel
  from cache_dit.caching.patch_functors import ErnieImagePatchFunctor

  _relaxed_assert(pipe.transformer, ErnieImageTransformer2DModel)
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.layers,
    forward_pattern=ForwardPattern.Pattern_3,
    check_forward_pattern=False,
    has_separate_cfg=False,
    patch_functor=ErnieImagePatchFunctor(),
    **kwargs,
  )


@BlockAdapterRegister.register("BooguImage")
def boogu_image_adapter(pipe, **kwargs) -> BlockAdapter:
  """BlockAdapter for Boogu-Image (third-party, non-diffusers model).

  Boogu-Image uses a dual-stream architecture similar to HiDream:
  - double_stream_layers: joint image+instruction processing (Pattern_0)
  - single_stream_layers: fused single-stream processing (Pattern_3)

  Block forward signatures are non-standard, so check_forward_pattern is disabled.
  The existing apply_cache_dit_caching() in boogu-image already validates this approach.
  """
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=[
      pipe.transformer.double_stream_layers,
      pipe.transformer.single_stream_layers,
    ],
    forward_pattern=[
      ForwardPattern.Pattern_0,
      ForwardPattern.Pattern_3,
    ],
    check_forward_pattern=False,
    has_separate_cfg=True,
    **kwargs,
  )


@BlockAdapterRegister.register("Krea2")
def krea2_adapter(pipe, **kwargs) -> BlockAdapter:
  from diffusers import Krea2Transformer2DModel

  _relaxed_assert(pipe.transformer, Krea2Transformer2DModel)
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.transformer_blocks,
    forward_pattern=ForwardPattern.Pattern_3,
    check_forward_pattern=True,
    has_separate_cfg=True,
    **kwargs,
  )


@BlockAdapterRegister.register("JoyImage")
def joy_image_adapter(pipe, **kwargs) -> BlockAdapter:
  """BlockAdapter for JoyImageEditPipeline (IE2I).

  JoyImage uses a double-stream architecture:
  - double_blocks: joint image+text processing (Pattern_0)
  - 24 MHA heads, no GQA, head_dim=128

  Extra block args (temb, image_rotary_emb) flow naturally through
  CachedBlocks_Pattern_Base's *args, **kwargs forwarding — no PatchFunctor needed.
  """
  from diffusers import JoyImageEditTransformer3DModel

  _relaxed_assert(pipe.transformer, JoyImageEditTransformer3DModel)
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.double_blocks,
    forward_pattern=ForwardPattern.Pattern_0,
    check_forward_pattern=True,
    check_num_outputs=True,
    has_separate_cfg=True,
    **kwargs,
  )


@BlockAdapterRegister.register("AnyFlow")
def anyflow_adapter(pipe, **kwargs) -> BlockAdapter:
  """BlockAdapter for AnyFlow (T2V) and AnyFlow-FAR (causal T2V/I2V).

  AnyFlow is based on the Wan v0.35.1 3D DiT backbone with a dual-timestep
  flow-map embedding.  The block forward takes (hidden_states, encoder_hidden_states)
  and returns only hidden_states -> Pattern_2.

  CFG is batch-concatenated (single forward pass), so has_separate_cfg=False.
  No PatchFunctor is needed — the block loop uses positional args with no extra
  operations inside the loop body.

  Note: AnyFlow-FAR (``AnyFlowFARTransformer3DModel``) shares the same
  ``AnyFlowTransformerBlock`` and Pattern_2 contract, so this adapter also
  resolves the FAR transformer for TP-only flows (``cache_config is None``).
  DBCache itself is NOT supported on FAR — the chunk-wise autoregressive loop
  (per-chunk scheduler reset + mandatory full cache-prefill passes) is
  incompatible with cache-dit's single-trajectory step tracking; enabling
  ``--cache`` on FAR will produce corrupted output.
  """
  from diffusers import AnyFlowTransformer3DModel
  try:
    from diffusers import AnyFlowFARTransformer3DModel
    supported_transformers = (AnyFlowTransformer3DModel, AnyFlowFARTransformer3DModel)
  except ImportError:
    supported_transformers = (AnyFlowTransformer3DModel, )

  _relaxed_assert(pipe.transformer, supported_transformers)
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.blocks,
    forward_pattern=ForwardPattern.Pattern_2,
    check_forward_pattern=True,
    has_separate_cfg=False,
    **kwargs,
  )


@BlockAdapterRegister.register("Ideogram4")
def ideogram4_adapter(pipe, **kwargs) -> BlockAdapter:
  """BlockAdapter for Ideogram 4 (asymmetric-CFG transformer pair).

  `Ideogram4Pipeline` denoises with two independent `Ideogram4Transformer2DModel` partitions:
  `transformer` runs the packed [text, image] sequence and `unconditional_transformer` runs the
  image-only positions with zeroed text features, then the two velocities are blended. Each
  partition owns its own `layers` stack and is called exactly once per denoising step, so both are
  cached as sibling entries (the Wan 2.2 `transformer` / `transformer_2` layout) instead of one
  merged stack: every cache context then sees one forward per step, which is why
  `has_separate_cfg` stays False and the per-branch step counters line up 1:1 with the steps.
  Checkpoints that ship without the unconditional partition (and transformer-only usage) fall back
  to a single cached stack.

  The block contract is `(hidden_states, attention_mask, image_rotary_emb, adaln_input)` ->
  `hidden_states` (Pattern_3) and the block loop carries no extra body ops, so no PatchFunctor is
  needed.

  :param pipe: `Ideogram4Pipeline`, or a wrapper exposing only the transformer partitions.
  :param kwargs: Additional keyword arguments forwarded to the `BlockAdapter`.
  :returns: The BlockAdapter covering the `layers` stack of every loaded partition.
  """
  from diffusers import Ideogram4Transformer2DModel

  _relaxed_assert(pipe.transformer, Ideogram4Transformer2DModel)

  transformers = [pipe.transformer]
  unconditional_transformer = getattr(pipe, "unconditional_transformer", None)
  if unconditional_transformer is not None:
    _relaxed_assert(unconditional_transformer, Ideogram4Transformer2DModel)
    transformers.append(unconditional_transformer)

  return BlockAdapter(
    pipe=pipe,
    transformer=transformers,
    blocks=[transformer.layers for transformer in transformers],
    blocks_name=["layers"] * len(transformers),
    forward_pattern=[ForwardPattern.Pattern_3] * len(transformers),
    check_forward_pattern=True,
    has_separate_cfg=False,
    **kwargs,
  )


@BlockAdapterRegister.register("MOSSImage")
def moss_image_adapter(pipe, **kwargs) -> BlockAdapter:
  """BlockAdapter for MOSS-Image (`MOSSImagePipeline` / `MOSSImageTransformer`).

  The transformer is cache-dit's built-in copy (`cache_dit.models.pipelines.moss_image`), so it is
  not a diffusers module and no strict type check applies. Its denoising stack is
  `transformer.layers` and each block takes `(hidden_states, rope, modulation, skip_layers,
  cp_config)` and returns `hidden_states` (Pattern_3).

  Two model specifics matter here:

  * `has_separate_cfg=True`: the pipeline runs one `transformer.forward()` per CFG branch (cond,
    then uncond with `uncond_skip_layers`), so the two branches need independent cache contexts and
    one shared step counter. The optional SLG forward is a third branch — keep it off when caching.
  * `check_forward_pattern=True`: the built-in block's first parameter is `hidden_states`, so the
    contract is verifiable, but cache-dit defaults that check off for non-diffusers transformers.

  Layer skipping lives inside the block (`skip_layers` argument) rather than in the block loop, so
  the cached-block wrapper reproduces the CFG skip exactly and no PatchFunctor is needed.

  :param pipe: `MOSSImagePipeline` instance, or a wrapper exposing only `transformer`.
  :param kwargs: Additional keyword arguments forwarded to the `BlockAdapter`.
  :returns: The BlockAdapter covering the `layers` denoising stack.
  """
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.layers,
    blocks_name="layers",
    forward_pattern=ForwardPattern.Pattern_3,
    check_forward_pattern=True,
    has_separate_cfg=True,
    **kwargs,
  )


@BlockAdapterRegister.register("NucleusMoE")
def nucleus_moe_adapter(pipe, **kwargs) -> BlockAdapter:
  """BlockAdapter for NucleusMoE-Image (`NucleusMoEImagePipeline`).

  The denoising stack is `transformer.transformer_blocks`, a single-stream DiT block list whose
  blocks take `(hidden_states, encoder_hidden_states, temb, image_rotary_emb, attention_kwargs)`
  and return only `hidden_states` (Pattern_2). The transformer calls them with keyword arguments
  whose first two names match the cache wrapper, so the cached loop forwards the extra arguments
  (`temb`, the rotary embedding, the joint text+image mask) unchanged and no PatchFunctor is
  needed.

  Two model specifics matter here:

  * `has_separate_cfg` follows the pipeline default (`guidance_scale=4.0`), i.e. one
    `transformer.forward()` per CFG branch, so the cond / uncond passes need independent cache
    contexts sharing one step counter.
  * The block stack mixes dense and Mixture-of-Experts layers (`dense_moe_strategy`), but the
    block I/O contract is identical for both: a skipped block also skips its routing, so the
    cached residual stays valid.

  `NucleusMoEImageTransformer2DModel` already declares `_repeated_blocks`, so regional
  quantization and `compile_repeated_blocks()` discover the stack without help.

  A third model-specific gotcha only shows up with `torch.compile`: upstream's grouped-GEMM
  offsets lose their `dtype` inside a compiled graph, so `aten._grouped_mm` rejects them. Set
  `CACHE_DIT_NUCLEUS_MOE_GROUPED_MM_OFFSETS_FIX=1` to install cache-dit's int32-safe copy of
  `SwiGLUExperts._run_experts_grouped_mm` (see `cache_dit.patches`). The install happens here so
  that users who call `torch.compile()` themselves are covered as well; the CLI `--compile` path
  hooks the compile funnel instead.

  :param pipe: `NucleusMoEImagePipeline` instance.
  :param kwargs: Additional keyword arguments forwarded to the `BlockAdapter`.
  :returns: The BlockAdapter covering the `transformer_blocks` denoising stack.
  """
  from diffusers import NucleusMoEImageTransformer2DModel

  _relaxed_assert(pipe.transformer, NucleusMoEImageTransformer2DModel)

  maybe_patch_nucleus_moe_grouped_mm_offsets()

  guidance_scale = _default_call_float(pipe, "guidance_scale")
  has_separate_cfg = True if guidance_scale is None else guidance_scale > 1.0

  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.transformer_blocks,
    blocks_name="transformer_blocks",
    forward_pattern=ForwardPattern.Pattern_2,
    check_forward_pattern=True,
    has_separate_cfg=has_separate_cfg,
    **kwargs,
  )


@BlockAdapterRegister.register("Kandinsky6")
def kandinsky6_adapter(pipe, **kwargs) -> BlockAdapter:
  """BlockAdapter for Kandinsky 6.0 (`Kandinsky6TI2VAPipeline`).

  Only the fused video/audio stack `visual_transformer_blocks` is cached: it holds the 60
  `Kandinsky6FusedTransformerDecoderBlock` layers that carry almost all of the FLOPs, while the
  4+4 text blocks (`video_text_transformer_blocks` / `audio_text_transformer_blocks`) run once per
  step over the prompt and are cheap.

  The block contract is `(vis, aud, text_v, text_a, time_embed, vis_rope, aud_rope, attn_mask)` ->
  `(vis, aud)` (Pattern_0): the second stream is the *evolving audio latent*, not a static text
  context, so it is cached as the pattern's "encoder" residual. `forward` calls the blocks
  positionally, which matches the cached wrapper's `block(hidden_states, encoder_hidden_states,
  *args)` call exactly, so no PatchFunctor is needed. The block parameters are named `vis` / `aud`
  instead of `hidden_states` / `encoder_hidden_states`, hence the disabled signature check.

  `has_separate_cfg` follows the pipeline default (`guidance_scale=5.0`, i.e. two forwards per
  step). The distilled checkpoints must run `guidance_scale=1.0` with `PiflowScheduler`, which is
  one forward per step: such a run has to set `enable_separate_cfg=False` in the cache config,
  otherwise cache-dit pairs two consecutive steps into one and almost nothing is cached.

  The VSR pipeline (`Kandinsky6SRPipeline`) drives a different, text-free transformer
  (`Kandinsky6SRTransformer3DModel`) and is **not** covered by this adapter yet.

  :param pipe: `Kandinsky6TI2VAPipeline` instance.
  :param kwargs: Additional keyword arguments forwarded to the `BlockAdapter`.
  :returns: The BlockAdapter covering the `visual_transformer_blocks` denoising stack.
  """
  from diffusers import Kandinsky6Transformer3DModel

  transformer = getattr(pipe, "transformer", None)
  if not isinstance(transformer, Kandinsky6Transformer3DModel):
    raise NotImplementedError(
      f"Kandinsky 6.0 caching needs a `Kandinsky6Transformer3DModel`, got "
      f"{type(transformer).__name__} on {pipe.__class__.__name__}. The VSR pipeline drives "
      "`Kandinsky6SRTransformer3DModel`, which needs its own adapter.")

  guidance_scale = _default_call_float(pipe, "guidance_scale")
  has_separate_cfg = True if guidance_scale is None else guidance_scale > 1.0
  if not has_separate_cfg:
    logger.info(
      f"Kandinsky 6.0 defaults to guidance_scale={guidance_scale}, i.e. a single transformer "
      "forward per step; separate CFG bookkeeping is disabled. Pass `enable_separate_cfg=True` "
      "in the cache config if the run enables guidance.")

  return BlockAdapter(
    pipe=pipe,
    transformer=transformer,
    blocks=transformer.visual_transformer_blocks,
    blocks_name="visual_transformer_blocks",
    forward_pattern=ForwardPattern.Pattern_0,
    # The block parameters are `vis` / `aud`, and the second output slot is the audio stream.
    check_forward_pattern=False,
    check_num_outputs=False,
    has_separate_cfg=has_separate_cfg,
    **kwargs,
  )


@BlockAdapterRegister.register("LingBotVideo")
def lingbot_video_adapter(pipe, **kwargs) -> BlockAdapter:
  """BlockAdapter for LingBot-Video Dense (`LingBotVideoPipeline`).

  The transformer is cache-dit's built-in copy (`cache_dit.models.pipelines.lingbot_video`), so it
  is not a diffusers module and no strict type check applies. Its denoising stack is
  `transformer.blocks`, and each block takes `(hidden_states, temb6, rotary_emb, attention_mask,
  cp_config)` and returns only `hidden_states` (Pattern_3).

  Two model specifics matter here:

  * `has_separate_cfg=True`: the pipeline runs one `transformer.forward()` per CFG branch, i.e. two
    forwards per sampling step at batch size 1, so the branches need independent cache contexts and
    one shared step counter.
  * `check_forward_pattern=True`: the built-in block's first parameter is `hidden_states`, so the
    contract is verifiable, but cache-dit defaults that check off for non-diffusers transformers.

  The token-wise modulation (`temb6`) and the RoPE table are computed before the block loop from the
  timestep and the position ids, so a cached step still gets a fresh signal and no PatchFunctor is
  needed to keep loop-body state alive.

  :param pipe: `LingBotVideoPipeline` instance, or a wrapper exposing only `transformer`.
  :param kwargs: Additional keyword arguments forwarded to the `BlockAdapter`.
  :returns: The BlockAdapter covering the `blocks` denoising stack.
  """
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.blocks,
    blocks_name="blocks",
    forward_pattern=ForwardPattern.Pattern_3,
    check_forward_pattern=True,
    has_separate_cfg=True,
    **kwargs,
  )
