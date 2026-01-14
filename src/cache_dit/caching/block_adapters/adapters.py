import torch
from typing import List, Tuple, Union, Optional
from ..forward_pattern import ForwardPattern
from .block_adapters import BlockAdapter
from .block_registers import BlockAdapterRegister
from cache_dit.envs import ENV
from cache_dit.logger import init_logger

logger = init_logger(__name__)


def _relaxed_assert(
    transformer: torch.nn.Module,
    allow_classes: Optional[
        Union[
            torch.nn.Module,
            List[torch.nn.Module],
            Tuple[torch.nn.Module],
        ]
    ] = None,
) -> None:
    if allow_classes is not None and not isinstance(allow_classes, (list, tuple)):
        allow_classes = (allow_classes,)
    _imported_module_ = transformer.__module__
    if _imported_module_.startswith("diffusers"):
        # Only apply strict check for Diffusers transformers
        if allow_classes is not None:
            assert isinstance(transformer, allow_classes), (
                f"Transformer class {transformer.__class__.__name__} not in "
                f"allowed classes: {[cls.__name__ for cls in allow_classes]}"
            )
        else:
            logger.warning(
                "No allowed classes provided for transformer strict type check "
                "in BlockAdapter. Skipping strict type check."
            )
    else:
        # Otherwise, just log a warning and skip strict type check, e.g:
        # sglang/multimodal_gen/runtime/models/dits/flux.py#L411
        logger.warning(
            f"Transformer class {transformer.__class__.__name__} is from "
            f"{_imported_module_.split('.')[0]} not diffusers, skipping strict type check "
            "in BlockAdapter."
        )


@BlockAdapterRegister.register("Flux")
def flux_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import FluxTransformer2DModel
    from cache_dit.utils import is_diffusers_at_least_0_3_5
    from cache_dit.caching.patch_functors import FluxPatchFunctor

    supported_transformers = (FluxTransformer2DModel,)
    try:
        from diffusers import Flux2Transformer2DModel

        supported_transformers += (Flux2Transformer2DModel,)
    except ImportError:
        Flux2Transformer2DModel = None  # requires diffusers>=0.36.dev0

    _relaxed_assert(pipe.transformer, supported_transformers)

    transformer_cls_name: str = pipe.transformer.__class__.__name__
    if (
        is_diffusers_at_least_0_3_5()
        and not transformer_cls_name.startswith("Nunchaku")
        and not transformer_cls_name.startswith("Flux2")
    ):
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
                blocks=(
                    pipe.transformer.transformer_blocks + pipe.transformer.single_transformer_blocks
                ),
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
    supported_transformers = (HunyuanVideoTransformer3DModel,)
    try:
        from diffusers import HunyuanVideo15Transformer3DModel

        supported_transformers += (HunyuanVideo15Transformer3DModel,)
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
            QwenImageControlNetPatchFunctor,
        )

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


@BlockAdapterRegister.register("LTX")
def ltxvideo_adapter(pipe, **kwargs) -> BlockAdapter:
    # LTX-1 (LTXVideoTransformer3DModel) and LTX-2 (LTX2VideoTransformer3DModel) share
    # the `transformer_blocks` structure, but differ in block forward IO:
    # - LTX-1 blocks return only `hidden_states`  -> Pattern_2
    # - LTX-2 blocks return `(hidden_states, audio_hidden_states)` -> Pattern_0
    from diffusers import LTXVideoTransformer3DModel

    cls_name: str = pipe.transformer.__class__.__name__
    is_ltx2: bool = cls_name.startswith("LTX2")
    forward_pattern = ForwardPattern.Pattern_0 if is_ltx2 else ForwardPattern.Pattern_2

    try:
        from diffusers import LTX2VideoTransformer3DModel
        from cache_dit.caching.patch_functors import LTX2PatchFunctor

        patch_functor = LTX2PatchFunctor() if is_ltx2 else None
    except Exception:
        LTX2VideoTransformer3DModel = None  # requires newer diffusers
        patch_functor = None

    supported_transformers = (LTXVideoTransformer3DModel,)
    if LTX2VideoTransformer3DModel is not None:
        supported_transformers = supported_transformers + (LTX2VideoTransformer3DModel,)

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
    from diffusers import CosmosTransformer3DModel

    _relaxed_assert(pipe.transformer, CosmosTransformer3DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.transformer_blocks,
        forward_pattern=ForwardPattern.Pattern_2,
        check_forward_pattern=True,
        has_separate_cfg=True,
        **kwargs,
    )


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
    from cache_dit.utils import is_diffusers_at_least_0_3_5

    _relaxed_assert(pipe.transformer, FluxTransformer2DModel)
    if is_diffusers_at_least_0_3_5():
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
