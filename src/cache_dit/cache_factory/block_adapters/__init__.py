from cache_dit.cache_factory.forward_pattern import ForwardPattern
from cache_dit.cache_factory.block_adapters.block_adapters import BlockAdapter
from cache_dit.cache_factory.block_adapters.block_adapters import ParamsModifier
from cache_dit.cache_factory.block_adapters.block_registers import (
    BlockAdapterRegistry,
)


@BlockAdapterRegistry.register("Flux")
def flux_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import FluxTransformer2DModel
    from cache_dit.utils import is_diffusers_at_least_0_3_5

    assert isinstance(pipe.transformer, FluxTransformer2DModel)
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
            **kwargs,
        )


@BlockAdapterRegistry.register("Mochi")
def mochi_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import MochiTransformer3DModel

    assert isinstance(pipe.transformer, MochiTransformer3DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.transformer_blocks,
        forward_pattern=ForwardPattern.Pattern_0,
        **kwargs,
    )


@BlockAdapterRegistry.register("CogVideoX")
def cogvideox_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import CogVideoXTransformer3DModel

    assert isinstance(pipe.transformer, CogVideoXTransformer3DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.transformer_blocks,
        forward_pattern=ForwardPattern.Pattern_0,
        **kwargs,
    )


@BlockAdapterRegistry.register("Wan")
def wan_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import (
        WanTransformer3DModel,
        WanVACETransformer3DModel,
    )

    assert isinstance(
        pipe.transformer,
        (WanTransformer3DModel, WanVACETransformer3DModel),
    )
    if getattr(pipe, "transformer_2", None):
        assert isinstance(
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
            has_separate_cfg=True,
            **kwargs,
        )
    else:
        # Wan 2.1
        return BlockAdapter(
            pipe=pipe,
            transformer=pipe.transformer,
            blocks=pipe.transformer.blocks,
            forward_pattern=ForwardPattern.Pattern_2,
            has_separate_cfg=True,
            **kwargs,
        )


@BlockAdapterRegistry.register("HunyuanVideo")
def hunyuanvideo_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import HunyuanVideoTransformer3DModel

    assert isinstance(pipe.transformer, HunyuanVideoTransformer3DModel)
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
        # The type hint in diffusers is wrong
        check_num_outputs=False,
        **kwargs,
    )


@BlockAdapterRegistry.register("QwenImage")
def qwenimage_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import QwenImageTransformer2DModel

    assert isinstance(pipe.transformer, QwenImageTransformer2DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.transformer_blocks,
        forward_pattern=ForwardPattern.Pattern_1,
        has_separate_cfg=True,
        **kwargs,
    )


@BlockAdapterRegistry.register("LTX")
def ltxvideo_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import LTXVideoTransformer3DModel

    assert isinstance(pipe.transformer, LTXVideoTransformer3DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.transformer_blocks,
        forward_pattern=ForwardPattern.Pattern_2,
        **kwargs,
    )


@BlockAdapterRegistry.register("Allegro")
def allegro_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import AllegroTransformer3DModel

    assert isinstance(pipe.transformer, AllegroTransformer3DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.transformer_blocks,
        forward_pattern=ForwardPattern.Pattern_2,
        **kwargs,
    )


@BlockAdapterRegistry.register("CogView3Plus")
def cogview3plus_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import CogView3PlusTransformer2DModel

    assert isinstance(pipe.transformer, CogView3PlusTransformer2DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.transformer_blocks,
        forward_pattern=ForwardPattern.Pattern_0,
        **kwargs,
    )


@BlockAdapterRegistry.register("CogView4")
def cogview4_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import CogView4Transformer2DModel

    assert isinstance(pipe.transformer, CogView4Transformer2DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.transformer_blocks,
        forward_pattern=ForwardPattern.Pattern_0,
        has_separate_cfg=True,
        **kwargs,
    )


@BlockAdapterRegistry.register("Cosmos")
def cosmos_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import CosmosTransformer3DModel

    assert isinstance(pipe.transformer, CosmosTransformer3DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.transformer_blocks,
        forward_pattern=ForwardPattern.Pattern_2,
        has_separate_cfg=True,
        **kwargs,
    )


@BlockAdapterRegistry.register("EasyAnimate")
def easyanimate_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import EasyAnimateTransformer3DModel

    assert isinstance(pipe.transformer, EasyAnimateTransformer3DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.transformer_blocks,
        forward_pattern=ForwardPattern.Pattern_0,
        **kwargs,
    )


@BlockAdapterRegistry.register("SkyReelsV2")
def skyreelsv2_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import SkyReelsV2Transformer3DModel

    assert isinstance(pipe.transformer, SkyReelsV2Transformer3DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.blocks,
        # NOTE: Use Pattern_3 instead of Pattern_2 because the
        # encoder_hidden_states will never change in the blocks
        # forward loop.
        forward_pattern=ForwardPattern.Pattern_3,
        has_separate_cfg=True,
        **kwargs,
    )


@BlockAdapterRegistry.register("StableDiffusion3")
def sd3_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import SD3Transformer2DModel

    assert isinstance(pipe.transformer, SD3Transformer2DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.transformer_blocks,
        forward_pattern=ForwardPattern.Pattern_1,
        **kwargs,
    )


@BlockAdapterRegistry.register("ConsisID")
def consisid_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import ConsisIDTransformer3DModel

    assert isinstance(pipe.transformer, ConsisIDTransformer3DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.transformer_blocks,
        forward_pattern=ForwardPattern.Pattern_0,
        **kwargs,
    )


@BlockAdapterRegistry.register("DiT")
def dit_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import DiTTransformer2DModel
    from cache_dit.cache_factory.patch_functors import DiTPatchFunctor

    assert isinstance(pipe.transformer, DiTTransformer2DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.transformer_blocks,
        forward_pattern=ForwardPattern.Pattern_3,
        patch_functor=DiTPatchFunctor(),
        **kwargs,
    )


@BlockAdapterRegistry.register("Amused")
def amused_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import UVit2DModel

    assert isinstance(pipe.transformer, UVit2DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.transformer_layers,
        forward_pattern=ForwardPattern.Pattern_3,
        **kwargs,
    )


@BlockAdapterRegistry.register("Bria")
def bria_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import BriaTransformer2DModel

    assert isinstance(pipe.transformer, BriaTransformer2DModel)
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
        **kwargs,
    )


@BlockAdapterRegistry.register("Lumina")
def lumina2_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import Lumina2Transformer2DModel
    from diffusers import LuminaNextDiT2DModel

    assert isinstance(
        pipe.transformer, (Lumina2Transformer2DModel, LuminaNextDiT2DModel)
    )
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.layers,
        forward_pattern=ForwardPattern.Pattern_3,
        **kwargs,
    )


@BlockAdapterRegistry.register("OmniGen")
def omnigen_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import OmniGenTransformer2DModel

    assert isinstance(pipe.transformer, OmniGenTransformer2DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.layers,
        forward_pattern=ForwardPattern.Pattern_3,
        **kwargs,
    )


@BlockAdapterRegistry.register("PixArt")
def pixart_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import PixArtTransformer2DModel

    assert isinstance(pipe.transformer, PixArtTransformer2DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.transformer_blocks,
        forward_pattern=ForwardPattern.Pattern_3,
        **kwargs,
    )


@BlockAdapterRegistry.register("Sana")
def sana_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import SanaTransformer2DModel

    assert isinstance(pipe.transformer, SanaTransformer2DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.transformer_blocks,
        forward_pattern=ForwardPattern.Pattern_3,
        **kwargs,
    )


@BlockAdapterRegistry.register("StableAudio")
def stabledudio_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import StableAudioDiTModel

    assert isinstance(pipe.transformer, StableAudioDiTModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.transformer_blocks,
        forward_pattern=ForwardPattern.Pattern_3,
        **kwargs,
    )


@BlockAdapterRegistry.register("VisualCloze")
def visualcloze_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import FluxTransformer2DModel
    from cache_dit.utils import is_diffusers_at_least_0_3_5

    assert isinstance(pipe.transformer, FluxTransformer2DModel)
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
            **kwargs,
        )


@BlockAdapterRegistry.register("AuraFlow")
def auraflow_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import AuraFlowTransformer2DModel

    assert isinstance(pipe.transformer, AuraFlowTransformer2DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.single_transformer_blocks,
        forward_pattern=ForwardPattern.Pattern_3,
        **kwargs,
    )


@BlockAdapterRegistry.register("Chroma")
def chroma_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import ChromaTransformer2DModel
    from cache_dit.cache_factory.patch_functors import ChromaPatchFunctor

    assert isinstance(pipe.transformer, ChromaTransformer2DModel)
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
        has_separate_cfg=True,
        **kwargs,
    )


@BlockAdapterRegistry.register("ShapE")
def shape_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import PriorTransformer

    assert isinstance(pipe.prior, PriorTransformer)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.prior,
        blocks=pipe.prior.transformer_blocks,
        forward_pattern=ForwardPattern.Pattern_3,
        **kwargs,
    )


@BlockAdapterRegistry.register("HiDream")
def hidream_adapter(pipe, **kwargs) -> BlockAdapter:
    # NOTE: Need to patch Transformer forward to fully support
    # double_stream_blocks and single_stream_blocks, namely, need
    # to remove the logics inside the blocks forward loop:
    # https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_hidream_image.py#L893
    # https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_hidream_image.py#L927
    from diffusers import HiDreamImageTransformer2DModel
    from cache_dit.cache_factory.patch_functors import HiDreamPatchFunctor

    assert isinstance(pipe.transformer, HiDreamImageTransformer2DModel)
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


@BlockAdapterRegistry.register("HunyuanDiT")
def hunyuandit_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import HunyuanDiT2DModel, HunyuanDiT2DControlNetModel
    from cache_dit.cache_factory.patch_functors import HunyuanDiTPatchFunctor

    assert isinstance(
        pipe.transformer,
        (HunyuanDiT2DModel, HunyuanDiT2DControlNetModel),
    )
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.blocks,
        forward_pattern=ForwardPattern.Pattern_3,
        patch_functor=HunyuanDiTPatchFunctor(),
        **kwargs,
    )


@BlockAdapterRegistry.register("HunyuanDiTPAG")
def hunyuanditpag_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import HunyuanDiT2DModel
    from cache_dit.cache_factory.patch_functors import HunyuanDiTPatchFunctor

    assert isinstance(pipe.transformer, HunyuanDiT2DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.blocks,
        forward_pattern=ForwardPattern.Pattern_3,
        patch_functor=HunyuanDiTPatchFunctor(),
        **kwargs,
    )
