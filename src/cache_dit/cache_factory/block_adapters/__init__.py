from cache_dit.cache_factory.forward_pattern import ForwardPattern
from cache_dit.cache_factory.block_adapters.block_adapters import BlockAdapter
from cache_dit.cache_factory.block_adapters.block_adapters import ParamsModifier
from cache_dit.cache_factory.block_adapters.block_registers import (
    BlockAdapterRegistry,
)


@BlockAdapterRegistry.register("Flux")
def flux_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import FluxTransformer2DModel
    from cache_dit.cache_factory.patch_functors import FluxPatchFunctor

    assert isinstance(pipe.transformer, FluxTransformer2DModel)

    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=(
            pipe.transformer.transformer_blocks
            + pipe.transformer.single_transformer_blocks
        ),
        blocks_name="transformer_blocks",
        dummy_blocks_names=["single_transformer_blocks"],
        patch_functor=FluxPatchFunctor(),
        forward_pattern=ForwardPattern.Pattern_1,
        disable_patch=kwargs.pop("disable_patch", False),
    )


@BlockAdapterRegistry.register("Mochi")
def mochi_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import MochiTransformer3DModel

    assert isinstance(pipe.transformer, MochiTransformer3DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.transformer_blocks,
        blocks_name="transformer_blocks",
        dummy_blocks_names=[],
        forward_pattern=ForwardPattern.Pattern_0,
    )


@BlockAdapterRegistry.register("CogVideoX")
def cogvideox_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import CogVideoXTransformer3DModel

    assert isinstance(pipe.transformer, CogVideoXTransformer3DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.transformer_blocks,
        blocks_name="transformer_blocks",
        dummy_blocks_names=[],
        forward_pattern=ForwardPattern.Pattern_0,
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
            blocks_name=[
                "blocks",
                "blocks",
            ],
            forward_pattern=[
                ForwardPattern.Pattern_2,
                ForwardPattern.Pattern_2,
            ],
            dummy_blocks_names=[],
            has_separate_cfg=True,
        )
    else:
        # Wan 2.1
        return BlockAdapter(
            pipe=pipe,
            transformer=pipe.transformer,
            blocks=pipe.transformer.blocks,
            blocks_name="blocks",
            dummy_blocks_names=[],
            forward_pattern=ForwardPattern.Pattern_2,
            has_separate_cfg=True,
        )


@BlockAdapterRegistry.register("HunyuanVideo")
def hunyuanvideo_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import HunyuanVideoTransformer3DModel

    assert isinstance(pipe.transformer, HunyuanVideoTransformer3DModel)
    return BlockAdapter(
        pipe=pipe,
        blocks=(
            pipe.transformer.transformer_blocks
            + pipe.transformer.single_transformer_blocks
        ),
        blocks_name="transformer_blocks",
        dummy_blocks_names=["single_transformer_blocks"],
        forward_pattern=ForwardPattern.Pattern_0,
    )


@BlockAdapterRegistry.register("QwenImage")
def qwenimage_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import QwenImageTransformer2DModel

    assert isinstance(pipe.transformer, QwenImageTransformer2DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.transformer_blocks,
        blocks_name="transformer_blocks",
        dummy_blocks_names=[],
        forward_pattern=ForwardPattern.Pattern_1,
        has_separate_cfg=True,
    )


@BlockAdapterRegistry.register("LTXVideo")
def ltxvideo_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import LTXVideoTransformer3DModel

    assert isinstance(pipe.transformer, LTXVideoTransformer3DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.transformer_blocks,
        blocks_name="transformer_blocks",
        dummy_blocks_names=[],
        forward_pattern=ForwardPattern.Pattern_2,
    )


@BlockAdapterRegistry.register("Allegro")
def allegro_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import AllegroTransformer3DModel

    assert isinstance(pipe.transformer, AllegroTransformer3DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.transformer_blocks,
        blocks_name="transformer_blocks",
        dummy_blocks_names=[],
        forward_pattern=ForwardPattern.Pattern_2,
    )


@BlockAdapterRegistry.register("CogView3Plus")
def cogview3plus_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import CogView3PlusTransformer2DModel

    assert isinstance(pipe.transformer, CogView3PlusTransformer2DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.transformer_blocks,
        blocks_name="transformer_blocks",
        dummy_blocks_names=[],
        forward_pattern=ForwardPattern.Pattern_0,
    )


@BlockAdapterRegistry.register("CogView4")
def cogview4_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import CogView4Transformer2DModel

    assert isinstance(pipe.transformer, CogView4Transformer2DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.transformer_blocks,
        blocks_name="transformer_blocks",
        dummy_blocks_names=[],
        forward_pattern=ForwardPattern.Pattern_0,
        has_separate_cfg=True,
    )


@BlockAdapterRegistry.register("Cosmos")
def cosmos_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import CosmosTransformer3DModel

    assert isinstance(pipe.transformer, CosmosTransformer3DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.transformer_blocks,
        blocks_name="transformer_blocks",
        dummy_blocks_names=[],
        forward_pattern=ForwardPattern.Pattern_2,
        has_separate_cfg=True,
    )


@BlockAdapterRegistry.register("EasyAnimate")
def easyanimate_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import EasyAnimateTransformer3DModel

    assert isinstance(pipe.transformer, EasyAnimateTransformer3DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.transformer_blocks,
        blocks_name="transformer_blocks",
        dummy_blocks_names=[],
        forward_pattern=ForwardPattern.Pattern_0,
    )


@BlockAdapterRegistry.register("SkyReelsV2")
def skyreelsv2_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import SkyReelsV2Transformer3DModel

    assert isinstance(pipe.transformer, SkyReelsV2Transformer3DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.blocks,
        blocks_name="blocks",
        dummy_blocks_names=[],
        forward_pattern=ForwardPattern.Pattern_2,
        has_separate_cfg=True,
    )


@BlockAdapterRegistry.register("SD3")
def sd3_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import SD3Transformer2DModel

    assert isinstance(pipe.transformer, SD3Transformer2DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.transformer_blocks,
        blocks_name="transformer_blocks",
        dummy_blocks_names=[],
        forward_pattern=ForwardPattern.Pattern_1,
    )


@BlockAdapterRegistry.register("ConsisID")
def consisid_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import ConsisIDTransformer3DModel

    assert isinstance(pipe.transformer, ConsisIDTransformer3DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.transformer_blocks,
        blocks_name="transformer_blocks",
        dummy_blocks_names=[],
        forward_pattern=ForwardPattern.Pattern_0,
    )


@BlockAdapterRegistry.register("DiT")
def dit_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import DiTTransformer2DModel

    assert isinstance(pipe.transformer, DiTTransformer2DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.transformer_blocks,
        blocks_name="transformer_blocks",
        dummy_blocks_names=[],
        forward_pattern=ForwardPattern.Pattern_3,
    )


@BlockAdapterRegistry.register("Amused")
def amused_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import UVit2DModel

    assert isinstance(pipe.transformer, UVit2DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.transformer_layers,
        blocks_name="transformer_layers",
        dummy_blocks_names=[],
        forward_pattern=ForwardPattern.Pattern_3,
    )


@BlockAdapterRegistry.register("Bria")
def bria_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import BriaTransformer2DModel

    assert isinstance(pipe.transformer, BriaTransformer2DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=(
            pipe.transformer.transformer_blocks
            + pipe.transformer.single_transformer_blocks
        ),
        blocks_name="transformer_blocks",
        dummy_blocks_names=["single_transformer_blocks"],
        forward_pattern=ForwardPattern.Pattern_0,
    )


@BlockAdapterRegistry.register("HunyuanDiT")
def hunyuandit_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import HunyuanDiT2DModel, HunyuanDiT2DControlNetModel

    assert isinstance(
        pipe.transformer,
        (HunyuanDiT2DModel, HunyuanDiT2DControlNetModel),
    )
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.blocks,
        blocks_name="blocks",
        dummy_blocks_names=[],
        forward_pattern=ForwardPattern.Pattern_3,
    )


@BlockAdapterRegistry.register("HunyuanDiTPAG")
def hunyuanditpag_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import HunyuanDiT2DModel

    assert isinstance(pipe.transformer, HunyuanDiT2DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.blocks,
        blocks_name="blocks",
        dummy_blocks_names=[],
        forward_pattern=ForwardPattern.Pattern_3,
    )


@BlockAdapterRegistry.register("Lumina")
def lumina_adapter(pipe) -> BlockAdapter:
    from diffusers import LuminaNextDiT2DModel

    assert isinstance(pipe.transformer, LuminaNextDiT2DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.layers,
        blocks_name="layers",
        dummy_blocks_names=[],
        forward_pattern=ForwardPattern.Pattern_3,
    )


@BlockAdapterRegistry.register("Lumina2")
def lumina2_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import Lumina2Transformer2DModel

    assert isinstance(pipe.transformer, Lumina2Transformer2DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.layers,
        blocks_name="layers",
        dummy_blocks_names=[],
        forward_pattern=ForwardPattern.Pattern_3,
    )


@BlockAdapterRegistry.register("OmniGen")
def omnigen_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import OmniGenTransformer2DModel

    assert isinstance(pipe.transformer, OmniGenTransformer2DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.layers,
        blocks_name="layers",
        dummy_blocks_names=[],
        forward_pattern=ForwardPattern.Pattern_3,
    )


@BlockAdapterRegistry.register("PixArt")
def pixart_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import PixArtTransformer2DModel

    assert isinstance(pipe.transformer, PixArtTransformer2DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.transformer_blocks,
        blocks_name="transformer_blocks",
        dummy_blocks_names=[],
        forward_pattern=ForwardPattern.Pattern_3,
    )


@BlockAdapterRegistry.register("Sana")
def sana_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import SanaTransformer2DModel

    assert isinstance(pipe.transformer, SanaTransformer2DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.transformer_blocks,
        blocks_name="transformer_blocks",
        dummy_blocks_names=[],
        forward_pattern=ForwardPattern.Pattern_3,
    )


@BlockAdapterRegistry.register("ShapE")
def shape_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import PriorTransformer

    assert isinstance(pipe.prior, PriorTransformer)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.prior,
        blocks=pipe.prior.transformer_blocks,
        blocks_name="transformer_blocks",
        dummy_blocks_names=[],
        forward_pattern=ForwardPattern.Pattern_3,
    )


@BlockAdapterRegistry.register("StableAudio")
def stabledudio_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import StableAudioDiTModel

    assert isinstance(pipe.transformer, StableAudioDiTModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.transformer_blocks,
        blocks_name="transformer_blocks",
        dummy_blocks_names=[],
        forward_pattern=ForwardPattern.Pattern_3,
    )


@BlockAdapterRegistry.register("VisualCloze")
def visualcloze_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import FluxTransformer2DModel
    from cache_dit.cache_factory.patch_functors import FluxPatchFunctor

    assert isinstance(pipe.transformer, FluxTransformer2DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=(
            pipe.transformer.transformer_blocks
            + pipe.transformer.single_transformer_blocks
        ),
        blocks_name="transformer_blocks",
        dummy_blocks_names=["single_transformer_blocks"],
        patch_functor=FluxPatchFunctor(),
        forward_pattern=ForwardPattern.Pattern_1,
    )


@BlockAdapterRegistry.register("AuraFlow")
def auraflow_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import AuraFlowTransformer2DModel

    assert isinstance(pipe.transformer, AuraFlowTransformer2DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=[
            # Only 4 joint blocks, apply no-cache
            # pipe.transformer.joint_transformer_blocks,
            pipe.transformer.single_transformer_blocks,
        ],
        blocks_name=[
            # "joint_transformer_blocks",
            "single_transformer_blocks",
        ],
        forward_pattern=[
            # ForwardPattern.Pattern_1,
            ForwardPattern.Pattern_3,
        ],
    )


@BlockAdapterRegistry.register("Chroma")
def chroma_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import ChromaTransformer2DModel

    assert isinstance(pipe.transformer, ChromaTransformer2DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=[
            pipe.transformer.transformer_blocks,
            pipe.transformer.single_transformer_blocks,
        ],
        blocks_name=[
            "transformer_blocks",
            "single_transformer_blocks",
        ],
        forward_pattern=[
            ForwardPattern.Pattern_1,
            ForwardPattern.Pattern_3,
        ],
        has_separate_cfg=True,
    )


@BlockAdapterRegistry.register("HiDream")
def hidream_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import HiDreamImageTransformer2DModel

    assert isinstance(pipe.transformer, HiDreamImageTransformer2DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=[
            pipe.transformer.double_stream_blocks,
            pipe.transformer.single_stream_blocks,
        ],
        blocks_name=[
            "double_stream_blocks",
            "single_stream_blocks",
        ],
        dummy_blocks_names=[],
        forward_pattern=[
            ForwardPattern.Pattern_4,
            ForwardPattern.Pattern_3,
        ],
        check_num_outputs=False,
    )
