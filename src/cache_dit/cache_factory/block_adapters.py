import torch

import inspect
import dataclasses

from typing import Any, Tuple, List, Optional, Dict

from diffusers import DiffusionPipeline
from cache_dit.cache_factory import ForwardPattern
from cache_dit.cache_factory import PatchFunctor

from cache_dit.logger import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass
class BlockAdapter:
    # Transformer configurations.
    pipe: DiffusionPipeline | Any = None
    transformer: torch.nn.Module = None
    blocks: torch.nn.ModuleList = None
    # transformer_blocks, blocks, etc.
    blocks_name: str = None
    dummy_blocks_names: List[str] = dataclasses.field(default_factory=list)
    # Forward pattern
    forward_pattern: ForwardPattern = None
    # Patch Functor: Flux, etc.
    patch_functor: Optional[PatchFunctor] = None
    # Flags for separate cfg
    has_separate_cfg: bool = False
    # Flags to control auto block adapter
    auto: bool = False
    allow_prefixes: List[str] = dataclasses.field(
        default_factory=lambda: [
            "transformer",
            "single_transformer",
            "blocks",
            "layers",
            "single_stream_blocks",
            "double_stream_blocks",
        ]
    )
    check_prefixes: bool = True
    allow_suffixes: List[str] = dataclasses.field(
        default_factory=lambda: ["TransformerBlock"]
    )
    check_suffixes: bool = False
    blocks_policy: str = dataclasses.field(
        default="max", metadata={"allowed_values": ["max", "min"]}
    )

    # TODO: Other flags.

    def __post_init__(self):
        assert any((self.pipe is not None, self.transformer is not None))
        self.patchify()

    def patchify(self, *args, **kwargs):
        # Process some specificial cases, specific for transformers
        # that has different forward patterns between single_transformer_blocks
        # and transformer_blocks , such as Flux (diffusers < 0.35.0).
        if self.patch_functor is not None:
            if self.transformer is not None:
                self.patch_functor.apply(self.transformer, *args, **kwargs)
            else:
                assert hasattr(self.pipe, "transformer")
                self.patch_functor.apply(self.pipe.transformer, *args, **kwargs)

    @staticmethod
    def auto_block_adapter(
        adapter: "BlockAdapter",
    ) -> "BlockAdapter":
        assert adapter.auto, (
            "Please manually set `auto` to True, or, manually "
            "set all the transformer blocks configuration."
        )
        assert adapter.pipe is not None, "adapter.pipe can not be None."
        assert (
            adapter.forward_pattern is not None
        ), "adapter.forward_pattern can not be None."
        pipe = adapter.pipe

        assert hasattr(pipe, "transformer"), "pipe.transformer can not be None."

        transformer = pipe.transformer

        # "transformer_blocks", "blocks", "single_transformer_blocks", "layers"
        blocks, blocks_name = BlockAdapter.find_blocks(
            transformer=transformer,
            allow_prefixes=adapter.allow_prefixes,
            allow_suffixes=adapter.allow_suffixes,
            check_prefixes=adapter.check_prefixes,
            check_suffixes=adapter.check_suffixes,
            blocks_policy=adapter.blocks_policy,
            forward_pattern=adapter.forward_pattern,
        )

        return BlockAdapter(
            pipe=pipe,
            transformer=transformer,
            blocks=blocks,
            blocks_name=blocks_name,
            forward_pattern=adapter.forward_pattern,
        )

    @staticmethod
    def check_block_adapter(adapter: "BlockAdapter") -> bool:
        if (
            # NOTE: pipe may not need to be DiffusionPipeline?
            # isinstance(adapter.pipe, DiffusionPipeline)
            adapter.pipe is not None
            and adapter.transformer is not None
            and adapter.blocks is not None
            and adapter.blocks_name is not None
            and adapter.forward_pattern is not None
            and isinstance(adapter.blocks, torch.nn.ModuleList)
        ):
            return True

        logger.warning("Check block adapter failed!")
        return False

    @staticmethod
    def find_blocks(
        transformer: torch.nn.Module,
        allow_prefixes: List[str] = [
            "transformer",
            "single_transformer",
            "blocks",
            "layers",
            "single_stream_blocks",
            "double_stream_blocks",
        ],
        allow_suffixes: List[str] = [
            "TransformerBlock",
        ],
        check_prefixes: bool = True,
        check_suffixes: bool = False,
        **kwargs,
    ) -> Tuple[torch.nn.ModuleList, str]:
        # Check prefixes
        if check_prefixes:
            blocks_names = []
            for attr_name in dir(transformer):
                for prefix in allow_prefixes:
                    if attr_name.startswith(prefix):
                        blocks_names.append(attr_name)
        else:
            blocks_names = dir(transformer)

        # Check ModuleList
        valid_names = []
        valid_count = []
        forward_pattern = kwargs.get("forward_pattern", None)
        for blocks_name in blocks_names:
            if blocks := getattr(transformer, blocks_name, None):
                if isinstance(blocks, torch.nn.ModuleList):
                    block = blocks[0]
                    block_cls_name = block.__class__.__name__
                    # Check suffixes
                    if isinstance(block, torch.nn.Module) and (
                        any(
                            (
                                block_cls_name.endswith(allow_suffix)
                                for allow_suffix in allow_suffixes
                            )
                        )
                        or (not check_suffixes)
                    ):
                        # May check forward pattern
                        if forward_pattern is not None:
                            if BlockAdapter.match_blocks_pattern(
                                blocks,
                                forward_pattern,
                                logging=False,
                            ):
                                valid_names.append(blocks_name)
                                valid_count.append(len(blocks))
                        else:
                            valid_names.append(blocks_name)
                            valid_count.append(len(blocks))

        if not valid_names:
            raise ValueError(
                "Auto selected transformer blocks failed, please set it manually."
            )

        final_name = valid_names[0]
        final_count = valid_count[0]
        block_policy = kwargs.get("blocks_policy", "max")

        for blocks_name, count in zip(valid_names, valid_count):
            blocks = getattr(transformer, blocks_name)
            logger.info(
                f"Auto selected transformer blocks: {blocks_name}, "
                f"class: {blocks[0].__class__.__name__}, "
                f"num blocks: {count}"
            )
            if block_policy == "max":
                if final_count < count:
                    final_count = count
                    final_name = blocks_name
            else:
                if final_count > count:
                    final_count = count
                    final_name = blocks_name

        final_blocks = getattr(transformer, final_name)

        logger.info(
            f"Final selected transformer blocks: {final_name}, "
            f"class: {final_blocks[0].__class__.__name__}, "
            f"num blocks: {final_count}, block_policy: {block_policy}."
        )

        return final_blocks, final_name

    @staticmethod
    def match_block_pattern(
        block: torch.nn.Module,
        forward_pattern: ForwardPattern,
    ) -> bool:
        assert (
            forward_pattern.Supported
            and forward_pattern in ForwardPattern.supported_patterns()
        ), f"Pattern {forward_pattern} is not support now!"

        forward_parameters = set(
            inspect.signature(block.forward).parameters.keys()
        )
        num_outputs = str(
            inspect.signature(block.forward).return_annotation
        ).count("torch.Tensor")

        in_matched = True
        out_matched = True
        if num_outputs > 0 and len(forward_pattern.Out) != num_outputs:
            # output pattern not match
            out_matched = False

        for required_param in forward_pattern.In:
            if required_param not in forward_parameters:
                in_matched = False

        return in_matched and out_matched

    @staticmethod
    def match_blocks_pattern(
        transformer_blocks: torch.nn.ModuleList,
        forward_pattern: ForwardPattern,
        logging: bool = True,
    ) -> bool:
        assert (
            forward_pattern.Supported
            and forward_pattern in ForwardPattern.supported_patterns()
        ), f"Pattern {forward_pattern} is not support now!"

        assert isinstance(transformer_blocks, torch.nn.ModuleList)

        pattern_matched_states = []
        for block in transformer_blocks:
            pattern_matched_states.append(
                BlockAdapter.match_block_pattern(
                    block,
                    forward_pattern,
                )
            )

        pattern_matched = all(pattern_matched_states)  # all block match
        if pattern_matched and logging:
            block_cls_name = transformer_blocks[0].__class__.__name__
            logger.info(
                f"Match Block Forward Pattern: {block_cls_name}, {forward_pattern}"
                f"\nIN:{forward_pattern.In}, OUT:{forward_pattern.Out})"
            )

        return pattern_matched


class BlockAdapterRegistry:
    _adapters: Dict[str, BlockAdapter] = {}
    _predefined_adapters_has_spearate_cfg: List[str] = {
        "QwenImage",
        "Wan",
        "CogView4",
        "Cosmos",
        "SkyReelsV2",
        "Chroma",
    }

    @classmethod
    def register(cls, name):
        def decorator(func):
            cls._adapters[name] = func
            return func

        return decorator

    @classmethod
    def get_adapter(
        cls,
        pipe: DiffusionPipeline | str | Any,
    ) -> BlockAdapter:
        if not isinstance(pipe, str):
            pipe_cls_name: str = pipe.__class__.__name__
        else:
            pipe_cls_name = pipe

        for name in cls._adapters:
            if pipe_cls_name.startswith(name):
                return cls._adapters[name](pipe)

        return BlockAdapter()

    @classmethod
    def has_separate_cfg(
        cls,
        pipe: DiffusionPipeline | str | Any,
    ) -> bool:
        if cls.get_adapter(pipe).has_separate_cfg:
            return True

        pipe_cls_name = pipe.__class__.__name__
        for name in cls._predefined_adapters_has_spearate_cfg:
            if pipe_cls_name.startswith(name):
                return True

        return False

    @classmethod
    def is_supported(cls, pipe) -> bool:
        pipe_cls_name: str = pipe.__class__.__name__

        for name in cls._adapters:
            if pipe_cls_name.startswith(name):
                return True
        return False

    @classmethod
    def supported_pipelines(cls, **kwargs) -> Tuple[int, List[str]]:
        val_pipelines = cls._adapters.keys()
        return len(val_pipelines), [p + "*" for p in val_pipelines]


@BlockAdapterRegistry.register("Flux")
def flux_adapter(pipe) -> BlockAdapter:
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


@BlockAdapterRegistry.register("Mochi")
def mochi_adapter(pipe) -> BlockAdapter:
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
def cogvideox_adapter(pipe) -> BlockAdapter:
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
def wan_adapter(pipe) -> BlockAdapter:
    from diffusers import (
        WanTransformer3DModel,
        WanVACETransformer3DModel,
    )

    assert isinstance(
        pipe.transformer,
        (WanTransformer3DModel, WanVACETransformer3DModel),
    )
    if getattr(pipe, "transformer_2", None):
        # Wan 2.2, cache for low-noise transformer
        return BlockAdapter(
            pipe=pipe,
            transformer=pipe.transformer_2,
            blocks=pipe.transformer_2.blocks,
            blocks_name="blocks",
            dummy_blocks_names=[],
            forward_pattern=ForwardPattern.Pattern_2,
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
def hunyuanvideo_adapter(pipe) -> BlockAdapter:
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
def qwenimage_adapter(pipe) -> BlockAdapter:
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
def ltxvideo_adapter(pipe) -> BlockAdapter:
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
def allegro_adapter(pipe) -> BlockAdapter:
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
def cogview3plus_adapter(pipe) -> BlockAdapter:
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
def cogview4_adapter(pipe) -> BlockAdapter:
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
def cosmos_adapter(pipe) -> BlockAdapter:
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
def easyanimate_adapter(pipe) -> BlockAdapter:
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
def skyreelsv2_adapter(pipe) -> BlockAdapter:
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
def sd3_adapter(pipe) -> BlockAdapter:
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
def consisid_adapter(pipe) -> BlockAdapter:
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
def dit_adapter(pipe) -> BlockAdapter:
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
def amused_adapter(pipe) -> BlockAdapter:
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
def bria_adapter(pipe) -> BlockAdapter:
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
def hunyuandit_adapter(pipe) -> BlockAdapter:
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
def hunyuanditpag_adapter(pipe) -> BlockAdapter:
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
def lumina2_adapter(pipe) -> BlockAdapter:
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
def omnigen_adapter(pipe) -> BlockAdapter:
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
def pixart_adapter(pipe) -> BlockAdapter:
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
def sana_adapter(pipe) -> BlockAdapter:
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
def shape_adapter(pipe) -> BlockAdapter:
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
def stabledudio_adapter(pipe) -> BlockAdapter:
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
def visualcloze_adapter(pipe) -> BlockAdapter:
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
def auraflow_adapter(pipe) -> BlockAdapter:
    from diffusers import AuraFlowTransformer2DModel

    assert isinstance(pipe.transformer, AuraFlowTransformer2DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        # Only support caching single_transformer_blocks for AuraFlow now.
        # TODO: Support AuraFlowPatchFunctor.
        blocks=pipe.transformer.single_transformer_blocks,
        blocks_name="single_transformer_blocks",
        dummy_blocks_names=[],
        forward_pattern=ForwardPattern.Pattern_3,
    )


@BlockAdapterRegistry.register("Chroma")
def chroma_adapter(pipe) -> BlockAdapter:
    from diffusers import ChromaTransformer2DModel
    from cache_dit.cache_factory.patch_functors import (
        ChromaPatchFunctor,
    )

    assert isinstance(pipe.transformer, ChromaTransformer2DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=(
            pipe.transformer.transformer_blocks
            + pipe.transformer.single_transformer_blocks
        ),
        blocks_name="transformer_blocks",
        dummy_blocks_names=["single_transformer_blocks"],
        patch_functor=ChromaPatchFunctor(),
        forward_pattern=ForwardPattern.Pattern_1,
        has_separate_cfg=True,
    )


@BlockAdapterRegistry.register("HiDream")
def hidream_adapter(pipe) -> BlockAdapter:
    from diffusers import HiDreamImageTransformer2DModel

    assert isinstance(pipe.transformer, HiDreamImageTransformer2DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        # Only support caching single_stream_blocks for HiDream now.
        # TODO: Support HiDreamPatchFunctor.
        blocks=pipe.transformer.single_stream_blocks,
        blocks_name="single_stream_blocks",
        dummy_blocks_names=[],
        forward_pattern=ForwardPattern.Pattern_3,
    )
