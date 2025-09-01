import torch

import inspect
import dataclasses

from typing import Any, Tuple, List, Optional

from diffusers import DiffusionPipeline
from cache_dit.cache_factory.forward_pattern import ForwardPattern
from cache_dit.cache_factory.patch_functors import PatchFunctor

from cache_dit.logger import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass
class BlockAdapter:
    # Transformer configurations.
    pipe: DiffusionPipeline | Any = None
    transformer: torch.nn.Module = None

    # ------------ Block Level Flags ------------
    blocks: torch.nn.ModuleList | List[torch.nn.ModuleList] = None
    # transformer_blocks, blocks, etc.
    blocks_name: str | List[str] = None
    dummy_blocks_names: List[str] = dataclasses.field(default_factory=list)
    forward_pattern: ForwardPattern | List[ForwardPattern] = None
    check_num_outputs: bool = True

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

    # NOTE: Other flags.
    disable_patch: bool = False

    # ------------ Pipeline Level Flags ------------
    # Patch Functor: Flux, etc.
    patch_functor: Optional[PatchFunctor] = None
    # Flags for separate cfg
    has_separate_cfg: bool = False

    def __post_init__(self):
        assert any((self.pipe is not None, self.transformer is not None))
        self.patchify()

    def patchify(self, *args, **kwargs):
        # Process some specificial cases, specific for transformers
        # that has different forward patterns between single_transformer_blocks
        # and transformer_blocks , such as Flux (diffusers < 0.35.0).
        if self.patch_functor is not None and not self.disable_patch:
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
            check_num_outputs=adapter.check_num_outputs,
        )

        return BlockAdapter(
            pipe=pipe,
            transformer=transformer,
            blocks=blocks,
            blocks_name=blocks_name,
            forward_pattern=adapter.forward_pattern,
        )

    @staticmethod
    def check_block_adapter(
        adapter: "BlockAdapter",
    ) -> bool:
        def _check_warning(attr: str):
            if getattr(adapter, attr, None) is None:
                logger.warning(f"{attr} is None!")
                return False
            return True

        if not _check_warning("pipe"):
            return False

        if not _check_warning("transformer"):
            return False

        if not _check_warning("blocks"):
            return False

        if not _check_warning("blocks_name"):
            return False

        if not _check_warning("forward_pattern"):
            return False

        if isinstance(adapter.blocks, list):
            for i, blocks in enumerate(adapter.blocks):
                if not isinstance(blocks, torch.nn.ModuleList):
                    logger.warning(f"blocks[{i}] is not ModuleList.")
                    return False
        else:
            if not isinstance(adapter.blocks, torch.nn.ModuleList):
                logger.warning("blocks is not ModuleList.")
                return False

        return True

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
        forward_pattern = kwargs.pop("forward_pattern", None)
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
                                **kwargs,
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
        **kwargs,
    ) -> bool:
        assert (
            forward_pattern.Supported
            and forward_pattern in ForwardPattern.supported_patterns()
        ), f"Pattern {forward_pattern} is not support now!"

        forward_parameters = set(
            inspect.signature(block.forward).parameters.keys()
        )

        in_matched = True
        out_matched = True

        if kwargs.get("check_num_outputs", True):
            num_outputs = str(
                inspect.signature(block.forward).return_annotation
            ).count("torch.Tensor")

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
        **kwargs,
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
                    **kwargs,
                )
            )

        pattern_matched = all(pattern_matched_states)  # all block match
        if pattern_matched and logging:
            block_cls_names = [
                block.__class__.__name__ for block in transformer_blocks
            ]
            block_cls_names = set(block_cls_names)
            logger.info(
                f"Match Block Forward Pattern: {block_cls_names}, {forward_pattern}"
                f"\nIN:{forward_pattern.In}, OUT:{forward_pattern.Out})"
            )

        return pattern_matched

    @staticmethod
    def normalize(
        adapter: "BlockAdapter",
    ) -> "BlockAdapter":
        if not isinstance(adapter.blocks, list):
            adapter.blocks = [adapter.blocks]
        if not isinstance(adapter.blocks_name, list):
            adapter.blocks_name = [adapter.blocks_name]
        if not isinstance(adapter.forward_pattern, list):
            adapter.forward_pattern = [adapter.forward_pattern]

        assert len(adapter.blocks) == len(adapter.blocks_name)
        assert len(adapter.blocks) == len(adapter.forward_pattern)

        return adapter
