import torch

import inspect
import dataclasses
from collections.abc import Iterable

from typing import Any, Tuple, List, Optional, Union

from diffusers import DiffusionPipeline
from cache_dit.cache_factory.forward_pattern import ForwardPattern
from cache_dit.cache_factory.patch_functors import PatchFunctor

from cache_dit.logger import init_logger

logger = init_logger(__name__)


class ParamsModifier:
    def __init__(self, **kwargs):
        self._context_kwargs = kwargs.copy()


@dataclasses.dataclass
class BlockAdapter:

    # Transformer configurations.
    pipe: Union[
        DiffusionPipeline,
        Any,
    ] = None

    # single transformer (most cases) or list of transformers (Wan2.2, etc)
    transformer: Union[
        torch.nn.Module,
        List[torch.nn.Module],
    ] = None

    # Block Level Flags
    # Each transformer contains a list of blocks-list,
    # blocks_name-list, dummy_blocks_names-list, etc.
    blocks: Union[
        torch.nn.ModuleList,
        List[torch.nn.ModuleList],
        List[List[torch.nn.ModuleList]],
    ] = None

    # transformer_blocks, blocks, etc.
    blocks_name: Union[
        str,
        List[str],
        List[List[str]],
    ] = None

    unique_blocks_name: Union[
        str,
        List[str],
        List[List[str]],
    ] = dataclasses.field(default_factory=list)

    dummy_blocks_names: Union[
        List[str],
        List[List[str]],
    ] = dataclasses.field(default_factory=list)

    forward_pattern: Union[
        ForwardPattern,
        List[ForwardPattern],
        List[List[ForwardPattern]],
    ] = None

    # modify cache context params for specific blocks.
    params_modifiers: Union[
        ParamsModifier,
        List[ParamsModifier],
        List[List[ParamsModifier]],
    ] = None

    check_num_outputs: bool = True

    # Pipeline Level Flags
    # Patch Functor: Flux, etc.
    patch_functor: Optional[PatchFunctor] = None
    # Flags for separate cfg
    has_separate_cfg: bool = False

    # Other Flags
    disable_patch: bool = False

    # Flags to control auto block adapter
    # NOTE: NOT support for multi-transformers.
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

    def __post_init__(self):
        assert any((self.pipe is not None, self.transformer is not None))
        self.maybe_fill_attrs()
        self.maybe_patchify()

    def maybe_fill_attrs(self):
        # NOTE: This func should be call before normalize.
        # Allow empty `blocks_names`, we will auto fill it.
        # TODO: preprocess more empty attrs.
        if (
            self.transformer is not None
            and self.blocks is not None
            and self.blocks_name is None
        ):

            def _find(transformer, blocks):
                attr_names = dir(transformer)
                assert isinstance(self.blocks, torch.nn.ModuleList)
                blocks_name = None
                for attr_name in attr_names:
                    if attr := getattr(self.transformer, attr_name, None):
                        if isinstance(attr, torch.nn.ModuleList) and id(
                            attr
                        ) == id(blocks):
                            blocks_name = attr
                            break
                assert (
                    blocks_name is not None
                ), "No blocks_name match, please set it manually!"
                return blocks_name

            if self.nested_depth(self.transformer) == 0:
                if self.nested_depth(self.blocks) == 0:  # str
                    self.blocks_name = _find(self.transformer, self.blocks)
                elif self.nested_depth(self.blocks) == 1:
                    self.blocks_name = [
                        _find(self.transformer, blocks)
                        for blocks in self.blocks
                    ]
                else:
                    raise ValueError(
                        "Blocks nested depth can't more than 1 if transformer "
                        f"is not a list, current is: {self.nested_depth(self.blocks)}"
                    )
            elif self.nested_depth(self.transformer) == 1:  # List[str]
                if self.nested_depth(self.blocks) == 1:  # List[str]
                    assert len(self.transformer) == len(self.blocks)
                    self.blocks_name = [
                        _find(transformer, blocks)
                        for transformer, blocks in zip(
                            self.transformer, self.blocks
                        )
                    ]
                if self.nested_depth(self.blocks) == 2:  # List[List[str]]
                    assert len(self.transformer) == len(self.blocks)
                    self.blocks_name = []
                    for i in range(len(self.blocks)):
                        self.blocks_name.append(
                            [
                                _find(self.transformer[i], blocks)
                                for blocks in self.blocks[i]
                            ]
                        )
                raise ValueError(
                    "Blocks nested depth can't more than 2 or less than 1 "
                    "if transformer is a list, current is: "
                    f"{self.nested_depth(self.blocks)}"
                )
            else:
                raise ValueError(
                    "transformer nested depth can't more than 1, "
                    f"current is: {self.nested_depth(self.transformer)}"
                )
            logger.info(f"Auto fill blocks_name: {self.blocks_name}.")

    def maybe_patchify(self, *args, **kwargs):
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
            block_cls_names = list(set(block_cls_names))
            if len(block_cls_names) == 1:
                block_cls_names = block_cls_names[0]
            logger.info(
                f"Match Block Forward Pattern: {block_cls_names}, {forward_pattern}"
                f"\nIN:{forward_pattern.In}, OUT:{forward_pattern.Out})"
            )

        return pattern_matched

    @staticmethod
    def normalize(
        adapter: "BlockAdapter",
    ) -> "BlockAdapter":

        if getattr(adapter, "_is_normalized", False):
            return adapter

        if not isinstance(adapter.transformer, list):
            adapter.transformer = [adapter.transformer]

        if isinstance(adapter.blocks, torch.nn.ModuleList):
            # blocks_0 = [[blocks_0,],] -> match [TRN_0,]
            adapter.blocks = [[adapter.blocks]]
        elif isinstance(adapter.blocks, list):
            if isinstance(adapter.blocks[0], torch.nn.ModuleList):
                # [blocks_0, blocks_1] -> [[blocks_0, blocks_1],] -> match [TRN_0,]
                if len(adapter.blocks) == len(adapter.transformer):
                    adapter.blocks = [[blocks] for blocks in adapter.blocks]
                else:
                    adapter.blocks = [adapter.blocks]
            elif isinstance(adapter.blocks[0], list):
                # [[blocks_0, blocks_1],[blocks_2, blocks_3],] -> match [TRN_0, TRN_1,]
                pass

        if isinstance(adapter.blocks_name, str):
            adapter.blocks_name = [[adapter.blocks_name]]
        elif isinstance(adapter.blocks_name, list):
            if isinstance(adapter.blocks_name[0], str):
                if len(adapter.blocks_name) == len(adapter.transformer):
                    adapter.blocks_name = [
                        [blocks_name] for blocks_name in adapter.blocks_name
                    ]
                else:
                    adapter.blocks_name = [adapter.blocks_name]
            elif isinstance(adapter.blocks_name[0], list):
                pass

        if isinstance(adapter.forward_pattern, ForwardPattern):
            adapter.forward_pattern = [[adapter.forward_pattern]]
        elif isinstance(adapter.forward_pattern, list):
            if isinstance(adapter.forward_pattern[0], ForwardPattern):
                if len(adapter.forward_pattern) == len(adapter.transformer):
                    adapter.forward_pattern = [
                        [forward_pattern]
                        for forward_pattern in adapter.forward_pattern
                    ]
                else:
                    adapter.forward_pattern = [adapter.forward_pattern]
            elif isinstance(adapter.forward_pattern[0], list):
                pass

        if isinstance(adapter.dummy_blocks_names, list):
            if len(adapter.dummy_blocks_names) > 0:
                if isinstance(adapter.dummy_blocks_names[0], str):
                    if len(adapter.dummy_blocks_names) == len(
                        adapter.transformer
                    ):
                        adapter.dummy_blocks_names = [
                            [dummy_blocks_names]
                            for dummy_blocks_names in adapter.dummy_blocks_names
                        ]
                    else:
                        adapter.dummy_blocks_names = [
                            adapter.dummy_blocks_names
                        ]
                elif isinstance(adapter.dummy_blocks_names[0], list):
                    pass
            else:
                # Empty dummy_blocks_names
                adapter.dummy_blocks_names = [
                    [] for _ in range(len(adapter.transformer))
                ]

        if adapter.params_modifiers is not None:
            if isinstance(adapter.params_modifiers, ParamsModifier):
                adapter.params_modifiers = [[adapter.params_modifiers]]
            elif isinstance(adapter.params_modifiers, list):
                if isinstance(adapter.params_modifiers[0], ParamsModifier):
                    if len(adapter.params_modifiers) == len(
                        adapter.transformer
                    ):
                        adapter.params_modifiers = [
                            [params_modifiers]
                            for params_modifiers in adapter.params_modifiers
                        ]
                    else:
                        adapter.params_modifiers = [adapter.params_modifiers]
                elif isinstance(adapter.params_modifiers[0], list):
                    pass

        assert len(adapter.transformer) == len(adapter.blocks)
        assert len(adapter.transformer) == len(adapter.blocks_name)
        assert len(adapter.transformer) == len(adapter.forward_pattern)
        assert len(adapter.transformer) == len(adapter.dummy_blocks_names)
        if adapter.params_modifiers is not None:
            assert len(adapter.transformer) == len(adapter.params_modifiers)

        for i in range(len(adapter.blocks)):
            assert len(adapter.blocks[i]) == len(adapter.blocks_name[i])
            assert len(adapter.blocks[i]) == len(adapter.forward_pattern[i])

        if len(adapter.unique_blocks_name) == 0:
            for i in range(len(adapter.transformer)):
                # Generate unique blocks names
                adapter.unique_blocks_name.append(
                    [
                        f"{name}_{hash(id(blocks))}"
                        for blocks, name in zip(
                            adapter.blocks[i],
                            adapter.blocks_name[i],
                        )
                    ]
                )

        assert len(adapter.transformer) == len(adapter.unique_blocks_name)

        # Match Forward Pattern
        for i in range(len(adapter.transformer)):
            for forward_pattern, blocks in zip(
                adapter.forward_pattern[i], adapter.blocks[i]
            ):
                assert BlockAdapter.match_blocks_pattern(
                    blocks,
                    forward_pattern=forward_pattern,
                    check_num_outputs=adapter.check_num_outputs,
                ), (
                    "No block forward pattern matched, "
                    f"supported lists: {ForwardPattern.supported_patterns()}"
                )

        adapter._is_normalized = True

        return adapter

    @classmethod
    def assert_normalized(cls, adapter: "BlockAdapter"):
        if not getattr(adapter, "_is_normalized", False):
            raise RuntimeError("block_adapter must be normailzed.")

    @classmethod
    def is_cached(cls, adapter: Any) -> bool:
        if isinstance(adapter, cls):
            cls.assert_normalized(adapter)
            return all(
                (
                    getattr(adapter.pipe, "_is_cached", False),
                    getattr(adapter.transformer[0], "_is_cached", False),
                )
            )
        elif isinstance(
            adapter,
            (DiffusionPipeline, torch.nn.Module),
        ):
            return getattr(adapter, "_is_cached", False)
        elif isinstance(adapter, list):  # [TRN_0,...]
            assert isinstance(adapter[0], torch.nn.Module)
            return getattr(adapter[0], "_is_cached", False)
        else:
            raise TypeError(f"Can't check this type: {adapter}!")

    @classmethod
    def nested_depth(cls, obj: Any):
        # str: 0; List[str]: 1; List[List[str]]: 2
        if isinstance(obj, (str, bytes)):
            return 0
        if not isinstance(obj, Iterable):
            return 0
        if isinstance(obj, dict):
            items = obj.values()
        else:
            items = obj

        max_depth = 0
        for item in items:
            current_depth = cls.nested_depth(item)
            if current_depth > max_depth:
                max_depth = current_depth
        return 1 + max_depth

    @classmethod
    def flatten(cls, attr: List[Any]) -> List[Any]:
        if not isinstance(attr, list):
            return attr
        flattened = []
        for item in attr:
            if isinstance(item, list) and not isinstance(item, (str, bytes)):
                flattened.extend(cls.flatten(item))
            else:
                flattened.append(item)
        return flattened
