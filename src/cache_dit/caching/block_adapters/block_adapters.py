import torch

import inspect
import dataclasses
from collections.abc import Iterable

from typing import Any, Tuple, List, Optional, Union

from diffusers import DiffusionPipeline, ModelMixin
from ..patch_functors import PatchFunctor
from ..forward_pattern import ForwardPattern
from ..params_modifier import ParamsModifier

from cache_dit.logger import init_logger

logger = init_logger(__name__)


class FakeDiffusionPipeline:
    # A placeholder for pipelines when pipe is None.
    def __init__(
        self,
        transformer: Optional[torch.nn.Module | ModelMixin] = None,
    ):
        self.transformer = transformer  # Reference only


@dataclasses.dataclass
class BlockAdapter:

    # Transformer configurations.
    pipe: Union[
        DiffusionPipeline,
        FakeDiffusionPipeline,
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
    params_modifiers: Optional[
        Union[
            ParamsModifier,
            List[ParamsModifier],
            List[List[ParamsModifier]],
        ]
    ] = None

    check_forward_pattern: Optional[bool] = None
    check_num_outputs: bool = False

    # Pipeline Level Flags
    # Patch Functor: Flux, etc.
    patch_functor: Optional[PatchFunctor] = None
    # Flags for separate cfg
    has_separate_cfg: bool = False

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
    allow_suffixes: List[str] = dataclasses.field(default_factory=lambda: ["TransformerBlock"])
    check_suffixes: bool = False
    blocks_policy: str = dataclasses.field(
        default="max", metadata={"allowed_values": ["max", "min"]}
    )

    # Other Flags
    skip_post_init: bool = False

    def __post_init__(self):
        if self.skip_post_init:
            return

        self.maybe_fake_pipe()
        if any((self.pipe is not None, self.transformer is not None)):
            self.maybe_fill_attrs()
            self.maybe_patchify()
            self.maybe_skip_checks()

    def maybe_fake_pipe(self):
        if self.pipe is None:
            self.pipe = FakeDiffusionPipeline()
            logger.warning("pipe is None, use FakeDiffusionPipeline instead.")

    def maybe_skip_checks(self):
        if self.check_forward_pattern is None:
            if self.transformer is not None:
                if self.nested_depth(self.transformer) == 0:
                    transformer = self.transformer
                elif self.nested_depth(self.transformer) == 1:
                    transformer = self.transformer[0]
                else:
                    raise ValueError(
                        "transformer nested depth can't more than 1, "
                        f"current is: {self.nested_depth(self.transformer)}"
                    )
                if transformer.__module__.startswith("diffusers"):
                    self.check_forward_pattern = True
                    logger.info(
                        f"Found transformer from diffusers: {transformer.__module__} "
                        "enable check_forward_pattern by default."
                    )
                else:
                    self.check_forward_pattern = False
                    logger.info(
                        f"Found transformer NOT from diffusers: {transformer.__module__} "
                        "disable check_forward_pattern by default."
                    )

        if getattr(self.transformer, "_hf_hook", None) is not None:
            logger.warning("_hf_hook is not None, force skip pattern check!")
            self.check_forward_pattern = False
            self.check_num_outputs = False
        elif getattr(self.transformer, "_diffusers_hook", None) is not None:
            logger.warning("_diffusers_hook is not None, force skip pattern check!")
            self.check_forward_pattern = False
            self.check_num_outputs = False

    def maybe_fill_attrs(self):
        # NOTE: This func should be call before normalize.
        # Allow empty `blocks_names`, we will auto fill it.
        # TODO: preprocess more empty attrs.
        if self.transformer is not None and self.blocks is not None and self.blocks_name is None:

            def _find(transformer, blocks):
                attr_names = dir(transformer)
                assert isinstance(blocks, torch.nn.ModuleList)
                blocks_name = None
                for attr_name in attr_names:
                    if (attr := getattr(transformer, attr_name, None)) is not None:
                        if isinstance(attr, torch.nn.ModuleList) and id(attr) == id(blocks):
                            blocks_name = attr_name
                            break
                assert blocks_name is not None, "No blocks_name match, please set it manually!"
                return blocks_name

            if self.nested_depth(self.transformer) == 0:
                if self.nested_depth(self.blocks) == 0:  # str
                    self.blocks_name = _find(self.transformer, self.blocks)
                elif self.nested_depth(self.blocks) == 1:
                    self.blocks_name = [_find(self.transformer, blocks) for blocks in self.blocks]
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
                        for transformer, blocks in zip(self.transformer, self.blocks)
                    ]
                elif self.nested_depth(self.blocks) == 2:  # List[List[str]]
                    assert len(self.transformer) == len(self.blocks)
                    self.blocks_name = []
                    for i in range(len(self.blocks)):
                        self.blocks_name.append(
                            [_find(self.transformer[i], blocks) for blocks in self.blocks[i]]
                        )
                else:
                    raise ValueError(
                        "Blocks nested depth can only be 1 or 2 "
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

        if self.patch_functor is not None:
            if self.transformer is not None:
                if self.nested_depth(self.transformer) == 0:
                    self.patch_functor.apply(self.transformer, *args, **kwargs)
                elif self.nested_depth(self.transformer) == 1:
                    for transformer in self.transformer:
                        self.patch_functor.apply(transformer, *args, **kwargs)
                else:
                    raise ValueError(
                        "transformer nested depth can't more than 1, "
                        f"current is: {self.nested_depth(self.transformer)}"
                    )
            else:
                assert hasattr(self.pipe, "transformer"), (
                    "pipe.transformer can not be None when patch_functor "
                    "is provided and transformer is None."
                )
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
        assert adapter.forward_pattern is not None, "adapter.forward_pattern can not be None."
        pipe = adapter.pipe
        if isinstance(pipe, FakeDiffusionPipeline):
            raise ValueError("Can not auto block adapter for FakeDiffusionPipeline.")

        assert hasattr(pipe, "transformer"), "pipe.transformer can not be None."

        transformer = pipe.transformer

        # "transformer_blocks", "blocks", "single_transformer_blocks", "layers"
        blocks, blocks_name = BlockAdapter.find_match_blocks(
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

        if getattr(adapter, "_is_normlized", False):
            return True

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

        if BlockAdapter.nested_depth(adapter.blocks) == 0:
            blocks = adapter.blocks
        else:
            blocks = BlockAdapter.flatten(adapter.blocks)[0]

        if not isinstance(blocks, torch.nn.ModuleList):
            logger.warning("blocks is not ModuleList.")
            return False

        return True

    @staticmethod
    def find_match_blocks(
        transformer: torch.nn.Module,
        allow_prefixes: List[str] = [
            "transformer_blocks",
            "single_transformer_blocks",
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
            if (blocks := getattr(transformer, blocks_name, None)) is not None:
                if isinstance(blocks, torch.nn.ModuleList):
                    block = blocks[0]
                    block_cls_name: str = block.__class__.__name__
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
            raise ValueError("Auto selected transformer blocks failed, please set it manually.")

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
    def find_blocks(
        transformer: torch.nn.Module,
    ) -> List[torch.nn.ModuleList]:
        total_blocks = []
        for attr in dir(transformer):
            if (blocks := getattr(transformer, attr, None)) is not None:
                if isinstance(blocks, torch.nn.ModuleList):
                    if isinstance(blocks[0], torch.nn.Module):
                        total_blocks.append(blocks)
        return total_blocks

    @staticmethod
    def match_block_pattern(
        block: torch.nn.Module,
        forward_pattern: ForwardPattern,
        **kwargs,
    ) -> bool:

        if not kwargs.get("check_forward_pattern", True):
            return True

        assert (
            forward_pattern.Supported and forward_pattern in ForwardPattern.supported_patterns()
        ), f"Pattern {forward_pattern} is not support now!"

        # NOTE: Special case for HiDreamBlock
        if hasattr(block, "block"):
            if isinstance(block.block, torch.nn.Module):
                block = block.block

        forward_parameters = set(inspect.signature(block.forward).parameters.keys())

        in_matched = True
        out_matched = True

        if kwargs.get("check_num_outputs", True):
            num_outputs = str(inspect.signature(block.forward).return_annotation).count(
                "torch.Tensor"
            )

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

        if not kwargs.get("check_forward_pattern", True):
            if logging:
                logger.warning(f"Skipped Forward Pattern Check: {forward_pattern}")
            return True

        assert (
            forward_pattern.Supported and forward_pattern in ForwardPattern.supported_patterns()
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
            block_cls_names = [block.__class__.__name__ for block in transformer_blocks]
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
        unique: bool = True,
    ) -> "BlockAdapter":

        if getattr(adapter, "_is_normalized", False):
            return adapter

        if BlockAdapter.nested_depth(adapter.transformer) == 0:
            adapter.transformer = [adapter.transformer]

        def _normalize_attr(attr: Any):
            normalized_attr = attr
            if attr is None:
                return normalized_attr

            if BlockAdapter.nested_depth(attr) == 0:
                normalized_attr = [[attr]]
            elif BlockAdapter.nested_depth(attr) == 1:  # List
                if attr:  # not-empty
                    if len(attr) == len(adapter.transformer):
                        normalized_attr = [[a] for a in attr]
                    else:
                        normalized_attr = [attr]
                else:  # [] empty
                    normalized_attr = [[] for _ in range(len(adapter.transformer))]

            assert len(adapter.transformer) == len(normalized_attr)
            return normalized_attr

        adapter.blocks = _normalize_attr(adapter.blocks)
        adapter.blocks_name = _normalize_attr(adapter.blocks_name)
        adapter.forward_pattern = _normalize_attr(adapter.forward_pattern)
        adapter.dummy_blocks_names = _normalize_attr(adapter.dummy_blocks_names)
        adapter.params_modifiers = _normalize_attr(adapter.params_modifiers)
        # Some times, the cache_config will be None.
        # So we do not perform unique check here.
        if unique:
            BlockAdapter.unique(adapter)

        adapter._is_normalized = True

        return adapter

    @classmethod
    def unique(cls, adapter: "BlockAdapter"):
        # NOTE: Users should never call this function
        for i in range(len(adapter.blocks)):
            assert len(adapter.blocks[i]) == len(adapter.blocks_name[i])
            assert len(adapter.blocks[i]) == len(adapter.forward_pattern[i])

        # Generate unique blocks names
        if len(adapter.unique_blocks_name) == 0:
            for i in range(len(adapter.transformer)):
                adapter.unique_blocks_name.append(
                    [
                        f"{name}_{hash(id(blocks))}"
                        for blocks, name in zip(
                            adapter.blocks[i],
                            adapter.blocks_name[i],
                        )
                    ]
                )
        else:
            assert len(adapter.transformer) == len(adapter.unique_blocks_name)

        # Also check Match Forward Pattern
        for i in range(len(adapter.transformer)):
            for forward_pattern, blocks in zip(adapter.forward_pattern[i], adapter.blocks[i]):
                assert BlockAdapter.match_blocks_pattern(
                    blocks,
                    forward_pattern=forward_pattern,
                    check_num_outputs=adapter.check_num_outputs,
                    check_forward_pattern=adapter.check_forward_pattern,
                ), (
                    "No block forward pattern matched, "
                    f"supported lists: {ForwardPattern.supported_patterns()}"
                )

    @classmethod
    def assert_normalized(cls, adapter: "BlockAdapter"):
        if not getattr(adapter, "_is_normalized", False):
            raise RuntimeError("block_adapter must be normailzed.")

    @classmethod
    def is_normalized(cls, adapter: "BlockAdapter") -> bool:
        return getattr(adapter, "_is_normalized", False)

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
            return getattr(adapter, "_is_cached", False)

    @classmethod
    def is_parallelized(cls, adapter: Any) -> bool:
        if isinstance(adapter, cls):
            cls.assert_normalized(adapter)
            return getattr(adapter.transformer[0], "_is_parallelized", False)
        elif isinstance(adapter, DiffusionPipeline):
            return getattr(adapter.transformer, "_is_parallelized", False)
        elif isinstance(adapter, torch.nn.Module):
            return getattr(adapter, "_is_parallelized", False)
        elif isinstance(adapter, list):  # [TRN_0,...]
            assert isinstance(adapter[0], torch.nn.Module)
            return getattr(adapter[0], "_is_parallelized", False)
        else:
            return getattr(adapter, "_is_parallelized", False)

    @classmethod
    def nested_depth(cls, obj: Any):
        # str: 0; List[str]: 1; List[List[str]]: 2
        atom_types = (
            str,
            bytes,
            torch.nn.ModuleList,
            torch.nn.Module,
            torch.Tensor,
        )
        if isinstance(obj, atom_types):
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
        atom_types = (
            str,
            bytes,
            torch.nn.ModuleList,
            torch.nn.Module,
            torch.Tensor,
        )
        if not isinstance(attr, list):
            return attr
        flattened = []
        for item in attr:
            if isinstance(item, list) and not isinstance(item, atom_types):
                flattened.extend(cls.flatten(item))
            else:
                flattened.append(item)
        return flattened
