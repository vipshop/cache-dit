import torch
import dataclasses
import diffusers
import builtins as __builtin__
import contextlib

import numpy as np
from pprint import pprint
from diffusers import DiffusionPipeline

from typing import Dict, Any, List, Union
from cache_dit.caching import CacheType
from cache_dit.caching import BlockAdapter
from cache_dit.caching import BasicCacheConfig
from cache_dit.caching import CalibratorConfig
from cache_dit.caching import FakeDiffusionPipeline
from cache_dit.parallelism import ParallelismConfig
from cache_dit.logger import init_logger


logger = init_logger(__name__)


def dummy_print(*args, **kwargs):
    pass


@contextlib.contextmanager
def disable_print():
    origin_print = __builtin__.print
    __builtin__.print = dummy_print
    yield
    __builtin__.print = origin_print


@torch.compiler.disable
def is_diffusers_at_least_0_3_5() -> bool:
    return diffusers.__version__ >= "0.35.0"


@dataclasses.dataclass
class CacheStats:
    cache_options: dict = dataclasses.field(default_factory=dict)
    # Dual Block Cache
    cached_steps: list[int] = dataclasses.field(default_factory=list)
    residual_diffs: dict[str, float] = dataclasses.field(default_factory=dict)
    cfg_cached_steps: list[int] = dataclasses.field(default_factory=list)
    cfg_residual_diffs: dict[str, float] = dataclasses.field(
        default_factory=dict
    )
    # Dynamic Block Prune
    pruned_steps: list[int] = dataclasses.field(default_factory=list)
    pruned_blocks: list[int] = dataclasses.field(default_factory=list)
    actual_blocks: list[int] = dataclasses.field(default_factory=list)
    pruned_ratio: float = None
    cfg_pruned_steps: list[int] = dataclasses.field(default_factory=list)
    cfg_pruned_blocks: list[int] = dataclasses.field(default_factory=list)
    cfg_actual_blocks: list[int] = dataclasses.field(default_factory=list)
    cfg_pruned_ratio: float = None
    # Parallelism Stats
    parallelism_config: ParallelismConfig = None


def summary(
    adapter_or_others: Union[
        BlockAdapter,
        DiffusionPipeline,
        FakeDiffusionPipeline,
        torch.nn.Module,
    ],
    details: bool = False,
    logging: bool = True,
    **kwargs,
) -> List[CacheStats]:
    if adapter_or_others is None:
        return [CacheStats()]

    if isinstance(adapter_or_others, FakeDiffusionPipeline):
        raise ValueError(
            "Please pass DiffusionPipeline, BlockAdapter or transfomer, "
            "not FakeDiffusionPipeline."
        )

    if not isinstance(adapter_or_others, BlockAdapter):
        if not isinstance(adapter_or_others, DiffusionPipeline):
            transformer = adapter_or_others  # transformer-only
            transformer_2 = None
        else:
            transformer = adapter_or_others.transformer
            transformer_2 = None  # Only for Wan2.2
            if hasattr(adapter_or_others, "transformer_2"):
                transformer_2 = adapter_or_others.transformer_2

        if all(
            (
                not BlockAdapter.is_cached(transformer),
                not BlockAdapter.is_parallelized(transformer),
            )
        ):
            return [CacheStats()]

        blocks_stats: List[CacheStats] = []
        if BlockAdapter.is_cached(transformer):
            for blocks in BlockAdapter.find_blocks(transformer):
                blocks_stats.append(
                    _summary(
                        blocks,
                        details=details,
                        logging=logging,
                        **kwargs,
                    )
                )

        if transformer_2 is not None and BlockAdapter.is_cached(transformer_2):
            for blocks in BlockAdapter.find_blocks(transformer_2):
                blocks_stats.append(
                    _summary(
                        blocks,
                        details=details,
                        logging=logging,
                        **kwargs,
                    )
                )

        blocks_stats.append(
            _summary(
                transformer,
                details=details,
                logging=logging,
                **kwargs,
            )
        )
        if transformer_2 is not None:
            blocks_stats.append(
                _summary(
                    transformer_2,
                    details=details,
                    logging=logging,
                    **kwargs,
                )
            )

        blocks_stats = [
            stats
            for stats in blocks_stats
            if (stats.cache_options or stats.parallelism_config)
        ]

        return blocks_stats if len(blocks_stats) else [CacheStats()]

    adapter = adapter_or_others
    if not BlockAdapter.check_block_adapter(adapter):
        return [CacheStats()]

    blocks_stats = []
    flatten_blocks = BlockAdapter.flatten(adapter.blocks)
    for blocks in flatten_blocks:
        blocks_stats.append(
            _summary(
                blocks,
                details=details,
                logging=logging,
                **kwargs,
            )
        )

    blocks_stats = [stats for stats in blocks_stats if stats.cache_options]

    return blocks_stats if len(blocks_stats) else [CacheStats()]


def strify(
    adapter_or_others: Union[
        BlockAdapter,
        DiffusionPipeline,
        FakeDiffusionPipeline,
        torch.nn.Module,
        CacheStats,
        List[CacheStats],
        Dict[str, Any],
    ],
) -> str:
    if isinstance(adapter_or_others, FakeDiffusionPipeline):
        raise ValueError(
            "Please pass DiffusionPipeline, BlockAdapter or transfomer, "
            "not FakeDiffusionPipeline."
        )

    parallelism_config: ParallelismConfig = None
    if isinstance(adapter_or_others, BlockAdapter):
        stats = summary(adapter_or_others, logging=False)[-1]
        cache_options = stats.cache_options
        cached_steps = len(stats.cached_steps)
    elif isinstance(adapter_or_others, DiffusionPipeline):
        stats = summary(adapter_or_others, logging=False)[-1]
        cache_options = stats.cache_options
        cached_steps = len(stats.cached_steps)
    elif isinstance(adapter_or_others, torch.nn.Module):
        stats = summary(adapter_or_others, logging=False)[-1]
        cache_options = stats.cache_options
        cached_steps = len(stats.cached_steps)
    elif isinstance(adapter_or_others, CacheStats):
        stats = adapter_or_others
        cache_options = stats.cache_options
        cached_steps = len(stats.cached_steps)
    elif isinstance(adapter_or_others, list):
        stats = adapter_or_others[0]
        cache_options = stats.cache_options
        cached_steps = len(stats.cached_steps)
    elif isinstance(adapter_or_others, dict):

        # Assume context_kwargs
        cache_options = adapter_or_others
        cached_steps = None
        cache_type = cache_options.get("cache_type", CacheType.NONE)
        stats = None
        parallelism_config = cache_options.get("parallelism_config", None)

        if cache_type == CacheType.NONE:
            return "NONE"
    else:
        raise ValueError(
            "Please set pipe_or_stats param as one of: "
            "DiffusionPipeline | CacheStats | Dict[str, Any] | List[CacheStats]"
            " | BlockAdapter | Transformer"
        )

    if stats is not None:
        parallelism_config = stats.parallelism_config

    if not cache_options and parallelism_config is None:
        return "NONE"

    def cache_str():
        cache_config: BasicCacheConfig = cache_options.get("cache_config", None)
        if cache_config is not None:
            if cache_config.cache_type == CacheType.NONE:
                return "NONE"
            elif cache_config.cache_type == CacheType.DBCache:
                return cache_config.strify()
            elif cache_config.cache_type == CacheType.DBPrune:
                pruned_ratio = stats.pruned_ratio
                if pruned_ratio is not None:
                    return f"{cache_config.strify()}_P{round(pruned_ratio * 100, 2)}"
                return cache_config.strify()
        return "NONE"

    def calibrator_str():
        calibrator_config: CalibratorConfig = cache_options.get(
            "calibrator_config", None
        )
        if calibrator_config is not None:
            return calibrator_config.strify()
        return "T0O0"

    def parallelism_str():
        if parallelism_config is not None:
            return f"_{parallelism_config.strify()}"
        return ""

    cache_type_str = f"{cache_str()}"
    if cache_type_str != "NONE":
        cache_type_str += f"_{calibrator_str()}"
    cache_type_str += f"{parallelism_str()}"

    if cached_steps:
        cache_type_str += f"_S{cached_steps}"

    return cache_type_str


def _summary(
    pipe_or_module: Union[
        DiffusionPipeline,
        torch.nn.Module,
    ],
    details: bool = False,
    logging: bool = True,
    **kwargs,
) -> CacheStats:
    cache_stats = CacheStats()

    # Get stats from transformer
    if not isinstance(pipe_or_module, torch.nn.Module):
        assert hasattr(pipe_or_module, "transformer")
        module = pipe_or_module.transformer
        cls_name = module.__class__.__name__
    else:
        module = pipe_or_module

    cls_name = module.__class__.__name__
    if isinstance(module, torch.nn.ModuleList):
        cls_name = module[0].__class__.__name__

    if hasattr(module, "_context_kwargs"):
        cache_options = module._context_kwargs
        cache_stats.cache_options = cache_options
        if logging:
            print(f"\n🤗Context Options: {cls_name}\n\n{cache_options}")
    else:
        if logging:
            logger.warning(f"Can't find Context Options for: {cls_name}")

    if hasattr(module, "_parallelism_config"):
        parallelism_config: ParallelismConfig = module._parallelism_config
        cache_stats.parallelism_config = parallelism_config
        if logging:
            print(
                f"\n🤖Parallelism Config: {cls_name}\n\n{parallelism_config.strify(True)}"
            )
    else:
        if logging:
            logger.warning(f"Can't find Parallelism Config for: {cls_name}")

    if hasattr(module, "_cached_steps"):
        cached_steps: list[int] = module._cached_steps
        residual_diffs: dict[str, list | float] = dict(module._residual_diffs)

        if hasattr(module, "_pruned_steps"):
            pruned_steps: list[int] = module._pruned_steps
            pruned_blocks: list[int] = module._pruned_blocks
            actual_blocks: list[int] = module._actual_blocks
            pruned_ratio: float = module._pruned_ratio
        else:
            pruned_steps = []
            pruned_blocks = []
            actual_blocks = []
            pruned_ratio = None

        cache_stats.cached_steps = cached_steps
        cache_stats.residual_diffs = residual_diffs

        cache_stats.pruned_steps = pruned_steps
        cache_stats.pruned_blocks = pruned_blocks
        cache_stats.actual_blocks = actual_blocks
        cache_stats.pruned_ratio = pruned_ratio

        if residual_diffs and logging:
            diffs_values = list(residual_diffs.values())
            if isinstance(diffs_values[0], list):
                diffs_values = [v for sublist in diffs_values for v in sublist]
            qmin = np.min(diffs_values)
            q0 = np.percentile(diffs_values, 0)
            q1 = np.percentile(diffs_values, 25)
            q2 = np.percentile(diffs_values, 50)
            q3 = np.percentile(diffs_values, 75)
            q4 = np.percentile(diffs_values, 95)
            qmax = np.max(diffs_values)

            if pruned_ratio is not None:
                print(
                    f"\n⚡️Pruned Blocks and Residual Diffs Statistics: {cls_name}\n"
                )

                print(
                    "| Pruned Blocks | Diffs P00 | Diffs P25 | Diffs P50 | Diffs P75 | Diffs P95 | Diffs Min | Diffs Max |"
                )
                print(
                    "|---------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|"
                )
                print(
                    f"| {sum(pruned_blocks):<13} | {round(q0, 3):<9} | {round(q1, 3):<9} "
                    f"| {round(q2, 3):<9} | {round(q3, 3):<9} | {round(q4, 3):<9} "
                    f"| {round(qmin, 3):<9} | {round(qmax, 3):<9} |"
                )
                print("")
            else:
                print(
                    f"\n⚡️Cache Steps and Residual Diffs Statistics: {cls_name}\n"
                )

                print(
                    "| Cache Steps | Diffs P00 | Diffs P25 | Diffs P50 | Diffs P75 | Diffs P95 | Diffs Min | Diffs Max |"
                )
                print(
                    "|-------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|"
                )
                print(
                    f"| {len(cached_steps):<11} | {round(q0, 3):<9} | {round(q1, 3):<9} "
                    f"| {round(q2, 3):<9} | {round(q3, 3):<9} | {round(q4, 3):<9} "
                    f"| {round(qmin, 3):<9} | {round(qmax, 3):<9} |"
                )
                print("")

            if pruned_ratio is not None:
                print(
                    f"Dynamic Block Prune Ratio: {round(pruned_ratio * 100, 2)}% ({sum(pruned_blocks)}/{sum(actual_blocks)})\n"
                )

            if details:
                if pruned_ratio is not None:
                    print(
                        f"📚Pruned Blocks and Residual Diffs Details: {cls_name}\n"
                    )
                    pprint(
                        f"Pruned Blocks: {len(pruned_blocks)}, {pruned_blocks}",
                    )
                    pprint(
                        f"Actual Blocks: {len(actual_blocks)}, {actual_blocks}",
                    )
                    pprint(
                        f"Residual Diffs: {len(residual_diffs)}, {residual_diffs}",
                        compact=True,
                    )
                else:
                    print(
                        f"📚Cache Steps and Residual Diffs Details: {cls_name}\n"
                    )
                    pprint(
                        f"Cache Steps: {len(cached_steps)}, {cached_steps}",
                    )
                    pprint(
                        f"Residual Diffs: {len(residual_diffs)}, {residual_diffs}",
                        compact=True,
                    )

    if hasattr(module, "_cfg_cached_steps"):
        cfg_cached_steps: list[int] = module._cfg_cached_steps
        cfg_residual_diffs: dict[str, list | float] = dict(
            module._cfg_residual_diffs
        )

        if hasattr(module, "_cfg_pruned_steps"):
            cfg_pruned_steps: list[int] = module._cfg_pruned_steps
            cfg_pruned_blocks: list[int] = module._cfg_pruned_blocks
            cfg_actual_blocks: list[int] = module._cfg_actual_blocks
            cfg_pruned_ratio: float = module._cfg_pruned_ratio
        else:
            cfg_pruned_steps = []
            cfg_pruned_blocks = []
            cfg_actual_blocks = []
            cfg_pruned_ratio = None

        cache_stats.cfg_cached_steps = cfg_cached_steps
        cache_stats.cfg_residual_diffs = cfg_residual_diffs
        cache_stats.cfg_pruned_steps = cfg_pruned_steps
        cache_stats.cfg_pruned_blocks = cfg_pruned_blocks
        cache_stats.cfg_actual_blocks = cfg_actual_blocks
        cache_stats.cfg_pruned_ratio = cfg_pruned_ratio

        if cfg_residual_diffs and logging:
            cfg_diffs_values = list(cfg_residual_diffs.values())
            if isinstance(cfg_diffs_values[0], list):
                cfg_diffs_values = [
                    v for sublist in cfg_diffs_values for v in sublist
                ]
            qmin = np.min(cfg_diffs_values)
            q0 = np.percentile(cfg_diffs_values, 0)
            q1 = np.percentile(cfg_diffs_values, 25)
            q2 = np.percentile(cfg_diffs_values, 50)
            q3 = np.percentile(cfg_diffs_values, 75)
            q4 = np.percentile(cfg_diffs_values, 95)
            qmax = np.max(cfg_diffs_values)

            if cfg_pruned_ratio is not None:
                print(
                    f"\n⚡️CFG Pruned Blocks and Residual Diffs Statistics: {cls_name}\n"
                )

                print(
                    "| CFG Pruned Blocks | Diffs P00 | Diffs P25 | Diffs P50 | Diffs P75 | Diffs P95 | Diffs Min | Diffs Max |"
                )
                print(
                    "|-------------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|"
                )
                print(
                    f"| {sum(cfg_pruned_blocks):<18} | {round(q0, 3):<9} | {round(q1, 3):<9} "
                    f"| {round(q2, 3):<9} | {round(q3, 3):<9} | {round(q4, 3):<9} "
                    f"| {round(qmin, 3):<9} | {round(qmax, 3):<9} |"
                )
                print("")
            else:
                print(
                    f"\n⚡️CFG Cache Steps and Residual Diffs Statistics: {cls_name}\n"
                )

                print(
                    "| CFG Cache Steps | Diffs P00 | Diffs P25 | Diffs P50 | Diffs P75 | Diffs P95 | Diffs Min | Diffs Max |"
                )
                print(
                    "|-----------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|"
                )
                print(
                    f"| {len(cfg_cached_steps):<15} | {round(q0, 3):<9} | {round(q1, 3):<9} "
                    f"| {round(q2, 3):<9} | {round(q3, 3):<9} | {round(q4, 3):<9} "
                    f"| {round(qmin, 3):<9} | {round(qmax, 3):<9} |"
                )
                print("")

            if cfg_pruned_ratio is not None:
                print(
                    f"CFG Dynamic Block Prune Ratio: {round(cfg_pruned_ratio * 100, 2)}% ({sum(cfg_pruned_blocks)}/{sum(cfg_actual_blocks)})\n"
                )

            if details:
                if cfg_pruned_ratio is not None:
                    print(
                        f"📚CFG Pruned Blocks and Residual Diffs Details: {cls_name}\n"
                    )
                    pprint(
                        f"CFG Pruned Blocks: {len(cfg_pruned_blocks)}, {cfg_pruned_blocks}",
                    )
                    pprint(
                        f"CFG Actual Blocks: {len(cfg_actual_blocks)}, {cfg_actual_blocks}",
                    )
                    pprint(
                        f"CFG Residual Diffs: {len(cfg_residual_diffs)}, {cfg_residual_diffs}",
                        compact=True,
                    )
                else:
                    print(
                        f"📚CFG Cache Steps and Residual Diffs Details: {cls_name}\n"
                    )
                    pprint(
                        f"CFG Cache Steps: {len(cfg_cached_steps)}, {cfg_cached_steps}",
                    )
                    pprint(
                        f"CFG Residual Diffs: {len(cfg_residual_diffs)}, {cfg_residual_diffs}",
                        compact=True,
                    )

    return cache_stats


@torch.compiler.disable
def print_tensor(
    x: torch.Tensor,
    name: str,
    dim: int = 1,
    shape_use_no_distributed: bool = True,
    disable: bool = False,
):
    if disable:
        return

    x = x.contiguous()
    if torch.distributed.is_initialized():
        # all gather hidden_states and check values mean
        gather_x = [
            torch.zeros_like(x)
            for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(gather_x, x)
        gather_x = torch.cat(gather_x, dim=dim)

        if not shape_use_no_distributed:
            x_shape = gather_x.shape
        else:
            x_shape = x.shape

        if torch.distributed.get_rank() == 0:
            print(
                f"{name}, mean: {gather_x.float().mean().item()}, "
                f"std: {gather_x.float().std().item()}, shape: {x_shape}"
            )
    else:
        print(
            f"{name}, mean: {x.float().mean().item()}, "
            f"std: {x.float().std().item()}, shape: {x.shape}"
        )


def supported_matrix() -> str | None:
    try:
        from cache_dit.caching.block_adapters.block_registers import (
            BlockAdapterRegistry,
        )

        _pipelines_supported_cache = BlockAdapterRegistry.supported_pipelines()[
            1
        ]
        _pipelines_supported_cache += [
            "LongCatVideo",  # not in diffusers, but supported
        ]
        from cache_dit.parallelism.backends.native_diffusers import (
            ContextParallelismPlannerRegister,
        )

        _pipelines_supported_context_parallelism = (
            ContextParallelismPlannerRegister.supported_planners()[1]
        )
        from cache_dit.parallelism.backends.native_pytorch import (
            TensorParallelismPlannerRegister,
        )

        _pipelines_supported_tensor_parallelism = (
            TensorParallelismPlannerRegister.supported_planners()[1]
        )
        # Add some special aliases since cp/tp planners use the name shortcut
        # of Transformer only.
        _pipelines_supported_context_parallelism += [
            "Wan",
            "LTX",
            "VisualCloze",
        ]
        _pipelines_supported_tensor_parallelism += [
            "Wan",
            "VisualCloze",
        ]

        # generate the supported matrix, markdown table format
        matrix_lines: List[str] = []
        header = "| Model Series | Cache Acceleration | Context Parallelism | Tensor Parallelism | Documentaion |"
        matrix_lines.append(header)
        matrix_lines.append("|---|---|---|---|---|")

        for pipeline in _pipelines_supported_cache:
            cp_support = (
                "✅"
                if pipeline in _pipelines_supported_context_parallelism
                else "✖️"
            )
            tp_support = (
                "✅"
                if pipeline in _pipelines_supported_tensor_parallelism
                else "✖️"
            )
            line = f"| 🎉{pipeline} | ✅ | {cp_support} | {tp_support} |"
            line += " 📚[docs](./docs/User_Guide.md), 📚[examples](./examples/pipelines) |"
            matrix_lines.append(line)
        # sort by '✅'
        matrix_lines = [matrix_lines[0], matrix_lines[1]] + sorted(
            matrix_lines[2:],
            key=lambda x: (x.count("✅"), x),
            reverse=True,
        )
        matrix_str = "\n".join(matrix_lines)

        print("\nSupported Cache and Parallelism Matrix:\n")
        print(matrix_str)
        return matrix_str
    except Exception:
        return None
