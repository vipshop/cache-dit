import inspect
import torch
from diffusers import DiffusionPipeline, ModelMixin
from cache_dit.cache_factory.adapters import CacheType
from cache_dit.cache_factory.adapters import apply_cache_on_pipe
from cache_dit.cache_factory.adapters import apply_cache_on_transformer
from cache_dit.cache_factory.utils import load_cache_options_from_yaml
from cache_dit.logger import init_logger

logger = init_logger(__name__)


def make_pattern(in_params: list, out_params: list) -> dict:
    return {"IN": in_params, "OUT": out_params}


def supported_patterns():
    return [
        make_pattern(
            ["hidden_states", "encoder_hidden_states"],
            ["hidden_states", "encoder_hidden_states"],
        ),
        # TODO: support more cache patterns.
    ]


def load_options(path: str):
    """cache_dit.load_options(cache_config.yaml)"""
    return load_cache_options_from_yaml(path)


def match_pattern(transformer_blocks: torch.nn.ModuleList) -> bool:
    pattern_matched = True
    pattern_ids = []
    for block in transformer_blocks:
        forward_parameters = set(
            inspect.signature(block.forward).parameters.keys()
        )

        matched_pattern_id = None
        for j, pattern in enumerate(supported_patterns()):
            param_matched = True
            for required_param in pattern["IN"]:
                if required_param not in forward_parameters:
                    param_matched = False
                    break
            if param_matched:
                matched_pattern_id = j  # last pattern
        if matched_pattern_id is not None:
            pattern_ids.append(matched_pattern_id)
        else:
            pattern_matched = False
            break

    if pattern_matched:
        unique_pattern_ids = set(pattern_ids)
        if len(unique_pattern_ids) > 1:
            pattern_matched = False
        else:
            pattern_id = unique_pattern_ids[0]
            logger.info(
                f"Match cache pattern: IN ({supported_patterns[pattern_id]['IN']}),"
                f"OUT ({supported_patterns[pattern_id]['OUT']})"
            )

    return pattern_matched


def enable_cache(
    pipe: DiffusionPipeline,
    *args,
    **kwargs,
) -> DiffusionPipeline:
    if transformer_blocks := kwargs.pop("transformer_blocks", None):
        assert isinstance(transformer_blocks, torch.nn.ModuleList)
        assert match_pattern(transformer_blocks), (
            "No block forward pattern matched, "
            f"supported lists: {supported_patterns()}"
        )
    if isinstance(pipe, DiffusionPipeline):
        return apply_cache_on_pipe(pipe, *args, **kwargs)
    else:
        raise ValueError("`pipe` must be a valid DiffusionPipeline")
