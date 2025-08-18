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
        # make_pattern(
        #     ["hidden_states", "encoder_hidden_states"],
        #     ["hidden_states"],
        # ),
        # make_pattern(
        #     ["hidden_states"],
        #     ["hidden_states"],
        # ),
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
    pipe_or_transformer: DiffusionPipeline,
    *args,
    **kwargs,
) -> DiffusionPipeline | ModelMixin:
    if transformer_blocks := kwargs.pop("transformer_blocks", None):
        assert isinstance(transformer_blocks, torch.nn.ModuleList)
        assert match_pattern(transformer_blocks), (
            "No block forward pattern matched, "
            f"supported lists: {supported_patterns()}"
        )
    # TODO: support caching for transformer module directly.
    if isinstance(pipe_or_transformer, DiffusionPipeline):
        return apply_cache_on_pipe(pipe_or_transformer, *args, **kwargs)
    elif isinstance(pipe_or_transformer, ModelMixin):
        # Assume you have pass a transformer (subclass of ModelMixin)
        return apply_cache_on_transformer(pipe_or_transformer, *args, **kwargs)
    else:
        return pipe_or_transformer  # do not-thing
