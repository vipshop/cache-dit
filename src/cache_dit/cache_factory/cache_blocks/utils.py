import torch

from cache_dit.cache_factory import cache_context


@torch.compiler.disable
def patch_cached_stats(
    transformer,
):
    # Patch the cached stats to the transformer, the cached stats
    # will be reset for each calling of pipe.__call__(**kwargs).
    if transformer is None:
        return

    # TODO: Patch more cached stats to the transformer
    transformer._cached_steps = cache_context.get_cached_steps()
    transformer._residual_diffs = cache_context.get_residual_diffs()
    transformer._cfg_cached_steps = cache_context.get_cfg_cached_steps()
    transformer._cfg_residual_diffs = cache_context.get_cfg_residual_diffs()
