import dataclasses
from typing import List
from ..cache_types import CacheType
from .cache_config import BasicCacheConfig

from cache_dit.logger import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass
class DBPruneConfig(BasicCacheConfig):
    # Dyanamic Block Prune specific configurations
    cache_type: CacheType = CacheType.DBPrune  # DBPrune

    # enable_dynamic_prune_threshold (`bool`, *required*, defaults to False):
    #     Whether to enable the dynamic prune threshold or not. If True, we will
    #     compute the dynamic prune threshold based on the mean of the residual
    #     diffs of the previous computed or pruned blocks.
    #     But, also limit mean_diff to be at least 2x the residual_diff_threshold
    #     to avoid too aggressive pruning.
    enable_dynamic_prune_threshold: bool = False
    # max_dynamic_prune_threshold (`float`, *optional*, defaults to None):
    #     The max dynamic prune threshold, if not None, the dynamic prune threshold
    #     will not exceed this value. If None, we will limit it to be at least 2x
    #     the residual_diff_threshold to avoid too aggressive pruning.
    max_dynamic_prune_threshold: float = None
    # dynamic_prune_threshold_relax_ratio (`float`, *optional*, defaults to 1.25):
    #     The relax ratio for dynamic prune threshold, the dynamic prune threshold
    #     will be set as:
    #         dynamic_prune_threshold = mean_diff * dynamic_prune_threshold_relax_ratio
    #     to avoid too aggressive pruning.
    #     The default value is 1.25, which means the dynamic prune threshold will
    #     be 1.25 times the mean of the residual diffs of the previous computed
    #     or pruned blocks.
    #     Users can tune this value to achieve a better trade-off between speedup
    #     and precision. A higher value leads to more aggressive pruning
    #     and faster speedup, but may also lead to lower precision.
    dynamic_prune_threshold_relax_ratio: float = 1.25
    # non_prune_block_ids (`List[int]`, *optional*, defaults to []):
    #     The list of block ids that will not be pruned, even if their residual
    #     diffs are below the prune threshold. This can be useful for the first
    #     few blocks, which are usually more important for the model performance.
    non_prune_block_ids: List[int] = dataclasses.field(default_factory=list)
    # force_reduce_calibrator_vram (`bool`, *optional*, defaults to True):
    #     Whether to force reduce the VRAM usage of the calibrator for Dynamic Block
    #     Prune. If True, we will set the downsample_factor of the extra_cache_config
    #     to at least 2 to reduce the VRAM usage of the calibrator.
    force_reduce_calibrator_vram: bool = False

    def strify(self) -> str:
        return (
            f"{self.cache_type}_"
            f"F{self.Fn_compute_blocks}"
            f"B{self.Bn_compute_blocks}_"
            f"W{self.max_warmup_steps}"
            f"I{self.warmup_interval}"
            f"M{max(0, self.max_cached_steps)}"
            f"MC{max(0, self.max_continuous_cached_steps)}_"
            f"R{self.residual_diff_threshold}"
        )
