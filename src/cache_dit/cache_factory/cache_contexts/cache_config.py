import torch
import dataclasses
from typing import Optional, Union
from cache_dit.cache_factory.cache_types import CacheType
from cache_dit.logger import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass
class BasicCacheConfig:
    # Default: Dual Block Cache with Flexible FnBn configuration.
    cache_type: CacheType = CacheType.DBCache  # DBCache, DBPrune, NONE

    # Fn_compute_blocks: (`int`, *required*, defaults to 8):
    #     Specifies that `DBCache` uses the **first n** Transformer blocks to fit the information
    #     at time step t, enabling the calculation of a more stable L1 diff and delivering more
    #     accurate information to subsequent blocks. Please check https://github.com/vipshop/cache-dit/blob/main/docs/DBCache.md
    #     for more details of DBCache.
    Fn_compute_blocks: int = 8
    # Bn_compute_blocks: (`int`, *required*, defaults to 0):
    #     Further fuses approximate information in the **last n** Transformer blocks to enhance
    #     prediction accuracy. These blocks act as an auto-scaler for approximate hidden states
    #     that use residual cache.
    Bn_compute_blocks: int = 0
    # residual_diff_threshold (`float`, *required*, defaults to 0.08):
    #     the value of residual diff threshold, a higher value leads to faster performance at the
    #     cost of lower precision.
    residual_diff_threshold: Union[torch.Tensor, float] = 0.08
    # max_warmup_steps (`int`, *required*, defaults to 8):
    #     DBCache does not apply the caching strategy when the number of running steps is less than
    #     or equal to this value, ensuring the model sufficiently learns basic features during warmup.
    max_warmup_steps: int = 8  # DON'T Cache in warmup steps
    # warmup_interval (`int`, *required*, defaults to 1):
    #     Skip interval in warmup steps, e.g., when warmup_interval is 2, only 0, 2, 4, ... steps
    #     in warmup steps will be computed, others will use dynamic cache.
    warmup_interval: int = 1  # skip interval in warmup steps
    # max_cached_steps (`int`, *required*, defaults to -1):
    #     DBCache disables the caching strategy when the previous cached steps exceed this value to
    #     prevent precision degradation.
    max_cached_steps: int = -1  # for both CFG and non-CFG
    # max_continuous_cached_steps (`int`, *required*, defaults to -1):
    #     DBCache disables the caching strategy when the previous continous cached steps exceed this value to
    #     prevent precision degradation.
    max_continuous_cached_steps: int = -1  # the max continuous cached steps
    # enable_separate_cfg (`bool`, *required*,  defaults to None):
    #     Whether to do separate cfg or not, such as Wan 2.1, Qwen-Image. For model that fused CFG
    #     and non-CFG into single forward step, should set enable_separate_cfg as False, for example:
    #     CogVideoX, HunyuanVideo, Mochi, etc.
    enable_separate_cfg: Optional[bool] = None
    # cfg_compute_first (`bool`, *required*,  defaults to False):
    #     Compute cfg forward first or not, default False, namely, 0, 2, 4, ..., -> non-CFG step;
    #     1, 3, 5, ... -> CFG step.
    cfg_compute_first: bool = False
    # cfg_diff_compute_separate (`bool`, *required*,  defaults to True):
    #     Compute separate diff values for CFG and non-CFG step, default True. If False, we will
    #     use the computed diff from current non-CFG transformer step for current CFG step.
    cfg_diff_compute_separate: bool = True

    def update(self, **kwargs) -> "BasicCacheConfig":
        for key, value in kwargs.items():
            if hasattr(self, key):
                if value is not None:
                    setattr(self, key, value)
        return self

    def empty(self, **kwargs) -> "BasicCacheConfig":
        # Set all fields to None
        for field in dataclasses.fields(self):
            if hasattr(self, field.name):
                setattr(self, field.name, None)
        if kwargs:
            self.update(**kwargs)
        return self

    def reset(self, **kwargs) -> "BasicCacheConfig":
        return self.empty(**kwargs)

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

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


@dataclasses.dataclass
class ExtraCacheConfig:
    # Some other not very important settings for Dual Block Cache.
    # NOTE: These flags maybe deprecated in the future and users
    # should never use these extra configurations in their cases.

    # l1_hidden_states_diff_threshold (`float`, *optional*, defaults to None):
    #     The hidden states diff threshold for DBCache if use hidden_states as
    #     cache (not residual).
    l1_hidden_states_diff_threshold: float = None
    # important_condition_threshold (`float`, *optional*, defaults to 0.0):
    #     Only select the most important tokens while calculating the l1 diff.
    important_condition_threshold: float = 0.0
    # downsample_factor (`int`, *optional*, defaults to 1):
    #     Downsample factor for Fn buffer, in order the save GPU memory.
    downsample_factor: int = 1
    # num_inference_steps (`int`, *optional*, defaults to -1):
    #     num_inference_steps for DiffusionPipeline, for future use.
    num_inference_steps: int = -1


@dataclasses.dataclass
class DBCacheConfig(BasicCacheConfig):
    pass  # Just an alias for BasicCacheConfig
