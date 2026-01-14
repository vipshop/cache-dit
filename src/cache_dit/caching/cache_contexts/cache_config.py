import torch
import dataclasses
from typing import Optional, Union, List
from ..cache_types import CacheType
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
    # max_accumulated_residual_diff_threshold (`float`, *optional*, defaults to None):
    #     The maximum accumulated relative l1 diff threshold for Cache. If set, when the
    #     accumulated relative l1 diff exceeds this threshold, the caching strategy will be
    #     disabled for current step. This is useful for some cases where the input condition
    #     changes significantly in a single step. Default None means this feature is disabled.
    max_accumulated_residual_diff_threshold: Optional[float] = None
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
    # num_inference_steps (`int`, *optional*, defaults to None):
    #     num_inference_steps for DiffusionPipeline, used to adjust some internal settings
    #     for better caching performance. For example, we will refresh the cache once the
    #     executed steps exceed num_inference_steps if num_inference_steps is provided.
    num_inference_steps: Optional[int] = None
    # steps_computation_mask (`List[int]`, *optional*, defaults to None):
    #     This param introduce LeMiCa/EasyCache style compute mask for steps. It is a list
    #     of length num_inference_steps indicating whether to compute each step or not.
    #     1 means must compute, 0 means use dynamic/static cache. If provided, will override
    #     other settings to decide whether to compute each step.
    steps_computation_mask: Optional[List[int]] = None
    # steps_computation_policy (`str`, *optional*, defaults to "dynamic"):
    #     The computation policy for steps when using steps_computation_mask. It can be
    #     "dynamic" or "static". "dynamic" means using dynamic cache for steps marked as 0
    #     in steps_computation_mask, while "static" means using static cache for those steps.
    steps_computation_policy: str = "dynamic"  # "dynamic" or "static"

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
        base_str = (
            f"{self.cache_type}_"
            f"F{self.Fn_compute_blocks}"
            f"B{self.Bn_compute_blocks}_"
            f"W{self.max_warmup_steps}"
            f"I{self.warmup_interval}"
            f"M{max(0, self.max_cached_steps)}"
            f"MC{max(0, self.max_continuous_cached_steps)}_"
            f"R{self.residual_diff_threshold}"
        )

        if self.steps_computation_mask is not None:
            base_str += f"_SCM{''.join(map(str, self.steps_computation_mask))}"
            base_str += f"_{self.steps_computation_policy}"

        if self.num_inference_steps is not None:
            base_str += f"_N{self.num_inference_steps}"

        if self.enable_separate_cfg is not None:
            base_str += f"_CFG{int(self.enable_separate_cfg)}"

        return base_str


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


@dataclasses.dataclass
class DBCacheConfig(BasicCacheConfig):
    pass  # Just an alias for BasicCacheConfig
