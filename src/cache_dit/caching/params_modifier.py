from typing import Optional

from .cache_contexts import BasicCacheConfig
from .cache_contexts import CalibratorConfig

from cache_dit.logger import init_logger

logger = init_logger(__name__)


class ParamsModifier:
    def __init__(
        self,
        # BasicCacheConfig, DBCacheConfig, DBPruneConfig, etc.
        cache_config: BasicCacheConfig = None,
        # Calibrator config: TaylorSeerCalibratorConfig, etc.
        calibrator_config: Optional[CalibratorConfig] = None,
        # Other cache context kwargs: Deprecated cache kwargs
        **kwargs,
    ):
        self._context_kwargs = {}

        # WARNING: Deprecated cache config params. These parameters are now retained
        # for backward compatibility but will be removed in the future.
        deprecated_kwargs = {
            "Fn_compute_blocks": kwargs.get("Fn_compute_blocks", None),
            "Bn_compute_blocks": kwargs.get("Bn_compute_blocks", None),
            "max_warmup_steps": kwargs.get("max_warmup_steps", None),
            "max_cached_steps": kwargs.get("max_cached_steps", None),
            "max_continuous_cached_steps": kwargs.get("max_continuous_cached_steps", None),
            "residual_diff_threshold": kwargs.get("residual_diff_threshold", None),
            "enable_separate_cfg": kwargs.get("enable_separate_cfg", None),
            "cfg_compute_first": kwargs.get("cfg_compute_first", None),
            "cfg_diff_compute_separate": kwargs.get("cfg_diff_compute_separate", None),
        }

        deprecated_kwargs = {k: v for k, v in deprecated_kwargs.items() if v is not None}

        if deprecated_kwargs:
            logger.warning(
                "Manually settup DBCache context without BasicCacheConfig is "
                "deprecated and will be removed in the future, please use "
                "`cache_config` parameter instead!"
            )
            if cache_config is not None:
                cache_config.update(**deprecated_kwargs)
            else:
                cache_config = BasicCacheConfig(**deprecated_kwargs)

        if cache_config is not None:
            self._context_kwargs["cache_config"] = cache_config
        # WARNING: Deprecated taylorseer params. These parameters are now retained
        # for backward compatibility but will be removed in the future.
        if (
            kwargs.get("enable_taylorseer", None) is not None
            or kwargs.get("enable_encoder_taylorseer", None) is not None
        ):
            logger.warning(
                "Manually settup TaylorSeer calibrator without TaylorSeerCalibratorConfig is "
                "deprecated and will be removed in the future, please use "
                "`calibrator_config` parameter instead!"
            )
            from cache_dit.caching.cache_contexts.calibrators import (
                TaylorSeerCalibratorConfig,
            )

            calibrator_config = TaylorSeerCalibratorConfig(
                enable_calibrator=kwargs.get("enable_taylorseer"),
                enable_encoder_calibrator=kwargs.get("enable_encoder_taylorseer"),
                calibrator_cache_type=kwargs.get("taylorseer_cache_type", "residual"),
                taylorseer_order=kwargs.get("taylorseer_order", 1),
            )

        if calibrator_config is not None:
            self._context_kwargs["calibrator_config"] = calibrator_config
