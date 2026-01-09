import yaml
import copy
from typing import Tuple, Optional, Union
from cache_dit.caching.cache_contexts import (
    DBCacheConfig,
    TaylorSeerCalibratorConfig,
    DBPruneConfig,
    CalibratorConfig,
)
from cache_dit.parallelism import ParallelismConfig, ParallelismBackend
from cache_dit.logger import init_logger

logger = init_logger(__name__)


def load_cache_options_from_dict(cache_kwargs: dict, reset: bool = False) -> dict:
    try:
        # deep copy to avoid modifying original kwargs
        kwargs: dict = copy.deepcopy(cache_kwargs)
        cache_context_kwargs = {}
        if kwargs.get("enable_taylorseer", False):
            cache_context_kwargs["calibrator_config"] = (
                TaylorSeerCalibratorConfig(
                    enable_calibrator=kwargs.get("enable_taylorseer"),
                    enable_encoder_calibrator=kwargs.get("enable_encoder_taylorseer", False),
                    calibrator_cache_type=kwargs.get("taylorseer_cache_type", "residual"),
                    taylorseer_order=kwargs.get("taylorseer_order", 1),
                )
                if not reset
                else TaylorSeerCalibratorConfig().reset(
                    enable_calibrator=kwargs.get("enable_taylorseer"),
                    enable_encoder_calibrator=kwargs.get("enable_encoder_taylorseer", False),
                    calibrator_cache_type=kwargs.get("taylorseer_cache_type", "residual"),
                    taylorseer_order=kwargs.get("taylorseer_order", 1),
                )
            )

        if "cache_type" not in kwargs:
            # Assume DBCache if cache_type is not specified
            cache_context_kwargs["cache_config"] = (
                DBCacheConfig() if not reset else DBCacheConfig().reset()
            )
            cache_context_kwargs["cache_config"].update(**kwargs)
        else:
            cache_type = str(kwargs.get("cache_type", None))
            if cache_type == "DBCache":

                cache_context_kwargs["cache_config"] = (
                    DBCacheConfig() if not reset else DBCacheConfig().reset()
                )
                cache_context_kwargs["cache_config"].update(**kwargs)
            elif cache_type == "DBPrune":

                cache_context_kwargs["cache_config"] = (
                    DBPruneConfig() if not reset else DBPruneConfig().reset()
                )
                cache_context_kwargs["cache_config"].update(**kwargs)
            else:
                raise ValueError(f"Unsupported cache_type: {cache_type}.")

        return cache_context_kwargs

    except Exception as e:
        raise ValueError(f"Error parsing cache configuration. {str(e)}")


def load_cache_options_from_yaml(yaml_file_path: str, reset: bool = False) -> dict:
    try:
        with open(yaml_file_path, "r") as f:
            kwargs: dict = yaml.safe_load(f)
        return load_cache_options_from_dict(kwargs, reset)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {yaml_file_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"YAML file parsing error: {str(e)}")


def load_options(path_or_dict: str | dict, reset: bool = False) -> dict:
    r"""
    Load cache options from a YAML file or a dictionary.
    Args:
        path_or_dict (`str` or `dict`):
            The file path to the YAML configuration file or a dictionary containing the configuration.
        reset (`bool`, *optional*, defaults to `False`):
            Whether to reset the configuration to default values to None before applying the loaded settings.
            This is useful when you want to ensure that only the settings specified in the file or dictionary
            are applied, without retaining any previous configurations (e.g., when using ParaModifier to modify
            existing configurations).
    Returns:
        `dict`: A dictionary containing the loaded cache options.
    """
    # Deprecated function warning
    logger.warning(
        "`load_options` is deprecated and will be removed in future versions. "
        "Please use `load_configs` instead."
    )
    if isinstance(path_or_dict, str):
        return load_cache_options_from_yaml(path_or_dict, reset)
    elif isinstance(path_or_dict, dict):
        return load_cache_options_from_dict(path_or_dict, reset)
    else:
        raise ValueError("Input must be a file path (str) or a configuration dictionary (dict).")


def load_cache_config(
    path_or_dict: str | dict, **kwargs
) -> Tuple[DBCacheConfig, Optional[CalibratorConfig]]:
    r"""
    New APU that only load cache configuration from a YAML file or a dictionary. Assumes
    that the yaml contains a 'cache_config' section, and returns only that section.
    Raise ValueError if not found.
    Args:
        path_or_dict (`str` or `dict`):
            The file path to the YAML configuration file or a dictionary containing the configuration.
        reset (`bool`, *optional*, defaults to `False`):
            Whether to reset the configuration to default values to None before applying the loaded settings.
            This is useful when you want to ensure that only the settings specified in the file or dictionary
            are applied, without retaining any previous configurations (e.g., when using ParaModifier to modify
            existing configurations).
    Returns:
        `dict`: A dictionary containing the loaded cache configuration.
    """
    if isinstance(path_or_dict, str):
        try:
            with open(path_or_dict, "r") as f:
                kwargs: dict = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {path_or_dict}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"YAML file parsing error: {str(e)}")
    elif isinstance(path_or_dict, dict):
        kwargs: dict = copy.deepcopy(path_or_dict)
    else:
        raise ValueError("Input must be a file path (str) or a configuration dictionary (dict).")

    if "cache_config" not in kwargs:
        raise ValueError("No 'cache_config' section found in the provided configuration.")

    cache_config_kwargs = kwargs["cache_config"]
    # Parse steps_mask if exists
    if "steps_mask" in cache_config_kwargs:
        step_mask = cache_config_kwargs["steps_mask"]
        if isinstance(step_mask, str):
            assert (
                "num_inference_steps" in cache_config_kwargs
            ), "To parse steps_mask from str, 'num_inference_steps' must be provided in cache_config."
            from .cache_interface import steps_mask

            num_inference_steps = cache_config_kwargs["num_inference_steps"]
            cache_config_kwargs["steps_mask"] = steps_mask(
                total_steps=num_inference_steps, mask_policy=step_mask
            )
    # Reuse load_cache_options_from_dict to parse cache_config
    cache_context_kwargs = load_cache_options_from_dict(
        cache_config_kwargs, kwargs.get("reset", False)
    )
    cache_config: DBCacheConfig = cache_context_kwargs.get("cache_config", None)
    calibrator_config = cache_context_kwargs.get("calibrator_config", None)
    if cache_config is None:
        raise ValueError("Failed to load 'cache_config'.")
    return cache_config, calibrator_config


def load_parallelism_config(path_or_dict: str | dict, **kwargs) -> Optional[ParallelismConfig]:
    r"""
    Load parallelism configuration from a YAML file or a dictionary. Assumes that the yaml
    contains a 'parallelism_config' section, and returns only that section. Raise ValueError
    if not found.
    Args:
        path_or_dict (`str` or `dict`):
            The file path to the YAML configuration file or a dictionary containing the configuration.
    Returns:
        `ParallelismConfig`: An instance of ParallelismConfig containing the loaded parallelism configuration.
    """
    if isinstance(path_or_dict, str):
        try:
            with open(path_or_dict, "r") as f:
                kwargs: dict = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {path_or_dict}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"YAML file parsing error: {str(e)}")
    elif isinstance(path_or_dict, dict):
        kwargs: dict = copy.deepcopy(path_or_dict)
    else:
        raise ValueError("Input must be a file path (str) or a configuration dictionary (dict).")

    # Allow missing parallelism_config
    if "parallelism_config" not in kwargs:
        return None

    parallelism_config_kwargs = kwargs["parallelism_config"]
    if "backend" in parallelism_config_kwargs:
        backend_str = parallelism_config_kwargs["backend"]
        parallelism_config_kwargs["backend"] = ParallelismBackend.from_str(backend_str)

    def _maybe_auto_parallel_size(size: str | int | None) -> Optional[int]:
        if size is None:
            return None
        if isinstance(size, int):
            return size
        if isinstance(size, str) and size.lower() == "auto":
            import torch.distributed as dist

            size = 1
            if dist.is_initialized():
                # Assume world size is the parallel size
                size = dist.get_world_size()
            logger.info(f"Auto selected parallel size to {size}.")
            return size
        raise ValueError(f"Invalid parallel size value: {size}. Must be int or 'auto'.")

    if "ulysses_size" in parallelism_config_kwargs:
        parallelism_config_kwargs["ulysses_size"] = _maybe_auto_parallel_size(
            parallelism_config_kwargs["ulysses_size"]
        )
    if "ring_size" in parallelism_config_kwargs:
        parallelism_config_kwargs["ring_size"] = _maybe_auto_parallel_size(
            parallelism_config_kwargs["ring_size"]
        )
    if "tp_size" in parallelism_config_kwargs:
        parallelism_config_kwargs["tp_size"] = _maybe_auto_parallel_size(
            parallelism_config_kwargs["tp_size"]
        )

    parallelism_config = ParallelismConfig(**parallelism_config_kwargs)
    return parallelism_config


def load_configs(
    path_or_dict: str | dict,
    return_dict: bool = True,
    **kwargs,
) -> Union[Tuple[DBCacheConfig, Optional[CalibratorConfig], ParallelismConfig], dict]:
    r"""
    Load both cache and parallelism configurations from a YAML file or a dictionary. For example,
    the YAML file can be structured as follows:
    ```yaml
    cache_config:
      max_warmup_steps: 8
      warmup_interval: 2
      max_cached_steps: -1
      max_continuous_cached_steps: 2
      Fn_compute_blocks: 1
      Bn_compute_blocks: 0
      residual_diff_threshold: 0.12
      enable_taylorseer: true
      taylorseer_order: 1
    parallelism_config:
      ulysses_size: 4
      parallel_kwargs:
        attention_backend: native
        experimental_ulysses_anything: true
        experimental_ulysses_float8: true
        extra_parallel_modules: ["text_encoder", "vae"]
    ```
    Args:
        path_or_dict (`str` or `dict`):
            The file path to the YAML configuration file or a dictionary containing the configuration.
    Returns:
        `Tuple[DBCacheConfig, Optional[CalibratorConfig], ParallelismConfig]`: A tuple containing the loaded
        cache configuration, optional calibrator configuration, and parallelism configuration. If `return_dict`
        is set to `True`, returns a dictionary with keys "cache_config", "calibrator_config", and "parallelism_config".
    """
    cache_config, calibrator_config = load_cache_config(path_or_dict, **kwargs)
    parallelism_config = load_parallelism_config(path_or_dict, **kwargs)
    if return_dict:
        return {
            "cache_config": cache_config,
            "calibrator_config": calibrator_config,
            "parallelism_config": parallelism_config,
        }
    return cache_config, calibrator_config, parallelism_config
