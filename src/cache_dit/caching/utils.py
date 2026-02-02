import yaml
import copy
from typing import Tuple, Optional, Union
from .cache_contexts import (
    DBCacheConfig,
    TaylorSeerCalibratorConfig,
    DBPruneConfig,
    CalibratorConfig,
)
from ..parallelism import ParallelismConfig, ParallelismBackend
from cache_dit.logger import init_logger

logger = init_logger(__name__)


def load_cache_options_from_dict(cache_kwargs: dict, reset: bool = False) -> dict:
    r"""
    Load cache options from a dictionary. We keep this function for backward compatibility.
    Args:
        cache_kwargs (`dict`):
            A dictionary containing the cache configuration.
        reset (`bool`, *optional*, defaults to `False`):
            Whether to reset the configuration to default values to None before applying the loaded settings.
            This is useful when you want to ensure that only the settings specified in the dictionary
            are applied, without retaining any previous configurations (e.g., when using ParaModifier to modify
            existing configurations).
    Returns:
        `dict`: A dictionary containing the loaded cache options.
    """
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
                cache_kwargs: dict = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {path_or_dict}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"YAML file parsing error: {str(e)}")
    elif isinstance(path_or_dict, dict):
        cache_kwargs: dict = copy.deepcopy(path_or_dict)
    else:
        raise ValueError("Input must be a file path (str) or a configuration dictionary (dict).")

    if "cache_config" not in cache_kwargs:
        if "parallelism_config" in cache_kwargs:
            # Allow missing cache_config for only parallelism_config checking
            return None, None
        # Try to load full cache options for backward compatibility if cache_config not found
        # and the parallelism_config is also not provided. This is to support old config files
        # and refresh_context api that only contains cache options (already used in vllm-omni).
        cache_context_kwargs = load_cache_options_from_dict(
            cache_kwargs, kwargs.get("reset", False)
        )
        cache_config: DBCacheConfig = cache_context_kwargs.get("cache_config", None)
        calibrator_config = cache_context_kwargs.get("calibrator_config", None)
        if cache_config is None:
            raise ValueError("Failed to load 'cache_config'. Got None.")
        return cache_config, calibrator_config

    cache_config_kwargs = cache_kwargs["cache_config"]
    # Parse steps_mask if exists
    if "steps_computation_mask" in cache_config_kwargs:
        steps_computation_mask = cache_config_kwargs["steps_computation_mask"]
        if isinstance(steps_computation_mask, str):
            assert (
                "num_inference_steps" in cache_config_kwargs
            ), "To parse steps_mask from str, 'num_inference_steps' must be provided in cache_config."
            from .cache_interface import steps_mask

            num_inference_steps = cache_config_kwargs["num_inference_steps"]
            cache_config_kwargs["steps_computation_mask"] = steps_mask(
                total_steps=num_inference_steps, mask_policy=steps_computation_mask
            )
    # Reuse load_cache_options_from_dict to parse cache_config
    cache_context_kwargs = load_cache_options_from_dict(
        cache_config_kwargs, kwargs.get("reset", False)
    )
    cache_config: DBCacheConfig = cache_context_kwargs.get("cache_config", None)
    calibrator_config = cache_context_kwargs.get("calibrator_config", None)
    if cache_config is None:
        raise ValueError("Failed to load 'cache_config'. Got None.")
    return cache_config, calibrator_config


def load_parallelism_config(
    path_or_dict: str | dict, **kwargs
) -> Optional[ParallelismConfig] | bool:
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
                parallel_kwargs: dict = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {path_or_dict}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"YAML file parsing error: {str(e)}")
    elif isinstance(path_or_dict, dict):
        parallel_kwargs: dict = copy.deepcopy(path_or_dict)
    else:
        raise ValueError("Input must be a file path (str) or a configuration dictionary (dict).")

    if kwargs.get("check_only", False):
        return "parallelism_config" in parallel_kwargs

    # Allow missing parallelism_config
    if "parallelism_config" not in parallel_kwargs:
        return None

    parallelism_config_kwargs = parallel_kwargs["parallelism_config"]
    if "backend" in parallelism_config_kwargs:
        backend_str = parallelism_config_kwargs["backend"]
        parallelism_config_kwargs["backend"] = ParallelismBackend.from_str(backend_str)

    def _maybe_auto_parallel_size(
        size: str | int | None, partial_max_size: Optional[int] = None
    ) -> Optional[int]:
        if size is None:
            return None
        if isinstance(size, int):
            return size
        if isinstance(size, str) and size.lower() == "auto":
            import torch.distributed as dist

            size = 1
            if dist.is_initialized():
                # Assume world size is the parallel size
                world_size = dist.get_world_size()
                if partial_max_size is not None:
                    size = world_size // partial_max_size
                else:
                    size = world_size
            if size == 1:
                logger.warning(
                    "Auto parallel size selected as 1. Make sure to run with torch.distributed "
                    "to utilize multiple devices for parallelism."
                )
            else:
                logger.info(f"Auto selected parallel size to {size}.")
            return size
        raise ValueError(f"Invalid parallel size value: {size}. Must be int or 'auto'.")

    def _maybe_auto_parallel_sizes(parallelism_config_kwargs: dict) -> dict:
        # Only allow one of the parallel size to be auto for simplicity
        auto_count = sum(
            1
            for key in ["ulysses_size", "ring_size", "tp_size"]
            if key in parallelism_config_kwargs and parallelism_config_kwargs[key] == "auto"
        )
        if auto_count > 1:
            raise ValueError(
                "Only one of 'ulysses_size', 'ring_size', or 'tp_size' can be set to 'auto'."
            )

        ulysses_size = parallelism_config_kwargs.get("ulysses_size", 1)
        ring_size = parallelism_config_kwargs.get("ring_size", 1)
        tp_size = parallelism_config_kwargs.get("tp_size", 1)
        partial_max_size = None
        if isinstance(ulysses_size, str) and ulysses_size.lower() == "auto":
            partial_max_size = ring_size * tp_size
        elif isinstance(ring_size, str) and ring_size.lower() == "auto":
            partial_max_size = ulysses_size * tp_size
        elif isinstance(tp_size, str) and tp_size.lower() == "auto":
            partial_max_size = ulysses_size * ring_size

        for key in ["ulysses_size", "ring_size", "tp_size"]:
            if key in parallelism_config_kwargs:
                parallelism_config_kwargs[key] = _maybe_auto_parallel_size(
                    parallelism_config_kwargs[key], partial_max_size=partial_max_size
                )
        return parallelism_config_kwargs

    if kwargs.get("auto_parallel_size", True):

        parallelism_config_kwargs = _maybe_auto_parallel_sizes(parallelism_config_kwargs)

    parallelism_config = ParallelismConfig(**parallelism_config_kwargs)
    return parallelism_config


def load_attn_backend_config(path_or_dict: str | dict, **kwargs) -> Optional[str]:
    r"""
    Load attention backend configuration from a YAML file or a dictionary. Assumes that the yaml
    contains an 'attention_backend' field, and returns only that field. Raise ValueError
    if not found.
    Args:
        path_or_dict (`str` or `dict`):
            The file path to the YAML configuration file or a dictionary containing the configuration.
    Returns:
        `str`: A string containing the loaded attention backend configuration.
    """
    if isinstance(path_or_dict, str):
        try:
            with open(path_or_dict, "r") as f:
                attn_kwargs: dict = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {path_or_dict}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"YAML file parsing error: {str(e)}")
    elif isinstance(path_or_dict, dict):
        attn_kwargs: dict = copy.deepcopy(path_or_dict)
    else:
        raise ValueError("Input must be a file path (str) or a configuration dictionary (dict).")
    if "attention_backend" not in attn_kwargs:
        return None
    attention_backend = attn_kwargs["attention_backend"]
    return attention_backend


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
    attention_backend = load_attn_backend_config(path_or_dict, **kwargs)
    if isinstance(parallelism_config, bool):
        parallelism_config = None
    if return_dict:
        return {
            "cache_config": cache_config,
            "calibrator_config": calibrator_config,
            "parallelism_config": parallelism_config,
            "attention_backend": attention_backend,
        }
    return cache_config, calibrator_config, parallelism_config
