import yaml
import copy


def load_cache_options_from_dict(cache_kwargs: dict, reset: bool = False) -> dict:
    try:
        # deep copy to avoid modifying original kwargs
        kwargs: dict = copy.deepcopy(cache_kwargs)
        cache_context_kwargs = {}
        if kwargs.get("enable_taylorseer", False):
            from cache_dit.caching.cache_contexts.calibrators import (
                TaylorSeerCalibratorConfig,
            )

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
            from cache_dit.caching.cache_contexts import BasicCacheConfig

            cache_context_kwargs["cache_config"] = (
                BasicCacheConfig() if not reset else BasicCacheConfig().reset()
            )
            cache_context_kwargs["cache_config"].update(**kwargs)
        else:
            cache_type = str(kwargs.get("cache_type", None))
            if cache_type == "DBCache":
                from cache_dit.caching.cache_contexts import DBCacheConfig

                cache_context_kwargs["cache_config"] = (
                    DBCacheConfig() if not reset else DBCacheConfig().reset()
                )
                cache_context_kwargs["cache_config"].update(**kwargs)
            elif cache_type == "DBPrune":
                from cache_dit.caching.cache_contexts import DBPruneConfig

                cache_context_kwargs["cache_config"] = (
                    DBPruneConfig() if not reset else DBPruneConfig().reset()
                )
                cache_context_kwargs["cache_config"].update(**kwargs)
            else:
                raise ValueError(f"Unsupported cache_type: {cache_type}.")

        if "parallelism_config" in kwargs:
            from cache_dit.parallelism import ParallelismConfig

            parallelism_kwargs = kwargs.get("parallelism_config", {})
            cache_context_kwargs["parallelism_config"] = ParallelismConfig(**parallelism_kwargs)

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
    if isinstance(path_or_dict, str):
        return load_cache_options_from_yaml(path_or_dict, reset)
    elif isinstance(path_or_dict, dict):
        return load_cache_options_from_dict(path_or_dict, reset)
    else:
        raise ValueError("Input must be a file path (str) or a configuration dictionary (dict).")
