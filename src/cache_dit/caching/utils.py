import yaml


def load_cache_options_from_dict(cache_kwargs: dict) -> dict:
    try:
        kwargs: dict = cache_kwargs

        required_keys = [
            "residual_diff_threshold",
        ]
        for key in required_keys:
            if key not in kwargs:
                raise ValueError(
                    f"Configuration file missing required item: {key}"
                )

        cache_context_kwargs = {}
        if kwargs.get("enable_taylorseer", False):
            from cache_dit.caching.cache_contexts.calibrators import (
                TaylorSeerCalibratorConfig,
            )

            cache_context_kwargs["calibrator_config"] = (
                TaylorSeerCalibratorConfig(
                    enable_calibrator=kwargs.pop("enable_taylorseer"),
                    enable_encoder_calibrator=kwargs.pop(
                        "enable_encoder_taylorseer", False
                    ),
                    calibrator_cache_type=kwargs.pop(
                        "taylorseer_cache_type", "residual"
                    ),
                    taylorseer_order=kwargs.pop("taylorseer_order", 1),
                )
            )

        if "cache_type" not in kwargs:
            from cache_dit.caching.cache_contexts import BasicCacheConfig

            cache_context_kwargs["cache_config"] = BasicCacheConfig()
            cache_context_kwargs["cache_config"].update(**kwargs)
        else:
            cache_type = str(kwargs.get("cache_type", None))
            if cache_type == "DBCache":
                from cache_dit.caching.cache_contexts import DBCacheConfig

                cache_context_kwargs["cache_config"] = DBCacheConfig()
                cache_context_kwargs["cache_config"].update(**kwargs)
            elif cache_type == "DBPrune":
                from cache_dit.caching.cache_contexts import DBPruneConfig

                cache_context_kwargs["cache_config"] = DBPruneConfig()
                cache_context_kwargs["cache_config"].update(**kwargs)
            else:
                raise ValueError(f"Unsupported cache_type: {cache_type}.")

        if "parallelism_config" in kwargs:
            from cache_dit.parallelism.parallel_config import (
                ParallelismConfig,
            )

            parallelism_kwargs = kwargs.get("parallelism_config", {})
            cache_context_kwargs["parallelism_config"] = ParallelismConfig(
                **parallelism_kwargs
            )

        return cache_context_kwargs

    except Exception as e:
        raise ValueError(f"Error parsing cache configuration. {str(e)}")


def load_cache_options_from_yaml(yaml_file_path: str) -> dict:
    try:
        with open(yaml_file_path, "r") as f:
            kwargs: dict = yaml.safe_load(f)
        return load_cache_options_from_dict(kwargs)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Configuration file not found: {yaml_file_path}"
        )
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"YAML file parsing error: {str(e)}")


def load_options(path_or_dict: str | dict) -> dict:
    if isinstance(path_or_dict, str):
        return load_cache_options_from_yaml(path_or_dict)
    elif isinstance(path_or_dict, dict):
        return load_cache_options_from_dict(path_or_dict)
    else:
        raise ValueError(
            "Input must be a file path (str) or a configuration dictionary (dict)."
        )
