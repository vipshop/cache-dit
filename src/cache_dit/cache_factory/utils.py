import yaml
from cache_dit.cache_factory import CacheType


def load_cache_options_from_yaml(yaml_file_path):
    try:
        with open(yaml_file_path, "r") as f:
            config = yaml.safe_load(f)

        required_keys = [
            "cache_type",
            "max_warmup_steps",
            "max_cached_steps",
            "Fn_compute_blocks",
            "Bn_compute_blocks",
            "residual_diff_threshold",
        ]
        for key in required_keys:
            if key not in config:
                raise ValueError(
                    f"Configuration file missing required item: {key}"
                )

        # Convert cache_type to CacheType enum
        if isinstance(config["cache_type"], str):
            try:
                config["cache_type"] = CacheType[config["cache_type"]]
            except KeyError:
                valid_types = [ct.name for ct in CacheType]
                raise ValueError(
                    f"Invalid cache_type value: {config['cache_type']}, "
                    f"valid values are: {valid_types}"
                )
        elif not isinstance(config["cache_type"], CacheType):
            raise ValueError(
                f"cache_type must be a string or CacheType enum, "
                f"got: {type(config['cache_type'])}"
            )

        # Handle default value for taylorseer_kwargs
        if "taylorseer_kwargs" not in config and config.get(
            "enable_taylorseer", False
        ):
            config["taylorseer_kwargs"] = {"n_derivatives": 2}

        return config

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Configuration file not found: {yaml_file_path}"
        )
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"YAML file parsing error: {str(e)}")


def load_options(path: str):
    return load_cache_options_from_yaml(path)
