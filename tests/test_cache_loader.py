from cache_dit import cache_factory

cache_options = cache_factory.load_cache_options_from_yaml(
    "cache_config.yaml",
)

print(f"cache_options from cache_config.yaml:\n {cache_options}")
