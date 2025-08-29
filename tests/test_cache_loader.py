import cache_dit

cache_options = cache_dit.load_options(
    "cache_config.yaml",
)

print(f"cache_options from cache_config.yaml:\n {cache_options}")
