try:
    import einops
except ImportError:
    raise ImportError(
        "Metrics functionality requires the 'parallelism' extra dependencies. "
        "Install with:\npip install cache-dit[parallelism]"
    )
