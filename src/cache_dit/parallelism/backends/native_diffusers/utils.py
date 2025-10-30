try:
    from diffusers import ContextParallelConfig

    def native_diffusers_parallelism_available() -> bool:
        return True

except ImportError:
    ContextParallelConfig = None

    def native_diffusers_parallelism_available() -> bool:
        return False
