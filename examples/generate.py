from cache_dit.logger import init_logger
from cache_dit.generate import entrypoint

logger = init_logger(__name__)

if __name__ == "__main__":
    entrypoint()
