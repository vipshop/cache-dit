from cache_dit.logger import init_logger

logger = init_logger(__name__)


_metrics_progress_verbose = False


def set_metrics_verbose(verbose: bool):
    global _metrics_progress_verbose
    _metrics_progress_verbose = verbose
    logger.debug(f"Metrics verbose: {verbose}")


def get_metrics_verbose() -> bool:
    global _metrics_progress_verbose
    return _metrics_progress_verbose


_IMAGE_EXTENSIONS = [
    "bmp",
    "jpg",
    "jpeg",
    "pgm",
    "png",
    "ppm",
    "tif",
    "tiff",
    "webp",
]

_VIDEO_EXTENSIONS = [
    "mp4",
]
