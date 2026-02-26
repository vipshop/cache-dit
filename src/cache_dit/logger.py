import logging
import os
import sys
import torch.distributed as dist
from .envs import ENV

_FORMAT = "[%(asctime)s] [Cache-DiT] %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"

_LOG_LEVEL = ENV.CACHE_DIT_LOG_LEVEL
_LOG_LEVEL = getattr(logging, _LOG_LEVEL.upper(), 0)
_LOG_DIR = ENV.CACHE_DIT_LOG_DIR


class NewLineFormatter(logging.Formatter):
    """Adds logging prefix to newlines to align multi-line messages."""

    def __init__(self, fmt, datefmt=None):
        logging.Formatter.__init__(self, fmt, datefmt)

    def format(self, record):
        msg = logging.Formatter.format(self, record)
        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\r\n" + parts[0])
        return msg


class Rank0Filter(logging.Filter):
    """Filter that only allows logs from rank 0 to pass through (real-time check)."""

    def filter(self, record):

        if not ENV.CACHE_DIT_FORCE_ONLY_RANK0_LOGGING:
            return True

        try:
            return not (dist.is_available() and dist.is_initialized() and dist.get_rank() != 0)
        except Exception:
            return True


_root_logger = logging.getLogger("CACHE_DIT")
_default_handler = None
_default_file_handler = None
_inference_log_file_handler = {}


def _setup_logger():
    """Setup the root logger with console and file handlers."""
    _root_logger.setLevel(_LOG_LEVEL)
    _root_logger.propagate = False
    fmt = NewLineFormatter(_FORMAT, datefmt=_DATE_FORMAT)
    rank_filter = Rank0Filter()

    # Setup console handler (always add, filter controls output)
    global _default_handler
    if _default_handler is None:
        _default_handler = logging.StreamHandler(sys.stdout)
        _default_handler.flush = sys.stdout.flush  # type: ignore
        _default_handler.setLevel(_LOG_LEVEL)
        _default_handler.setFormatter(fmt)
        _default_handler.addFilter(rank_filter)
        _root_logger.addHandler(_default_handler)

    # Setup default file handler (always add if dir exists, filter controls output)
    global _default_file_handler
    if _default_file_handler is None and _LOG_DIR is not None:
        if not os.path.exists(_LOG_DIR):
            try:
                os.makedirs(_LOG_DIR)
            except OSError as e:
                _root_logger.warning(f"Error creating directory {_LOG_DIR} : {e}")
        _default_file_handler = logging.FileHandler(_LOG_DIR + "/default.log")
        _default_file_handler.setLevel(_LOG_LEVEL)
        _default_file_handler.setFormatter(fmt)
        _default_file_handler.addFilter(rank_filter)
        _root_logger.addHandler(_default_file_handler)


# Initialize logger when module is imported
_setup_logger()


def init_logger(name: str):
    """Initialize a logger with the given name."""
    logger = logging.getLogger(name)
    logger.setLevel(_LOG_LEVEL)
    logger.propagate = False
    rank_filter = Rank0Filter()

    # Add console handler
    if _default_handler is not None:
        logger.addHandler(_default_handler)

    # Add file handlers if log directory is configured
    if _LOG_DIR is not None:
        pid = os.getpid()
        if _inference_log_file_handler.get(pid, None) is not None:
            logger.addHandler(_inference_log_file_handler[pid])
        else:
            if not os.path.exists(_LOG_DIR):
                try:
                    os.makedirs(_LOG_DIR)
                except OSError as e:
                    _root_logger.warning(f"Error creating directory {_LOG_DIR} : {e}")
            file_handler = logging.FileHandler(_LOG_DIR + f"/process.{pid}.log")
            file_handler.setLevel(_LOG_LEVEL)
            file_handler.setFormatter(NewLineFormatter(_FORMAT, datefmt=_DATE_FORMAT))
            file_handler.addFilter(rank_filter)
            _inference_log_file_handler[pid] = file_handler
            logger.addHandler(file_handler)

    return logger
