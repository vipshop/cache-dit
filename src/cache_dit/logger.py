import logging
import warnings
import contextlib
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
_warning_once_messages: set[tuple[str, str]] = set()


def _warning_once(self: logging.Logger, msg, *args, **kwargs) -> None:
  message = logging.LogRecord(
    name=self.name,
    level=logging.WARNING,
    pathname="",
    lineno=0,
    msg=msg,
    args=args,
    exc_info=None,
  ).getMessage()
  key = (self.name, message)
  if key in _warning_once_messages:
    return
  _warning_once_messages.add(key)
  self.warning(msg, *args, **kwargs)


logging.Logger.warning_once = _warning_once  # type: ignore[attr-defined]


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
  """Initialize a logger with the given name.

  :param name: Logger name, usually `__name__` from the caller module.
  :returns: The configured logger instance.
  """
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


# Adapted from: https://github.com/vllm-project/vllm/blob/a11f4a81e027efd9ef783b943489c222950ac989/vllm/utils/system_utils.py#L60
@contextlib.contextmanager
def suppress_stdout():
  """Suppress stdout from C libraries at the file descriptor level.

  Only suppresses stdout, not stderr, to preserve error messages.

  Example::

    with suppress_stdout():
      # C library calls that would normally print to stdout
      torch.distributed.new_group(ranks, backend="gloo")
  """
  # Don't suppress if logging level is DEBUG
  if _LOG_LEVEL == logging.DEBUG:
    yield
    return

  stdout_fd = sys.stdout.fileno()
  stdout_dup = os.dup(stdout_fd)
  devnull_fd = os.open(os.devnull, os.O_WRONLY)

  try:
    sys.stdout.flush()
    os.dup2(devnull_fd, stdout_fd)
    yield
  finally:
    sys.stdout.flush()
    os.dup2(stdout_dup, stdout_fd)
    os.close(stdout_dup)
    os.close(devnull_fd)


# Adapted from: https://github.com/sgl-project/sglang/blob/17119a697de72910d77d3bfffc34d097cf6cad09/python/sglang/multimodal_gen/runtime/utils/logging_utils.py#L385
def suppress_loggers(loggers_to_suppress: list[str], level: int = logging.ERROR) -> dict[str, int]:
  original_levels = {}

  # Support prefix wildcard patterns like "diffusers*".
  prefixes = [name[:-1] for name in loggers_to_suppress if name.endswith("*")]
  exact_names = [name for name in loggers_to_suppress if not name.endswith("*")]

  for logger_name in exact_names:
    try:
      logger = logging.getLogger(logger_name)
      original_levels[logger_name] = logger.level
      logger.setLevel(level)
    except Exception:
      pass

  if prefixes:
    for prefix in prefixes:
      # Also set base prefix logger (e.g. "diffusers") so children inherit.
      base_name = prefix.rstrip(".")
      try:
        base_logger = logging.getLogger(base_name)
        original_levels[base_name] = base_logger.level
        base_logger.setLevel(level)
      except Exception:
        pass

    for logger_name, logger_obj in logging.root.manager.loggerDict.items():
      if not isinstance(logger_obj, logging.Logger):
        continue
      if any(logger_name.startswith(prefix) for prefix in prefixes):
        try:
          original_levels[logger_name] = logger_obj.level
          logger_obj.setLevel(level)
        except Exception:
          pass

  return original_levels


def suppress_torch_compile_loggers() -> dict[str, int]:
  """Set specified torch loggers to ERROR level to suppress warnings.

  :returns: A mapping from logger names to their original levels before suppression.
  """
  try:
    import torch._inductor.config as inductor_config
    import torch._inductor.select_algorithm as select_algorithm

    select_algorithm.PRINT_AUTOTUNE = False
    inductor_config.max_autotune_report_choices_stats = False

    # select_algorithm emits many max-autotune messages through log.error/log.warning,
    # so setting the logger level to ERROR is insufficient. Disable this logger directly.
    select_algorithm_logger = logging.getLogger("torch._inductor.select_algorithm")
    select_algorithm_logger.disabled = True
  except Exception:
    pass

  # Suppress specific warnings from torch._dynamo and torch._inductor when using compile
  modules_to_suppress_warnings = [
    r"torch\._dynamo.*",
    r"torch\._inductor.*",
    r"torch\.compiler.*",
    r"torch\._functorch.*",
    r"torch\._export.*",
    r"torch\._fx.*",
    r"torch\._meta_tracing.*",
    r"diffusers\.*",
  ]
  for module in modules_to_suppress_warnings:
    warnings.filterwarnings("ignore", category=UserWarning, module=module)
    warnings.filterwarnings("ignore", category=FutureWarning, module=module)
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=module)
  # diffusers emits this deprecation warning from its own modules, so module-based
  # filters for torch.* do not match. Filter by warning message to suppress it.
  messages_to_suppress_warnings = [
    r".*torch\._dynamo\.allow_in_graph*",
    r"AffineQuantizedTensor does not implement*",
  ]
  for message in messages_to_suppress_warnings:
    warnings.filterwarnings("ignore", category=UserWarning, message=message)
    warnings.filterwarnings("ignore", category=FutureWarning, message=message)
    warnings.filterwarnings("ignore", category=DeprecationWarning, message=message)
  compile_loggers_names = [
    "torch._dynamo",
    "torch._inductor",
    "torch._functorch",
    "torch._export",
    "torch._fx",
    "torch._meta_tracing",
    "torch.compiler",
  ]
  original_levels = suppress_loggers(compile_loggers_names, level=logging.ERROR)
  return original_levels


# Adapted from: https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/utils/logging_utils.py#L396
def globally_suppress_loggers() -> dict[str, int]:
  """Set specified loggers to ERROR level to suppress logs globally.

  :returns: A mapping from logger names to their original levels before suppression.
  """
  warnings.filterwarnings("ignore", category=UserWarning, module=r"torchao*")
  warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"torchao*")
  # diffusers manages its own root logger and can reset level to WARNING.
  # Force its library-wide verbosity to ERROR when available.
  try:
    from diffusers.utils import logging as diffusers_logging  # type: ignore

    diffusers_logging.set_verbosity_error()
  except Exception:
    pass
  loggers_to_suppress = [
    "torchao",
    "torch.distributed.run",
    # * wildcard to suppress all diffusers loggers, which can be very verbose,
    # especially at INFO level, and are often not relevant to users of Cache-DiT.
    # This includes loggers like "diffusers", "diffusers.quantizers", etc.
    "diffusers*",
    "imageio",
    "imageio_ffmpeg",
    "PIL",
    "PIL_Image",
    "python_multipart.multipart",
    "filelock",
    "urllib3",
  ]
  original_levels = suppress_loggers(loggers_to_suppress, level=logging.ERROR)
  return original_levels
