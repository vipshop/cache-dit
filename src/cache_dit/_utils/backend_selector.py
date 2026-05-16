import torch
from cache_dit.envs import ENV
from cache_dit.logger import init_logger

logger = init_logger(__name__)


class BackendSelector:
  _attn_backend: str | None = None
  _selected: bool = False

  @classmethod
  def auto_select(cls, pipe_or_adapter) -> str | None:
    if cls._selected:
      return cls._attn_backend

    enable = ENV.CACHE_DIT_ENABLE_MINDIESD_ATTN
    if enable == "0":
      cls._attn_backend = "_native_npu"
      cls._selected = True
      return cls._attn_backend

    device = cls._detect_device(pipe_or_adapter)
    if device.type == "npu":
      try:
        import mindiesd  # noqa F401

        cls._attn_backend = "_mindiesd_laser"
        logger.info(f"Auto-selected MindIE-SD attention backend: {cls._attn_backend}")
      except Exception:
        cls._attn_backend = "_native_npu"
        logger.info(f"MindIE-SD not found, fallback attention backend: {cls._attn_backend}")
    cls._selected = True
    return cls._attn_backend

  @classmethod
  def auto_select_kernel_backend(cls) -> str | None:
    try:
      import mindiesd  # noqa F401

      from cache_dit.kernels.backend import KernelBackend

      return KernelBackend.MINDIESD
    except Exception:
      return None

  @staticmethod
  def _detect_device(pipe_or_adapter):
    try:
      if hasattr(pipe_or_adapter, "device"):
        return pipe_or_adapter.device
      param = next(pipe_or_adapter.parameters())
      return param.device
    except (StopIteration, AttributeError):
      return torch.device("cpu")
