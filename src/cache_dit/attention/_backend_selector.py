import torch
from cache_dit.logger import init_logger

logger = init_logger(__name__)


class AttnBackendSelector:
  _attn_backend: str | None = None
  _selected: bool = False

  @classmethod
  def auto_select(cls, pipe_or_adapter) -> str | None:
    try:
      if cls._selected:
        return cls._attn_backend
      device = cls._detect_device(pipe_or_adapter)
      if device.type == "npu":
        cls._attn_backend = "_native_npu"
      cls._selected = True
      return cls._attn_backend
    except Exception as e:
      logger.warning(f"Failed to auto-select attention backend: {e}")
      return None

  @classmethod
  def auto_select_kernel_backend(cls) -> str | None:
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
