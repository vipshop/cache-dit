import torch
import importlib
from typing import TYPE_CHECKING
from .platform import BasePlatform, CudaPlatform, CpuPlatform, NPUPlatform  # noqa: F401


def resolve_obj_by_qualname(qualname: str) -> BasePlatform:
    """
    Resolve an object by its fully-qualified class name.
    """
    module_name, obj_name = qualname.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)


def resolve_current_platform_cls_qualname() -> str:
    if torch.cuda.is_available():
        return "cache_dit.platforms.platform.CudaPlatform"
    try:
        import torch_npu  # type: ignore  # noqa

        return "cache_dit.platforms.platform.NPUPlatform"
    except ImportError:
        return "cache_dit.platforms.platform.CpuPlatform"


_current_platform: BasePlatform = None


if TYPE_CHECKING:
    current_platform: BasePlatform


def __getattr__(name: str):
    if name == "current_platform":
        global _current_platform
        if _current_platform is None:
            platform_cls_qualname = resolve_current_platform_cls_qualname()
            _current_platform = resolve_obj_by_qualname(platform_cls_qualname)()
        return _current_platform
    elif name in globals():
        return globals()[name]
    else:
        raise AttributeError(f"No attribute named '{name}' exists in {__name__}.")


def __setattr__(name: str, value):
    if name == "current_platform":
        global _current_platform
        _current_platform = value
    elif name in globals():
        globals()[name] = value
    else:
        raise AttributeError(f"No attribute named '{name}' exists in {__name__}.")


__all__ = ["BasePlatform", "current_platform"]
