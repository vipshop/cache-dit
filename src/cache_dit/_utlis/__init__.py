import importlib
from typing import Callable
from ..logger import init_logger
from .registers import Example
from .generate import entrypoint

logger = init_logger(__name__)


def import_error_example(
    *args,
    **kwargs,
) -> Example:
    raise ImportError(
        "This Example requires latest diffusers to be installed. "
        "Please install diffusers from source."
    )


def _safe_import(module_name: str, func_name: str) -> Callable[..., Example]:
    try:
        # e.g., module_name = ".examples", func_name = "flux_example"
        package = __package__ if __package__ is not None else ""
        module = importlib.import_module(module_name, package=package)
        target_func = getattr(module, func_name)
        return target_func
    except (ImportError, AttributeError) as e:
        logger.debug(f"Failed to import {func_name} from {module_name}: {e}")
        return import_error_example


flux_example = _safe_import(".examples", "flux_example")
flux_fill_example = _safe_import(".examples", "flux_fill_example")
flux2_klein_example = _safe_import(".examples", "flux2_klein_example")
flux2_example = _safe_import(".examples", "flux2_example")
qwen_image_example = _safe_import(".examples", "qwen_image_example")
qwen_image_controlnet_example = _safe_import(".examples", "qwen_image_controlnet_example")
qwen_image_edit_example = _safe_import(".examples", "qwen_image_edit_example")
qwen_image_layered_example = _safe_import(".examples", "qwen_image_layered_example")
skyreels_v2_example = _safe_import(".examples", "skyreels_v2_example")
ltx2_t2v_example = _safe_import(".examples", "ltx2_t2v_example")
ltx2_i2v_example = _safe_import(".examples", "ltx2_i2v_example")
wan_example = _safe_import(".examples", "wan_example")
wan_i2v_example = _safe_import(".examples", "wan_i2v_example")
wan_vace_example = _safe_import(".examples", "wan_vace_example")
ovis_image_example = _safe_import(".examples", "ovis_image_example")
zimage_example = _safe_import(".examples", "zimage_example")
longcat_image_example = _safe_import(".examples", "longcat_image_example")
longcat_image_edit_example = _safe_import(".examples", "longcat_image_edit_example")
zimage_controlnet_example = _safe_import(".examples", "zimage_controlnet_example")
