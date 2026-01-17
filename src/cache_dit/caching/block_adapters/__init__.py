import importlib
from typing import Callable
from .block_adapters import BlockAdapter
from .block_adapters import FakeDiffusionPipeline
from .block_adapters import ParamsModifier
from .block_registers import BlockAdapterRegister
from cache_dit.logger import init_logger

logger = init_logger(__name__)


def import_error_adapter(
    *args,
    **kwargs,
) -> BlockAdapter:
    raise ImportError(
        "This BlockAdapter requires latest diffusers to be installed. "
        "Please install diffusers from source."
    )


def _safe_import(module_name: str, func_name: str) -> Callable[..., BlockAdapter]:
    try:
        # e.g., module_name = ".adapters", func_name = "flux_adapter"
        package = __package__ if __package__ is not None else ""
        module = importlib.import_module(module_name, package=package)
        target_func = getattr(module, func_name)
        return target_func
    except (ImportError, AttributeError) as e:
        logger.debug(f"Failed to import {func_name} from {module_name}: {e}")
        return import_error_adapter


flux_adapter = _safe_import(".adapters", "flux_adapter")
mochi_adapter = _safe_import(".adapters", "mochi_adapter")
cogvideox_adapter = _safe_import(".adapters", "cogvideox_adapter")
wan_adapter = _safe_import(".adapters", "wan_adapter")
hunyuanvideo_adapter = _safe_import(".adapters", "hunyuanvideo_adapter")
qwenimage_adapter = _safe_import(".adapters", "qwenimage_adapter")
ltxvideo_adapter = _safe_import(".adapters", "ltxvideo_adapter")
allegro_adapter = _safe_import(".adapters", "allegro_adapter")
cogview3plus_adapter = _safe_import(".adapters", "cogview3plus_adapter")
cogview4_adapter = _safe_import(".adapters", "cogview4_adapter")
cosmos_adapter = _safe_import(".adapters", "cosmos_adapter")
easyanimate_adapter = _safe_import(".adapters", "easyanimate_adapter")
skyreelsv2_adapter = _safe_import(".adapters", "skyreelsv2_adapter")
sd3_adapter = _safe_import(".adapters", "sd3_adapter")
consisid_adapter = _safe_import(".adapters", "consisid_adapter")
dit_adapter = _safe_import(".adapters", "dit_adapter")
amused_adapter = _safe_import(".adapters", "amused_adapter")
bria_adapter = _safe_import(".adapters", "bria_adapter")
lumina2_adapter = _safe_import(".adapters", "lumina2_adapter")
omnigen_adapter = _safe_import(".adapters", "omnigen_adapter")
pixart_adapter = _safe_import(".adapters", "pixart_adapter")
sana_adapter = _safe_import(".adapters", "sana_adapter")
stabledudio_adapter = _safe_import(".adapters", "stabledudio_adapter")
visualcloze_adapter = _safe_import(".adapters", "visualcloze_adapter")
auraflow_adapter = _safe_import(".adapters", "auraflow_adapter")
chroma_adapter = _safe_import(".adapters", "chroma_adapter")
shape_adapter = _safe_import(".adapters", "shape_adapter")
hidream_adapter = _safe_import(".adapters", "hidream_adapter")
hunyuandit_adapter = _safe_import(".adapters", "hunyuandit_adapter")
hunyuanditpag_adapter = _safe_import(".adapters", "hunyuanditpag_adapter")
kandinsky5_adapter = _safe_import(".adapters", "kandinsky5_adapter")
prx_adapter = _safe_import(".adapters", "prx_adapter")
hunyuan_image_adapter = _safe_import(".adapters", "hunyuan_image_adapter")
chronoedit_adapter = _safe_import(".adapters", "chronoedit_adapter")
zimage_adapter = _safe_import(".adapters", "zimage_adapter")
ovis_image_adapter = _safe_import(".adapters", "ovis_image_adapter")
longcat_image_adapter = _safe_import(".adapters", "longcat_image_adapter")
