import importlib
from cache_dit.logger import init_logger
from .functor_base import PatchFunctor

logger = init_logger(__name__)


class ImportErrorPatchFunctor(PatchFunctor):
    def _apply(
        self,
        transformer,
        **kwargs,
    ):
        raise ImportError(
            "This PatchFunctor requires latest diffusers to be installed. "
            "Please install diffusers from source."
        )


def _safe_import(module_name: str, class_name: str) -> type[PatchFunctor]:
    try:
        # e.g., module_name = ".functor_dit", class_name = "DiTPatchFunctor"
        package = __package__ if __package__ is not None else ""
        module = importlib.import_module(module_name, package=package)
        target_class = getattr(module, class_name)
        return target_class
    except (ImportError, AttributeError) as e:
        logger.debug(f"Warning: Failed to import {class_name} from {module_name}: {e}")
        return ImportErrorPatchFunctor


DiTPatchFunctor = _safe_import(".functor_dit", "DiTPatchFunctor")
FluxPatchFunctor = _safe_import(".functor_flux", "FluxPatchFunctor")
ChromaPatchFunctor = _safe_import(".functor_chroma", "ChromaPatchFunctor")
HiDreamPatchFunctor = _safe_import(".functor_hidream", "HiDreamPatchFunctor")
HunyuanDiTPatchFunctor = _safe_import(".functor_hunyuan_dit", "HunyuanDiTPatchFunctor")
QwenImageControlNetPatchFunctor = _safe_import(
    ".functor_qwen_image_controlnet", "QwenImageControlNetPatchFunctor"
)
WanVACEPatchFunctor = _safe_import(".functor_wan_vace", "WanVACEPatchFunctor")
LTX2PatchFunctor = _safe_import(".functor_ltx2", "LTX2PatchFunctor")
ZImageControlNetPatchFunctor = _safe_import(
    ".functor_zimage_controlnet", "ZImageControlNetPatchFunctor"
)
