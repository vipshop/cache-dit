import importlib
from cache_dit.logger import init_logger
from cache_dit.caching.patch_functors.functor_base import PatchFunctor

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


def __safe_import__(module_name: str, class_name: str) -> type[PatchFunctor]:
    try:
        # e.g., module_name = ".functor_dit", class_name = "DiTPatchFunctor"
        package = __package__ if __package__ is not None else ""
        module = importlib.import_module(module_name, package=package)
        target_class = getattr(module, class_name)
        return target_class
    except (ImportError, AttributeError) as e:
        logger.warning(f"Warning: Failed to import {class_name} from {module_name}: {e}")
        return ImportErrorPatchFunctor


DiTPatchFunctor = __safe_import__(".functor_dit", "DiTPatchFunctor")
FluxPatchFunctor = __safe_import__(".functor_flux", "FluxPatchFunctor")
ChromaPatchFunctor = __safe_import__(".functor_chroma", "ChromaPatchFunctor")
HiDreamPatchFunctor = __safe_import__(".functor_hidream", "HiDreamPatchFunctor")
HunyuanDiTPatchFunctor = __safe_import__(".functor_hunyuan_dit", "HunyuanDiTPatchFunctor")
QwenImageControlNetPatchFunctor = __safe_import__(
    ".functor_qwen_image_controlnet", "QwenImageControlNetPatchFunctor"
)
WanVACEPatchFunctor = __safe_import__(".functor_wan_vace", "WanVACEPatchFunctor")
LTX2PatchFunctor = __safe_import__(".functor_ltx2", "LTX2PatchFunctor")
ZImageControlNetPatchFunctor = __safe_import__(
    ".functor_zimage_controlnet", "ZImageControlNetPatchFunctor"
)
