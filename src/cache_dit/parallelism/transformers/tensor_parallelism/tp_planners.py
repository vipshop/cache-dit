import importlib
from cache_dit.logger import init_logger
from .tp_plan_registers import TensorParallelismPlanner

logger = init_logger(__name__)


class ImportErrorTensorParallelismPlanner(TensorParallelismPlanner):
    def plan(
        self,
        transformer,
        **kwargs,
    ):
        raise ImportError(
            "This TensorParallelismPlanner requires latest diffusers to be installed. "
            "Please install diffusers from source."
        )


def _safe_import(module_name: str, class_name: str) -> type[TensorParallelismPlanner]:
    try:
        # e.g., module_name = ".tp_plan_dit", class_name = "DiTTensorParallelismPlanner"
        package = __package__ if __package__ is not None else ""
        module = importlib.import_module(module_name, package=package)
        target_class = getattr(module, class_name)
        return target_class
    except (ImportError, AttributeError) as e:
        logger.debug(f"Failed to import {class_name} from {module_name}: {e}")
        return ImportErrorTensorParallelismPlanner


def _activate_tp_planners():
    """Function to register all built-in tensor parallelism planners."""
    CogViewTensorParallelismPlanner = _safe_import(  # noqa: F841
        ".tp_plan_cogview", "CogViewTensorParallelismPlanner"
    )
    FluxTensorParallelismPlanner = _safe_import(  # noqa: F841
        ".tp_plan_flux", "FluxTensorParallelismPlanner"
    )
    Flux2TensorParallelismPlanner = _safe_import(  # noqa: F841
        ".tp_plan_flux2", "Flux2TensorParallelismPlanner"
    )
    HunyuanDiTTensorParallelismPlanner = _safe_import(  # noqa: F841
        ".tp_plan_hunyuan_dit", "HunyuanDiTTensorParallelismPlanner"
    )
    Kandinsky5TensorParallelismPlanner = _safe_import(  # noqa: F841
        ".tp_plan_kandinsky5", "Kandinsky5TensorParallelismPlanner"
    )
    MochiTensorParallelismPlanner = _safe_import(  # noqa: F841
        ".tp_plan_mochi", "MochiTensorParallelismPlanner"
    )
    LTXVideoTensorParallelismPlanner = _safe_import(  # noqa: F841
        ".tp_plan_ltx_video", "LTXVideoTensorParallelismPlanner"
    )
    LTX2VideoTensorParallelismPlanner = _safe_import(  # noqa: F841
        ".tp_plan_ltx2_video", "LTX2VideoTensorParallelismPlanner"
    )
    PixArtTensorParallelismPlanner = _safe_import(  # noqa: F841
        ".tp_plan_pixart", "PixArtTensorParallelismPlanner"
    )
    QwenImageTensorParallelismPlanner = _safe_import(  # noqa: F841
        ".tp_plan_qwen_image", "QwenImageTensorParallelismPlanner"
    )
    WanTensorParallelismPlanner = _safe_import(  # noqa: F841
        ".tp_plan_wan", "WanTensorParallelismPlanner"
    )
    SkyReelsV2TensorParallelismPlanner = _safe_import(  # noqa: F841
        ".tp_plan_skyreels", "SkyReelsV2TensorParallelismPlanner"
    )
    ZImageTensorParallelismPlanner = _safe_import(  # noqa: F841
        ".tp_plan_zimage", "ZImageTensorParallelismPlanner"
    )
    OvisImageTensorParallelismPlanner = _safe_import(  # noqa: F841
        ".tp_plan_ovis_image", "OvisImageTensorParallelismPlanner"
    )
    LongCatImageTensorParallelismPlanner = _safe_import(  # noqa: F841
        ".tp_plan_longcat_image", "LongCatImageTensorParallelismPlanner"
    )


__all__ = ["_activate_tp_planners"]
