import importlib
from cache_dit.logger import init_logger
from .dp_plan_registers import AutoEncoderDataParallelismPlanner

logger = init_logger(__name__)


class ImportErrorAutoEncoderDataParallelismPlanner(AutoEncoderDataParallelismPlanner):
    def plan(
        self,
        auto_encoder,
        **kwargs,
    ):
        raise ImportError(
            "This AutoEncoderDataParallelismPlanner requires latest diffusers to be installed. "
            "Please install diffusers from source."
        )


def _safe_import(module_name: str, class_name: str) -> type[AutoEncoderDataParallelismPlanner]:
    try:
        # e.g., module_name = ".dp_plan_autoencoder_kl", class_name = "AutoencoderKLDataParallelismPlanner"
        package = __package__ if __package__ is not None else ""
        module = importlib.import_module(module_name, package=package)
        target_class = getattr(module, class_name)
        return target_class
    except (ImportError, AttributeError) as e:
        logger.debug(f"Failed to import {class_name} from {module_name}: {e}")
        return ImportErrorAutoEncoderDataParallelismPlanner


def _activate_auto_encoder_dp_planners():
    """Function to register all built-in auto encoder data parallelism planners."""
    AutoencoderKLDataParallelismPlanner = _safe_import(  # noqa: F841
        ".dp_plan_autoencoder_kl", "AutoencoderKLDataParallelismPlanner"
    )
    AutoencoderKLLTX2VideoDataParallelismPlanner = _safe_import(  # noqa: F841
        ".dp_plan_autoencoder_kl_ltx2", "AutoencoderKLLTX2VideoDataParallelismPlanner"
    )
    AutoencoderKLQwenImageDataParallelismPlanner = _safe_import(  # noqa: F841
        ".dp_plan_autoencoder_kl_qwen_image", "AutoencoderKLQwenImageDataParallelismPlanner"
    )
    AutoencoderKLWanDataParallelismPlanner = _safe_import(  # noqa: F841
        ".dp_plan_autoencoder_kl_wan", "AutoencoderKLWanDataParallelismPlanner"
    )
    AutoencoderKLHunyuanVideoDataParallelismPlanner = _safe_import(  # noqa: F841
        ".dp_plan_autoencoder_kl_hunyuanvideo", "AutoencoderKLHunyuanVideoDataParallelismPlanner"
    )
    AutoencoderKLFlux2DataParallelismPlanner = _safe_import(  # noqa: F841
        ".dp_plan_autoencoder_kl_flux2", "AutoencoderKLFlux2DataParallelismPlanner"
    )


__all__ = ["_activate_auto_encoder_dp_planners"]
