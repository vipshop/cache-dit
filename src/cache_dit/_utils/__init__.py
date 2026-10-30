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
  raise ImportError("This Example requires latest diffusers to be installed. "
                    "Please install diffusers from source.")


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
flux2_klein_edit_example = _safe_import(".examples", "flux2_klein_edit_example")
flux2_klein_kv_edit_example = _safe_import(".examples", "flux2_klein_kv_edit_example")
flux3_example = _safe_import(".examples", "flux3_example")
flux3_edit_example = _safe_import(".examples", "flux3_edit_example")
flux2_example = _safe_import(".examples", "flux2_example")
qwen_image_example = _safe_import(".examples", "qwen_image_example")
qwen_image_controlnet_example = _safe_import(".examples", "qwen_image_controlnet_example")
qwen_image_edit_example = _safe_import(".examples", "qwen_image_edit_example")
qwen_image_layered_example = _safe_import(".examples", "qwen_image_layered_example")
skyreels_v2_example = _safe_import(".examples", "skyreels_v2_example")
ltx2_t2v_example = _safe_import(".examples", "ltx2_t2v_example")
ltx2_i2v_example = _safe_import(".examples", "ltx2_i2v_example")
ltx25_t2v_example = _safe_import(".examples", "ltx25_t2v_example")
ltx25_i2v_example = _safe_import(".examples", "ltx25_i2v_example")
ltx25_dfr_example = _safe_import(".examples", "ltx25_dfr_example")
wan_example = _safe_import(".examples", "wan_example")
wan_i2v_example = _safe_import(".examples", "wan_i2v_example")
wan_vace_example = _safe_import(".examples", "wan_vace_example")
ovis_image_example = _safe_import(".examples", "ovis_image_example")
zimage_example = _safe_import(".examples", "zimage_example")
longcat_audio_dit_example = _safe_import(".examples", "longcat_audio_dit_example")
stable_audio_3_example = _safe_import(".examples", "stable_audio_3_example")
longcat_image_example = _safe_import(".examples", "longcat_image_example")
longcat_image_edit_example = _safe_import(".examples", "longcat_image_edit_example")
zimage_controlnet_example = _safe_import(".examples", "zimage_controlnet_example")
glm_image_example = _safe_import(".examples", "glm_image_example")
glm_image_edit_example = _safe_import(".examples", "glm_image_edit_example")
firered_image_edit_example = _safe_import(".examples", "firered_image_edit_example")
helios_t2v_example = _safe_import(".examples", "helios_t2v_example")
helios_t2v_distill_example = _safe_import(".examples", "helios_t2v_distill_example")
cosmos3_omni_example = _safe_import(".examples", "cosmos3_omni_example")
boogu_image_edit_example = _safe_import(".examples", "boogu_image_edit_example")
ernie_image_example = _safe_import(".examples", "ernie_image_example")
ernie_image_turbo_example = _safe_import(".examples", "ernie_image_turbo_example")
krea2_example = _safe_import(".examples", "krea2_example")
bria_fibo_example = _safe_import(".examples", "bria_fibo_example")
anyflow_t2v_example = _safe_import(".examples", "anyflow_t2v_example")
anyflow_far_example = _safe_import(".examples", "anyflow_far_example")
minimax_h3_example = _safe_import(".examples", "minimax_h3_example")
minimax_h3_fl2va_example = _safe_import(".examples", "minimax_h3_fl2va_example")
minimax_h3_ref2va_example = _safe_import(".examples", "minimax_h3_ref2va_example")
fasth3_example = _safe_import(".examples", "fasth3_example")
fasth3_v2_example = _safe_import(".examples", "fasth3_v2_example")
joy_image_edit_example = _safe_import(".examples", "joy_image_edit_example")
prx_pixel_example = _safe_import(".examples", "prx_pixel_example")
anima_example = _safe_import(".examples", "anima_example")
motif_video_t2v_example = _safe_import(".examples", "motif_video_t2v_example")
motif_video_i2v_example = _safe_import(".examples", "motif_video_i2v_example")
ideogram4_example = _safe_import(".examples", "ideogram4_example")
moss_image_sft_example = _safe_import(".examples", "moss_image_sft_example")
moss_image_base_example = _safe_import(".examples", "moss_image_base_example")
ming_image_design_example = _safe_import(".examples", "ming_image_design_example")
ming_image_design_layer_example = _safe_import(".examples", "ming_image_design_layer_example")
hidream_o1_image_example = _safe_import(".examples", "hidream_o1_image_example")
nucleus_image_example = _safe_import(".examples", "nucleus_image_example")
kandinsky6_ti2va_example = _safe_import(".examples", "kandinsky6_ti2va_example")
wan_animate_2_example = _safe_import(".examples", "wan_animate_2_example")
wan_s2v_example = _safe_import(".examples", "wan_s2v_example")
wan_animate_2_distilled_example = _safe_import(".examples", "wan_animate_2_distilled_example")
sana_wm_example = _safe_import(".examples", "sana_wm_example")
lingbot_video_example = _safe_import(".examples", "lingbot_video_example")
