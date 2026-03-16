import gc
import time
import torch
import diffusers
import builtins as __builtin__
import contextlib
from typing import Tuple, List, Union, Optional, Any
from diffusers import DiffusionPipeline

from .platforms import current_platform
from .logger import init_logger


logger = init_logger(__name__)


def dummy_print(*args, **kwargs):
    pass


@contextlib.contextmanager
def disable_print():
    origin_print = __builtin__.print
    __builtin__.print = dummy_print
    yield
    __builtin__.print = origin_print


def is_diffusers_at_least_0_3_5() -> bool:
    return diffusers.__version__ >= "0.35.0"


def maybe_empty_cache():
    try:
        time.sleep(1)
        gc.collect()
        current_platform.empty_cache()
        current_platform.ipc_collect()
        time.sleep(1)
        gc.collect()
        current_platform.empty_cache()
        current_platform.ipc_collect()
    except Exception:
        pass


def _is_text_encoder(module: torch.nn.Module) -> bool:
    _import_module = module.__class__.__module__
    # Including the cases for normal text encoder and vision-language
    # model (e.g, GLM Image) in transformers
    return _import_module.startswith("transformers")


def _is_controlnet(module: torch.nn.Module) -> bool:
    _import_module = module.__class__.__module__
    return _import_module.startswith("diffusers.models.controlnet")


def _is_auto_encoder(module: torch.nn.Module) -> bool:
    _import_module = module.__class__.__module__
    return _import_module.startswith("diffusers.models.autoencoder")


def check_text_encoder(module: torch.nn.Module | DiffusionPipeline | Any) -> bool:
    """Check if the given pipeline has text encoder."""
    if isinstance(module, torch.nn.Module) and not isinstance(module, DiffusionPipeline):
        return _is_text_encoder(module)

    if not isinstance(module, DiffusionPipeline):
        pipe = getattr(module, "pipe", None)
    else:
        pipe = module
    if hasattr(pipe, "text_encoder") and getattr(pipe, "text_encoder") is not None:
        return True
    # For some pipelines (e.g., FLUX), the text encoder may have different names.
    for attr_name in dir(pipe):
        attr = getattr(pipe, attr_name)
        if isinstance(attr, torch.nn.Module) and _is_text_encoder(attr):
            return True
    return False


def check_controlnet(module: torch.nn.Module | DiffusionPipeline | Any) -> bool:
    """Check if the given pipeline has ControlNet."""
    if isinstance(module, torch.nn.Module) and not isinstance(module, DiffusionPipeline):
        return _is_controlnet(module)

    if not isinstance(module, DiffusionPipeline):
        pipe = getattr(module, "pipe", None)
    else:
        pipe = module
    if hasattr(pipe, "controlnet") and getattr(pipe, "controlnet") is not None:
        return True
    return False


def check_auto_encoder(module: torch.nn.Module | DiffusionPipeline | Any) -> bool:
    """Check if the given pipeline has auto encoder."""
    if isinstance(module, torch.nn.Module) and not isinstance(module, DiffusionPipeline):
        return _is_auto_encoder(module)

    if not isinstance(module, DiffusionPipeline):
        pipe = getattr(module, "pipe", None)
    else:
        pipe = module
    if hasattr(pipe, "vae") and getattr(pipe, "vae") is not None:
        return True
    return False


def check_parallelized(module: torch.nn.Module) -> bool:
    """Check if the given module is already parallelized."""
    return getattr(module, "_is_parallelized", False)


def check_quantized(module: torch.nn.Module) -> bool:
    """Check if the given module is already quantized."""
    return getattr(module, "_is_quantized", False)


def check_cached(module: torch.nn.Module) -> bool:
    """Check if the given module is already cached."""
    return getattr(module, "_is_cached", False)


def parse_text_encoder(
    pipe: DiffusionPipeline,
) -> Tuple[Optional[torch.nn.Module], Optional[str]]:
    pipe_cls_name = pipe.__class__.__name__
    if (
        hasattr(pipe, "text_encoder_2")
        and not pipe_cls_name.startswith("Hunyuan")
        and not pipe_cls_name.startswith("Kandinsky")
    ):
        # Specific for FluxPipeline, FLUX.1-dev
        return getattr(pipe, "text_encoder_2"), "text_encoder_2"
    elif hasattr(pipe, "text_encoder_3"):  # HiDream pipeline
        return getattr(pipe, "text_encoder_3"), "text_encoder_3"
    elif hasattr(pipe, "vision_language_encoder") and pipe_cls_name.startswith(
        "GlmImage"
    ):  # GLM Image pipeline
        return getattr(pipe, "vision_language_encoder"), "vision_language_encoder"
    elif hasattr(pipe, "text_encoder"):  # General case
        return getattr(pipe, "text_encoder"), "text_encoder"
    else:
        return None, None


def parse_extra_modules(
    pipe_or_adapter: DiffusionPipeline | Any,
    extra_modules: List[str | torch.nn.Module],
) -> Union[List[torch.nn.Module], List]:
    """Parse extra modules according to the given names in extra_modules to
    actual modules in the pipeline. Useful for extra parallelism and extra
    quantization outside of the transformer module, e.g., applying parallelism
    or quantization to text encoder and vae at the same time. For example,
    when extra_modules is set to ['text_encoder', 'vae'], we will try to find
    the text encoder and vae modules in the pipeline and apply parallelism or
    quantization to these modules as well. Note that the supported extra module
    names may vary for different pipelines, but generally include common components
    such as 'text_encoder', 'vae', 'unet', etc. User can also directly pass the
    actual module objects in extra_modules for more precise control.
    Args:
        pipe_or_adapter: The DiffusionPipeline or BlockAdapter to parse the extra modules from.
        extra_modules: A list of module names or actual module objects to be parsed.
    Returns:
        A list of parsed extra modules as actual module objects. If a module name is not found
        in the pipeline, it will be skipped with a warning.
    """
    if not isinstance(pipe_or_adapter, DiffusionPipeline):
        pipe = getattr(pipe_or_adapter, "pipe", None)
    else:
        pipe = pipe_or_adapter

    if not extra_modules or pipe is None:  # empty list
        return []

    parsed_extra_modules: List[torch.nn.Module] = []
    for module_or_name in extra_modules:
        if isinstance(module_or_name, torch.nn.Module):
            parsed_extra_modules.append(module_or_name)
            continue

        if hasattr(pipe, module_or_name):
            if module_or_name.lower() == "text_encoder":
                # Special handling for text encoder
                text_encoder, text_encoder_name = parse_text_encoder(pipe)
                if text_encoder is not None:
                    text_encoder._actual_module_name = text_encoder_name
                    parsed_extra_modules.append(text_encoder)
                else:
                    logger.warning("Text encoder not found in the pipeline for extra modules.")
            else:
                parsed_extra_modules.append(getattr(pipe, module_or_name))
        else:
            logger.warning(f"Extra module name {module_or_name} not found in the pipeline.")
    return parsed_extra_modules
