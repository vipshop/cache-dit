try:
    from diffusers.utils.import_utils import is_torch_npu_available

except:
    def is_torch_npu_available():
        try:
            import torch
            import torch_npu
            return torch.npu.is_available()
        except:
            return False

from cache_dit.logger import init_logger

logger = init_logger(__name__)


def log_replace_info(ori_module, npu_module):
    logger.info(f"Replaced '{ori_module}' with '{npu_module}' successfully")


def npu_optimize(optim_modules: dict = None):
    if is_torch_npu_available():
        from .npu_ops import NPU_OPTIM_MAP
        from .models_wrapper import NPU_MODELS_WRAPPER_MAP

        if optim_modules is None:
            optim_modules = list(NPU_OPTIM_MAP.keys())

        for module in optim_modules:
            try:
                if module in NPU_OPTIM_MAP:
                    NPU_OPTIM_MAP[module]()
                elif module in NPU_MODELS_WRAPPER_MAP:
                    NPU_MODELS_WRAPPER_MAP[module]()
                else:
                    logger.warning(f"Module {module} not found in NPU_OPTIM_MAP, will not apply npu optimizations")
            except:
                logger.warning(f"Apply {module} failed, will still use original module")
    else:
        logger.warning("NPU is unavailable, will not apply npu optimizations")