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
        if optim_modules is None:
            optim_modules = list(NPU_OPTIM_MAP.keys())

        for module in optim_modules:
            try:
                NPU_OPTIM_MAP[module]()
            except:
                logger.warning(f"Apply {module} failed, will still use original module")
    else:
        logger.warning("NPU is unavailable, will not apply npu optimizations")