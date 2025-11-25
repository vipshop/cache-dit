import torch
import torch_npu

from ..utils import log_replace_info

from cache_dit.logger import init_logger
logger = init_logger(__name__)

def _transpose_to_nz(model):
    torch.npu.config.allow_internal_format=True
    if not hasattr(model, "named_modules"):
        return
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if module.weight.data.device.type == "cpu":
                module.weight.data = module.weight.data.to("npu")
            try:
                weight = torch_npu.npu_format_cast(module.weight.data.contiguous(), 29)
                module.weight.data = weight
            except Exception as e:
                logger.warning(f"Failed to transpose {name} to NZ, skipping: {e}")

def _replace_diffusers_from_pretrained():
    import diffusers
    from_pretrained_func = diffusers.DiffusionPipeline.from_pretrained
    def new_from_pretrained(self, *args, **kwargs):
        pipe = from_pretrained_func(self, *args, **kwargs)
        # 遍历 pipe 中的所有属性
        for attr in dir(pipe):
            if attr.startswith("_") or not hasattr(pipe, attr):
                continue
            if hasattr(getattr(pipe, attr), "named_modules"):
                _transpose_to_nz(getattr(pipe, attr))
        return pipe
    diffusers.DiffusionPipeline.from_pretrained = new_from_pretrained

def replace_npu_weight_nz():
    _replace_diffusers_from_pretrained()
    log_replace_info("from_pretrained of Diffusers", "npu_weight_nz")