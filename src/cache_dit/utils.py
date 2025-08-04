import torch
import diffusers


@torch.compiler.disable
def is_diffusers_at_least_0_3_5() -> bool:
    return diffusers.__version__ >= "0.35.0"
