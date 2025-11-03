from .npu_fast_gelu import replace_npu_fast_gelu
from .npu_rms_norm import replace_npu_rms_norm
from .npu_layer_norm_eval import replace_npu_layer_norm_eval
from .npu_rotary_mul import replace_npu_rotary_mul


NPU_OPTIM_MAP = {
    "npu_fast_gelu": replace_npu_fast_gelu,
    "npu_rms_norm": replace_npu_rms_norm,
    "npu_layer_norm_eval": replace_npu_layer_norm_eval,
    "npu_rotary_mul": replace_npu_rotary_mul,
}