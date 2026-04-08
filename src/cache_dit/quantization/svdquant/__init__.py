"""SVDQ W4A4 quantization entry points.

Call chain overview
-------------------

Offline quantization path used by tests and model conversion:

    tests.quantization._svdq_test_utils.quantize_toy_model(...)
        -> quantize_linear_svdq_w4a4(...)
            -> standardize_calibration_activations(...)
            -> compute_smooth_scale(...)
            -> decompose_lowrank_residual(...)
            -> _compute_group_scales(...)
            -> export_raw_svdq_w4a4_state_dict(...)
                -> pack_svdq_w4a4_linear_tensors(...)
                    -> residual / scale
                    -> round to INT4 in [-8, 7]
                    -> SVDQWeightPacker.pack_weight(...)
                    -> SVDQWeightPacker.pack_scale(...)
            -> adapt_svdq_module_state_dict(...)
            -> SVDQW4A4Linear.from_linear(...)
            -> load_state_dict(...)

Runtime path after the packed tensors have been loaded:

    SVDQW4A4Linear.forward(...)
        -> quantize(...)
            -> svdq_quantize_w4a4_act_fuse_lora(...)
        -> forward_quant(...)
            -> svdq_gemm_w4a4(...)

Notes:
- The main flow quantizes weights offline in Python and stores packed
    qweight/wscales tensors in the module state dict.
- SVDQWeightPacker exists to serialize tensors into the warp-oriented W4A4
    kernel layout, not to define an arbitrary checkpoint packing format.
- svdq_quantize_w4a4_wgt is a low-level CUDA helper used by runtime tests,
    but it is not part of quantize_linear_svdq_w4a4's main call chain.
"""

from ...kernels import svdq_extension_is_available as svdq_is_available
from ...kernels import svdq_get_load_error
from .linear import SVDQW4A4Linear
from .quantizer import CalibrationInputs
from .quantizer import compute_smooth_scale
from .quantizer import quantize_linear_svdq_w4a4
from .quantizer import standardize_calibration_activations
from .quantizer import validate_svdq_linear_geometry

__all__ = [
    "CalibrationInputs",
    "SVDQW4A4Linear",
    "compute_smooth_scale",
    "svdq_get_load_error",
    "svdq_is_available",
    "quantize_linear_svdq_w4a4",
    "standardize_calibration_activations",
    "validate_svdq_linear_geometry",
]
