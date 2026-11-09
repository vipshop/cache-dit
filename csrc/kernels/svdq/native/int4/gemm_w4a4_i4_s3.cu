#include "gemm_w4a4_impl.cuh"

namespace svdq::native_kernels {

SVDQ_DEFINE_GEMM_W4A4_LAUNCH(bf16, GEMMConfig_W4A4_BF16, 3);
SVDQ_DEFINE_GEMM_W4A4_LAUNCH(fp16, GEMMConfig_W4A4_FP16, 3);

}  // namespace svdq::native_kernels
