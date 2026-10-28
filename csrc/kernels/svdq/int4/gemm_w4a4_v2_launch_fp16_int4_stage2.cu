#include "gemm_w4a4_v2_launch_impl.cuh"

namespace svdq::kernels {

SVDQ_DEFINE_GEMM_W4A4_V2_LAUNCH(fp16, GEMMConfig_W4A4_FP16, 2);

}  // namespace svdq::kernels
