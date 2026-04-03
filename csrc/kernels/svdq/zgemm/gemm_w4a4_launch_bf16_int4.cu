// Copied from nunchaku/src/kernels/zgemm/gemm_w4a4_launch_bf16_int4.cu
#include "gemm_w4a4_launch_impl.cuh"

namespace svdq::kernels {
template class GEMM_W4A4_Launch<GEMMConfig_W4A4_BF16, false>;
};
