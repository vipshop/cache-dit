#include "gemm_w4a4_impl.cuh"

namespace svdq::native_kernels {
template class GEMM_W4A4_Launch<GEMMConfig_W4A4_BF16, false>;
template class GEMM_W4A4_Launch<GEMMConfig_W4A4_FP16, false>;
}  // namespace svdq::native_kernels
