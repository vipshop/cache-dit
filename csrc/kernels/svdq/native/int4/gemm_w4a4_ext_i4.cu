#include "../core/gemm_w4a4_ext_impl.cuh"

namespace svdq::native_kernels {
template class GEMM_W4A4_Ext_Launch<GEMMConfig_W4A4_BF16, false>;
template class GEMM_W4A4_Ext_Launch<GEMMConfig_W4A4_FP16, false>;
template class GEMM_W4A4_Ext_Launch<GEMMConfig_W4A4_FP16_FasterI2F, false>;
}  // namespace svdq::native_kernels
