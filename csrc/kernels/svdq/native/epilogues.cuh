// Adapted from nunchaku/src/kernels/zgemm/epilogues.cuh
#pragma once

#include "gemm_base.cuh"

namespace svdq::native_kernels {

template <typename Config>
class Epilogues;

#ifndef __INTELLISENSE__
template <typename Config>
class Epilogues : public GEMMBase<Config> {
#else
template <>
class Epilogues<GEMMConfig_W4A4_FP16> : public GEMMBase<GEMMConfig_W4A4_FP16> {
  using Config = GEMMConfig_W4A4_FP16;
#endif
 public:
  IMPORT_GEMM_BASE(Config);

 public:
  struct EpilogueGelu {
    struct Arguments {
      size_t unused;
    };

    __device__ __forceinline__ void operator()(const BlockInfo binfo, fpsum_warp &fpsum, int M,
                                               int N, int K, const Arguments &args) {
#pragma unroll
      for (int i = 0; i < WARP_M_TILES; i++) {
#pragma unroll
        for (int j = 0; j < WARP_N_TILES; j++) {
#pragma unroll
          for (int k = 0; k < 4; k++) {
            half2_t &data = fpsum[i * WARP_N_TILES + j].data[k];
            data = gelu_half2(data);
          }
        }
      }
    }
  };
};

};  // namespace svdq::native_kernels
