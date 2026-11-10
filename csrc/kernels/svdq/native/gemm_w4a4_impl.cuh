// Adapted from nunchaku/src/kernels/zgemm/gemm_w4a4_launch_impl.cuh
#pragma once

#include "gemm_w4a4_launch.cuh"

namespace svdq::native_kernels {

#ifndef __INTELLISENSE__
template <typename Config, bool USE_FP4>
void GEMM_W4A4_Launch<Config, USE_FP4>::gemm_w4a4(
#else
template <>
void GEMM_W4A4_Launch<GEMMConfig_W4A4_FP16, false>::gemm_w4a4(
#endif
  Tensor act,            // packed act [M, K / 2]
  Tensor wgt,            // packed wgt [N, K / 2]
  Tensor out,            // linear     [M, N]
  Tensor qout,           // packed act [M, N / 2]
  Tensor ascales,        // packed as  [K / 64, M]
  Tensor wscales,        // packed ws  [K / 64, N]
  Tensor oscales,        // packed as  [N / 64, M]
  Tensor lora_act_in,    // packed lora_act [M, R]
  Tensor lora_up,        // packed lora_wgt [N, R]
  Tensor lora_down,      // packed lora_wgt [N, R]
  Tensor lora_act_out,   // packed lora_act [M, R]
  Tensor bias,           // linear     [N]
  Tensor smooth_factor,  // packed ws  [N], for quantization of the next layer
  bool act_unsigned,
  std::vector<float> lora_scales,  // [R / 16]
  bool fuse_silu, bool fp4, float alpha, Tensor wcscales) {
#ifdef __INTELLISENSE__
  static constexpr bool USE_FP4 = false;
#endif
  assert(fp4 == USE_FP4);

  int M = act.numel() / act.shape[-1];
  int N = wgt.shape[0];
  int K = act.shape[-1] * 2;
  assert(K == wgt.shape[1] * 2);

  int actualM = 0;
  int actualN = 0;
  if (out.valid()) {
    actualM = out.numel() / out.shape[-1];
    actualN = out.shape[-1];

    assert(actualM <= M && M - actualM < GEMM::BLOCK_M);
    assert(actualN <= N && N - actualN < GEMM::BLOCK_N);
  }

  spdlog::trace("gemm_w4a4: M={} N={} K={}", M, N, K);
  spdlog::trace("act at {}", act.data_ptr());
  spdlog::trace("wgt at {}", wgt.data_ptr());
  spdlog::trace("ascales at {}", ascales.data_ptr());
  spdlog::trace("wscales at {}", wscales.data_ptr());
  if (bias.valid()) {
    spdlog::trace("bias at {}", bias.data_ptr());
  }

  auto launch = [&]<typename Epilogue>(Epilogue::Arguments args) {
    assert(M % GEMM::BLOCK_M == 0);
    assert(N % GEMM::BLOCK_N == 0);
    dim3 grid(M / GEMM::BLOCK_M, N / GEMM::BLOCK_N);

    bool swapBlockMN = M > N * 2;
    if (swapBlockMN) {
      std::swap(grid.x, grid.y);
    }

    if constexpr (!USE_FP4) {
      dispatchBool(act_unsigned, [&]<bool ACT_UNSIGNED>() {
        auto func = invoke_kernel<typename GEMM::gemm_w4a4_kernel<Epilogue, ACT_UNSIGNED>,
                                  const packed_act_t *, const packed_wgt_t *,
                                  const packed_ascale_t *, const packed_wscale_t *, int, int, int,
                                  typename Epilogue::Arguments, bool, bool>;

        assert(alpha == 1.0f);

        func<<<grid, GEMM::WARP_SIZE * GEMM::NUM_WARPS, 0, getCurrentCUDAStream()>>>(
          act.data_ptr<packed_act_t>(), wgt.data_ptr<packed_wgt_t>(),
          ascales.data_ptr<packed_ascale_t>(), wscales.data_ptr<packed_wscale_t>(), M, N, K, args,
          swapBlockMN, false);
        checkCUDA(cudaGetLastError());
      });
      return;
    }

    if constexpr (USE_FP4) {
      dispatchBool(alpha != 1.0f, [&]<bool USE_ALPHA>() {
        assert(!act_unsigned);

        auto func = invoke_kernel<typename GEMM::gemm_w4a4_fp4_kernel<Epilogue, USE_ALPHA>,
                                  const packed_act_t *, const packed_wgt_t *,
                                  const packed_amscale_t *, const packed_wmscale_t *, float, int,
                                  int, int, typename Epilogue::Arguments, bool, bool>;

        assert(ascales.dtype() == Tensor::FP8_E4M3);
        assert(wscales.dtype() == Tensor::FP8_E4M3);

        func<<<grid, GEMM::WARP_SIZE * GEMM::NUM_WARPS, 0, getCurrentCUDAStream()>>>(
          act.data_ptr<packed_act_t>(), wgt.data_ptr<packed_wgt_t>(),
          ascales.data_ptr<packed_amscale_t>(), wscales.data_ptr<packed_wmscale_t>(), alpha, M, N,
          K, args, swapBlockMN, false);
        checkCUDA(cudaGetLastError());
      });

      return;
    }
  };

  auto launch_bias = [&]<typename NextEpilogue>(NextEpilogue::Arguments nextArgs) {
    assert(!bias.valid() || bias.numel() == N);
    assert(!wcscales.valid() || wcscales.numel() == N);

    dispatchBool(bias.valid(), [&]<bool USE_BIAS>() {
      dispatchBool(wcscales.valid(), [&]<bool USE_SCALE>() {
        using EpilogueBias = typename GEMM::EpilogueBias<USE_BIAS, USE_SCALE>;
        // append EpilgoueNop to workaround mismatched memory layout of std::tuple between device
        // and host code on Windows
        // ** sizeof(std::tuple<std::tuple<int>>) == 8 on device **
        using Epilogue = typename GEMM::EpilogueCombination<EpilogueBias, NextEpilogue,
                                                            typename GEMM::EpilogueNop>;
        return launch.template operator()<Epilogue>(
          {typename EpilogueBias::Arguments{
             .bias = USE_BIAS ? bias.data_ptr<packed_wscale_t>() : nullptr,
             .scale = USE_SCALE ? wcscales.data_ptr<packed_wscale_t>() : nullptr,
           },
           nextArgs,
           {}});
      });
    });
  };

  auto launch_lora = [&]<typename NextEpilogue, typename MidEpilogue>(
                       NextEpilogue::Arguments nextArgs, MidEpilogue::Arguments midArgs) {
    assert(lora_up.valid() == lora_act_in.valid());
    assert(lora_down.valid() == lora_act_out.valid());

    const int rank_up = lora_up.valid() ? lora_up.shape[1] : 0;
    const int rank_down = lora_down.valid() ? lora_down.shape[1] : 0;

    if (rank_up == 0) {
      assert(rank_down == 0);
      return launch_bias
        .template operator()<typename GEMM::EpilogueCombination<MidEpilogue, NextEpilogue>>(
          {midArgs, nextArgs});
    }

    assert(rank_up % 16 == 0);

    assert(lora_up.shape[0] == N);
    assert(lora_act_in.shape[0] == M);
    assert(lora_act_in.shape[1] == rank_up);

    using LoraUp = Lora;
    using scale_t = typename LoraUp::scale_t;

    scale_t scales;
    if constexpr (scales.size() > 0) {
      for (size_t i = 0; i < scales.size(); i++) {
        scales[i] = i < lora_scales.size() ? lora_scales[i] : 0.0f;
      }
    }

    if (rank_down == 0) {
      using Epilogue =
        typename GEMM::EpilogueCombination<typename LoraUp::EpilogueLoraUp, MidEpilogue,
                                           NextEpilogue, typename GEMM::EpilogueNop>;
      return launch_bias.template operator()<Epilogue>(
        {typename LoraUp::EpilogueLoraUp::Arguments{
           .lora_act = lora_act_in.data_ptr<float>(),
           .lora_wgt_up = lora_up.data_ptr<packed_fpsum_t>(),
           .rank = rank_up,
           .scales = scales,
           .alwaysfalse = false,
         },
         midArgs,
         nextArgs,
         {}});
    }

    assert(rank_down % 16 == 0);

    assert(lora_down.shape[0] == N);
    assert(lora_act_out.shape[0] == M);
    assert(lora_act_out.shape[1] == rank_down);

    lora_act_out.zero_();

    using LoraDown = LoraUp;
    using Epilogue =
      typename GEMM::EpilogueCombination<typename LoraUp::EpilogueLoraUp, MidEpilogue,
                                         typename LoraDown::EpilogueLoraDown, NextEpilogue,
                                         typename GEMM::EpilogueNop>;
    return launch_bias.template operator()<Epilogue>(
      {typename LoraUp::EpilogueLoraUp::Arguments{
         .lora_act = lora_act_in.data_ptr<float>(),
         .lora_wgt_up = lora_up.data_ptr<packed_fpsum_t>(),
         .rank = rank_up,
         .scales = scales,
         .alwaysfalse = false,
       },
       midArgs,
       typename LoraDown::EpilogueLoraDown::Arguments{
         .lora_wgt_down = lora_down.data_ptr<packed_fpsum_t>(),
         .lora_act = lora_act_out.data_ptr<float>(),
         .rank = rank_down,
         .alwaysfalse = false,
       },
       nextArgs,
       {}});
  };

  assert(qout.valid() == oscales.valid());

  if (qout.valid() && oscales.valid()) {
    // The fused-GELU path always writes the next-layer activation pack, so the smoothing vector
    // must be present; `_impl` dereferences it unconditionally below.
    assert(smooth_factor.valid());

    // Use signed INT4 (matching fc2.quantize output).
    // No unsigned shift — the non-fused path never applies +0.171875
    // before fc2.quantize, so we must match that behaviour exactly.
    constexpr bool USE_UNSIGNED = false;

    // GELU is always applied in-place via EpilogueGelu (MidEpilogue)
    // so that *every* downstream epilogue — including EpilogueDefault
    // (fp16 `out`) and EpilogueLoraDown (`lora_act_out`) — sees
    // post-GELU accumulator values.  EpilogueQuantize only quantizes.
    constexpr bool FUSE_GELU = false;
    using EpilogueQuantize = typename GEMM::EpilogueQuantize<FUSE_GELU, USE_UNSIGNED, USE_FP4>;
    auto argsQuantize = typename EpilogueQuantize::Arguments{
      .qout = qout.data_ptr<packed_act_t>(),
      .oscales = oscales.data_ptr<typename EpilogueQuantize::oscales_t>(),
      .shift_value = 0.0f,
      .smooth_factor = smooth_factor.data_ptr<packed_wscale_t>()};

    if (out.valid()) {
      launch_lora.template operator()<
        typename GEMM::EpilogueCombination<typename GEMM::EpilogueDefault, EpilogueQuantize>,
        typename Epilogues::EpilogueGelu>({typename GEMM::EpilogueDefault::Arguments{
                                             .out = out.data_ptr<half_t>(),
                                             .actualM = actualM,
                                             .actualN = actualN,
                                           },
                                           argsQuantize},
                                          {});
    } else {
      launch_lora.template operator()<EpilogueQuantize, typename Epilogues::EpilogueGelu>(
        argsQuantize, {});
    }
  } else if (out.valid()) {
    using Epilogue = typename GEMM::EpilogueDefault;
    typename Epilogue::Arguments args{
      .out = out.data_ptr<half_t>(),
      .actualM = actualM,
      .actualN = actualN,
    };

    if (fuse_silu) {
      launch_lora.template operator()<Epilogue, typename GEMM::EpilogueSilu>(args, {});
    } else {
      launch_lora.template operator()<Epilogue, typename GEMM::EpilogueNop>(args, {});
    }
  } else {
    assert(false);
  }
}

template <typename Config, bool USE_FP4>
void GEMM_W4A4_Launch<Config, USE_FP4>::quantize_w4a4(Tensor input, Tensor output, Tensor oscales,
                                                      Tensor lora_down, Tensor lora_act_out,
                                                      Tensor smooth, bool fuse_glu, bool fp4) {
  const int actualM = input.numel() / input.shape[-1];
  const int actualN = input.shape[-1];

  const int M = ceilDiv(actualM, GEMM::BLOCK_M) * GEMM::BLOCK_M;
  const int N = ceilDiv(actualN / (fuse_glu ? 2 : 1), GEMM::BLOCK_N) * GEMM::BLOCK_N;

  assert(output.dtype() == Tensor::INT8);
  assert(output.numel() / output.shape[-1] == M);
  assert(output.shape[-1] == N / 2);

  if (fp4) {
    assert(oscales.dtype() == Tensor::FP8_E4M3);
    assert(oscales.numel() == M * N / GEMM::WARP_K * 4);
  } else {
    assert(isTypeMatch<half_t>(oscales.dtype()));
    assert(oscales.numel() == M * N / GEMM::WARP_K);
  }

  const int rank = lora_down.shape[1];

  assert(rank % 16 == 0);

  assert(lora_down.shape[0] == N);
  assert(lora_act_out.shape[0] == M);
  assert(lora_act_out.shape[1] == rank);

  lora_act_out.zero_();

  dim3 grid(M / GEMM::BLOCK_M, N / GEMM::BLOCK_N);

  dispatchBool(fuse_glu, [&]<bool FUSE_GLU>() {
    using kernel = typename GEMM::quantize_w4a4_fuse_lora_kernel<FUSE_GLU, USE_FP4>;

    auto func = invoke_kernel<kernel, typename kernel::Arguments>;

    checkCUDA(
      cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, kernel::SHMEM_SIZE));

    func<<<grid, GEMM::WARP_SIZE * GEMM::NUM_WARPS, kernel::SHMEM_SIZE, getCurrentCUDAStream()>>>(
      typename kernel::Arguments{
        .input = input.data_ptr<half_t>(),
        .smooth_factor = smooth.valid() ? smooth.data_ptr<packed_wscale_t>() : nullptr,
        .output = output.data_ptr<packed_act_t>(),
        .oscales = oscales.data_ptr<typename kernel::oscales_t>(),
        .lora_wgt_down = lora_down.data_ptr<packed_fpsum_t>(),
        .lora_act = lora_act_out.data_ptr<float>(),
        .lora_rank = rank,
        .M = M,
        .N = N,
        .actualM = actualM,
        .actualN = actualN,
        .alwaysfalse = false,
      });
    checkCUDA(cudaGetLastError());
  });
}

};  // namespace svdq::native_kernels
