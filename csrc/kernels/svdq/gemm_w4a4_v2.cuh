#pragma once

#include "gemm_w4a4.cuh"

namespace svdq::kernels {

// v2 keeps the original packed tensor contract and the same per-warp MMA decomposition as v1.
// The dedicated launch wrapper is retained so the v2 path can evolve independently. The packed
// tensors are still produced with the original 256-row activation layout, while the v2 runtime can
// reinterpret that contiguous warp-group sequence as smaller logical row blocks on Ada.

template <typename PackedConfig, int LogicalBlockM>
class GEMMConfig_W4A4_V2_Block {
 public:
  static_assert(LogicalBlockM > 0);

  static constexpr int PACKED_BLOCK_M = PackedConfig::BLOCK_M;
  static constexpr int PACKED_NUM_WARPS = PackedConfig::NUM_WARPS;
  static constexpr int WARP_M_ROWS = PACKED_BLOCK_M / PACKED_NUM_WARPS;

  static constexpr int BLOCK_M = LogicalBlockM;
  static constexpr int BLOCK_N = PackedConfig::BLOCK_N;
  static constexpr int WARP_SIZE = PackedConfig::WARP_SIZE;
  static constexpr int NUM_WARPS = LogicalBlockM / WARP_M_ROWS;

  static constexpr int INSN_M = PackedConfig::INSN_M;
  static constexpr int INSN_N = PackedConfig::INSN_N;
  static constexpr int INSN_K = PackedConfig::INSN_K;
  static constexpr bool FASTER_I2F = PackedConfig::FASTER_I2F;

  using half_t = typename PackedConfig::half_t;
  using half2_t = typename PackedConfig::half2_t;

  static_assert(PACKED_BLOCK_M % BLOCK_M == 0);
  static_assert(NUM_WARPS > 0);
  static_assert(PACKED_NUM_WARPS % NUM_WARPS == 0);
  static_assert(BLOCK_M % NUM_WARPS == 0);
  static_assert(BLOCK_M / NUM_WARPS == WARP_M_ROWS);
};

template <typename kernel, typename... T>
__global__ static void invoke_kernel_v2(T... args) {
#ifdef __CUDA_ARCH__
  if constexpr (__CUDA_ARCH__ >= min_arch<kernel>() && __CUDA_ARCH__ <= max_arch<kernel>()) {
    __shared__ typename kernel::SharedStorage shared_storage;
    kernel()(shared_storage, args...);
  } else {
    trap_unsupported_arch();
  }
#else
  __shared__ typename kernel::SharedStorage shared_storage;
  kernel()(shared_storage, args...);
#endif
}

template <typename Config, int NumPipelineStages>
class GEMM_W4A4_V2 : public GEMM_W4A4<Config> {
 public:
  using Parent = GEMM_W4A4<Config>;
  using Base = GEMMBase<Config>;
  using LoraKernel = Lora<Config>;
  static_assert(NumPipelineStages >= 1 && NumPipelineStages <= 3,
                "svdq_gemm_w4a4_v2 supports pipeline stages in [1, 3].");
  // More stages can hide additional load latency, but each extra stage keeps more packed tiles
  // live per thread and therefore raises register pressure. Keep the kernel body stage-agnostic so
  // the runtime wrapper can dispatch among multiple compiled stage variants.
  static constexpr int NUM_PIPELINE_STAGES = NumPipelineStages;
  static constexpr int PACKED_BLOCK_M = Config::PACKED_BLOCK_M;
  static constexpr int PACKED_NUM_WARPS = Config::PACKED_NUM_WARPS;
  static constexpr bool USE_PACKED_LAYOUT_COMPAT = PACKED_BLOCK_M != Base::BLOCK_M;
  static constexpr int LOGICAL_BLOCKS_PER_PACKED_BLOCK = PACKED_BLOCK_M / Base::BLOCK_M;
  using Base::ASCALES_NUM_PACKS;
  using Base::ASCALES_VALID_LANES;
  using Base::BLOCK_M;
  using Base::BLOCK_N;
  using Base::NUM_WARPS;
  using Base::WARP_K;
  using Base::WARP_M_TILES;
  using Base::WARP_N_TILES;
  using Base::WARP_SIZE;
  using Base::WSCALES_NUM_PACKS;
  using Base::WSCALES_VALID_LANES;
  using typename Base::BlockInfo;
  using typename Base::act_warp;
  using typename Base::ascale_warp;
  using typename Base::f32psum_warp;
  using typename Base::fpsum_warp;
  using typename Base::packed_act_t;
  using typename Base::packed_ascale_t;
  using typename Base::packed_f32psum_t;
  using typename Base::packed_fpsum_t;
  using typename Base::packed_wgt_t;
  using typename Base::packed_wscale_t;
  using typename Base::wgt_warp;
  using typename Base::wscale_warp;
  using typename Base::EpilogueDefault;
  using typename Base::EpilogueNop;
  template <bool USE_BIAS, bool USE_SCALE>
  using EpilogueBias = typename Base::template EpilogueBias<USE_BIAS, USE_SCALE>;
  template <typename... Epilogues>
  using EpilogueCombination = typename Base::template EpilogueCombination<Epilogues...>;
  using Base::load_act;
  using Base::load_ascale;
  using Base::load_wgt;
  using Base::load_wscale;
  using Base::mma_f16xf16_f32;
  using Base::packed_fp16_to_fp32;
  using Base::packed_fp32_to_fp16;

  struct SharedStorage {
    alignas(16) packed_act_t act_tiles[NUM_PIPELINE_STAGES][NUM_WARPS * WARP_M_TILES * WARP_SIZE];
    alignas(16) packed_wgt_t wgt_tiles[NUM_PIPELINE_STAGES][WARP_N_TILES * WARP_SIZE];
  };

  __device__ __forceinline__ static int packed_block_index(int logical_block_m) {
    return logical_block_m / LOGICAL_BLOCKS_PER_PACKED_BLOCK;
  }

  __device__ __forceinline__ static int packed_warp_base(int logical_block_m) {
    return (logical_block_m % LOGICAL_BLOCKS_PER_PACKED_BLOCK) * NUM_WARPS;
  }

  __device__ __forceinline__ static void load_act_compat(const packed_act_t *ptr,
                                                         int logical_block_m, int k, int K,
                                                         act_warp &out, bool pred) {
    const int laneId = threadIdx.x % WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;
    const int packed_bm = packed_block_index(logical_block_m);
    const int packed_warp = packed_warp_base(logical_block_m) + warpId;

#pragma unroll
    for (int tile = 0; tile < WARP_M_TILES; tile++) {
      out[tile] = load_pred(
        &ptr[(((packed_bm * (K / WARP_K) + k) * PACKED_NUM_WARPS + packed_warp) * WARP_M_TILES +
               tile) *
                WARP_SIZE +
              laneId],
        pred);
    }
  }

  __device__ __forceinline__ static void load_ascale_compat(const packed_ascale_t *ptr,
                                                            int logical_block_m, int group, int K,
                                                            ascale_warp &out, bool pred) {
    const int laneId = threadIdx.x % WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;
    const int packed_bm = packed_block_index(logical_block_m);
    const int packed_warp = packed_warp_base(logical_block_m) + warpId;

#pragma unroll
    for (int pack = 0; pack < ASCALES_NUM_PACKS; pack++) {
      out[pack] = load_pred(
        &ptr[(((packed_bm * (K / WARP_K) + group) * PACKED_NUM_WARPS + packed_warp) *
                ASCALES_NUM_PACKS +
              pack) *
                ASCALES_VALID_LANES +
              laneId],
        pred && laneId < ASCALES_VALID_LANES);
    }
  }

  __device__ __forceinline__ static void load_lora_act_compat(
    const float *ptr, int logical_block_m, int rtile, int rank,
    typename LoraKernel::lora_act_warp &result, bool pred) {
    constexpr int LORA_TILE_SIZE =
      LoraKernel::LORA_M_TILES * LoraKernel::LORA_R_TILES * 8 * WARP_SIZE;

    const int laneId = threadIdx.x % WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;
    const int packed_bm = packed_block_index(logical_block_m);
    const int packed_warp = packed_warp_base(logical_block_m) + warpId;

    const float *ptrlane =
      &ptr[(((packed_bm * (rank / LoraKernel::WARP_R) + rtile) * PACKED_NUM_WARPS + packed_warp) *
              LORA_TILE_SIZE) +
           laneId];

    unrolled_loop<LoraKernel::LORA_M_TILES>([&]<int m>() {
      unrolled_loop<LoraKernel::LORA_R_TILES>([&]<int r>() {
        constexpr int i = m * LoraKernel::LORA_R_TILES + r;
        unrolled_loop<8>([&]<int j>() {
          constexpr int offset = i * 8 * WARP_SIZE + j * WARP_SIZE;
          result[i].data[j] = load_pred(ptrlane + offset, pred);
        });
      });
    });
  }

  template <typename Warp>
  __device__ __forceinline__ static void clear_warp(Warp &warp) {
    warp = {};
  }

  __device__ __forceinline__ static int act_shared_offset(int warpId, int tile, int laneId) {
    return ((warpId * WARP_M_TILES + tile) * WARP_SIZE) + laneId;
  }

  __device__ __forceinline__ static int wgt_shared_offset(int tile, int laneId) {
    return (tile * WARP_SIZE) + laneId;
  }

  __device__ __forceinline__ static void prefetch_act_compat_to_shared(
    SharedStorage &shared, int stage_slot, const packed_act_t *ptr, int logical_block_m, int k,
    int K, bool pred) {
    const int laneId = threadIdx.x % WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;
    const int packed_bm = packed_block_index(logical_block_m);
    const int packed_warp = packed_warp_base(logical_block_m) + warpId;

#pragma unroll
    for (int tile = 0; tile < WARP_M_TILES; tile++) {
      const int shared_offset = act_shared_offset(warpId, tile, laneId);
      const packed_act_t *src =
        &ptr[(((packed_bm * (K / WARP_K) + k) * PACKED_NUM_WARPS + packed_warp) * WARP_M_TILES +
               tile) *
                WARP_SIZE +
              laneId];
      cp_async_ca(&shared.act_tiles[stage_slot][shared_offset], src, pred);
    }
  }

  __device__ __forceinline__ static void prefetch_wgt_to_shared(
    SharedStorage &shared, int stage_slot, const packed_wgt_t *ptr, int k, int K, bool pred) {
    const int laneId = threadIdx.x % WARP_SIZE;
    unused_var(K, false);

#pragma unroll
    for (int tile = 0; tile < WARP_N_TILES; tile++) {
      const int shared_offset = wgt_shared_offset(tile, laneId);
      const packed_wgt_t *src = &ptr[(k * WARP_N_TILES + tile) * WARP_SIZE + laneId];
      cp_async_ca(&shared.wgt_tiles[stage_slot][shared_offset], src, pred);
    }
  }

  __device__ __forceinline__ static void load_act_from_shared(const SharedStorage &shared,
                                                              int stage_slot,
                                                              act_warp &out) {
    const int laneId = threadIdx.x % WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;

#pragma unroll
    for (int tile = 0; tile < WARP_M_TILES; tile++) {
      out[tile] = load<true>(&shared.act_tiles[stage_slot][act_shared_offset(warpId, tile, laneId)]);
    }
  }

  __device__ __forceinline__ static void load_wgt_from_shared(const SharedStorage &shared,
                                                              int stage_slot,
                                                              wgt_warp &out) {
    const int laneId = threadIdx.x % WARP_SIZE;

#pragma unroll
    for (int tile = 0; tile < WARP_N_TILES; tile++) {
      out[tile] = load<true>(&shared.wgt_tiles[stage_slot][wgt_shared_offset(tile, laneId)]);
    }
  }

  __device__ __forceinline__ static void load_act_compat_safe(const packed_act_t *ptr,
                                                              int logical_block_m,
                                                              int k,
                                                              int K,
                                                              act_warp &out,
                                                              bool pred) {
    if (!pred) {
      clear_warp(out);
      return;
    }
    load_act_compat(ptr, logical_block_m, k, K, out, true);
  }

  __device__ __forceinline__ static void load_wgt_safe(const packed_wgt_t *ptr,
                                                       int k,
                                                       int K,
                                                       wgt_warp &out,
                                                       bool pred) {
    if (!pred) {
      clear_warp(out);
      return;
    }
    load_wgt(ptr, k, K, out, true);
  }

  __device__ __forceinline__ static void load_ascale_compat_safe(const packed_ascale_t *ptr,
                                                                 int logical_block_m,
                                                                 int group,
                                                                 int K,
                                                                 ascale_warp &out,
                                                                 bool pred) {
    if (!pred) {
      clear_warp(out);
      return;
    }
    load_ascale_compat(ptr, logical_block_m, group, K, out, true);
  }

  __device__ __forceinline__ static void load_wscale_safe(const packed_wscale_t *ptr,
                                                          int group,
                                                          int N,
                                                          wscale_warp &out,
                                                          bool pred) {
    if (!pred) {
      clear_warp(out);
      return;
    }
    load_wscale(ptr, group, N, out, true);
  }

  __device__ __forceinline__ static void load_lora_act_safe(const float *ptr,
                                                            int rtile,
                                                            typename LoraKernel::lora_act_warp &out,
                                                            bool pred) {
    if (!pred) {
      clear_warp(out);
      return;
    }
    LoraKernel::load_lora_act(ptr, rtile, out, true);
  }

  __device__ __forceinline__ static void load_lora_act_compat_safe(
    const float *ptr, int logical_block_m, int rtile, int rank,
    typename LoraKernel::lora_act_warp &out, bool pred) {
    if (!pred) {
      clear_warp(out);
      return;
    }
    load_lora_act_compat(ptr, logical_block_m, rtile, rank, out, true);
  }

  __device__ __forceinline__ static void load_lora_wgt_safe(
    const packed_fpsum_t *ptr, int rtile, int rank, typename LoraKernel::lora_wgt_warp &out,
    bool pred) {
    if (!pred) {
      clear_warp(out);
      return;
    }
    LoraKernel::load_lora_wgt(ptr, rtile, rank, out, true);
  }

  template <typename Epilogue, bool ACT_UNSIGNED>
  __device__ __forceinline__ static void gemm_w4a4_v2_block_compat(
    const BlockInfo binfo, const packed_act_t *act, const packed_wgt_t *wgt,
    const packed_ascale_t *ascales, const packed_wscale_t *wscales, int M, int N, int K,
    const typename Epilogue::Arguments &epilogueArgs, bool alwaysfalse) {
    act_warp A[NUM_PIPELINE_STAGES];
    wgt_warp W[NUM_PIPELINE_STAGES];
    ascale_warp ascale[NUM_PIPELINE_STAGES];
    wscale_warp wscale[NUM_PIPELINE_STAGES];
    fpsum_warp fpsum;

    for (auto &pack : fpsum) {
      for (int i = 0; i < 4; i++) {
        pack.data[i].x = 0;
        pack.data[i].y = 0;
      }
    }

    const int num_k_tiles = K / WARP_K;

    for (int stage = 0; stage < NUM_PIPELINE_STAGES - 1; stage++) {
      const bool pred = stage < num_k_tiles;
      load_act_compat_safe(act, binfo.bm, stage, K, A[stage], pred);
      load_wgt_safe(wgt, stage, K, W[stage], pred);
      load_ascale_compat_safe(ascales, binfo.bm, stage, K, ascale[stage], pred);
      load_wscale_safe(wscales, stage, N, wscale[stage], pred);
    }

    int dummy = 0;

    for (int k1 = 0; k1 < num_k_tiles; k1 += NUM_PIPELINE_STAGES) {
#pragma unroll
      for (int k2 = 0; k2 < NUM_PIPELINE_STAGES; k2++) {
        const int nextk = k1 + k2 + NUM_PIPELINE_STAGES - 1;
        const int idx = (k2 + NUM_PIPELINE_STAGES - 1) % NUM_PIPELINE_STAGES;
        const bool pred = nextk < num_k_tiles;

        load_act_compat_safe(act, binfo.bm, nextk, K, A[idx], pred);
        load_wgt_safe(wgt, nextk, K, W[idx], pred);
        load_ascale_compat_safe(ascales, binfo.bm, nextk, K, ascale[idx], pred);
        load_wscale_safe(wscales, nextk, N, wscale[idx], pred);

        Parent::template compute<ACT_UNSIGNED>(A[k2], W[k2], ascale[k2], wscale[k2], fpsum);

        if (alwaysfalse) {
          dummy = clock();
        }
      }
    }

    unused_var(dummy, alwaysfalse);
    CHECK_NAN(fpsum, "fpsum_v2_compat");
    Epilogue()(binfo, fpsum, M, N, K, epilogueArgs);
  }

  template <typename Epilogue, bool ACT_UNSIGNED>
  __device__ __forceinline__ static void gemm_w4a4_v2_block_compat_smem(
    SharedStorage &shared, const BlockInfo binfo, const packed_act_t *act,
    const packed_wgt_t *wgt, const packed_ascale_t *ascales, const packed_wscale_t *wscales,
    int M, int N, int K, const typename Epilogue::Arguments &epilogueArgs, bool alwaysfalse) {
    fpsum_warp fpsum;

    for (auto &pack : fpsum) {
      for (int i = 0; i < 4; i++) {
        pack.data[i].x = 0;
        pack.data[i].y = 0;
      }
    }

    const int num_k_tiles = K / WARP_K;
    const int preload_stages = min(NUM_PIPELINE_STAGES, num_k_tiles);

    for (int stage = 0; stage < preload_stages; stage++) {
      prefetch_act_compat_to_shared(shared, stage, act, binfo.bm, stage, K, true);
      prefetch_wgt_to_shared(shared, stage, wgt, stage, K, true);
      cp_async_commit();
    }

    if (preload_stages > 0) {
      cp_async_wait<0>();
      __syncthreads();
    }

    for (int k = 0; k < num_k_tiles; k++) {
      const int stage_slot = k % NUM_PIPELINE_STAGES;
      const int nextk = k + NUM_PIPELINE_STAGES;
      {
        act_warp current_act;
        wgt_warp current_wgt;
        load_act_from_shared(shared, stage_slot, current_act);
        load_wgt_from_shared(shared, stage_slot, current_wgt);

        // All warps must finish consuming the current shared-memory stage before any
        // warp starts overwriting the same stage slot with the next tile.
        __syncthreads();

        if (nextk < num_k_tiles) {
          prefetch_act_compat_to_shared(shared, stage_slot, act, binfo.bm, nextk, K, true);
          prefetch_wgt_to_shared(shared, stage_slot, wgt, nextk, K, true);
          cp_async_commit();
        }

        ascale_warp current_ascale;
        wscale_warp current_wscale;
        load_ascale_compat_safe(ascales, binfo.bm, k, K, current_ascale, true);
        load_wscale_safe(wscales, k, N, current_wscale, true);

        Parent::template compute<ACT_UNSIGNED>(
          current_act, current_wgt, current_ascale, current_wscale, fpsum);
      }

      if (nextk < num_k_tiles) {
        cp_async_wait<0>();
        __syncthreads();
      }
    }

    unused_var(alwaysfalse, alwaysfalse);
    CHECK_NAN(fpsum, "fpsum_v2_compat_smem");
    Epilogue()(binfo, fpsum, M, N, K, epilogueArgs);
  }

  template <typename Epilogue, bool ACT_UNSIGNED>
  struct gemm_w4a4_v2_kernel {
    static constexpr int MIN_ARCH = 750;
    using SharedStorage = typename GEMM_W4A4_V2<Config, NumPipelineStages>::SharedStorage;
    static constexpr bool USE_COMPAT_SMEM_OPT = false;

    __device__ void operator()(SharedStorage &shared, const packed_act_t *act,
                               const packed_wgt_t *wgt,
                               const packed_ascale_t *ascales, const packed_wscale_t *wscales,
                               int M, int N, int K, typename Epilogue::Arguments epilogueArgs,
                               bool swapBlockXY, bool alwaysfalse) {
      BlockInfo binfo = {
        .bm = static_cast<int>(blockIdx.x),
        .bn = static_cast<int>(blockIdx.y),
        .numBlocksM = static_cast<int>(gridDim.x),
        .numBlocksN = static_cast<int>(gridDim.y),
      };

      if (swapBlockXY) {
        std::swap(binfo.bm, binfo.bn);
        std::swap(binfo.numBlocksM, binfo.numBlocksN);
      }

      const int bm = binfo.bm;
      const int bn = binfo.bn;

      (void)shared;
      if constexpr (USE_PACKED_LAYOUT_COMPAT) {
        if constexpr (USE_COMPAT_SMEM_OPT && NumPipelineStages >= 2) {
          GEMM_W4A4_V2<Config, NumPipelineStages>::template gemm_w4a4_v2_block_compat_smem<
            Epilogue, ACT_UNSIGNED>(
            shared,
            binfo,
            act,
            wgt + bn * (K / WARP_K) * WARP_N_TILES * WARP_SIZE,
            ascales,
            wscales + bn * (K / WARP_K) * WSCALES_NUM_PACKS * WSCALES_VALID_LANES,
            M,
            N,
            K,
            epilogueArgs,
            alwaysfalse);
        } else {
          GEMM_W4A4_V2<Config, NumPipelineStages>::template gemm_w4a4_v2_block_compat<
            Epilogue, ACT_UNSIGNED>(
            binfo,
            act,
            wgt + bn * (K / WARP_K) * WARP_N_TILES * WARP_SIZE,
            ascales,
            wscales + bn * (K / WARP_K) * WSCALES_NUM_PACKS * WSCALES_VALID_LANES,
            M,
            N,
            K,
            epilogueArgs,
            alwaysfalse);
        }
      } else {
        Parent::template gemm_w4a4_block<Epilogue, ACT_UNSIGNED, false>(
          binfo,
          act + bm * (K / WARP_K) * NUM_WARPS * WARP_M_TILES * WARP_SIZE,
          wgt + bn * (K / WARP_K) * WARP_N_TILES * WARP_SIZE,
          ascales + bm * (K / WARP_K) * NUM_WARPS * ASCALES_NUM_PACKS * ASCALES_VALID_LANES,
          wscales + bn * (K / WARP_K) * WSCALES_NUM_PACKS * WSCALES_VALID_LANES,
          M,
          N,
          K,
          epilogueArgs,
          alwaysfalse);
      }
    }
  };

  struct EpilogueLoraUpRuntime {
    using Arguments = typename LoraKernel::EpilogueLoraUp::Arguments;

    __device__ __forceinline__ static void apply_lora_up(
      fpsum_warp &fpsum, const float *act, const packed_fpsum_t *wgt,
      const typename LoraKernel::scale_t &scales, int rank, bool alwaysfalse) {
      constexpr int NUM_STAGES = NUM_PIPELINE_STAGES;

      typename LoraKernel::lora_act_warp lora_act[NUM_STAGES];
      typename LoraKernel::lora_wgt_warp lora_wgt[NUM_STAGES];

      int dummy = 0;
      const int num_rank_tiles = rank / LoraKernel::WARP_R;

#pragma unroll
      for (int stage = 0; stage < NUM_STAGES - 1; stage++) {
        const bool pred = stage < num_rank_tiles;
        load_lora_act_safe(act, stage, lora_act[stage], pred);
        load_lora_wgt_safe(wgt, stage, rank, lora_wgt[stage], pred);
      }

      f32psum_warp f32psum = packed_fp16_to_fp32(fpsum);

      auto compute = [&scales](const typename LoraKernel::lora_act_warp &A,
                               const typename LoraKernel::lora_wgt_warp &W,
                               f32psum_warp &accum, int rtile) ALWAYSINLINE {
        typename LoraKernel::lora_act16_warp A_fp16;

#pragma unroll
        for (int m = 0; m < LoraKernel::LORA_M_TILES; m++) {
#pragma unroll
          for (int r = 0; r < LoraKernel::LORA_R_TILES; r++) {
            packed_f32psum_t pack = A[m * LoraKernel::LORA_R_TILES + r];
#pragma unroll
            for (int j = 0; j < 8; j++) {
              pack.data[j] *= scales[rtile * LoraKernel::LORA_R_TILES + r];
            }
            A_fp16[m * LoraKernel::LORA_R_TILES + r] = packed_fp32_to_fp16(pack);
          }
        }

#pragma unroll
        for (int m = 0; m < LoraKernel::LORA_M_TILES; m++) {
#pragma unroll
          for (int n = 0; n < LoraKernel::LORA_N_TILES; n++) {
#pragma unroll
            for (int r = 0; r < LoraKernel::LORA_R_TILES; r++) {
              accum[m * WARP_N_TILES + n] = mma_f16xf16_f32(
                A_fp16[m * LoraKernel::LORA_R_TILES + r],
                W[n * LoraKernel::LORA_R_TILES + r],
                accum[m * WARP_N_TILES + n]);
            }
          }
        }
      };

      for (int k1 = 0; k1 < num_rank_tiles; k1 += NUM_STAGES) {
#pragma unroll
        for (int k2 = 0; k2 < NUM_STAGES; k2++) {
          if (k1 + k2 >= num_rank_tiles) {
            break;
          }

          const int nextk = k1 + k2 + NUM_STAGES - 1;
          const int idx = (k2 + NUM_STAGES - 1) % NUM_STAGES;
          const bool pred = nextk < num_rank_tiles;

          if (alwaysfalse) {
            act += kernels::bit_cast<int>(lora_act[k2][0].data[0]);
            dummy = clock();
          }

          load_lora_act_safe(act, nextk, lora_act[idx], pred);
          load_lora_wgt_safe(wgt, nextk, rank, lora_wgt[idx], pred);

          compute(lora_act[k2], lora_wgt[k2], f32psum, k1 + k2);
        }
      }

#pragma unroll
      for (int stage = 0; stage < NUM_STAGES - 1; stage++) {
#pragma unroll
        for (auto &&data : lora_act[stage]) {
#pragma unroll
          for (int i = 0; i < 8; i++) {
            dummy ^= kernels::bit_cast<int>(data.data[i]);
          }
        }
#pragma unroll
        for (auto &&data : lora_wgt[stage]) {
#pragma unroll
          for (int i = 0; i < 4; i++) {
            dummy ^= kernels::bit_cast<int>(data.data[i]);
          }
        }
      }

      unused_var(dummy, alwaysfalse);
      fpsum = packed_fp32_to_fp16(f32psum);
    }

    __device__ __forceinline__ void operator()(const BlockInfo binfo, fpsum_warp &fpsum, int M,
                                               int N, int K, const Arguments &args) {
      const int bm = binfo.bm;
      const int bn = binfo.bn;

      (void)M;
      (void)N;
      (void)K;
      CHECK_NAN(fpsum, "fpsum");

      apply_lora_up(
        fpsum,
        args.lora_act + bm * (args.rank / LoraKernel::WARP_R) *
                          (NUM_WARPS * LoraKernel::LORA_M_TILES * LoraKernel::LORA_R_TILES * 8 *
                           WARP_SIZE),
        args.lora_wgt_up + bn * (BLOCK_N / 16) * (args.rank / 16) * WARP_SIZE,
        args.scales,
        args.rank,
        args.alwaysfalse);

      CHECK_NAN(fpsum, "fpsum");
    }
  };

  struct EpilogueLoraUpCompat {
    using Arguments = typename LoraKernel::EpilogueLoraUp::Arguments;

    // The LoRA up path reuses the original packed lora_act tensor layout from the quantizer.
    // When v2 shrinks the logical row block on Ada, only the base GEMM block shape changes; the
    // LoRA activations must still be read with the original packed warp-group stride.
    __device__ __forceinline__ static void apply_lora_up(
      fpsum_warp &fpsum, const float *act, const packed_fpsum_t *wgt,
      const typename LoraKernel::scale_t &scales, int rank, int logical_block_m,
      bool alwaysfalse) {
      constexpr int NUM_STAGES = NUM_PIPELINE_STAGES;

      typename LoraKernel::lora_act_warp lora_act[NUM_STAGES];
      typename LoraKernel::lora_wgt_warp lora_wgt[NUM_STAGES];

      int dummy = 0;
      const int num_rank_tiles = rank / LoraKernel::WARP_R;

#pragma unroll
      for (int stage = 0; stage < NUM_STAGES - 1; stage++) {
        const bool pred = stage < num_rank_tiles;
        load_lora_act_compat_safe(act, logical_block_m, stage, rank, lora_act[stage], pred);
        load_lora_wgt_safe(wgt, stage, rank, lora_wgt[stage], pred);
      }

      f32psum_warp f32psum = packed_fp16_to_fp32(fpsum);

      auto compute = [&scales](const typename LoraKernel::lora_act_warp &A,
                               const typename LoraKernel::lora_wgt_warp &W,
                               f32psum_warp &accum, int rtile) ALWAYSINLINE {
        typename LoraKernel::lora_act16_warp A_fp16;

#pragma unroll
        for (int m = 0; m < LoraKernel::LORA_M_TILES; m++) {
#pragma unroll
          for (int r = 0; r < LoraKernel::LORA_R_TILES; r++) {
            packed_f32psum_t pack = A[m * LoraKernel::LORA_R_TILES + r];
#pragma unroll
            for (int j = 0; j < 8; j++) {
              pack.data[j] *= scales[rtile * LoraKernel::LORA_R_TILES + r];
            }
            A_fp16[m * LoraKernel::LORA_R_TILES + r] = packed_fp32_to_fp16(pack);
          }
        }

#pragma unroll
        for (int m = 0; m < LoraKernel::LORA_M_TILES; m++) {
#pragma unroll
          for (int n = 0; n < LoraKernel::LORA_N_TILES; n++) {
#pragma unroll
            for (int r = 0; r < LoraKernel::LORA_R_TILES; r++) {
              accum[m * WARP_N_TILES + n] = mma_f16xf16_f32(
                A_fp16[m * LoraKernel::LORA_R_TILES + r],
                W[n * LoraKernel::LORA_R_TILES + r],
                accum[m * WARP_N_TILES + n]);
            }
          }
        }
      };

      for (int k1 = 0; k1 < num_rank_tiles; k1 += NUM_STAGES) {
#pragma unroll
        for (int k2 = 0; k2 < NUM_STAGES; k2++) {
          if (k1 + k2 >= num_rank_tiles) {
            break;
          }

          const int nextk = k1 + k2 + NUM_STAGES - 1;
          const int idx = (k2 + NUM_STAGES - 1) % NUM_STAGES;
          const bool pred = nextk < num_rank_tiles;

          if (alwaysfalse) {
            act += kernels::bit_cast<int>(lora_act[k2][0].data[0]);
            dummy = clock();
          }

          load_lora_act_compat_safe(act, logical_block_m, nextk, rank, lora_act[idx], pred);
          load_lora_wgt_safe(wgt, nextk, rank, lora_wgt[idx], pred);

          compute(lora_act[k2], lora_wgt[k2], f32psum, k1 + k2);
        }
      }

#pragma unroll
      for (int stage = 0; stage < NUM_STAGES - 1; stage++) {
#pragma unroll
        for (auto &&data : lora_act[stage]) {
#pragma unroll
          for (int i = 0; i < 8; i++) {
            dummy ^= kernels::bit_cast<int>(data.data[i]);
          }
        }
#pragma unroll
        for (auto &&data : lora_wgt[stage]) {
#pragma unroll
          for (int i = 0; i < 4; i++) {
            dummy ^= kernels::bit_cast<int>(data.data[i]);
          }
        }
      }

      unused_var(dummy, alwaysfalse);
      fpsum = packed_fp32_to_fp16(f32psum);
    }

    __device__ __forceinline__ void operator()(const BlockInfo binfo, fpsum_warp &fpsum, int M,
                                               int N, int K, const Arguments &args) {
      const int bn = binfo.bn;

      (void)M;
      (void)N;
      (void)K;
      CHECK_NAN(fpsum, "fpsum");

      apply_lora_up(
        fpsum,
        args.lora_act,
        args.lora_wgt_up + bn * (BLOCK_N / 16) * (args.rank / 16) * WARP_SIZE,
        args.scales,
        args.rank,
        binfo.bm,
        args.alwaysfalse);

      CHECK_NAN(fpsum, "fpsum");
    }
  };
};

template <typename Config, int NumPipelineStages>
class GEMM_W4A4_V2_Launch {
  using GEMM = GEMM_W4A4_V2<Config, NumPipelineStages>;
  using LoraKernel = Lora<Config>;

  using packed_act_t = typename GEMM::packed_act_t;
  using packed_wgt_t = typename GEMM::packed_wgt_t;
  using packed_ascale_t = typename GEMM::packed_ascale_t;
  using packed_wscale_t = typename GEMM::packed_wscale_t;
  using packed_fpsum_t = typename GEMM::packed_fpsum_t;
  using half_t = typename GEMM::half_t;

 public:
  static void gemm_w4a4(Tensor act, Tensor wgt, Tensor out, Tensor ascales, Tensor wscales,
                        Tensor lora_act_in, Tensor lora_up, Tensor bias, bool act_unsigned,
                        float alpha, Tensor wcscales) {
    int M = act.numel() / act.shape[-1];
    int N = wgt.shape[0];
    int K = act.shape[-1] * 2;
    assert(K == wgt.shape[1] * 2);
    assert(out.valid());

    int actualM = out.numel() / out.shape[-1];
    int actualN = out.shape[-1];
    assert(actualM <= M && M - actualM < GEMM::BLOCK_M);
    assert(actualN <= N && N - actualN < GEMM::BLOCK_N);
    assert(alpha == 1.0f);
    auto launch = [&]<typename Epilogue>(typename Epilogue::Arguments args) {
      assert(M % GEMM::BLOCK_M == 0);
      assert(N % GEMM::BLOCK_N == 0);

      dim3 grid(M / GEMM::BLOCK_M, N / GEMM::BLOCK_N);
      bool swapBlockMN = M > N * 2;
      if (swapBlockMN) {
        std::swap(grid.x, grid.y);
      }

      dispatchBool(act_unsigned, [&]<bool ACT_UNSIGNED>() {
        using kernel_t = typename GEMM::template gemm_w4a4_v2_kernel<Epilogue, ACT_UNSIGNED>;
        auto func = invoke_kernel_v2<kernel_t, const packed_act_t *, const packed_wgt_t *,
                                     const packed_ascale_t *, const packed_wscale_t *, int, int,
                                     int, typename Epilogue::Arguments, bool, bool>;

        func<<<grid, GEMM::WARP_SIZE * GEMM::NUM_WARPS, 0, getCurrentCUDAStream()>>>(
          act.data_ptr<packed_act_t>(),
          wgt.data_ptr<packed_wgt_t>(),
          ascales.data_ptr<packed_ascale_t>(),
          wscales.data_ptr<packed_wscale_t>(),
          M,
          N,
          K,
          args,
          swapBlockMN,
          false);
        checkCUDA(cudaGetLastError());
      });
    };

    auto launch_bias = [&]<typename NextEpilogue>(typename NextEpilogue::Arguments nextArgs) {
      assert(!bias.valid() || bias.numel() == N);
      assert(!wcscales.valid() || wcscales.numel() == N);

      dispatchBool(bias.valid(), [&]<bool USE_BIAS>() {
        dispatchBool(wcscales.valid(), [&]<bool USE_SCALE>() {
          using EpilogueBias = typename GEMM::template EpilogueBias<USE_BIAS, USE_SCALE>;
          using Epilogue = typename GEMM::template EpilogueCombination<EpilogueBias,
                                                                       NextEpilogue,
                                                                       typename GEMM::EpilogueNop>;
          launch.template operator()<Epilogue>({typename EpilogueBias::Arguments{
                                                 .bias = USE_BIAS
                                                   ? bias.data_ptr<packed_wscale_t>()
                                                   : nullptr,
                                                 .scale = USE_SCALE
                                                   ? wcscales.data_ptr<packed_wscale_t>()
                                                   : nullptr,
                                               },
                                               nextArgs,
                                               {}});
        });
      });
    };

    auto launch_lora = [&]<typename NextEpilogue>(typename NextEpilogue::Arguments nextArgs) {
      assert(lora_up.valid() == lora_act_in.valid());

      int rank = lora_up.valid() ? lora_up.shape[1] : 0;
      if (rank == 0) {
        return launch_bias.template operator()<NextEpilogue>(nextArgs);
      }

      assert(rank % 16 == 0);
      assert(lora_up.shape[0] == N);
      assert(lora_act_in.shape[0] == M);
      assert(lora_act_in.shape[1] == rank);

      typename LoraKernel::scale_t scales{};
      for (int group = 0; group < rank / 16; group++) {
        scales[group] = 1.0f;
      }

      if constexpr (GEMM::USE_PACKED_LAYOUT_COMPAT) {
        using Epilogue = typename GEMM::template EpilogueCombination<
          typename GEMM::EpilogueLoraUpCompat,
          NextEpilogue,
          typename GEMM::EpilogueNop>;

        return launch_bias.template operator()<Epilogue>(
          {typename GEMM::EpilogueLoraUpCompat::Arguments{
             .lora_act = lora_act_in.data_ptr<float>(),
             .lora_wgt_up = lora_up.data_ptr<packed_fpsum_t>(),
             .rank = rank,
             .scales = scales,
             .alwaysfalse = false,
           },
           nextArgs,
           {}});
      }

      using Epilogue = typename GEMM::template EpilogueCombination<
        typename GEMM::EpilogueLoraUpRuntime,
        NextEpilogue,
        typename GEMM::EpilogueNop>;

      return launch_bias.template operator()<Epilogue>(
        {typename LoraKernel::EpilogueLoraUp::Arguments{
           .lora_act = lora_act_in.data_ptr<float>(),
           .lora_wgt_up = lora_up.data_ptr<packed_fpsum_t>(),
           .rank = rank,
           .scales = scales,
           .alwaysfalse = false,
         },
         nextArgs,
         {}});
    };

    launch_lora.template operator()<typename GEMM::EpilogueDefault>(
      typename GEMM::EpilogueDefault::Arguments{
        .out = out.data_ptr<half_t>(),
        .actualM = actualM,
        .actualN = actualN,
      });
  }
};

}  // namespace svdq::kernels
