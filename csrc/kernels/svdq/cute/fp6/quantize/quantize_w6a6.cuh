#pragma once

#include <algorithm>
#include <cstdint>

#include <cuda_bf16.h>

#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "common.h"
#include "../gemm/gemm_w6a6.cuh"

namespace svdq::cute_kernels {

using namespace cute;

// Fused SVDQ W6A6 activation producer: MX block-scaled E2M3 quantization plus the BF16 LoRA-down
// projection, from a single read of the activation tile.
//
//   asf[m,g] = ue8m0(2^ceil(log2(amax_{k in g} |X[m,k] / s[k]| / 7.5)))       g = k / 32
//   qA[m,k]  = satfinite_e2m3((X[m,k] / s[k]) / 2^(asf[m,g]-127))
//   D[m,r]   = BF16(sum_k X[m,k] * lora_down[r,k])
//
// `lora_down` is L2/s, so D consumes the *raw* activation while qA consumes the smoothed one; both
// come from the same shared-memory tile. D accumulates in FP32 across the whole K loop and is
// rounded to BF16 exactly once. There is no standalone down-projection GEMM anywhere.
//
// One CTA owns 128 rows and a slice of the K extent: it streams 64-wide K tiles, computes each
// 32-wide block's amax with a two-shuffle warp reduction (four consecutive lanes own one block),
// writes the UE8M0 scale in the tile-blocked layout and packs the codes, all in the same pass, and
// accumulates the BF16 MMA for D. M must be a multiple of 128 and K a multiple of 128. The packed
// destination keeps four codes per three bytes, so one 8-element chunk of a row is exactly six
// bytes and never straddles a byte boundary.

constexpr int kW6A6QuantThreads = 256;
constexpr int kW6A6QuantBM = 128;
constexpr int kW6A6QuantBK = 64;
constexpr int kW6A6SfVec = 32;
constexpr float kW6A6CodeMax = 7.5f;

using W6A6QuantSmemAtom = UMMA::Layout_K_SW128_Atom<cutlass::bfloat16_t>;
using W6A6QuantSmemLayoutX =
  decltype(tile_to_shape(W6A6QuantSmemAtom{}, Shape<Int<kW6A6QuantBM>, Int<kW6A6QuantBK>>{}));

union W6A6Vec8BF16 {
  uint4 raw;
  cutlass::bfloat16_t elem[8];
};

/**
 * UE8M0 code of the smallest power-of-two scale that keeps a 32-wide block inside [-7.5, 7.5].
 *
 * :param amax: Largest smoothed magnitude in the block.
 * :return: Exponent byte with bias 127; 127 (scale 1.0) for an all-zero block.
 */
__device__ __forceinline__ uint8_t w6a6_sf_code(float amax) {
  const float target = amax / kW6A6CodeMax;
  if (!(target > 0.0f)) {
    return 127;
  }
  int exponent = 0;
  const float mantissa = frexpf(target, &exponent);
  // frexp returns target = mantissa * 2^exponent with mantissa in [0.5, 1), so ceil(log2(target))
  // is exponent unless target is an exact power of two (mantissa == 0.5).
  const int shift = mantissa == 0.5f ? exponent - 1 : exponent;
  const int code = shift + 127;
  return static_cast<uint8_t>(code < 1 ? 1 : (code > 254 ? 254 : code));
}

// Dequantization multiplier of a UE8M0 code, i.e. 2^(code-127) read straight out of the exponent
// bits.
__device__ __forceinline__ float w6a6_sf_scale(uint8_t code) {
  return __uint_as_float(static_cast<uint32_t>(code) << 23);
}

// Reciprocal of w6a6_sf_scale without a division: 2^(127-code).
__device__ __forceinline__ float w6a6_sf_inverse(uint8_t code) {
  return __uint_as_float(static_cast<uint32_t>(254 - code) << 23);
}

// Round eight smoothed FP32 values to E2M3 and pack them as six bytes, four codes per three bytes
// (OCP MX packing: code j occupies bits [6j, 6j+6) of the little-endian stream).
__device__ __forceinline__ void w6a6_store_e2m3x8(uint8_t *output, float const *values,
                                                  float inv_scale) {
  cutlass::Array<float, 4> source;
  uint32_t words[2];
#pragma unroll
  for (int half = 0; half < 2; ++half) {
#pragma unroll
    for (int i = 0; i < 4; ++i) source[i] = values[half * 4 + i] * inv_scale;
    auto codes =
      cutlass::NumericArrayConverter<cutlass::detail::float_e2m3_unpack8bits_t, float, 4,
                                     cutlass::FloatRoundStyle::round_to_nearest>()(source);
    words[half] = *reinterpret_cast<uint32_t const *>(codes.data());
  }
  // One container byte per code; fold four 6-bit codes into 24 bits per word.
  uint32_t packed[2];
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const uint32_t w = words[half];
    packed[half] =
      (w & 0x3Fu) | ((w >> 8 & 0x3Fu) << 6) | ((w >> 16 & 0x3Fu) << 12) | ((w >> 24 & 0x3Fu) << 18);
  }
  // Six bytes: the low 24 bits of both words. The caller guarantees an even destination offset, so
  // three 16-bit stores cover the chunk without straddling another chunk.
  const uint32_t lo = packed[0];
  const uint32_t hi = packed[1];
  *reinterpret_cast<uint16_t *>(output) = static_cast<uint16_t>(lo & 0xFFFFu);
  *reinterpret_cast<uint16_t *>(output + 2) =
    static_cast<uint16_t>((lo >> 16) | ((hi & 0xFFu) << 8));
  *reinterpret_cast<uint16_t *>(output + 4) = static_cast<uint16_t>((hi >> 8) & 0xFFFFu);
}

template <int kRank>
struct W6A6QuantTraits {
  static constexpr int kBM = kW6A6QuantBM;
  static constexpr int kBK = kW6A6QuantBK;
  static constexpr int kThreads = kW6A6QuantThreads;
  static constexpr int kXBytes = kBM * kBK * 2;
  static constexpr int kLBytes = kRank * kBK * 2;
  static constexpr int kSmemBytes = kXBytes + kLBytes;

  using Element = cutlass::bfloat16_t;
  using MmaAtom = MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>;
  using AtomLayoutMNK = Layout<Shape<_4, _2, _1>>;
  using PermutationMNK = Tile<Int<kBM>, Int<(kRank > 0 ? kRank : 16)>, Int<kBK>>;
  using TiledMma = decltype(make_tiled_mma(MmaAtom{}, AtomLayoutMNK{}, PermutationMNK{}));
  using SmemLayoutX = W6A6QuantSmemLayoutX;
  using SmemLayoutL =
    decltype(tile_to_shape(W6A6QuantSmemAtom{}, Shape<Int<(kRank > 0 ? kRank : 8)>, Int<kBK>>{}));
  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
};

template <int kRank>
__global__ void __launch_bounds__(kW6A6QuantThreads, 1)
  quantize_w6a6_kernel(cutlass::bfloat16_t const *__restrict__ input,
                       cutlass::bfloat16_t const *__restrict__ smooth,
                       cutlass::bfloat16_t const *__restrict__ lora_down,
                       uint8_t *__restrict__ output, uint8_t *__restrict__ asf,
                       cutlass::bfloat16_t *__restrict__ lora_act, float *__restrict__ lora_scratch,
                       int K, int tiles_per_split) {
  using Traits = W6A6QuantTraits<kRank>;
  constexpr int kBM = Traits::kBM;
  constexpr int kBK = Traits::kBK;
  const int tid = threadIdx.x;
  const int m0 = blockIdx.x * kBM;
  const int nKt_total = K / kBK;
  const int kt_begin = blockIdx.y * tiles_per_split;
  const int kt_end = min(nKt_total, kt_begin + tiles_per_split);
  if (kt_begin >= kt_end) return;

  extern __shared__ __align__(1024) uint8_t shm[];
  auto *x_bytes = reinterpret_cast<cutlass::bfloat16_t *>(shm);
  auto *l_bytes = reinterpret_cast<cutlass::bfloat16_t *>(shm + Traits::kXBytes);

  auto sX = make_tensor(make_smem_ptr(x_bytes), typename Traits::SmemLayoutX{});
  auto sL = make_tensor(make_smem_ptr(l_bytes), typename Traits::SmemLayoutL{});

  // Row base of the packed scale rows; each chunk below indexes its own 32-wide group.
  const int64_t out_row_bytes = static_cast<int64_t>(K) * 3 / 4;

  typename Traits::TiledMma mma;
  auto thr_mma = mma.get_thread_slice(tid);
  auto tCrD = partition_fragment_C(mma, Shape<Int<kBM>, Int<(kRank > 0 ? kRank : 16)>>{});
  clear(tCrD);
  auto tCrX = thr_mma.partition_fragment_A(sX);
  auto tCrL = thr_mma.partition_fragment_B(sL);
  auto s2r_copy_x = make_tiled_copy_A(typename Traits::SmemCopyAtom{}, mma);
  auto s2r_copy_l = make_tiled_copy_B(typename Traits::SmemCopyAtom{}, mma);
  auto s2r_thr_x = s2r_copy_x.get_thread_slice(tid);
  auto s2r_thr_l = s2r_copy_l.get_thread_slice(tid);
  auto tCsX = s2r_thr_x.partition_S(as_position_independent_swizzle_tensor(sX));
  auto tCsL = s2r_thr_l.partition_S(as_position_independent_swizzle_tensor(sL));
  auto tCrX_view = s2r_thr_x.retile_D(tCrX);
  auto tCrL_view = s2r_thr_l.retile_D(tCrL);

  constexpr int kChunksPerRow = kBK / 8;
  constexpr int kXChunks = kBM * kChunksPerRow;
  constexpr int kLChunks = (kRank > 0 ? kRank : 1) * kChunksPerRow;
  constexpr int kXPerThread = (kXChunks + kW6A6QuantThreads - 1) / kW6A6QuantThreads;
  constexpr int kLPerThread = (kLChunks + kW6A6QuantThreads - 1) / kW6A6QuantThreads;

  // Software pipeline: tile kt+1 is loaded into a second register set right after tile kt reaches
  // shared memory, so its HBM latency overlaps the MMA and the quantization of tile kt.
  uint4 x_regs[2][kXPerThread];
  uint4 l_regs[2][kLPerThread];
  auto load_tile = [&](int kt, auto buf_tag) {
    constexpr int buf = decltype(buf_tag)::value;
    const int k0 = kt * kBK;
#pragma unroll
    for (int i = 0; i < kXPerThread; ++i) {
      const int chunk = tid + i * kW6A6QuantThreads;
      if constexpr (kXChunks % kW6A6QuantThreads != 0) {
        if (chunk >= kXChunks) break;
      }
      const int row = chunk / kChunksPerRow;
      const int col = (chunk % kChunksPerRow) * 8;
      x_regs[buf][i] =
        *reinterpret_cast<uint4 const *>(input + static_cast<int64_t>(m0 + row) * K + k0 + col);
    }
    if constexpr (kRank > 0) {
#pragma unroll
      for (int i = 0; i < kLPerThread; ++i) {
        const int chunk = tid + i * kW6A6QuantThreads;
        if constexpr (kLChunks % kW6A6QuantThreads != 0) {
          if (chunk >= kLChunks) break;
        }
        const int row = chunk / kChunksPerRow;
        const int col = (chunk % kChunksPerRow) * 8;
        l_regs[buf][i] =
          *reinterpret_cast<uint4 const *>(lora_down + static_cast<int64_t>(row) * K + k0 + col);
      }
    }
  };

  auto process_tile = [&](int kt, auto buf_tag) {
    constexpr int buf = decltype(buf_tag)::value;
    const int k0 = kt * kBK;
#pragma unroll
    for (int i = 0; i < kXPerThread; ++i) {
      const int chunk = tid + i * kW6A6QuantThreads;
      if constexpr (kXChunks % kW6A6QuantThreads != 0) {
        if (chunk >= kXChunks) break;
      }
      const int row = chunk / kChunksPerRow;
      const int col = (chunk % kChunksPerRow) * 8;
      *reinterpret_cast<uint4 *>(&sX(row, col)) = x_regs[buf][i];
    }
    if constexpr (kRank > 0) {
#pragma unroll
      for (int i = 0; i < kLPerThread; ++i) {
        const int chunk = tid + i * kW6A6QuantThreads;
        if constexpr (kLChunks % kW6A6QuantThreads != 0) {
          if (chunk >= kLChunks) break;
        }
        const int row = chunk / kChunksPerRow;
        const int col = (chunk % kChunksPerRow) * 8;
        *reinterpret_cast<uint4 *>(&sL(row, col)) = l_regs[buf][i];
      }
    }
    if (kt + 1 < kt_end) load_tile(kt + 1, Int<buf ^ 1>{});
    __syncthreads();
    if constexpr (kRank > 0) {
      copy(s2r_copy_x, tCsX, tCrX_view);
      copy(s2r_copy_l, tCsL, tCrL_view);
      cute::gemm(mma, tCrX, tCrL, tCrD);
    }
#pragma unroll
    for (int i = 0; i < kXPerThread; ++i) {
      const int chunk = tid + i * kW6A6QuantThreads;
      if constexpr (kXChunks % kW6A6QuantThreads != 0) {
        if (chunk >= kXChunks) break;
      }
      const int row = chunk / kChunksPerRow;
      const int col = (chunk % kChunksPerRow) * 8;
      const int64_t offset = static_cast<int64_t>(m0 + row) * out_row_bytes + (k0 + col) * 3 / 4;
      W6A6Vec8BF16 x, s;
      x.raw = x_regs[buf][i];
      s.raw = *reinterpret_cast<uint4 const *>(smooth + k0 + col);
      float values[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        values[j] = static_cast<float>(x.elem[j]) / static_cast<float>(s.elem[j]);
      }
      // One 32-wide scale group spans exactly four consecutive lanes of this warp, so the block
      // amax is a two-shuffle reduction. That removes the separate amax pass over the activation
      // (and its kernel launch) entirely: no per-token second-level scale exists to reduce for.
      float amax = 0.0f;
#pragma unroll
      for (int j = 0; j < 8; ++j) amax = fmaxf(amax, fabsf(values[j]));
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));
      const uint8_t sf = w6a6_sf_code(amax);
      if ((tid & 3) == 0) {
        asf[tile_blocked_sf_offset(m0 + row, (k0 + col) / kW6A6SfVec, K / 128)] = sf;
      }
      const float inv_scale = w6a6_sf_inverse(sf);
      w6a6_store_e2m3x8(output + offset, values, inv_scale);
    }
    __syncthreads();
  };

  load_tile(kt_begin, Int<0>{});
  for (int kt = kt_begin; kt < kt_end; kt += 2) {
    process_tile(kt, Int<0>{});
    if (kt + 1 < kt_end) process_tile(kt + 1, Int<1>{});
  }

  if constexpr (kRank > 0) {
    auto cD = make_identity_tensor(Shape<Int<kBM>, Int<kRank>>{});
    auto tScD = thr_mma.partition_C(cD);
#pragma unroll
    for (int i = 0; i < size(tCrD); ++i) {
      const int row = m0 + get<0>(tScD(i));
      const int col = get<1>(tScD(i));
      const int64_t index = static_cast<int64_t>(row) * kRank + col;
      if (lora_scratch != nullptr) {
        // Split-K: partial D fragments are combined with relaxed FP32 atomics and rounded once by
        // the finalize kernel, so D is still produced inside this operator.
        atomicAdd(lora_scratch + index, tCrD(i));
      } else {
        lora_act[index] = cutlass::bfloat16_t(__float2bfloat16_rn(tCrD(i)));
      }
    }
  }
}

// Round the FP32 split-K scratch into the BF16 D tensor. Only used when more than one K split
// contributes to the same row block.
static __global__ void quantize_w6a6_finalize_kernel(float const *__restrict__ lora_scratch,
                                                     cutlass::bfloat16_t *__restrict__ lora_act,
                                                     int64_t count) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    lora_act[index] = cutlass::bfloat16_t(__float2bfloat16_rn(lora_scratch[index]));
  }
}

// Number of K splits for the fused producer, chosen to expose enough CTAs to hide HBM latency. This
// is shared with the host dispatcher, which allocates the split-K scratch when it exceeds 1.
inline int w6a6_quant_k_splits(int M, int K) {
  const int row_blocks = std::max(1, M / kW6A6QuantBM);
  const int nKt = std::max(1, K / kW6A6QuantBK);
  const int target = (256 + row_blocks - 1) / row_blocks;
  return std::max(1, std::min(nKt, target));
}

/**
 * Launch the fused W6A6 activation producer for one rank.
 *
 * :param input: BF16 activation [M,K].
 * :param smooth: BF16 per-input-channel divisor [K].
 * :param lora_down: BF16 [rank,K] down projection already divided by smooth; null for rank 0.
 * :param output: Packed E2M3 destination [M, K*3/4].
 * :param asf: UE8M0 block scales [M, K/32].
 * :param lora_act: BF16 LoRA-down output [M,rank]; null for rank 0.
 * :param M: Row count, a multiple of 128.
 * :param K: Channel count, a multiple of 128.
 * :param stream: CUDA stream for the launch.
 */
template <int kRank>
void launch_quantize_w6a6(void const *input, void const *smooth, void const *lora_down,
                          void *output, void *asf, void *lora_act, float *lora_scratch, int M,
                          int K, cudaStream_t stream) {
  using Traits = W6A6QuantTraits<kRank>;
  // Rank 0 has no D to reduce, so it always splits K for the extra CTAs; ranked runs split exactly
  // when the dispatcher supplied the FP32 scratch that the finalize pass reduces.
  const int splits = kRank > 0 && lora_scratch == nullptr ? 1 : w6a6_quant_k_splits(M, K);
  float *scratch = splits > 1 ? lora_scratch : nullptr;
  const int nKt = K / kW6A6QuantBK;
  const int tiles_per_split = (nKt + splits - 1) / splits;
  quantize_w6a6_kernel<kRank>
    <<<dim3(M / kW6A6QuantBM, splits), kW6A6QuantThreads, Traits::kSmemBytes, stream>>>(
      static_cast<cutlass::bfloat16_t const *>(input),
      static_cast<cutlass::bfloat16_t const *>(smooth),
      static_cast<cutlass::bfloat16_t const *>(lora_down), static_cast<uint8_t *>(output),
      static_cast<uint8_t *>(asf), static_cast<cutlass::bfloat16_t *>(lora_act), scratch, K,
      tiles_per_split);
  if (kRank > 0 && splits > 1) {
    const int64_t count = static_cast<int64_t>(M) * kRank;
    const int blocks = static_cast<int>((count + kW6A6QuantThreads - 1) / kW6A6QuantThreads);
    quantize_w6a6_finalize_kernel<<<blocks, kW6A6QuantThreads, 0, stream>>>(
      lora_scratch, static_cast<cutlass::bfloat16_t *>(lora_act), count);
  }
}

}  // namespace svdq::cute_kernels
