#include "common.h"
#include "gemm_w4a4.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace svdq::cute_kernels {
namespace {

// Independent CuTe activation producer. One warp owns 32 rows and the whole K reduction:
//   qact/ascales are written directly in the TMA-compatible NVFP4 layouts;
//   D = input @ lora_down is accumulated in registers and written once as row-major BF16.
// There is no v1 activation fragment, FP32 LoRA scratch, atomic, or post-processing kernel.
struct PackedBF16 {
  __nv_bfloat162 data[4];
};

struct PackedF32 {
  float data[8];
};

struct LoraScaleParams {
  float values[64];
};

constexpr int kAStageFloats = 16 * (64 + 8) / 2;

void require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

__device__ __forceinline__ float rcp_approx(float x) {
  float result;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
  return result;
}

__device__ __forceinline__ uint32_t quantize_float2_fp4(float2 value) {
  uint32_t result;
  asm volatile("{ .reg .b8 tmp; cvt.rn.satfinite.e2m1x2.f32 tmp, %1, %2; cvt.u32.u8 %0, tmp; }"
               : "=r"(result)
               : "f"(value.y), "f"(value.x));
  return result;
}

__device__ __forceinline__ uint32_t quantize_float4_fp8(float4 value) {
  uint16_t lo, hi;
  asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(value.y), "f"(value.x));
  asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(hi) : "f"(value.w), "f"(value.z));
  return uint32_t(lo) | (uint32_t(hi) << 16);
}

__device__ __forceinline__ __nv_bfloat162 load_bf16x2(const __nv_bfloat16 *ptr) {
  return *reinterpret_cast<const __nv_bfloat162 *>(ptr);
}

__device__ __forceinline__ __nv_bfloat162 div_bf16x2(__nv_bfloat162 a, __nv_bfloat162 b) {
  float2 af = __bfloat1622float2(a);
  float2 bf = __bfloat1622float2(b);
  return __float22bfloat162_rn(make_float2(__fdividef(af.x, bf.x), __fdividef(af.y, bf.y)));
}

// Load a 16x16 BF16 fragment from the warp-private row-major staging tile. The tile is padded by
// eight BF16 columns, matching the native ldmatrix layout and avoiding shared-memory bank conflicts.
__device__ __forceinline__ void ldmatrix_x4(const void *ptr, uint4 &out) {
  asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w)
               : "l"(__cvta_generic_to_shared(ptr)));
}

// SM80_16x8x16_F32BF16BF16F32_TN has BLayout
// ((4,8),(2,2)):((16,1),(8,64)). Each uint32 register packs two BF16 values that are adjacent
// in K at the same N/rank column. lora_down is row-major [K,R], so those two values are separated
// by R elements, not adjacent in memory. Reading ptr[0],ptr[1] would silently select two adjacent
// rank columns and produce the observed even/odd-column permutation.
__device__ __forceinline__ __nv_bfloat162 load_lora_pair(const __nv_bfloat16 *lora, int R, int k,
                                                          int r) {
  __nv_bfloat162 out;
  out.x = lora[size_t(k) * R + r];
  out.y = lora[size_t(k + 1) * R + r];
  return out;
}

// Load the B fragment for two m16n8k16 atoms from row-major lora_down[K,R]. data[0]/[1] are the
// K and K+8 pairs for rank columns 0..7; data[2]/[3] are the same pairs for rank columns 8..15.
// Scalar loads also avoid the 4-byte alignment assumption on odd rank columns.
__device__ __forceinline__ PackedBF16 load_lora_fragment(const __nv_bfloat16 *lora, int R, int row,
                                                          int col, int lane) {
  const int k = row + (lane % 4) * 2;
  const int r = col + lane / 4;
  PackedBF16 out;
  out.data[0] = load_lora_pair(lora, R, k, r);
  out.data[1] = load_lora_pair(lora, R, k + 8, r);
  out.data[2] = load_lora_pair(lora, R, k, r + 8);
  out.data[3] = load_lora_pair(lora, R, k + 8, r + 8);
  return out;
}

__device__ __forceinline__ uint4 mma_bf16(uint4 a, uint2 b, uint4 c) {
  uint4 d;
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
               : "=r"(d.x), "=r"(d.y), "=r"(d.z), "=r"(d.w)
               : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y), "r"(c.x), "r"(c.y),
                 "r"(c.z), "r"(c.w));
  return d;
}

__device__ __forceinline__ PackedF32 mma_lora(PackedBF16 a, PackedBF16 b, PackedF32 acc) {
  uint4 aa = *reinterpret_cast<uint4 *>(&a);
  uint2 b0 = *reinterpret_cast<uint2 *>(&b.data[0]);
  uint2 b1 = *reinterpret_cast<uint2 *>(&b.data[2]);
  uint4 c0 = make_uint4(__float_as_uint(acc.data[0]), __float_as_uint(acc.data[1]),
                        __float_as_uint(acc.data[2]), __float_as_uint(acc.data[3]));
  uint4 c1 = make_uint4(__float_as_uint(acc.data[4]), __float_as_uint(acc.data[5]),
                        __float_as_uint(acc.data[6]), __float_as_uint(acc.data[7]));
  uint4 d0 = mma_bf16(aa, b0, c0);
  uint4 d1 = mma_bf16(aa, b1, c1);
  acc.data[0] = __uint_as_float(d0.x);
  acc.data[1] = __uint_as_float(d0.y);
  acc.data[2] = __uint_as_float(d0.z);
  acc.data[3] = __uint_as_float(d0.w);
  acc.data[4] = __uint_as_float(d1.x);
  acc.data[5] = __uint_as_float(d1.y);
  acc.data[6] = __uint_as_float(d1.z);
  acc.data[7] = __uint_as_float(d1.w);
  return acc;
}

__device__ __forceinline__ size_t sf_offset(int row, int group, int groups) {
  return size_t(row / 128) * (groups * 128) + size_t(group / 4) * 512 + 16 * (row % 32) +
         4 * ((row / 32) % 4) + group % 4;
}

// Quantize four K16 groups and write standard row-major qact plus CUTLASS 128x4 SFA.
// The max/shuffle/rounding sequence matches the native producer, but the stores bypass its v1
// fragment layout completely. Each lane writes two u32 per row half, so the result is linear TMA
// input rather than an MMA register image in gmem.
__device__ __forceinline__ void quantize_group(PackedBF16 (&tmp)[4], uint8_t *qact,
                                                uint8_t *ascales, int M, int K, int row0, int k0,
                                                int lane) {
  __nv_bfloat162 input[2][8];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    input[0][i * 2] = tmp[i].data[0];
    input[0][i * 2 + 1] = tmp[i].data[2];
    input[1][i * 2] = tmp[i].data[1];
    input[1][i * 2 + 1] = tmp[i].data[3];
  }

  float maxvalue[2][4] = {};
#pragma unroll
  for (int j = 0; j < 2; ++j) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float2 a = __bfloat1622float2(input[j][i * 2]);
      float2 b = __bfloat1622float2(input[j][i * 2 + 1]);
      maxvalue[j][i] = fmaxf(fabsf(a.x), fmaxf(fabsf(a.y), fmaxf(fabsf(b.x), fabsf(b.y))));
    }
  }
#pragma unroll
  for (int mask = 2; mask > 0; mask /= 2) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        maxvalue[j][i] = fmaxf(maxvalue[j][i], __shfl_xor_sync(~0u, maxvalue[j][i], mask));
      }
    }
  }

  float rscale[2][4];
  uint32_t fp8scale[2];
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    float scale[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      scale[i] = fminf(maxvalue[j][i] / 6.0f, 448.0f);
      rscale[j][i] = rcp_approx(scale[i]);
    }
    fp8scale[j] = quantize_float4_fp8(make_float4(scale[0], scale[1], scale[2], scale[3]));
  }

  uint32_t qpacks[2][8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      float2 fval = __bfloat1622float2(input[j][i]);
      fval.x *= rscale[j][i / 2];
      fval.y *= rscale[j][i / 2];
      qpacks[j][i] = quantize_float2_fp4(fval) << ((lane % 4) * 8);
    }
  }
#pragma unroll
  for (int mask = 1; mask <= 2; mask *= 2) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        qpacks[j][i] |= __shfl_xor_sync(~0u, qpacks[j][i], mask);
      }
    }
  }

  // Two lanes per row emit the full 32-byte K64 qact slice as adjacent 16-byte stores. The old
  // four-lane/4-byte form produced half-used sectors; this keeps the same standard row-major
  // bytes while letting the coalescer fill complete 32-byte sectors.
  const int i = lane % 4;
  if (i < 2) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int row = row0 + lane / 4 + j * 8;
      const uint4 packed = make_uint4(qpacks[j][i * 4 + 0], qpacks[j][i * 4 + 1],
                                      qpacks[j][i * 4 + 2], qpacks[j][i * 4 + 3]);
      *reinterpret_cast<uint4 *>(qact + size_t(row) * (K / 2) + k0 / 2 + i * 16) = packed;
    }
  }

  // All four lanes in a row group own the same four scales. Let lane%4==0 write the packed u32.
  if (lane % 4 == 0) {
    const int group = k0 / 16;
    const int groups = K / 16;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int row = row0 + lane / 4 + j * 8;
      *reinterpret_cast<uint32_t *>(ascales + sf_offset(row, group, groups)) = fp8scale[j];
    }
  }
}

// TODO(cute-producer-migration): This fused activation-quant + LoRA-down producer currently uses
// manual mma.sync/ldmatrix and explicit shared-memory staging. Once the native-matching schedule
// is locked, re-express the same tile/reduction/coalesced-output plan with CuTe TiledMMA/TiledCopy
// atoms. Keep the standard row-major A/SFA and row-major BF16 D contract unchanged.
template <int kRankTiles, int kWarps>
__global__ void quantize_w4a4_fused_lora_kernel(
  const __nv_bfloat16 *__restrict__ input,
  const __nv_bfloat16 *__restrict__ smooth,
  const __nv_bfloat16 *__restrict__ lora_down,
  uint8_t *__restrict__ qact,
  uint8_t *__restrict__ ascales,
  __nv_bfloat16 *__restrict__ lora_act,
  LoraScaleParams scales,
  int M,
  int K) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  constexpr int kAccTiles = kRankTiles > 0 ? kRankTiles : 1;
  constexpr int kRank = kRankTiles * 16;
  constexpr int kRankStride = kRankTiles == 2 ? kRank + 4 : kRank + 1;
  constexpr int kDStageFloats = kRankTiles > 0 ? 32 * kRankStride : 0;
  constexpr int kSharedFloats = kAStageFloats > kDStageFloats ? kAStageFloats : kDStageFloats;
  constexpr int kAStride = 64 + 8;
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int row0 = blockIdx.x * 32;
  extern __shared__ float shared_scratch[];
  auto *warp_stage = reinterpret_cast<__nv_bfloat16 *>(shared_scratch + warp * kSharedFloats);
  float *warp_partial = shared_scratch + warp * kSharedFloats;

  PackedF32 acc[2][kAccTiles];
  if constexpr (kRankTiles > 0) {
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
      for (int rt = 0; rt < kRankTiles; ++rt) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          acc[mt][rt].data[j] = 0.0f;
        }
      }
    }
  }

  // Native uses 24 K-split CTAs for M=4096. Here the same K parallelism stays inside one CTA:
  // all kWarps warps share 32 rows, cover disjoint K64 slices, and reduce D in shared memory.
  // A is staged as a 32x64 row before ldmatrix so every global load is a fully coalesced
  // 128-byte row transaction instead of four half-sector 4-byte loads per lane.
  for (int k0 = warp * 64; k0 < K; k0 += kWarps * 64) {
    PackedBF16 raw_all[2][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      // Eight lanes per row move 16B each. Four rows are staged per instruction, reducing the
      // original 32 4-byte shared stores per k64 to eight 16-byte stores without changing layout.
#pragma unroll
      for (int row = 0; row < 16; row += 4) {
        const int stage_row = row + lane / 8;
        const int stage_col = (lane % 8) * 8;
        const uint4 pack = *reinterpret_cast<const uint4 *>(
          input + size_t(row0 + mt * 16 + stage_row) * K + k0 + stage_col);
        *reinterpret_cast<uint4 *>(warp_stage + stage_row * kAStride + stage_col) = pack;
      }
      __syncwarp();

#pragma unroll
      for (int n = 0; n < 4; ++n) {
        uint4 tmp;
        const int row = lane % 16;
        const int col = n * 16 + lane / 16 * 8;
        ldmatrix_x4(warp_stage + row * kAStride + col, tmp);
        raw_all[mt][n] = *reinterpret_cast<PackedBF16 *>(&tmp);
      }
      __syncwarp();
    }

    __nv_bfloat162 smooth0[4];
    __nv_bfloat162 smooth1[4];
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      const int col = k0 + n * 16 + (lane % 4) * 2;
      smooth0[n] = load_bf16x2(smooth + col);
      smooth1[n] = load_bf16x2(smooth + col + 8);
    }

#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      PackedBF16 quant[4];
#pragma unroll
      for (int n = 0; n < 4; ++n) {
        quant[n] = raw_all[mt][n];
        quant[n].data[0] = div_bf16x2(quant[n].data[0], smooth0[n]);
        quant[n].data[1] = div_bf16x2(quant[n].data[1], smooth0[n]);
        quant[n].data[2] = div_bf16x2(quant[n].data[2], smooth1[n]);
        quant[n].data[3] = div_bf16x2(quant[n].data[3], smooth1[n]);
      }
      quantize_group(quant, qact, ascales, M, K, row0 + mt * 16, k0, lane);
    }

    if constexpr (kRankTiles > 0) {
      // B weights are identical for both 16-row halves. Load each fragment once and apply it to
      // both accumulators, halving lora_down global loads without changing the K reduction.
#pragma unroll
      for (int rt = 0; rt < kRankTiles; ++rt) {
#pragma unroll
        for (int n = 0; n < 4; ++n) {
          PackedBF16 b = load_lora_fragment(lora_down, kRank, k0 + n * 16, rt * 16, lane);
          acc[0][rt] = mma_lora(raw_all[0][n], b, acc[0][rt]);
          acc[1][rt] = mma_lora(raw_all[1][n], b, acc[1][rt]);
        }
      }
    }
  }

  if constexpr (kRankTiles > 0) {
    // rank=32 uses a hand-checked XOR layout: 4 words of row padding plus shared_col=col^row
    // gives 32 distinct banks for every fragment store and for the full-row reduction read.
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
      for (int rt = 0; rt < kRankTiles; ++rt) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int row = mt * 16 + lane / 4 + 8 * ((j / 2) % 2);
          const int col = rt * 16 + 2 * (lane % 4) + (j % 2) + 8 * (j / 4);
          const int shared_col = kRankTiles == 2 ? (col ^ row) : col;
          warp_partial[row * kRankStride + shared_col] = acc[mt][rt].data[j];
        }
      }
    }
    __syncthreads();

    // All threads cooperate on the 32xR reduction. Each output element is rounded once, after
    // summing every warp's K partial and folding lora_scales in FP32. The rank=32 swizzle was
    // checked offline for both the MMA fragment write pattern and this row-major readback.
    for (int index = threadIdx.x; index < 32 * kRank; index += kWarps * 32) {
      const int row = index / kRank;
      const int col = index % kRank;
      const int shared_col = kRankTiles == 2 ? (col ^ row) : col;
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < kWarps; ++w) {
        sum += shared_scratch[w * kSharedFloats + row * kRankStride + shared_col];
      }
      lora_act[size_t(row0 + row) * kRank + col] =
        __float2bfloat16_rn(sum * scales.values[col / 16]);
    }
  }
#else
  asm volatile("trap;");
#endif
}

template <int kRankTiles, int kWarps>
void launch_quantize_w4a4(const __nv_bfloat16 *input, const __nv_bfloat16 *smooth,
                               const __nv_bfloat16 *lora_down, uint8_t *qact, uint8_t *ascales,
                               __nv_bfloat16 *lora_act, LoraScaleParams scales, int M, int K,
                               cudaStream_t stream) {
  constexpr int kRank = kRankTiles * 16;
  constexpr int kRankStride = kRankTiles == 2 ? kRank + 4 : kRank + 1;
  constexpr int kDStageFloats = kRankTiles > 0 ? 32 * kRankStride : 0;
  constexpr int kSharedFloats = kAStageFloats > kDStageFloats ? kAStageFloats : kDStageFloats;
  constexpr size_t kSmemBytes = static_cast<size_t>(kWarps) * kSharedFloats * sizeof(float);
  auto kernel = quantize_w4a4_fused_lora_kernel<kRankTiles, kWarps>;
  if constexpr (kSmemBytes > 48 * 1024) {
    checkCUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  }
  kernel<<<dim3(M / 32), dim3(kWarps * 32), kSmemBytes, stream>>>(
    input, smooth, lora_down, qact, ascales, lora_act, scales, M, K);
}

}  // namespace

void quantize_w4a4(::Tensor input, ::Tensor output, ::Tensor ascales, ::Tensor lora_down,
                        ::Tensor lora_act, ::Tensor smooth, std::vector<float> lora_scales) {
  require(input.valid() && output.valid() && ascales.valid() && smooth.valid(),
          "quantize_w4a4 requires input/output/ascales/smooth");
  require(input.dtype() == ::Tensor::BF16 && smooth.dtype() == ::Tensor::BF16,
          "quantize_w4a4 currently requires BF16 input and smooth");
  require(output.dtype() == ::Tensor::INT8 && ascales.dtype() == ::Tensor::FP8_E4M3,
          "quantize_w4a4 output must be INT8 and ascales FP8_E4M3");
  require(input.is_contiguous() && smooth.is_contiguous(), "input and smooth must be contiguous");
  const int K = input.shape[-1];
  const int M = input.numel() / K;
  const int rank = lora_down.valid() ? lora_down.shape[-1] : 0;
  require(M % 32 == 0 && K % 64 == 0, "quantize_w4a4 requires M%32==0 and K%64==0");
  require(rank == 0 || (rank % 16 == 0 && rank <= 128),
          "quantize_w4a4 rank must be 0 or a multiple of 16 up to 128");
  require(static_cast<int>(lora_scales.size()) == std::max(1, rank / 16),
          "lora_scales must have one value per 16-rank tile");
  if (rank > 0) {
    require(lora_down.is_contiguous() && lora_act.valid() && lora_act.dtype() == ::Tensor::BF16,
            "CuTe LoRA-down must be row-major BF16 and lora_act must be BF16");
  }

  LoraScaleParams scales{};
  for (size_t i = 0; i < lora_scales.size(); ++i) {
    scales.values[i] = lora_scales[i];
  }
  const auto *input_ptr = input.data_ptr<__nv_bfloat16>();
  const auto *smooth_ptr = smooth.data_ptr<__nv_bfloat16>();
  const auto *lora_down_ptr = rank > 0 ? lora_down.data_ptr<__nv_bfloat16>() : nullptr;
  auto *qact_ptr = output.data_ptr<uint8_t>();
  auto *ascales_ptr = ascales.data_ptr<uint8_t>();
  auto *lora_act_ptr = rank > 0 ? lora_act.data_ptr<__nv_bfloat16>() : nullptr;
  const cudaStream_t stream = getCurrentCUDAStream();

  if (rank == 0) {
    launch_quantize_w4a4<0, 8>(input_ptr, smooth_ptr, lora_down_ptr, qact_ptr, ascales_ptr,
                                    lora_act_ptr, scales, M, K, stream);
  } else if (rank == 16) {
    launch_quantize_w4a4<1, 8>(input_ptr, smooth_ptr, lora_down_ptr, qact_ptr, ascales_ptr,
                                    lora_act_ptr, scales, M, K, stream);
  } else if (rank == 32) {
    launch_quantize_w4a4<2, 8>(input_ptr, smooth_ptr, lora_down_ptr, qact_ptr, ascales_ptr,
                                    lora_act_ptr, scales, M, K, stream);
  } else if (rank == 48) {
    launch_quantize_w4a4<3, 8>(input_ptr, smooth_ptr, lora_down_ptr, qact_ptr, ascales_ptr,
                                    lora_act_ptr, scales, M, K, stream);
  } else if (rank == 64) {
    launch_quantize_w4a4<4, 8>(input_ptr, smooth_ptr, lora_down_ptr, qact_ptr, ascales_ptr,
                                    lora_act_ptr, scales, M, K, stream);
  } else if (rank == 80) {
    launch_quantize_w4a4<5, 4>(input_ptr, smooth_ptr, lora_down_ptr, qact_ptr, ascales_ptr,
                                    lora_act_ptr, scales, M, K, stream);
  } else if (rank == 96) {
    launch_quantize_w4a4<6, 4>(input_ptr, smooth_ptr, lora_down_ptr, qact_ptr, ascales_ptr,
                                    lora_act_ptr, scales, M, K, stream);
  } else if (rank == 112) {
    launch_quantize_w4a4<7, 4>(input_ptr, smooth_ptr, lora_down_ptr, qact_ptr, ascales_ptr,
                                    lora_act_ptr, scales, M, K, stream);
  } else {
    launch_quantize_w4a4<8, 4>(input_ptr, smooth_ptr, lora_down_ptr, qact_ptr, ascales_ptr,
                                    lora_act_ptr, scales, M, K, stream);
  }
  checkCUDA(cudaGetLastError());
}

}  // namespace svdq::cute_kernels
