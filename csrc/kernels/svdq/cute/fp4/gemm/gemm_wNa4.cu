#include "gemm_wNa4.h"

#include <stdexcept>

#include "common.h"
#include "gemm_w4a4.cuh"

// Packed W1/W2/W3 codec and its compatibility with standard two-stage NVFP4.
//
// Two-stage reference (W3, and W1/W2 when stages=2):
//   s_c[n]   = BF16(max_g A[n,g] / (448 * C_b))
//   s_g[n,g] = FP8_E4M3(A[n,g] / (s_c[n] * C_b))
//   q        = nearest code in Q_b after division by s_c * s_g
//   R_hat    = s_c * s_g * Q_b[q]
// C_b is 6 for W3 and 1.5 for W1/W2. `s_c` is rounded to BF16 before deriving
// `s_g`, exactly like the Python encoder and the W4A4 epilogue.
//
// Three-stage W1/W2 keeps the same `s_c`/`s_g` and adds four 2-bit exponents per
// 16-wide group. Each exponent selects a 4-wide table of legal E2M1 magnitudes:
//   W1 e=0/1/2/3 -> {0.5,1.5}/{1.5}/{3}/{6}
//   W2 e=0/1/2/3 -> {0.5,1.5}/{1,3}/{2,6}/{2,6}
// The tables fold the exponent into an E2M1 nibble, so the MMA path and its
// accumulator still see only the standard NVFP4 product; no extra scale is
// applied after the tensor-core instruction.
//
// This TU expands packed rows for the production path and instantiates the
// direct packed GEMM used by tests/benchmarks. Both call the same
// `packed_a4_nibble(bits, code, inner)` table in gemm_w4a4.cuh.
namespace svdq::cute_kernels {
namespace {

// Reference path. Keep it for G=2 and non-32-aligned test shapes so the fast G=4 path cannot
// silently change the packed bit order.
__global__ void expand_wna4_reference_kernel(uint8_t const *__restrict__ lo,
                                             uint8_t const *__restrict__ hi,
                                             uint8_t const *__restrict__ micro,
                                             uint8_t *__restrict__ out, int rows, int K, int bits,
                                             int row_bytes_lo, int row_bytes_hi,
                                             int row_bytes_micro, int inner_group) {
  const int groups_per_row = K / 8;
  const int group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group >= rows * groups_per_row) return;
  const int row = group / groups_per_row;
  const int g = group % groups_per_row;
  const int lo_bits = bits == 3 ? 2 : bits;
  const int lo_bytes = 8 * lo_bits / 8;
  const int k0 = g * 8;
  const uint8_t *lo_ptr = lo + row * row_bytes_lo + (k0 * lo_bits) / 8;
  uint32_t raw = 0;
  if (lo_bytes == 2) {
    raw = __ldg(reinterpret_cast<uint16_t const *>(lo_ptr));
  } else {
#pragma unroll
    for (int b = 0; b < 3; ++b) {
      if (b < lo_bytes) raw |= static_cast<uint32_t>(__ldg(lo_ptr + b)) << (8 * b);
    }
  }
  uint32_t sign_bits = 0;
  if (bits == 3) sign_bits = hi[row * row_bytes_hi + k0 / 8];
  uint32_t micro_bits = 0;
  if (micro != nullptr) {
    micro_bits = __ldg(micro + row * row_bytes_micro + k0 / (4 * inner_group));
  }
  uint32_t packed = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int code = (raw >> (i * lo_bits)) & ((1 << lo_bits) - 1);
    if (bits == 3) {
      packed |= static_cast<uint32_t>(1 + 2 * code + 8 * ((sign_bits >> i) & 1)) << (4 * i);
      continue;
    }
    const int shift = (((k0 % (4 * inner_group)) / inner_group) + i / inner_group) * 2;
    const int inner = micro != nullptr ? static_cast<int>((micro_bits >> shift) & 3) : -1;
    packed |= static_cast<uint32_t>(packed_a4_nibble(bits, code, inner)) << (4 * i);
  }
  *reinterpret_cast<uint32_t *>(out + static_cast<int64_t>(row) * (K / 2) + g * 4) = packed;
}

// NCU on the reference kernel: SM 85.9%, DRAM 12.6%, 35.6 instructions/thread, 39 regs, zero
// spills. The kernel is instruction- and branch-bound, not HBM-bound. The fast path below
// processes 32 K elements per thread, loads the packed codes as 32/64-bit words, writes one
// aligned uint4, and replaces the per-weight code/exponent branches with a 4-weight shared LUT.
// G=4 only: G=2 keeps the reference path, whose exact per-2 exponent order is already covered by
// tests and is not on the production (default per-4) path.
template <int kBits, bool kHasMicro>
__global__ void __launch_bounds__(256)
  expand_wna4_g4_kernel(uint8_t const *__restrict__ lo, uint8_t const *__restrict__ micro,
                        uint8_t *__restrict__ out, int rows, int K, int row_bytes_lo,
                        int row_bytes_micro) {
  static_assert(kBits == 1 || kBits == 2, "W1/W2 only");
  __shared__ uint16_t lut[64];

  const int tid = threadIdx.x;
  if (tid < 64) {
    const int e = tid >> 4;
    const int j = tid & 15;
    uint16_t v = 0;
    if constexpr (kBits == 1) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int nibble = packed_a4_nibble(1, (j >> i) & 1, kHasMicro ? e : 1);
        v |= static_cast<uint16_t>(nibble) << (4 * i);
      }
    } else {
      const int inner = kHasMicro ? e : 0;
      v = static_cast<uint16_t>(packed_a4_nibble(2, j & 3, inner) |
                                (packed_a4_nibble(2, (j >> 2) & 3, inner) << 4));
    }
    lut[tid] = v;
  }
  __syncthreads();

  const int chunks_per_row = K / 32;
  const int chunk = blockIdx.x * blockDim.x + tid;
  if (chunk >= rows * chunks_per_row) return;
  const int row = chunk / chunks_per_row;
  const int seg = chunk - row * chunks_per_row;
  const int k0 = seg * 32;
  const uint8_t *lo_ptr = lo + static_cast<int64_t>(row) * row_bytes_lo + (k0 * kBits) / 8;
  uint32_t raw0 = 0;
  uint32_t raw1 = 0;
  if constexpr (kBits == 1) {
    raw0 = __ldg(reinterpret_cast<uint32_t const *>(lo_ptr));
  } else {
    const uint2 raw = *reinterpret_cast<uint2 const *>(lo_ptr);
    raw0 = raw.x;
    raw1 = raw.y;
  }
  uint16_t micro_bits = 0;
  if constexpr (kHasMicro) {
    micro_bits = __ldg(reinterpret_cast<uint16_t const *>(
      micro + static_cast<int64_t>(row) * row_bytes_micro + k0 / 16));
  }

  uint32_t packed[4] = {0, 0, 0, 0};
  if constexpr (kBits == 1) {
#pragma unroll
    for (int p = 0; p < 8; ++p) {
      const int shift = 2 * (p & 3) + 8 * (p >> 2);
      const int inner = kHasMicro ? ((micro_bits >> shift) & 3) : 1;
      const uint16_t nibbles = lut[inner * 16 + ((raw0 >> (4 * p)) & 0xF)];
      packed[p >> 1] |= static_cast<uint32_t>(nibbles) << (16 * (p & 1));
    }
  } else {
#pragma unroll
    for (int p = 0; p < 16; ++p) {
      const int group = p >> 1;
      const int shift = 2 * (group & 3) + 8 * (group >> 2);
      const int inner = kHasMicro ? ((micro_bits >> shift) & 3) : 0;
      const uint32_t nibble = p < 8 ? (raw0 >> (4 * p)) : (raw1 >> (4 * (p - 8)));
      const uint16_t pair = lut[inner * 16 + (nibble & 0xF)];
      packed[p >> 2] |= static_cast<uint32_t>(pair) << (8 * (p & 3));
    }
  }
  *reinterpret_cast<uint4 *>(out + static_cast<int64_t>(row) * (K / 2) + k0 / 2) =
    make_uint4(packed[0], packed[1], packed[2], packed[3]);
}

__global__ void __launch_bounds__(256)
  expand_wna4_w3_kernel(uint8_t const *__restrict__ lo, uint8_t const *__restrict__ hi,
                        uint8_t *__restrict__ out, int rows, int K, int row_bytes_lo,
                        int row_bytes_hi) {
  __shared__ uint16_t low_lut[256];
  __shared__ uint16_t sign_lut[16];
  const int tid = threadIdx.x;
  for (int i = tid; i < 256; i += blockDim.x) {
    uint16_t v = 0;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      v |= static_cast<uint16_t>(1 + 2 * ((i >> (2 * p)) & 3)) << (4 * p);
    }
    low_lut[i] = v;
  }
  if (tid < 16) {
    uint16_t v = 0;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      if ((tid >> p) & 1) v |= static_cast<uint16_t>(8) << (4 * p);
    }
    sign_lut[tid] = v;
  }
  __syncthreads();

  const int chunks_per_row = K / 32;
  const int chunk = blockIdx.x * blockDim.x + tid;
  if (chunk >= rows * chunks_per_row) return;
  const int row = chunk / chunks_per_row;
  const int seg = chunk - row * chunks_per_row;
  const int k0 = seg * 32;
  const uint2 low =
    *reinterpret_cast<uint2 const *>(lo + static_cast<int64_t>(row) * row_bytes_lo + k0 / 4);
  const uint32_t signs = __ldg(
    reinterpret_cast<uint32_t const *>(hi + static_cast<int64_t>(row) * row_bytes_hi + k0 / 8));

  uint32_t packed[4] = {0, 0, 0, 0};
#pragma unroll
  for (int p = 0; p < 8; ++p) {
    const uint8_t low_byte =
      p < 4 ? ((low.x >> (8 * p)) & 0xFF) : ((low.y >> (8 * (p - 4))) & 0xFF);
    const uint16_t nibbles = low_lut[low_byte] | sign_lut[(signs >> (4 * p)) & 0xF];
    packed[p >> 1] |= static_cast<uint32_t>(nibbles) << (16 * (p & 1));
  }
  *reinterpret_cast<uint4 *>(out + static_cast<int64_t>(row) * (K / 2) + k0 / 2) =
    make_uint4(packed[0], packed[1], packed[2], packed[3]);
}

template <int kBits, bool kHasMicro>
void launch_expand_wna4_g4(uint8_t const *lo, uint8_t const *micro, uint8_t *out, int rows, int K,
                           int row_bytes_lo, int row_bytes_micro, cudaStream_t stream) {
  const int total = rows * (K / 32);
  expand_wna4_g4_kernel<kBits, kHasMicro><<<(total + 255) / 256, 256, 0, stream>>>(
    lo, micro, out, rows, K, row_bytes_lo, row_bytes_micro);
}

template <int kBits>
void launch_expand_wna4_g4_dispatch(uint8_t const *lo, uint8_t const *micro, uint8_t *out, int rows,
                                    int K, int row_bytes_lo, int row_bytes_micro, bool has_micro,
                                    cudaStream_t stream) {
  if (has_micro) {
    launch_expand_wna4_g4<kBits, true>(lo, micro, out, rows, K, row_bytes_lo, row_bytes_micro,
                                       stream);
  } else {
    launch_expand_wna4_g4<kBits, false>(lo, micro, out, rows, K, row_bytes_lo, row_bytes_micro,
                                        stream);
  }
}

template <int kBits>
void launch_wna4(svdq::Tensor act, svdq::Tensor wgt_lo, svdq::Tensor out, svdq::Tensor ascales,
                 svdq::Tensor wscales, svdq::Tensor lora_act_in, svdq::Tensor lora_up,
                 svdq::Tensor bias, svdq::Tensor wcscales, float alpha, int K, svdq::Tensor wgt_hi,
                 svdq::Tensor wgt_micro, int inner_group) {
  auto require = [](bool condition, const char *message) {
    if (!condition) {
      throw std::runtime_error(spdlog::fmt_lib::format("gemm_wNa4: {}", message));
    }
  };
  require(act.valid() && act.ndims() == 2 && wgt_lo.valid() && wgt_lo.ndims() == 2,
          "act/wgt_lo must be valid 2D tensors");
  require(out.valid() && out.ndims() == 2, "out must be a valid 2D tensor");
  require(ascales.valid() && wscales.valid(), "ascales/wscales must be valid");
  require(wgt_lo.dtype() == svdq::Tensor::INT8 && act.dtype() == svdq::Tensor::INT8,
          "activation and packed weights must be byte tensors");
  require(!wgt_hi.valid() || wgt_hi.dtype() == svdq::Tensor::INT8,
          "wgt_hi must be a uint8/INT8 byte tensor");
  require(!wgt_micro.valid() || wgt_micro.dtype() == svdq::Tensor::INT8,
          "wgt_micro must be a uint8/INT8 byte tensor");
  require(inner_group == 2 || inner_group == 4, "inner_group must be 2 or 4");
  if constexpr (kBits == 3) {
    require(inner_group == 4, "W3 requires inner_group=4");
  }
  require(ascales.dtype() == svdq::Tensor::FP8_E4M3 && wscales.dtype() == svdq::Tensor::FP8_E4M3,
          "scales must be FP8_E4M3");
  const int M = act.size(0);
  const int N = wgt_lo.size(0);
  const int rank = lora_up.valid() ? lora_up.size(1) : 0;
  require(M > 0 && M % 256 == 0, "M must be a positive multiple of 256");
  require(N > 0 && N % 256 == 0, "N must be a positive multiple of 256");
  if constexpr (kBits == 3) {
    require(rank == 0 || rank == 32 || rank == 64 || rank == 128,
            "W3 rank must be 0, 32, 64 or 128");
  } else {
    require(rank == 0 || rank == 32 || rank == 64 || rank == 128 || rank == 256 || rank == 512,
            "W1/W2 rank must be 0, 32, 64, 128, 256 or 512");
  }
  require(lora_up.valid() == lora_act_in.valid(), "LoRA tensors must be supplied together");
  require(
    act.is_cuda() && wgt_lo.is_cuda() && out.is_cuda() && ascales.is_cuda() && wscales.is_cuda(),
    "all tensors must be CUDA tensors");
  require(act.is_contiguous() && wgt_lo.is_contiguous() && out.is_contiguous() &&
            ascales.is_contiguous() && wscales.is_contiguous(),
          "all tensors must be contiguous");
  cudaDeviceProp prop;
  checkCUDA(cudaGetDeviceProperties(&prop, act.get_device()));
  require(prop.major >= 12, "packed W1/W2/W3 requires sm120 or newer");
  require(K == act.size(1) * 2, "K must match the packed activation extent");
  require(K > 0 && K % 128 == 0, "K must be a multiple of 128");
  if constexpr (kBits == 1) {
    require(wgt_lo.size(1) * 8 >= K, "W1 packed row is too short");
  } else if constexpr (kBits == 2) {
    require(wgt_lo.size(1) * 4 >= K, "W2 packed row is too short");
  } else {
    require(wgt_lo.size(1) * 4 >= K && wgt_hi.valid() && wgt_hi.size(1) * 8 >= K,
            "W3 packed magnitude/sign rows are too short");
  }
  if (wgt_micro.valid()) {
    require(kBits != 3, "W3 does not support the three-stage inner scale");
    require(
      wgt_micro.ndims() == 2 && wgt_micro.size(0) == N && wgt_micro.size(1) * 4 * inner_group >= K,
      "micro scales must be uint8 [N,ceil(K/(4*inner_group))]");
    require(wgt_micro.is_cuda() && wgt_micro.is_contiguous(),
            "micro scales must be a contiguous CUDA tensor");
  }
  require(wscales.numel() == sf_buffer_bytes(N, K), "wscales must be a 128x4 SF buffer");
  require(out.dtype() == svdq::Tensor::BF16 && out.size(0) == M && out.size(1) == N,
          "out must be BF16 [M,N]");
  if (wcscales.valid()) {
    require(wcscales.numel() == static_cast<size_t>(N) && wcscales.dtype() == svdq::Tensor::BF16,
            "wcscales must be BF16 [N]");
  }
  F32DLoraScales scales{};
  for (float &value : scales.values) value = 1.0f;
  const auto *lora_act =
    rank > 0 ? static_cast<cutlass::bfloat16_t const *>(lora_act_in.data_ptr()) : nullptr;
  const auto *lora_up_ptr =
    rank > 0 ? static_cast<cutlass::bfloat16_t const *>(lora_up.data_ptr()) : nullptr;
  launch_gemm_w4a4<256, 3, 128, true, false, false, false, kBits>(
    act.data_ptr(), wgt_lo.data_ptr(), out.data_ptr(),
    static_cast<cutlass::float_ue4m3_t const *>(ascales.data_ptr()),
    static_cast<cutlass::float_ue4m3_t const *>(wscales.data_ptr()), lora_act, nullptr, lora_up_ptr,
    bias.valid() ? static_cast<cutlass::bfloat16_t const *>(bias.data_ptr()) : nullptr,
    wcscales.valid() ? static_cast<cutlass::bfloat16_t const *>(wcscales.data_ptr()) : nullptr,
    scales, alpha, M, N, K, rank, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    F32DLoraScales{}, 0, false, false, getCurrentCUDAStream(),
    static_cast<uint8_t const *>(wgt_lo.data_ptr()),
    wgt_hi.valid() ? static_cast<uint8_t const *>(wgt_hi.data_ptr()) : nullptr,
    static_cast<int>(wgt_lo.size(1)), wgt_hi.valid() ? static_cast<int>(wgt_hi.size(1)) : 0,
    wgt_micro.valid() ? static_cast<uint8_t const *>(wgt_micro.data_ptr()) : nullptr,
    wgt_micro.valid() ? static_cast<int>(wgt_micro.size(1)) : 0, inner_group);
  checkCUDA(cudaGetLastError());
}

}  // namespace

void expand_wna4(svdq::Tensor wgt_lo, svdq::Tensor wgt_hi, svdq::Tensor out, int bits, int K,
                 svdq::Tensor wgt_micro, int inner_group) {
  auto require = [](bool condition, const char *message) {
    if (!condition) {
      throw std::runtime_error(spdlog::fmt_lib::format("expand_wna4: {}", message));
    }
  };
  require(wgt_lo.valid() && wgt_lo.ndims() == 2 && out.valid() && out.ndims() == 2,
          "packed weight and output must be 2D");
  require(bits >= 1 && bits <= 3, "bits must be 1, 2 or 3");
  require(inner_group == 2 || inner_group == 4, "inner_group must be 2 or 4");
  require(bits != 3 || inner_group == 4, "W3 requires inner_group=4");
  require(bits != 3 || (wgt_hi.valid() && wgt_hi.ndims() == 2),
          "W3 requires a valid 2D sign plane");
  require(!wgt_hi.valid() || wgt_hi.dtype() == svdq::Tensor::INT8,
          "wgt_hi must be a uint8/INT8 byte tensor");
  require(!wgt_micro.valid() || wgt_micro.dtype() == svdq::Tensor::INT8,
          "wgt_micro must be a uint8/INT8 byte tensor");
  require(wgt_lo.is_cuda() && out.is_cuda() && wgt_lo.is_contiguous() && out.is_contiguous(),
          "packed weight and output must be contiguous CUDA tensors");
  require(K > 0 && K % 8 == 0 && wgt_lo.size(1) * 8 >= K, "K must match the packed weight");
  const int rows = wgt_lo.size(0);
  require(out.size(0) == rows && out.size(1) * 2 == K, "out must be packed E2M1 [rows,K/2]");
  if (wgt_micro.valid()) {
    require(bits != 3 && K % (4 * inner_group) == 0,
            "three-stage expansion requires W1/W2 and K divisible by 4*inner_group");
    require(wgt_micro.ndims() == 2 && wgt_micro.size(0) == rows &&
              wgt_micro.size(1) * 4 * inner_group >= K && wgt_micro.is_cuda() &&
              wgt_micro.is_contiguous(),
            "micro scales must be uint8 [rows,ceil(K/(4*inner_group))] on CUDA");
  }
  const auto *lo_ptr = static_cast<uint8_t const *>(wgt_lo.data_ptr());
  const auto *hi_ptr = wgt_hi.valid() ? static_cast<uint8_t const *>(wgt_hi.data_ptr()) : nullptr;
  const auto *micro_ptr =
    wgt_micro.valid() ? static_cast<uint8_t const *>(wgt_micro.data_ptr()) : nullptr;
  auto *out_ptr = static_cast<uint8_t *>(out.data_ptr());
  const int row_bytes_lo = static_cast<int>(wgt_lo.size(1));
  const int row_bytes_hi = wgt_hi.valid() ? static_cast<int>(wgt_hi.size(1)) : 0;
  const int row_bytes_micro = wgt_micro.valid() ? static_cast<int>(wgt_micro.size(1)) : 0;
  const cudaStream_t stream = getCurrentCUDAStream();

  // The fast path performs 4/8/16-byte vector loads and a 16-byte store. Keep the public op safe
  // for padded row strides or offset views: unless every participating byte address meets the
  // vector alignment, fall back to the scalar reference kernel.
  const int lo_align = bits == 1 ? 4 : 8;
  const bool fast_aligned =
    inner_group == 4 && K % 32 == 0 && row_bytes_lo % lo_align == 0 &&
    (bits != 3 || row_bytes_hi % 4 == 0) && (micro_ptr == nullptr || row_bytes_micro % 2 == 0) &&
    reinterpret_cast<uintptr_t>(lo_ptr) % lo_align == 0 &&
    (hi_ptr == nullptr || reinterpret_cast<uintptr_t>(hi_ptr) % 4 == 0) &&
    (micro_ptr == nullptr || reinterpret_cast<uintptr_t>(micro_ptr) % 2 == 0) &&
    reinterpret_cast<uintptr_t>(out_ptr) % 16 == 0;

  if (fast_aligned && bits != 3) {
    if (bits == 1) {
      launch_expand_wna4_g4_dispatch<1>(lo_ptr, micro_ptr, out_ptr, rows, K, row_bytes_lo,
                                        row_bytes_micro, micro_ptr != nullptr, stream);
    } else {
      launch_expand_wna4_g4_dispatch<2>(lo_ptr, micro_ptr, out_ptr, rows, K, row_bytes_lo,
                                        row_bytes_micro, micro_ptr != nullptr, stream);
    }
    checkCUDA(cudaGetLastError());
    return;
  }
  if (fast_aligned && bits == 3) {
    const int total = rows * (K / 32);
    expand_wna4_w3_kernel<<<(total + 255) / 256, 256, 0, stream>>>(lo_ptr, hi_ptr, out_ptr, rows, K,
                                                                   row_bytes_lo, row_bytes_hi);
    checkCUDA(cudaGetLastError());
    return;
  }

  const int groups = rows * (K / 8);
  const int threads = 256;
  expand_wna4_reference_kernel<<<(groups + threads - 1) / threads, threads, 0, stream>>>(
    lo_ptr, hi_ptr, micro_ptr, out_ptr, rows, K, bits, row_bytes_lo, row_bytes_hi, row_bytes_micro,
    inner_group);
  checkCUDA(cudaGetLastError());
}

void gemm_w1a4(svdq::Tensor act, svdq::Tensor wgt_lo, svdq::Tensor out, svdq::Tensor ascales,
               svdq::Tensor wscales, svdq::Tensor lora_act_in, svdq::Tensor lora_up,
               svdq::Tensor bias, svdq::Tensor wcscales, float alpha, int K, svdq::Tensor wgt_hi,
               svdq::Tensor wgt_micro, int inner_group) {
  launch_wna4<1>(act, wgt_lo, out, ascales, wscales, lora_act_in, lora_up, bias, wcscales, alpha, K,
                 wgt_hi, wgt_micro, inner_group);
}

void gemm_w2a4(svdq::Tensor act, svdq::Tensor wgt_lo, svdq::Tensor out, svdq::Tensor ascales,
               svdq::Tensor wscales, svdq::Tensor lora_act_in, svdq::Tensor lora_up,
               svdq::Tensor bias, svdq::Tensor wcscales, float alpha, int K, svdq::Tensor wgt_hi,
               svdq::Tensor wgt_micro, int inner_group) {
  launch_wna4<2>(act, wgt_lo, out, ascales, wscales, lora_act_in, lora_up, bias, wcscales, alpha, K,
                 wgt_hi, wgt_micro, inner_group);
}

void gemm_w3a4(svdq::Tensor act, svdq::Tensor wgt_lo, svdq::Tensor out, svdq::Tensor ascales,
               svdq::Tensor wscales, svdq::Tensor lora_act_in, svdq::Tensor lora_up,
               svdq::Tensor bias, svdq::Tensor wcscales, float alpha, int K, svdq::Tensor wgt_hi,
               svdq::Tensor wgt_micro, int inner_group) {
  launch_wna4<3>(act, wgt_lo, out, ascales, wscales, lora_act_in, lora_up, bias, wcscales, alpha, K,
                 wgt_hi, wgt_micro, inner_group);
}

}  // namespace svdq::cute_kernels
