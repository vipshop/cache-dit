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

__global__ void expand_wna4_kernel(uint8_t const *__restrict__ lo, uint8_t const *__restrict__ hi,
                                   uint8_t const *__restrict__ micro, uint8_t *__restrict__ out,
                                   int rows, int K, int bits, int row_bytes_lo, int row_bytes_hi,
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
  const int groups = rows * (K / 8);
  const int threads = 256;
  expand_wna4_kernel<<<(groups + threads - 1) / threads, threads, 0, getCurrentCUDAStream()>>>(
    static_cast<uint8_t const *>(wgt_lo.data_ptr()),
    wgt_hi.valid() ? static_cast<uint8_t const *>(wgt_hi.data_ptr()) : nullptr,
    wgt_micro.valid() ? static_cast<uint8_t const *>(wgt_micro.data_ptr()) : nullptr,
    static_cast<uint8_t *>(out.data_ptr()), rows, K, bits, static_cast<int>(wgt_lo.size(1)),
    wgt_hi.valid() ? static_cast<int>(wgt_hi.size(1)) : 0,
    wgt_micro.valid() ? static_cast<int>(wgt_micro.size(1)) : 0, inner_group);
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
