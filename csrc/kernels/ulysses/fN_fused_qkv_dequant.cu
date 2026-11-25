#include <type_traits>

#include "fp4_common.cuh"
#include "fp6_common.cuh"

// Inverse of fN_fused_qkv_quant.cu. After all-to-all, `packed` has S_GLOBAL sequence rows and the
// same per-row Q||K||V byte layout. This kernel reads each segment directly and writes three
// contiguous BF16 tensors [B, S_GLOBAL, H_LOCAL, D]; it never materializes per-tensor packed views
// or concatenated temporary outputs.

// Multirow uses the same row-segment mapping as the quantizer. At D=128, FP8 uses two 16-lane
// segments and FP6 uses four 8-lane segments per warp. `row_in_warp` selects the packed row and
// `row_lane` selects elements/scales within that row. Group-scale broadcasts use the physical lane
// of the corresponding group leader, so lanes from different rows cannot exchange scales. The
// resulting BF16 output is bitwise identical to one-row-per-warp dequantization.

namespace cache_dit::ulysses {
namespace {

template <int kLanesPerRow>
__global__ void fused_qkv_dequant_kernel(const uint8_t* packed, __nv_bfloat16* q, __nv_bfloat16* k,
                                         __nv_bfloat16* v, int64_t batches, int64_t seq_len,
                                         int64_t q_heads, int64_t k_heads, int64_t v_heads,
                                         int64_t head_size, int64_t q_width, int64_t k_width,
                                         int64_t v_width, int64_t chunk_row_bytes, int64_t q_rows,
                                         int64_t k_rows, int64_t v_rows, int group_size, int bits,
                                         bool quantize_k) {
  // Tensor-local rows are ordered [S_GLOBAL, B, H_LOCAL], matching the legacy dequant input
  // [S_GLOBAL, B, H_LOCAL, W]. Q/K/V again have separate warp ranges and tails. Each lane segment
  // restores one complete row so packed scale reads and BF16 stores never cross row boundaries.
  static_assert(kLanesPerRow == kWarpSize || kLanesPerRow == 16 || kLanesPerRow == 8);
  constexpr int kRowsPerWarp = kWarpSize / kLanesPerRow;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int row_in_warp = lane / kLanesPerRow;
  const int row_lane = lane % kLanesPerRow;
  const int64_t q_warps = (q_rows + kRowsPerWarp - 1) / kRowsPerWarp;
  const int64_t k_warps = (k_rows + kRowsPerWarp - 1) / kRowsPerWarp;
  const int64_t v_warps = (v_rows + kRowsPerWarp - 1) / kRowsPerWarp;
  const int64_t global_warp = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  if (global_warp >= q_warps + k_warps + v_warps) return;

  // tensor_id identifies which output component is being restored: 0=Q, 1=K, 2=V. It selects the
  // output pointer, local head count, packed width and segment base. In particular, tensor_id==1
  // means K, so quantize_k==false must read raw BF16 bytes instead of an FP4/FP6/FP8 payload+scale.
  int tensor_id = 0;
  int64_t tensor_row = global_warp * kRowsPerWarp + row_in_warp;
  if (global_warp >= q_warps + k_warps) {
    tensor_id = 2;
    tensor_row = (global_warp - q_warps - k_warps) * kRowsPerWarp + row_in_warp;
    if (tensor_row >= v_rows) return;
  } else if (global_warp >= q_warps) {
    tensor_id = 1;
    tensor_row = (global_warp - q_warps) * kRowsPerWarp + row_in_warp;
    if (tensor_row >= k_rows) return;
  } else if (tensor_row >= q_rows) {
    return;
  }

  __nv_bfloat16* output = q;
  int64_t heads = q_heads;
  int64_t width = q_width;
  int64_t segment_bytes = 0;
  if (tensor_id == 2) {
    output = v;
    heads = v_heads;
    width = v_width;
    segment_bytes = batches * (q_heads * q_width + k_heads * k_width);
  } else if (tensor_id == 1) {
    output = k;
    heads = k_heads;
    width = k_width;
    segment_bytes = batches * q_heads * q_width;
  }

  // Recover [s, b, n]. The packed address walks to this sequence row, then to the Q/K/V segment,
  // then to the [b, n] row inside that segment. The BF16 output address uses the equivalent
  // [B, S_GLOBAL, H_LOCAL, D] contiguous layout, which also restores the old permute for free.
  const int64_t n = tensor_row % heads;
  int64_t index = tensor_row / heads;
  const int64_t s = index % seq_len;
  const int64_t b = index / seq_len;
  const uint8_t* input = packed + s * chunk_row_bytes + segment_bytes + (b * heads + n) * width;
  output += (b * seq_len + s) * heads * head_size + n * head_size;

  // Quantized Q/K/V rows use FP32 scales: FP8 per-token scales follow the D-byte payload, FP6
  // scales follow its 4-byte-aligned 6-bit payload, and FP4 scales follow its nibble payload.
  // Grouped scales start at the same payload boundary. BF16 K skips scales and drops packed_D
  // padding.
  if (tensor_id == 1 && !quantize_k) {
    for (int64_t d = row_lane; d < head_size; d += kLanesPerRow) {
      // The packed K segment may start at an odd byte address after an odd-width FP4 Q segment.
      std::memcpy(output + d, input + d * sizeof(__nv_bfloat16), sizeof(__nv_bfloat16));
    }
  } else if (bits == 8) {
    if (group_size == 0) {
      float stored_scale;
      std::memcpy(&stored_scale, input + head_size, sizeof(stored_scale));
      dequantize_row(input, output, head_size, row_lane, stored_scale, kLanesPerRow);
    } else {
      dequantize_row_groups<false>(input, output, head_size, row_lane, group_size, kLanesPerRow);
    }
  } else if (bits == 6) {
#if CACHE_DIT_ULYSSES_HAS_FP6
    if (group_size == 0) {
      float scale;
      std::memcpy(&scale, input + fp6_payload_bytes(head_size), sizeof(scale));
      dequantize_row_fp6<false>(input, output, head_size, row_lane, scale, kLanesPerRow);
    } else {
      dequantize_row_groups_fp6<false>(input, output, head_size, row_lane, group_size,
                                       kLanesPerRow);
    }
#endif
  } else {
#if CACHE_DIT_ULYSSES_HAS_FP4
    dequantize_row_groups_fp4<false>(input, output, head_size, row_lane, group_size, kLanesPerRow);
#endif
  }
}

}  // namespace

void ulysses_fN_fused_qkv_dequant(at::Tensor packed, at::Tensor q, at::Tensor k, at::Tensor v,
                                  int64_t q_heads, int64_t k_heads, int64_t v_heads,
                                  int64_t head_size, int64_t q_width, int64_t k_width,
                                  int64_t v_width, int64_t chunk_row_bytes, int64_t group_size,
                                  int64_t bits, bool quantize_k) {
  TORCH_CHECK(packed.is_cuda() && q.is_cuda() && k.is_cuda() && v.is_cuda(),
              "fused QKV dequant requires CUDA tensors");
  TORCH_CHECK(packed.scalar_type() == at::kByte && packed.is_contiguous(),
              "fused QKV dequant input must be contiguous uint8");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16 &&
                v.scalar_type() == at::kBFloat16,
              "fused QKV dequant outputs must be BF16");
  TORCH_CHECK(packed.dim() == 2 && q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "fused QKV dequant expects a 2D input and [B,S,H,D] outputs");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
              "fused QKV dequant outputs must be contiguous");
  const int64_t batches = q.size(0);
  const int64_t seq_len = q.size(1);
  TORCH_CHECK(k.size(0) == batches && v.size(0) == batches && k.size(1) == seq_len &&
                v.size(1) == seq_len && q.size(2) == q_heads && k.size(2) == k_heads &&
                v.size(2) == v_heads && q.size(3) == head_size && k.size(3) == head_size &&
                v.size(3) == head_size,
              "invalid fused QKV dequant output shapes");
  TORCH_CHECK(packed.size(0) == seq_len && packed.size(1) == chunk_row_bytes,
              "invalid fused QKV dequant input shape");
  TORCH_CHECK(
    chunk_row_bytes == batches * (q_heads * q_width + k_heads * k_width + v_heads * v_width),
    "invalid fused QKV chunk row width");
  TORCH_CHECK(bits == 4 || bits == 6 || bits == 8, "fused QKV bits must be 4, 6, or 8");
  check_fp8_group_size(group_size);
  if (bits == 4) {
#if CACHE_DIT_ULYSSES_HAS_FP4
    check_fp4_device(packed);
    check_fp4_group_size(group_size);
#else
    TORCH_CHECK(false, "FP4 fused QKV requires CUDA >= 12.8 with cuda_fp4.h");
#endif
  }
  if (bits == 6) {
    check_fp6_device(packed);
    TORCH_CHECK(q_width % 4 == 0 && k_width % 4 == 0 && v_width % 4 == 0,
                "FP6 fused QKV row widths must be multiples of 4");
  }
  int64_t min_width = 0;
  if (bits == 8) {
    min_width = fp8_packed_min_width(head_size, group_size, 4);
  } else if (bits == 6) {
    min_width = fp6_packed_min_width(head_size, group_size, 4);
  }
#if CACHE_DIT_ULYSSES_HAS_FP4
  else if (bits == 4) {
    min_width = fp4_packed_min_width(head_size, group_size, 4);
  }
#endif
  TORCH_CHECK(q_width >= min_width && v_width >= min_width && (!quantize_k || k_width >= min_width),
              "fused QKV row widths leave no room for payload and scales");
  if (!quantize_k) {
    TORCH_CHECK(k_width == ((head_size * 2 + 15) / 16) * 16,
                "BF16 K width must be the legacy 16-byte-aligned row width");
  }
  const int64_t q_rows = seq_len * batches * q_heads;
  const int64_t k_rows = seq_len * batches * k_heads;
  const int64_t v_rows = seq_len * batches * v_heads;
  const int64_t total_rows = q_rows + k_rows + v_rows;
  if (total_rows == 0) return;

  const c10::cuda::CUDAGuard guard(packed.device());
  const auto stream = at::cuda::getCurrentCUDAStream(packed.device().index());
  // The host repeats the device's per-tensor ceil(rows/R) calculation. D=128 and 16-byte-aligned
  // BF16 outputs allow vectorized stores for every row; other shapes retain the full-warp fallback
  // rather than risking a partial segment or an unaligned vector store.
  auto launch = [&](auto lanes_tag) {
    constexpr int kLanes = decltype(lanes_tag)::value;
    constexpr int kRowsPerWarp = kWarpSize / kLanes;
    const int64_t warps = (q_rows + kRowsPerWarp - 1) / kRowsPerWarp +
                          (k_rows + kRowsPerWarp - 1) / kRowsPerWarp +
                          (v_rows + kRowsPerWarp - 1) / kRowsPerWarp;
    const dim3 grid((warps + kWarpsPerBlock - 1) / kWarpsPerBlock);
    fused_qkv_dequant_kernel<kLanes><<<grid, kThreads, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(packed.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(q.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(k.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(v.data_ptr()), batches, seq_len, q_heads, k_heads, v_heads,
      head_size, q_width, k_width, v_width, chunk_row_bytes, q_rows, k_rows, v_rows, group_size,
      bits, quantize_k);
  };
  const bool aligned_outputs = reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0 &&
                               reinterpret_cast<uintptr_t>(k.data_ptr()) % 16 == 0 &&
                               reinterpret_cast<uintptr_t>(v.data_ptr()) % 16 == 0;
  const bool use_multirow = head_size == 128 && total_rows >= 2048 && aligned_outputs;
  if (use_multirow && bits == 8) {
    launch(std::integral_constant<int, 16>{});
  } else if (use_multirow) {
    launch(std::integral_constant<int, 8>{});
  } else {
    launch(std::integral_constant<int, kWarpSize>{});
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace cache_dit::ulysses
