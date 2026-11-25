#include <type_traits>

#include "fp4_common.cuh"
#include "fp6_common.cuh"

// Fused QKV wire format (all offsets and widths are in bytes).
//
// The output is a 2D uint8 tensor [P * S_LOCAL, chunk_row_bytes]. Each sequence row contains all
// three tensors for one (partition p, local sequence s), in this exact order:
//
//   +-------------------------------+-------------------------------+-------------------------------+
//   | Q[p, s]                     | K[p, s]                       | V[p, s]                       |
//   | [B, Hq_LOCAL, Wq]           | [B, Hk_LOCAL, Wk]             | [B, Hv_LOCAL, Wv]             |
//   +-------------------------------+-------------------------------+-------------------------------+
//
// Q/K/V widths may differ (GQA, or BF16 K mixed with FP4/FP6/FP8 Q/V). Keeping the three legacy
// chunks inside each sequence row is what makes one all-to-all equivalent to the three old
// collectives. The collective splits the first dimension in sequence rows, never in the middle of
// this Q||K||V row. Every individual chunk is 16-byte aligned, so their sum is aligned as well.

// Multirow execution principle (independent of the wire format above).
//
// A full warp has 32 physical lanes, but one D=128 row does not need all of them on the vectorized
// path. FP8 processes 8 elements/lane, so only 16 lanes are required per row; FP6 processes 16
// elements/lane, so only 8 lanes are required. We therefore split one warp into independent row
// segments instead of leaving the remaining lanes idle:
//
//   FP8: lanes  0..15 -> row 0, lanes 16..31 -> row 1       (2 rows/warp)
//   FP6: lanes  0.. 7 -> row 0, lanes  8..15 -> row 1       (4 rows/warp)
//        lanes 16..23 -> row 2, lanes 24..31 -> row 3
//
// For kLanesPerRow=L and physical lane `lane`:
//
//   row_in_warp = lane / L       // which row inside this warp
//   row_lane    = lane % L       // lane index inside that row, always in [0, L)
//   tensor_row  = tensor_warp_index * (32 / L) + row_in_warp
//
// `row_lane`, rather than the physical lane, is passed to quant helpers. Per-token absmax is
// reduced only within the L lanes of each row; group reductions are smaller nested segments inside
// that row. Thus every row computes and writes its own scale(s), with no shuffle crossing rows.
// Smaller blocks and fewer idle lanes reduce launch/index overhead; packed bytes and row ordering
// are exactly the same as the one-row-per-warp implementation.

namespace cache_dit::ulysses {
namespace {

template <int kLanesPerRow>
__global__ void fused_qkv_quant_kernel(const __nv_bfloat16* q, const __nv_bfloat16* k,
                                       const __nv_bfloat16* v, uint8_t* y, int64_t partitions,
                                       int64_t batches, int64_t seq_len, int64_t q_heads,
                                       int64_t k_heads, int64_t v_heads, int64_t head_size,
                                       int64_t q_stride_b, int64_t q_stride_s, int64_t q_stride_h,
                                       int64_t k_stride_b, int64_t k_stride_s, int64_t k_stride_h,
                                       int64_t v_stride_b, int64_t v_stride_s, int64_t v_stride_h,
                                       int64_t q_width, int64_t k_width, int64_t v_width,
                                       int64_t chunk_row_bytes, int64_t q_rows, int64_t k_rows,
                                       int64_t v_rows, int group_size, int bits, bool quantize_k) {
  // A row means one [B, S, H, D] head vector of length D. Each tensor has P*S*B*H rows in
  // [P, S, B, H] order. Q, K, and V get independent ceil(rows / R) warp ranges, so a warp never
  // crosses a tensor boundary. R=1 is the generic path; D=128 uses R=2 for FP8 and R=4 for FP6.
  // This both fills all 32 lanes and keeps mixed Q/K/V formats uniform within each warp.
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

  // tensor_id identifies the QKV component: 0=Q, 1=K, 2=V. It is not a partition/head index.
  // Each component has its own local rows, input pointer/strides, head count, packed width and
  // byte-segment offset. tensor_id==1 is needed below to apply the optional BF16-K bypass only
  // to K; Q and V always use the selected FP4/FP6/FP8 format.
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

  const __nv_bfloat16* input = q;
  int64_t heads = q_heads;
  int64_t width = q_width;
  int64_t stride_b = q_stride_b;
  int64_t stride_s = q_stride_s;
  int64_t stride_h = q_stride_h;
  int64_t segment_bytes = 0;
  if (tensor_id == 2) {
    input = v;
    heads = v_heads;
    width = v_width;
    stride_b = v_stride_b;
    stride_s = v_stride_s;
    stride_h = v_stride_h;
    segment_bytes = batches * (q_heads * q_width + k_heads * k_width);
  } else if (tensor_id == 1) {
    input = k;
    heads = k_heads;
    width = k_width;
    stride_b = k_stride_b;
    stride_s = k_stride_s;
    stride_h = k_stride_h;
    segment_bytes = batches * q_heads * q_width;
  }

  // Recover [p, s, b, n] from the tensor-local row. The input is still the original padded
  // [B, S_LOCAL, H_GLOBAL, D] tensor, so partition p selects global head p*H_LOCAL+n directly;
  // no reshaped input copy is needed. The output row is [p*S+s][segment][b*H_LOCAL+n].
  const int64_t n = tensor_row % heads;
  int64_t index = tensor_row / heads;
  const int64_t b = index % batches;
  index /= batches;
  const int64_t s = index % seq_len;
  const int64_t p = index / seq_len;
  input += b * stride_b + s * stride_s + (p * heads + n) * stride_h;
  uint8_t* output =
    y + (p * seq_len + s) * chunk_row_bytes + segment_bytes + (b * heads + n) * width;

  // row_lane is relative to the current row, not the physical warp. Per-token absmax and group
  // reductions are segmented by kLanesPerRow, and each row's lane 0 writes its own scale(s).
  // BF16 K is the only bypass: it uses the legacy 16-byte-aligned packed_D and zeroes its D tail.
  // QKV scales are FP32 for FP4, FP6, and FP8; BF16 scales belong to separate O/generic paths.
  if (tensor_id == 1 && !quantize_k) {
    const int64_t packed_head_size = width / 2;
    for (int64_t d = row_lane; d < packed_head_size; d += kLanesPerRow) {
      // FP4 permits odd packed widths, so a preceding Q segment can leave K on an odd byte address.
      if (d < head_size) {
        std::memcpy(output + d * sizeof(__nv_bfloat16), input + d, sizeof(__nv_bfloat16));
      } else {
        std::memset(output + d * sizeof(__nv_bfloat16), 0, sizeof(__nv_bfloat16));
      }
    }
  } else if (bits == 8) {
    if (group_size == 0) {
      quantize_row_per_token<false>(input, output, head_size, width, row_lane, 1e-6f, kLanesPerRow);
    } else {
      quantize_row_groups<false>(input, output, head_size, width, row_lane, group_size, 1e-6f,
                                 kLanesPerRow);
    }
  } else if (bits == 6) {
#if CACHE_DIT_ULYSSES_HAS_FP6
    if (group_size == 0) {
      quantize_row_per_token_fp6<false>(input, output, head_size, width, row_lane, 1e-6f,
                                        kLanesPerRow);
    } else {
      quantize_row_groups_fp6<false>(input, output, head_size, width, row_lane, group_size, 1e-6f,
                                     kLanesPerRow);
    }
#endif
  } else {
#if CACHE_DIT_ULYSSES_HAS_FP4
    quantize_row_groups_fp4<false>(input, output, head_size, width, row_lane, group_size, 1e-6f,
                                   kLanesPerRow);
#endif
  }
}

}  // namespace

void ulysses_fN_fused_qkv_quant(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor y,
                                int64_t partitions, int64_t q_heads, int64_t k_heads,
                                int64_t v_heads, int64_t q_width, int64_t k_width, int64_t v_width,
                                int64_t chunk_row_bytes, int64_t group_size, int64_t bits,
                                bool quantize_k) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && y.is_cuda(),
              "fused QKV quant requires CUDA tensors");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16 &&
                v.scalar_type() == at::kBFloat16,
              "fused QKV quant expects BF16 inputs");
  TORCH_CHECK(y.scalar_type() == at::kByte && y.is_contiguous(),
              "fused QKV quant output must be contiguous uint8");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && y.dim() == 2,
              "fused QKV quant expects [B,S,H,D] inputs and a 2D output");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0) && q.size(1) == k.size(1) &&
                q.size(1) == v.size(1) && q.size(3) == k.size(3) && q.size(3) == v.size(3),
              "fused QKV inputs must share B, S, and D");
  TORCH_CHECK(q.stride(3) == 1 && k.stride(3) == 1 && v.stride(3) == 1,
              "fused QKV head dimensions must be contiguous");
  TORCH_CHECK(partitions > 0 && q_heads > 0 && k_heads > 0 && v_heads > 0,
              "fused QKV partitions and local heads must be positive");
  TORCH_CHECK(bits == 4 || bits == 6 || bits == 8, "fused QKV bits must be 4, 6, or 8");
  check_fp8_group_size(group_size);
  if (bits == 4) {
#if CACHE_DIT_ULYSSES_HAS_FP4
    check_fp4_device(q);
    check_fp4_group_size(group_size);
#else
    TORCH_CHECK(false, "FP4 fused QKV requires CUDA >= 12.8 with cuda_fp4.h");
#endif
  }
  if (bits == 6) {
    check_fp6_device(q);
    TORCH_CHECK(q_width % 4 == 0 && k_width % 4 == 0 && v_width % 4 == 0,
                "FP6 fused QKV row widths must be multiples of 4");
  }
  const int64_t batches = q.size(0);
  const int64_t seq_len = q.size(1);
  const int64_t head_size = q.size(3);
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
  const int64_t q_rows = partitions * seq_len * batches * q_heads;
  const int64_t k_rows = partitions * seq_len * batches * k_heads;
  const int64_t v_rows = partitions * seq_len * batches * v_heads;
  TORCH_CHECK(y.size(0) == partitions * seq_len && y.size(1) == chunk_row_bytes,
              "invalid fused QKV output shape");
  TORCH_CHECK(
    chunk_row_bytes == batches * (q_heads * q_width + k_heads * k_width + v_heads * v_width),
    "invalid fused QKV chunk row width");
  if (!quantize_k) {
    TORCH_CHECK(k_width == ((head_size * 2 + 15) / 16) * 16,
                "BF16 K width must be the legacy 16-byte-aligned row width");
  }
  const int64_t total_rows = q_rows + k_rows + v_rows;
  if (total_rows == 0) return;

  const c10::cuda::CUDAGuard guard(q.device());
  const auto stream = at::cuda::getCurrentCUDAStream(q.device().index());
  // Keep host warp accounting identical to the device mapping: rounding is per tensor, not over
  // q_rows+k_rows+v_rows as one pooled range. Multirow is enabled only when the aligned D=128 fast
  // helper is valid for every input row; small grids and arbitrary strides use the full-warp path.
  auto launch = [&](auto lanes_tag) {
    constexpr int kLanes = decltype(lanes_tag)::value;
    constexpr int kRowsPerWarp = kWarpSize / kLanes;
    const int64_t warps = (q_rows + kRowsPerWarp - 1) / kRowsPerWarp +
                          (k_rows + kRowsPerWarp - 1) / kRowsPerWarp +
                          (v_rows + kRowsPerWarp - 1) / kRowsPerWarp;
    const dim3 grid((warps + kWarpsPerBlock - 1) / kWarpsPerBlock);
    fused_qkv_quant_kernel<kLanes><<<grid, kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
      reinterpret_cast<uint8_t*>(y.data_ptr()), partitions, batches, seq_len, q_heads, k_heads,
      v_heads, head_size, q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1),
      k.stride(2), v.stride(0), v.stride(1), v.stride(2), q_width, k_width, v_width,
      chunk_row_bytes, q_rows, k_rows, v_rows, group_size, bits, quantize_k);
  };
  const bool aligned_inputs =
    reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0 &&
    reinterpret_cast<uintptr_t>(k.data_ptr()) % 16 == 0 &&
    reinterpret_cast<uintptr_t>(v.data_ptr()) % 16 == 0 && q.stride(0) % kPackSize == 0 &&
    q.stride(1) % kPackSize == 0 && q.stride(2) % kPackSize == 0 && k.stride(0) % kPackSize == 0 &&
    k.stride(1) % kPackSize == 0 && k.stride(2) % kPackSize == 0 && v.stride(0) % kPackSize == 0 &&
    v.stride(1) % kPackSize == 0 && v.stride(2) % kPackSize == 0;
  const bool use_multirow = head_size == 128 && total_rows >= 2048 && aligned_inputs;
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
