// FLUX.2-Klein q/k RMSNorm + GPT-J (interleaved-pair) RoPE, one read and one write.
//
// Eager reference (diffusers `transformers/transformer_flux2.py:353-370` and
// `models/embeddings.py:973-999`):
//   n   = attn.norm_q(q)                                 # torch.nn.RMSNorm(128, affine)
//   out = (n.float() * cos + rotate(n).float() * sin).to(bf16)
// with `rotate` swapping interleaved pairs: `stack([-x_imag, x_real]).flatten()`.
//
// Two roundings happen in that chain and both are replayed here:
//   1. `torch.nn.RMSNorm` keeps everything in fp32 and rounds once, after the affine. This is
//      not the diffusers `RMSNorm` schedule, which rounds twice; mixing them up costs ~2.8e-3
//      rel-L2, so they are not interchangeable (schedule table:
//      `xDiT .../fused_qk_rope_flydsl.py:726-757`).
//   2. `apply_rotary_emb` rotates in fp32 and rounds once at the end, and it consumes the
//      *rounded* bf16 normalised value, not the fp32 one.
// Both are verified against the eager expression: the rounding sequence matches, and the RMS
// reduction tree leaves roughly 0.5% of elements one bf16 ulp off (worst rel-L2 2.8e-6).
//
// Layout: 16 lanes cover one (token, head) row with 8 channels (4 interleaved pairs) per lane,
// so every global access is a 16-byte vector and a GPT-J pair never straddles two lanes; a
// warp covers two rows, which keeps the reduction inside one warp.
//
// Reading and writing once is the point. Inductor's fused kernel for this chain measured a
// 12:1 load-to-store sector ratio (NCU: 21.45M vs 1.77M sectors) because the row is loaded
// again once the statistics are known; holding the row in registers removes that pass.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "flux2_klein.h"

namespace cache_dit::models {
namespace {

constexpr int kWarpsPerBlock = 4;
constexpr int kLanesPerRow = 16;
constexpr int kPairsPerLane = 4;  // 4 pairs = 8 channels = 16 bytes
constexpr int kPairs = 64;        // head_dim 128 -> 64 interleaved pairs

struct Strides {
  int64_t token;
  int64_t head;
};

__global__ void flux2_klein_qk_norm_rope_kernel(
  const __nv_bfloat162* __restrict__ x, const __nv_bfloat162* __restrict__ weight,
  const float* __restrict__ cos, const float* __restrict__ sin, __nv_bfloat162* __restrict__ out,
  Strides in_strides, Strides strides, int64_t rows, int seq_len, int heads, int rope_row_offset,
  int64_t out_token_offset, float eps) {
  const int lane = threadIdx.x & 31;
  const int row_in_pair = lane >> 4;
  const int lid = lane & 15;
  const int warp = threadIdx.x >> 5;
  const int64_t row = ((int64_t)blockIdx.x * kWarpsPerBlock + warp) * 2 + row_in_pair;
  const unsigned group_mask = (row_in_pair == 0) ? 0x0000FFFFu : 0xFFFF0000u;

  if (row >= rows) {
    return;
  }

  // Row order is (token, head) with the head varying fastest. The source is addressed through
  // its own strides because the real callers hand over a chunk of a fused projection
  // (`to_qkv_mlp_proj(...).chunk(3, -1)`), whose rows are packed in `(H, D)` but whose token
  // stride spans the whole fused width. Reading it in place avoids a `contiguous()` copy.
  const int head = (int)(row % heads);
  const int64_t token = row / heads;
  const int pos = (int)(token % seq_len);
  const __nv_bfloat162* x_row = x + token * in_strides.token + (int64_t)head * in_strides.head;

  const uint4 xv = *reinterpret_cast<const uint4*>(x_row + lid * kPairsPerLane);
  const __nv_bfloat162* xq = reinterpret_cast<const __nv_bfloat162*>(&xv);

  float ss = 0.0f;
#pragma unroll
  for (int j = 0; j < kPairsPerLane; ++j) {
    const float2 v = __bfloat1622float2(xq[j]);
    ss += v.x * v.x;
    ss += v.y * v.y;
  }
  // Down-tree broadcast by offset, so every lane of the row ends with the same total.
#pragma unroll
  for (int offset = kLanesPerRow / 2; offset > 0; offset >>= 1) {
    ss += __shfl_down_sync(group_mask, ss, offset, kLanesPerRow);
  }
  ss = __shfl_sync(group_mask, ss, 0, kLanesPerRow);
  const float rstd = rsqrtf(ss / (float)(kPairs * 2) + eps);

  const uint4 wv = *reinterpret_cast<const uint4*>(weight + lid * kPairsPerLane);
  const __nv_bfloat162* wq = reinterpret_cast<const __nv_bfloat162*>(&wv);
  // The tables are full width (`repeat_interleave_real=True`), one entry per *channel*, so this
  // lane needs 8 floats - channels [8 * lid, 8 * lid + 8) - and pair `j` reads the entries of
  // channels `2j` and `2j + 1`, exactly like the elementwise eager rotation.
  const int64_t table = ((int64_t)(rope_row_offset + pos) * kPairs * 2) + lid * (kPairsPerLane * 2);
  const float4 c40 = *reinterpret_cast<const float4*>(cos + table);
  const float4 c41 = *reinterpret_cast<const float4*>(cos + table + 4);
  const float4 s40 = *reinterpret_cast<const float4*>(sin + table);
  const float4 s41 = *reinterpret_cast<const float4*>(sin + table + 4);
  const float cs[8] = {c40.x, c40.y, c40.z, c40.w, c41.x, c41.y, c41.z, c41.w};
  const float sn[8] = {s40.x, s40.y, s40.z, s40.w, s41.x, s41.y, s41.z, s41.w};

  uint4 rv;
  __nv_bfloat162* rq = reinterpret_cast<__nv_bfloat162*>(&rv);
#pragma unroll
  for (int j = 0; j < kPairsPerLane; ++j) {
    const float2 e = __bfloat1622float2(xq[j]);
    const float2 w = __bfloat1622float2(wq[j]);
    // Rounding 1: fp32 norm and affine, one bf16 round at the end.
    const __nv_bfloat162 normed = {__float2bfloat16_rn(e.x * rstd * w.x),
                                   __float2bfloat16_rn(e.y * rstd * w.y)};
    // The rotation consumes the rounded pair, exactly like the eager chain.
    const float r = __bfloat162float(normed.x);
    const float i = __bfloat162float(normed.y);
    // Rounding 2: fp32 rotate, one bf16 round. `__fmul_rn`/add keep nvcc from contracting a
    // product into an FMA that the eager expression does not perform.
    // Interleaved (GPT-J) pairs: `x_rotated` is `[-x_imag, x_real]`, so the even channel
    // subtracts the rotated sine and the odd channel adds its own sine term.
    rq[j].x = __float2bfloat16_rn(__fadd_rn(__fmul_rn(r, cs[2 * j]), __fmul_rn(-i, sn[2 * j])));
    rq[j].y =
      __float2bfloat16_rn(__fadd_rn(__fmul_rn(r, sn[2 * j + 1]), __fmul_rn(i, cs[2 * j + 1])));
  }

  const int64_t dst = (out_token_offset + token) * strides.token + (int64_t)head * strides.head +
                      (int64_t)lid * kPairsPerLane;
  *reinterpret_cast<uint4*>(out + dst) = rv;
}

bool is_aligned(const void* p, uintptr_t bytes) {
  return (reinterpret_cast<uintptr_t>(p) & (bytes - 1)) == 0;
}

void check_stream(const at::Tensor& x, const at::Tensor& weight, const at::Tensor& cos,
                  const at::Tensor& sin, const at::Tensor& out, int64_t rope_row_offset,
                  int64_t out_tokens, int64_t out_token_offset, const char* op) {
  // The source is read through its own strides, so a chunk of a fused projection (`[1, S,
  // 3 * inner]` split on the last dim) is valid input: rows are packed in `(H, D)`, which is
  // all the kernel needs, and the 16-byte vector load only requires aligned row strides.
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 4, op,
              ": x must be a CUDA bf16 [B,S,H,D] tensor");
  TORCH_CHECK(x.size(0) == 1, op, ": only batch 1 is supported, got ", x.size(0));
  TORCH_CHECK(x.size(3) == 128, op, ": only head_dim 128 is supported, got ", x.size(3));
  TORCH_CHECK(x.stride(3) == 1 && x.stride(2) % 8 == 0 && x.stride(1) % 8 == 0, op,
              ": x must be dense along D with 16-byte aligned token/head strides, got ",
              x.stride(1), "/", x.stride(2), "/", x.stride(3));
  const int64_t seq_len = x.size(1);
  TORCH_CHECK(weight.is_cuda() && weight.scalar_type() == at::kBFloat16 && weight.is_contiguous() &&
                weight.numel() == x.size(3) && weight.storage_offset() == 0,
              op, ": weight must be a contiguous CUDA bf16 [D] tensor");
  TORCH_CHECK(cos.is_cuda() && cos.scalar_type() == at::kFloat && cos.is_contiguous() &&
                cos.dim() == 2 && cos.size(1) == x.size(3) && cos.storage_offset() == 0 &&
                rope_row_offset + seq_len <= cos.size(0),
              op, ": cos must be a contiguous fp32 [rope_rows, D] table covering this stream");
  TORCH_CHECK(sin.is_cuda() && sin.scalar_type() == at::kFloat && sin.is_contiguous() &&
                sin.numel() == cos.numel() && sin.storage_offset() == 0,
              op, ": sin must have the same layout as cos");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kBFloat16 && out.dim() == 3, op,
              ": out must be a CUDA bf16 [S_total, H, D] tensor");
  TORCH_CHECK(out.size(0) == out_tokens && out.size(1) == x.size(2) && out.size(2) == x.size(3), op,
              ": out must be [S_total, H, D] matching heads and head_dim");
  TORCH_CHECK(out_token_offset >= 0 && out_token_offset + seq_len <= out.size(0), op,
              ": out token range does not fit");
  TORCH_CHECK(out.stride(2) == 1 && out.stride(1) % 8 == 0 && out.stride(0) % 8 == 0, op,
              ": out must be dense along D with 16-byte aligned outer strides");
  TORCH_CHECK(is_aligned(x.const_data_ptr(), 16) && is_aligned(weight.const_data_ptr(), 16) &&
                is_aligned(cos.const_data_ptr(), 16) && is_aligned(sin.const_data_ptr(), 16) &&
                is_aligned(out.data_ptr(), 16),
              op, ": buffers must be 16-byte aligned for the uint4 path");
}

void launch(const at::Tensor& x, const at::Tensor& weight, const at::Tensor& cos,
            const at::Tensor& sin, at::Tensor& out, int64_t rope_row_offset,
            int64_t out_token_offset, double eps, cudaStream_t stream) {
  const int64_t seq_len = x.size(1);
  const int64_t heads = x.size(2);
  const int64_t rows = seq_len * heads;
  if (rows == 0) {
    return;
  }
  // Strides are expressed in bf16 pairs, which is the unit every pointer here is advanced in.
  const Strides in_strides{x.stride(1) / 2, x.stride(2) / 2};
  const Strides strides{out.stride(0) / 2, out.stride(1) / 2};
  const int64_t warp_count = (rows + 1) / 2;
  const int64_t grid = (warp_count + kWarpsPerBlock - 1) / kWarpsPerBlock;
  flux2_klein_qk_norm_rope_kernel<<<grid, kWarpsPerBlock * 32, 0, stream>>>(
    reinterpret_cast<const __nv_bfloat162*>(x.const_data_ptr()),
    reinterpret_cast<const __nv_bfloat162*>(weight.const_data_ptr()), cos.const_data_ptr<float>(),
    sin.const_data_ptr<float>(), reinterpret_cast<__nv_bfloat162*>(out.data_ptr()), in_strides,
    strides, rows, (int)seq_len, (int)heads, (int)rope_row_offset, out_token_offset, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void flux2_klein_qk_norm_rope(at::Tensor x, at::Tensor weight, at::Tensor cos, at::Tensor sin,
                              int64_t rope_row_offset, double eps, at::Tensor out,
                              int64_t out_token_offset) {
  // The single-stream form owns the whole buffer, so the token range is not a free parameter;
  // stating it here keeps `check_stream`'s `out_tokens` argument meaningful (the joined form
  // passes the two-segment total instead).
  TORCH_CHECK(out_token_offset == 0 && out.size(0) == x.size(1), "flux2_klein_qk_norm_rope",
              ": the single-stream form writes all ", x.size(1), " tokens at offset 0, got offset ",
              out_token_offset, " into a buffer of ", out.size(0), " tokens");
  check_stream(x, weight, cos, sin, out, rope_row_offset, out.size(0), out_token_offset,
               "flux2_klein_qk_norm_rope");
  if (x.size(1) == 0) {
    return;
  }
  const c10::cuda::CUDAGuard guard(x.device());
  launch(x, weight, cos, sin, out, rope_row_offset, out_token_offset, eps,
         at::cuda::getCurrentCUDAStream());
}

void flux2_klein_qk_norm_rope_join(at::Tensor image, at::Tensor image_weight, at::Tensor text,
                                   at::Tensor text_weight, at::Tensor cos, at::Tensor sin,
                                   double eps, at::Tensor out) {
  const char* op = "flux2_klein_qk_norm_rope_join";
  const int64_t text_tokens = text.size(1);
  const int64_t image_tokens = image.size(1);
  // The text stream owns [0, S_text) and reads the table from row 0; the image stream follows
  // and reads it from row S_text, which is exactly the `cat([text_rope, image_rope])` table.
  const int64_t total = text_tokens + image_tokens;
  check_stream(text, text_weight, cos, sin, out, 0, total, 0, op);
  check_stream(image, image_weight, cos, sin, out, text_tokens, total, text_tokens, op);
  TORCH_CHECK(
    text.size(0) == image.size(0) && text.size(2) == image.size(2) && text.size(3) == image.size(3),
    op, ": both streams must share batch, heads and head_dim");
  if (text_tokens + image_tokens == 0) {
    return;
  }
  const c10::cuda::CUDAGuard guard(image.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  launch(text, text_weight, cos, sin, out, 0, 0, eps, stream);
  launch(image, image_weight, cos, sin, out, text_tokens, text_tokens, eps, stream);
}

}  // namespace cache_dit::models
