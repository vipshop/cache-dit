// FLUX.3 Klein LayerNorm + `(1 + scale) * x + shift`, and its two-stream concatenating variant.
//
// Numerics: the block norms are `nn.LayerNorm(dim, elementwise_affine=False)` and upstream then
// evaluates `(1 + mod_scale) * norm_hidden_states + mod_shift` as two separate aten calls, i.e.
// with a bf16 rounding in between. The compiled step fuses that chain into one kernel and keeps
// the intermediate in fp32, so this kernel does the same: `(x - mean) * rstd` in fp32, then the
// modulation in fp32, then one bf16 store. Both the sum and the sum of squared deviations are
// accumulated in fp32 over registers (a re-read would double the traffic).
//
// Layout: one row per CTA, 128 threads, `uint4` (8 bf16) per access, `kVecsPerThread` accesses
// per thread. That keeps a whole row in registers (24 floats at dim=3072) and turns the
// reduction into one warp shuffle tree plus a four-value shared array. ncu measured inductor's
// equivalent kernel at 93-95% of DRAM peak, so this operator is about bytes, not access
// patterns: the concatenating variant is the one that can remove a pass, by writing the joint
// `[text, content]` buffer directly instead of materialising two normalized tensors and copying
// both again.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "flux3_klein.h"

namespace cache_dit::models {
namespace {

constexpr int kThreads = 128;
constexpr int kWarps = kThreads / 32;
constexpr int kChannelsPerVec = 8;
constexpr int kMaxVecsPerThread = 8;

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ float block_reduce_sum(float value, float* scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = warp_reduce_sum(value);
  if (lane == 0) {
    scratch[warp] = value;
  }
  __syncthreads();
  value = (warp == 0) ? ((lane < kWarps) ? scratch[lane] : 0.0f) : 0.0f;
  if (warp == 0) {
    value = warp_reduce_sum(value);
    if (lane == 0) {
      scratch[0] = value;
    }
  }
  __syncthreads();
  const float total = scratch[0];
  __syncthreads();
  return total;
}

template <int kVecsPerThread>
__device__ __forceinline__ void ln_modulate_row(const __nv_bfloat162* __restrict__ x,
                                                const __nv_bfloat162* __restrict__ scale,
                                                const __nv_bfloat162* __restrict__ shift,
                                                __nv_bfloat162* __restrict__ out, int64_t dim,
                                                float eps, float* scratch) {
  const int tid = threadIdx.x;
  float values[kVecsPerThread][kChannelsPerVec];

  float local_sum = 0.0f;
#pragma unroll
  for (int v = 0; v < kVecsPerThread; ++v) {
    const uint4 raw = *reinterpret_cast<const uint4*>(x + tid * kVecsPerThread * 4 + v * 4);
    const __nv_bfloat162* pair = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = __bfloat1622float2(pair[j]);
      values[v][2 * j] = f.x;
      values[v][2 * j + 1] = f.y;
      local_sum += f.x;
      local_sum += f.y;
    }
  }

  const float mean = block_reduce_sum(local_sum, scratch) / float(dim);

  float local_var = 0.0f;
#pragma unroll
  for (int v = 0; v < kVecsPerThread; ++v) {
#pragma unroll
    for (int j = 0; j < kChannelsPerVec; ++j) {
      const float centered = values[v][j] - mean;
      local_var = __fmaf_rn(centered, centered, local_var);
    }
  }
  const float variance = block_reduce_sum(local_var, scratch) / float(dim);
  const float rstd = rsqrtf(variance + eps);

#pragma unroll
  for (int v = 0; v < kVecsPerThread; ++v) {
    const int64_t offset = int64_t(tid) * kVecsPerThread * 4 + v * 4;
    const uint4 scale_raw = *reinterpret_cast<const uint4*>(scale + offset);
    const uint4 shift_raw = *reinterpret_cast<const uint4*>(shift + offset);
    const __nv_bfloat162* scale_pair = reinterpret_cast<const __nv_bfloat162*>(&scale_raw);
    const __nv_bfloat162* shift_pair = reinterpret_cast<const __nv_bfloat162*>(&shift_raw);
    uint4 result;
    __nv_bfloat162* result_pair = reinterpret_cast<__nv_bfloat162*>(&result);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 s = __bfloat1622float2(scale_pair[j]);
      const float2 b = __bfloat1622float2(shift_pair[j]);
      const float y0 =
        __fadd_rn(__fmul_rn(__fmul_rn(values[v][2 * j] - mean, rstd), 1.0f + s.x), b.x);
      const float y1 =
        __fadd_rn(__fmul_rn(__fmul_rn(values[v][2 * j + 1] - mean, rstd), 1.0f + s.y), b.y);
      result_pair[j].x = __float2bfloat16_rn(y0);
      result_pair[j].y = __float2bfloat16_rn(y1);
    }
    *reinterpret_cast<uint4*>(out + offset) = result;
  }
}

template <int kVecsPerThread>
__global__ void ln_modulate_kernel(const __nv_bfloat162* __restrict__ x,
                                   const __nv_bfloat162* __restrict__ scale,
                                   const __nv_bfloat162* __restrict__ shift,
                                   __nv_bfloat162* __restrict__ out, int64_t seq_len, int64_t dim,
                                   int64_t scale_stride_b, int64_t scale_stride_s, float eps) {
  __shared__ float scratch[kWarps];
  const int64_t row = blockIdx.x;
  const int64_t batch = row / seq_len;
  const int64_t token = row - batch * seq_len;
  const int64_t pairs = dim / 2;
  const int64_t modulation_row = batch * scale_stride_b + token * scale_stride_s;
  ln_modulate_row<kVecsPerThread>(x + row * pairs, scale + modulation_row, shift + modulation_row,
                                  out + row * pairs, dim, eps, scratch);
}

template <int kVecsPerThread>
__global__ void ln_modulate_cat_kernel(const __nv_bfloat162* __restrict__ hidden_states,
                                       const __nv_bfloat162* __restrict__ encoder_hidden_states,
                                       const __nv_bfloat162* __restrict__ scale,
                                       const __nv_bfloat162* __restrict__ shift,
                                       const __nv_bfloat162* __restrict__ context_scale,
                                       const __nv_bfloat162* __restrict__ context_shift,
                                       __nv_bfloat162* __restrict__ out, int64_t prefix_len,
                                       int64_t seq_len, int64_t dim, int64_t scale_stride_b,
                                       int64_t scale_stride_s, int64_t context_stride_b,
                                       int64_t context_stride_s, float eps) {
  __shared__ float scratch[kWarps];
  const int64_t joint = prefix_len + seq_len;
  const int64_t row = blockIdx.x;
  const int64_t pairs = dim / 2;
  const int64_t batch = row / joint;
  const int64_t token = row - batch * joint;
  if (token < prefix_len) {
    const int64_t modulation_row = batch * context_stride_b + token * context_stride_s;
    ln_modulate_row<kVecsPerThread>(encoder_hidden_states + (batch * prefix_len + token) * pairs,
                                    context_scale + modulation_row, context_shift + modulation_row,
                                    out + row * pairs, dim, eps, scratch);
  } else {
    const int64_t content = token - prefix_len;
    const int64_t modulation_row = batch * scale_stride_b + content * scale_stride_s;
    ln_modulate_row<kVecsPerThread>(hidden_states + (batch * seq_len + content) * pairs,
                                    scale + modulation_row, shift + modulation_row,
                                    out + row * pairs, dim, eps, scratch);
  }
}

int64_t vecs_per_thread(int64_t dim) { return dim / kChannelsPerVec / kThreads; }

template <int kVecsPerThread>
void launch_ln_modulate(const at::Tensor& x, const at::Tensor& scale, const at::Tensor& shift,
                        at::Tensor& out, int64_t scale_stride_b, int64_t scale_stride_s, double eps,
                        cudaStream_t stream) {
  const int64_t seq_len = x.size(1);
  const int64_t dim = x.size(2);
  const int64_t rows = x.numel() / dim;
  auto* x_ptr = reinterpret_cast<const __nv_bfloat162*>(x.const_data_ptr());
  auto* scale_ptr = reinterpret_cast<const __nv_bfloat162*>(scale.const_data_ptr());
  auto* shift_ptr = reinterpret_cast<const __nv_bfloat162*>(shift.const_data_ptr());
  auto* out_ptr = reinterpret_cast<__nv_bfloat162*>(out.data_ptr());
  ln_modulate_kernel<kVecsPerThread>
    <<<rows, kThreads, 0, stream>>>(x_ptr, scale_ptr, shift_ptr, out_ptr, seq_len, dim,
                                    scale_stride_b / 2, scale_stride_s / 2, float(eps));
}

template <int kVecsPerThread>
void launch_ln_modulate_cat(const at::Tensor& hidden_states,
                            const at::Tensor& encoder_hidden_states, const at::Tensor& scale,
                            const at::Tensor& shift, const at::Tensor& context_scale,
                            const at::Tensor& context_shift, at::Tensor& out,
                            int64_t scale_stride_b, int64_t scale_stride_s,
                            int64_t context_stride_b, int64_t context_stride_s, double eps,
                            cudaStream_t stream) {
  const int64_t prefix_len = encoder_hidden_states.size(1);
  const int64_t seq_len = hidden_states.size(1);
  const int64_t dim = hidden_states.size(2);
  const int64_t batch = hidden_states.size(0);
  auto* hidden_ptr = reinterpret_cast<const __nv_bfloat162*>(hidden_states.const_data_ptr());
  auto* encoder_ptr =
    reinterpret_cast<const __nv_bfloat162*>(encoder_hidden_states.const_data_ptr());
  auto* out_ptr = reinterpret_cast<__nv_bfloat162*>(out.data_ptr());
  ln_modulate_cat_kernel<kVecsPerThread><<<batch*(prefix_len + seq_len), kThreads, 0, stream>>>(
    hidden_ptr, encoder_ptr, reinterpret_cast<const __nv_bfloat162*>(scale.const_data_ptr()),
    reinterpret_cast<const __nv_bfloat162*>(shift.const_data_ptr()),
    reinterpret_cast<const __nv_bfloat162*>(context_scale.const_data_ptr()),
    reinterpret_cast<const __nv_bfloat162*>(context_shift.const_data_ptr()), out_ptr, prefix_len,
    seq_len, dim, scale_stride_b / 2, scale_stride_s / 2, context_stride_b / 2,
    context_stride_s / 2, float(eps));
}

bool is_aligned(const void* pointer, uintptr_t bytes) {
  return (reinterpret_cast<uintptr_t>(pointer) & (bytes - 1)) == 0;
}

void check_modulation(const at::Tensor& modulation, const at::Tensor& like, const char* name) {
  // `temb_mod.chunk(3, dim=-1)[i]` is what every caller passes: dense along the channels, three
  // widths apart along the rows. Only the inner axis has to be contiguous, and the row stride is
  // handed to the kernel.
  TORCH_CHECK(modulation.is_cuda() && modulation.scalar_type() == like.scalar_type() &&
                modulation.dim() == 3 && modulation.stride(2) == 1,
              "flux3_klein_ln_modulate: ", name,
              " must be a CUDA tensor of the model dtype with three "
              "dimensions and a dense channel axis");
  TORCH_CHECK(modulation.size(0) == like.size(0) && modulation.size(2) == like.size(2),
              "flux3_klein_ln_modulate: ", name, " must match the batch and channel axes of x");
  TORCH_CHECK(modulation.size(1) == 1 || modulation.size(1) == like.size(1),
              "flux3_klein_ln_modulate: ", name, " must have one row per sample or per token");
  TORCH_CHECK(
    is_aligned(modulation.const_data_ptr(), 16) &&
      (modulation.stride(0) * modulation.element_size()) % 16 == 0 &&
      (modulation.size(1) == 1 || (modulation.stride(1) * modulation.element_size()) % 16 == 0),
    "flux3_klein_ln_modulate: ", name, " rows must start on a 16-byte boundary");
}

void check_ln_input(const at::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 3,
              "flux3_klein_ln_modulate: ", name, " must be a contiguous CUDA bf16 [B,S,D] tensor");
  const int64_t dim = x.size(2);
  TORCH_CHECK(dim % (kChannelsPerVec * kThreads) == 0,
              "flux3_klein_ln_modulate: the channel dimension must be a multiple of ",
              kChannelsPerVec * kThreads, ", got ", dim);
  TORCH_CHECK(vecs_per_thread(dim) <= kMaxVecsPerThread,
              "flux3_klein_ln_modulate: the channel dimension ", dim,
              " exceeds the largest supported row");
}

int64_t row_stride(const at::Tensor& modulation) {
  return modulation.size(1) == 1 ? 0 : modulation.stride(1);
}

}  // namespace

void flux3_klein_ln_modulate(at::Tensor x, at::Tensor scale, at::Tensor shift, double eps,
                             at::Tensor out) {
  check_ln_input(x, "x");
  check_modulation(scale, x, "scale");
  check_modulation(shift, x, "shift");
  TORCH_CHECK(scale.sizes() == shift.sizes(),
              "flux3_klein_ln_modulate: scale and shift must have the same shape");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kBFloat16 && out.is_contiguous() &&
                out.sizes() == x.sizes(),
              "flux3_klein_ln_modulate: out must be a contiguous CUDA bf16 buffer with the shape "
              "of x");
  TORCH_CHECK(is_aligned(x.const_data_ptr(), 16) && is_aligned(scale.const_data_ptr(), 16) &&
                is_aligned(shift.const_data_ptr(), 16) && is_aligned(out.data_ptr(), 16),
              "flux3_klein_ln_modulate: buffers must be 16-byte aligned for the uint4 path");

  if (x.numel() == 0) {
    return;
  }

  const c10::cuda::CUDAGuard guard(x.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t scale_stride_b = scale.stride(0);
  const int64_t scale_stride_s = row_stride(scale);
  switch (vecs_per_thread(x.size(2))) {
    case 1:
      launch_ln_modulate<1>(x, scale, shift, out, scale_stride_b, scale_stride_s, eps, stream);
      break;
    case 2:
      launch_ln_modulate<2>(x, scale, shift, out, scale_stride_b, scale_stride_s, eps, stream);
      break;
    case 3:
      launch_ln_modulate<3>(x, scale, shift, out, scale_stride_b, scale_stride_s, eps, stream);
      break;
    case 4:
      launch_ln_modulate<4>(x, scale, shift, out, scale_stride_b, scale_stride_s, eps, stream);
      break;
    case 5:
      launch_ln_modulate<5>(x, scale, shift, out, scale_stride_b, scale_stride_s, eps, stream);
      break;
    case 6:
      launch_ln_modulate<6>(x, scale, shift, out, scale_stride_b, scale_stride_s, eps, stream);
      break;
    case 7:
      launch_ln_modulate<7>(x, scale, shift, out, scale_stride_b, scale_stride_s, eps, stream);
      break;
    default:
      launch_ln_modulate<8>(x, scale, shift, out, scale_stride_b, scale_stride_s, eps, stream);
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void flux3_klein_ln_modulate_cat(at::Tensor hidden_states, at::Tensor encoder_hidden_states,
                                 at::Tensor scale, at::Tensor shift, at::Tensor context_scale,
                                 at::Tensor context_shift, double eps, at::Tensor out) {
  check_ln_input(hidden_states, "hidden_states");
  check_ln_input(encoder_hidden_states, "encoder_hidden_states");
  check_modulation(scale, hidden_states, "scale");
  check_modulation(shift, hidden_states, "shift");
  check_modulation(context_scale, encoder_hidden_states, "context_scale");
  check_modulation(context_shift, encoder_hidden_states, "context_shift");
  TORCH_CHECK(hidden_states.size(2) == encoder_hidden_states.size(2) &&
                hidden_states.size(0) == encoder_hidden_states.size(0),
              "flux3_klein_ln_modulate_cat: both streams must share the batch and channel axes");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kBFloat16 && out.is_contiguous() &&
                out.dim() == 3 && out.size(0) == hidden_states.size(0) &&
                out.size(1) == hidden_states.size(1) + encoder_hidden_states.size(1) &&
                out.size(2) == hidden_states.size(2),
              "flux3_klein_ln_modulate_cat: out must be a contiguous CUDA bf16 buffer of shape "
              "[B, Se + Sc, D]");
  TORCH_CHECK(is_aligned(hidden_states.const_data_ptr(), 16) &&
                is_aligned(encoder_hidden_states.const_data_ptr(), 16) &&
                is_aligned(scale.const_data_ptr(), 16) && is_aligned(shift.const_data_ptr(), 16) &&
                is_aligned(context_scale.const_data_ptr(), 16) &&
                is_aligned(context_shift.const_data_ptr(), 16) && is_aligned(out.data_ptr(), 16),
              "flux3_klein_ln_modulate_cat: buffers must be 16-byte aligned for the uint4 path");

  if (out.numel() == 0) {
    return;
  }

  const c10::cuda::CUDAGuard guard(hidden_states.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t scale_stride_b = scale.stride(0);
  const int64_t scale_stride_s = row_stride(scale);
  const int64_t context_stride_b = context_scale.stride(0);
  const int64_t context_stride_s = row_stride(context_scale);
  switch (vecs_per_thread(hidden_states.size(2))) {
    case 1:
      launch_ln_modulate_cat<1>(hidden_states, encoder_hidden_states, scale, shift, context_scale,
                                context_shift, out, scale_stride_b, scale_stride_s,
                                context_stride_b, context_stride_s, eps, stream);
      break;
    case 2:
      launch_ln_modulate_cat<2>(hidden_states, encoder_hidden_states, scale, shift, context_scale,
                                context_shift, out, scale_stride_b, scale_stride_s,
                                context_stride_b, context_stride_s, eps, stream);
      break;
    case 3:
      launch_ln_modulate_cat<3>(hidden_states, encoder_hidden_states, scale, shift, context_scale,
                                context_shift, out, scale_stride_b, scale_stride_s,
                                context_stride_b, context_stride_s, eps, stream);
      break;
    case 4:
      launch_ln_modulate_cat<4>(hidden_states, encoder_hidden_states, scale, shift, context_scale,
                                context_shift, out, scale_stride_b, scale_stride_s,
                                context_stride_b, context_stride_s, eps, stream);
      break;
    case 5:
      launch_ln_modulate_cat<5>(hidden_states, encoder_hidden_states, scale, shift, context_scale,
                                context_shift, out, scale_stride_b, scale_stride_s,
                                context_stride_b, context_stride_s, eps, stream);
      break;
    case 6:
      launch_ln_modulate_cat<6>(hidden_states, encoder_hidden_states, scale, shift, context_scale,
                                context_shift, out, scale_stride_b, scale_stride_s,
                                context_stride_b, context_stride_s, eps, stream);
      break;
    case 7:
      launch_ln_modulate_cat<7>(hidden_states, encoder_hidden_states, scale, shift, context_scale,
                                context_shift, out, scale_stride_b, scale_stride_s,
                                context_stride_b, context_stride_s, eps, stream);
      break;
    default:
      launch_ln_modulate_cat<8>(hidden_states, encoder_hidden_states, scale, shift, context_scale,
                                context_shift, out, scale_stride_b, scale_stride_s,
                                context_stride_b, context_stride_s, eps, stream);
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace cache_dit::models
