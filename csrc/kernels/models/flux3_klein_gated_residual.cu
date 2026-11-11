// FLUX.3 Klein gated residual, with the branch sum folded in.
//
// Upstream evaluates `output = attn_output + ff_output` and then `hidden + gate * output` as
// two pointwise passes: the sum is written once and read back once. Keeping both branch outputs
// in registers removes a full `(B, S, D)` write plus read per block, which is the only byte
// reduction available here - ncu measured inductor's version of both passes at 92.2% of DRAM
// peak, so the access pattern is not the problem.
//
// Everything is fp32 in registers with one bf16 store, matching what the compiled step does
// after it fuses the same chain. Out-of-place is a hard requirement: cache-dit derives a block's
// residual as `output - input`, so an in-place update would corrupt the cache.
//
// The gate is read through explicit strides, so a per-sample `(B, 1, D)` row is as cheap as it
// should be instead of being expanded to `(B, S, D)`.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "flux3_klein.h"

namespace cache_dit::models {
namespace {

constexpr int kThreads = 256;
constexpr int kChannelsPerVec = 8;
constexpr int kPairsPerVec = kChannelsPerVec / 2;

__device__ __forceinline__ uint4 gated_vec(const uint4& hidden, const uint4& attn, const uint4& ff,
                                           const uint4& gate) {
  const __nv_bfloat162* h_pair = reinterpret_cast<const __nv_bfloat162*>(&hidden);
  const __nv_bfloat162* a_pair = reinterpret_cast<const __nv_bfloat162*>(&attn);
  const __nv_bfloat162* f_pair = reinterpret_cast<const __nv_bfloat162*>(&ff);
  const __nv_bfloat162* g_pair = reinterpret_cast<const __nv_bfloat162*>(&gate);
  uint4 result;
  __nv_bfloat162* r_pair = reinterpret_cast<__nv_bfloat162*>(&result);
#pragma unroll
  for (int j = 0; j < kPairsPerVec; ++j) {
    const float2 h = __bfloat1622float2(h_pair[j]);
    const float2 a = __bfloat1622float2(a_pair[j]);
    const float2 f = __bfloat1622float2(f_pair[j]);
    const float2 g = __bfloat1622float2(g_pair[j]);
    // The branch sum is a materialised tensor upstream (`output = attn_output + ff_output`,
    // transformer_flux3.py:284) and stock+compile keeps it that way - inductor emits it as its
    // own `triton_poi_..._add_24` kernel. Rounding here keeps the kernel bit-identical to that
    // graph instead of silently dropping one bf16 rounding point.
    const float t0 = __bfloat162float(__float2bfloat16_rn(__fadd_rn(a.x, f.x)));
    const float t1 = __bfloat162float(__float2bfloat16_rn(__fadd_rn(a.y, f.y)));
    const float y0 = __fadd_rn(h.x, __fmul_rn(g.x, t0));
    const float y1 = __fadd_rn(h.y, __fmul_rn(g.y, t1));
    r_pair[j].x = __float2bfloat16_rn(y0);
    r_pair[j].y = __float2bfloat16_rn(y1);
  }
  return result;
}

__global__ void gated_residual_kernel(const __nv_bfloat162* __restrict__ hidden_states,
                                      const __nv_bfloat162* __restrict__ gate,
                                      const __nv_bfloat162* __restrict__ attn_output,
                                      const __nv_bfloat162* __restrict__ ff_output,
                                      __nv_bfloat162* __restrict__ out, int64_t vecs,
                                      int64_t vecs_per_row, int64_t seq_len, int64_t gate_stride_b,
                                      int64_t gate_stride_s) {
  const int64_t vec = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (vec >= vecs) {
    return;
  }
  const int64_t row = vec / vecs_per_row;
  const int64_t batch = row / seq_len;
  const int64_t token = row - batch * seq_len;
  // The gate strides are already in bf16-pair units and already step by one whole row.
  const int64_t gate_base = batch * gate_stride_b + token * gate_stride_s;
  const int64_t in_row_base = row * vecs_per_row * kPairsPerVec;
  const int64_t offset = in_row_base + (vec - row * vecs_per_row) * kPairsPerVec;
  const int64_t gate_offset = gate_base + (vec - row * vecs_per_row) * kPairsPerVec;

  const uint4 h = *reinterpret_cast<const uint4*>(hidden_states + offset);
  const uint4 a = *reinterpret_cast<const uint4*>(attn_output + offset);
  const uint4 f = *reinterpret_cast<const uint4*>(ff_output + offset);
  const uint4 g = *reinterpret_cast<const uint4*>(gate + gate_offset);
  *reinterpret_cast<uint4*>(out + offset) = gated_vec(h, a, f, g);
}

__global__ void gated_residual_split_kernel(
  const __nv_bfloat162* __restrict__ hidden_states,
  const __nv_bfloat162* __restrict__ encoder_hidden_states, const __nv_bfloat162* __restrict__ gate,
  const __nv_bfloat162* __restrict__ context_gate, const __nv_bfloat162* __restrict__ attn_output,
  const __nv_bfloat162* __restrict__ ff_output, __nv_bfloat162* __restrict__ out_hidden,
  __nv_bfloat162* __restrict__ out_encoder, int64_t vecs, int64_t vecs_per_row, int64_t prefix_len,
  int64_t seq_len, int64_t gate_stride_b, int64_t gate_stride_s, int64_t context_stride_b,
  int64_t context_stride_s) {
  const int64_t vec = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (vec >= vecs) {
    return;
  }
  const int64_t joint = prefix_len + seq_len;
  const int64_t vec_in_row = vec - (vec / vecs_per_row) * vecs_per_row;
  const int64_t row = vec / vecs_per_row;
  const int64_t batch = row / joint;
  const int64_t token = row - batch * joint;
  const int64_t joint_offset = row * vecs_per_row * kPairsPerVec + vec_in_row * kPairsPerVec;
  const uint4 a = *reinterpret_cast<const uint4*>(attn_output + joint_offset);
  const uint4 f = *reinterpret_cast<const uint4*>(ff_output + joint_offset);

  if (token < prefix_len) {
    // The gate strides are already in bf16-pair units and already step by one whole row.
    const int64_t gate_row = batch * context_stride_b + token * context_stride_s;
    const uint4 g =
      *reinterpret_cast<const uint4*>(context_gate + gate_row + vec_in_row * kPairsPerVec);
    const int64_t offset =
      (batch * prefix_len + token) * vecs_per_row * kPairsPerVec + vec_in_row * kPairsPerVec;
    const uint4 h = *reinterpret_cast<const uint4*>(encoder_hidden_states + offset);
    *reinterpret_cast<uint4*>(out_encoder + offset) = gated_vec(h, a, f, g);
  } else {
    const int64_t content = token - prefix_len;
    const int64_t gate_row = batch * gate_stride_b + content * gate_stride_s;
    const uint4 g = *reinterpret_cast<const uint4*>(gate + gate_row + vec_in_row * kPairsPerVec);
    const int64_t offset =
      (batch * seq_len + content) * vecs_per_row * kPairsPerVec + vec_in_row * kPairsPerVec;
    const uint4 h = *reinterpret_cast<const uint4*>(hidden_states + offset);
    *reinterpret_cast<uint4*>(out_hidden + offset) = gated_vec(h, a, f, g);
  }
}

bool is_aligned(const void* pointer, uintptr_t bytes) {
  return (reinterpret_cast<uintptr_t>(pointer) & (bytes - 1)) == 0;
}

void check_activation(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.scalar_type() == at::kBFloat16 && tensor.is_contiguous() &&
                tensor.dim() == 3,
              "flux3_klein_gated_residual: ", name,
              " must be a contiguous CUDA bf16 [B,S,D] "
              "tensor");
  TORCH_CHECK((tensor.size(2) / 2) % kPairsPerVec == 0,
              "flux3_klein_gated_residual: the channel dimension must be a multiple of ",
              kChannelsPerVec, ", got ", tensor.size(2));
  TORCH_CHECK(is_aligned(tensor.const_data_ptr(), 16), "flux3_klein_gated_residual: ", name,
              " must be 16-byte aligned");
}

void check_gate(const at::Tensor& gate, const at::Tensor& like, const char* name) {
  // The gate is `temb_mod.chunk(3, dim=-1)[2]`: dense channels, three widths apart per row.
  TORCH_CHECK(gate.is_cuda() && gate.dim() == 3 && gate.stride(2) == 1 &&
                gate.scalar_type() == like.scalar_type(),
              "flux3_klein_gated_residual: ", name,
              " must be a CUDA tensor of the model dtype with three "
              "dimensions and a dense channel axis");
  TORCH_CHECK(gate.size(0) == like.size(0) && gate.size(2) == like.size(2),
              "flux3_klein_gated_residual: ", name, " must match the batch and channel axes");
  TORCH_CHECK(gate.size(1) == 1 || gate.size(1) == like.size(1),
              "flux3_klein_gated_residual: ", name, " must have one row per sample or per token");
  TORCH_CHECK(is_aligned(gate.const_data_ptr(), 16) &&
                (gate.stride(0) * gate.element_size()) % 16 == 0 &&
                (gate.size(1) == 1 || (gate.stride(1) * gate.element_size()) % 16 == 0),
              "flux3_klein_gated_residual: ", name, " rows must start on a 16-byte boundary");
}

int64_t row_stride(const at::Tensor& gate) { return gate.size(1) == 1 ? 0 : gate.stride(1); }

int64_t grid_for(int64_t vecs) { return (vecs + kThreads - 1) / kThreads; }

}  // namespace

void flux3_klein_gated_residual(at::Tensor hidden_states, at::Tensor gate, at::Tensor attn_output,
                                at::Tensor ff_output, at::Tensor out) {
  check_activation(hidden_states, "hidden_states");
  check_activation(attn_output, "attn_output");
  check_activation(ff_output, "ff_output");
  check_gate(gate, hidden_states, "gate");
  TORCH_CHECK(
    attn_output.sizes() == hidden_states.sizes() && ff_output.sizes() == hidden_states.sizes(),
    "flux3_klein_gated_residual: the branch outputs must have the shape of the residual "
    "stream");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kBFloat16 && out.is_contiguous() &&
                out.sizes() == hidden_states.sizes() && is_aligned(out.data_ptr(), 16),
              "flux3_klein_gated_residual: out must be a contiguous 16-byte aligned CUDA bf16 "
              "buffer with the shape of the residual stream");

  const int64_t dim = hidden_states.size(2);
  const int64_t vecs_per_row = dim / kChannelsPerVec;
  const int64_t vecs = hidden_states.numel() / kChannelsPerVec;
  if (vecs == 0) {
    return;
  }

  const c10::cuda::CUDAGuard guard(hidden_states.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  gated_residual_kernel<<<grid_for(vecs), kThreads, 0, stream>>>(
    reinterpret_cast<const __nv_bfloat162*>(hidden_states.const_data_ptr()),
    reinterpret_cast<const __nv_bfloat162*>(gate.const_data_ptr()),
    reinterpret_cast<const __nv_bfloat162*>(attn_output.const_data_ptr()),
    reinterpret_cast<const __nv_bfloat162*>(ff_output.const_data_ptr()),
    reinterpret_cast<__nv_bfloat162*>(out.data_ptr()), vecs, vecs_per_row, hidden_states.size(1),
    gate.stride(0) / 2, row_stride(gate) / 2);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void flux3_klein_gated_residual_split(at::Tensor hidden_states, at::Tensor encoder_hidden_states,
                                      at::Tensor gate, at::Tensor context_gate,
                                      at::Tensor attn_output, at::Tensor ff_output,
                                      int64_t prefix_len, at::Tensor out_hidden,
                                      at::Tensor out_encoder) {
  check_activation(hidden_states, "hidden_states");
  check_activation(encoder_hidden_states, "encoder_hidden_states");
  check_activation(attn_output, "attn_output");
  check_activation(ff_output, "ff_output");
  check_gate(gate, hidden_states, "gate");
  check_gate(context_gate, encoder_hidden_states, "context_gate");

  const int64_t dim = hidden_states.size(2);
  const int64_t seq_len = hidden_states.size(1);
  const int64_t batch = hidden_states.size(0);
  TORCH_CHECK(encoder_hidden_states.size(0) == batch && encoder_hidden_states.size(2) == dim &&
                encoder_hidden_states.size(1) == prefix_len,
              "flux3_klein_gated_residual_split: encoder_hidden_states must be [B, prefix_len, D]");
  TORCH_CHECK(
    attn_output.sizes() == ff_output.sizes() && attn_output.size(0) == batch &&
      attn_output.size(2) == dim && attn_output.size(1) == prefix_len + seq_len,
    "flux3_klein_gated_residual_split: the branch outputs must be [B, prefix_len + Sc, D]");
  TORCH_CHECK(out_hidden.is_cuda() && out_hidden.scalar_type() == at::kBFloat16 &&
                out_hidden.is_contiguous() && out_hidden.sizes() == hidden_states.sizes() &&
                is_aligned(out_hidden.data_ptr(), 16),
              "flux3_klein_gated_residual_split: out_hidden must be a contiguous 16-byte aligned "
              "CUDA bf16 buffer of the shape of hidden_states");
  TORCH_CHECK(out_encoder.is_cuda() && out_encoder.scalar_type() == at::kBFloat16 &&
                out_encoder.is_contiguous() &&
                out_encoder.sizes() == encoder_hidden_states.sizes() &&
                is_aligned(out_encoder.data_ptr(), 16),
              "flux3_klein_gated_residual_split: out_encoder must be a contiguous 16-byte aligned "
              "CUDA bf16 buffer of the shape of encoder_hidden_states");

  const int64_t vecs_per_row = dim / kChannelsPerVec;
  const int64_t vecs = attn_output.numel() / kChannelsPerVec;
  if (vecs == 0) {
    return;
  }

  const c10::cuda::CUDAGuard guard(hidden_states.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  gated_residual_split_kernel<<<grid_for(vecs), kThreads, 0, stream>>>(
    reinterpret_cast<const __nv_bfloat162*>(hidden_states.const_data_ptr()),
    reinterpret_cast<const __nv_bfloat162*>(encoder_hidden_states.const_data_ptr()),
    reinterpret_cast<const __nv_bfloat162*>(gate.const_data_ptr()),
    reinterpret_cast<const __nv_bfloat162*>(context_gate.const_data_ptr()),
    reinterpret_cast<const __nv_bfloat162*>(attn_output.const_data_ptr()),
    reinterpret_cast<const __nv_bfloat162*>(ff_output.const_data_ptr()),
    reinterpret_cast<__nv_bfloat162*>(out_hidden.data_ptr()),
    reinterpret_cast<__nv_bfloat162*>(out_encoder.data_ptr()), vecs, vecs_per_row, prefix_len,
    seq_len, gate.stride(0) / 2, row_stride(gate) / 2, context_gate.stride(0) / 2,
    row_stride(context_gate) / 2);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace cache_dit::models
