#include "gemm_w4a4.h"

#include <algorithm>

#include "native/gemm_base.cuh"
#include "native/gemm_w4a4.cuh"
#include "native/lora.cuh"

namespace svdq::cute_kernels {
namespace {

// Large-K specialization of the non-fused row-major-D GEMM. The regular CuTe path uses a TMA
// producer and a shared-memory stage ring, which is effective on the wider/shorter shapes but pays
// producer, mbarrier and stage-drain work on every K tile. For large K that repeated scheduling
// cost dominates; native-style G2R instead loads fragment-packed A/B/SF directly into registers
// and double-buffers them inside each warp. Reusing that mainloop removes the TMA pipeline from
// the consumer GEMM while keeping the public runtime and LoRA-D layout unchanged.
//
// Only the epilogue needs a new implementation because native expects its own FP32 fragment D,
// whereas this path deliberately keeps the CuTe producer's row-major BF16 D[M,R]. All other
// operands use the native fragment pack, and the dispatcher selects this file only for the proven
// large-K geometry; the TMA implementation remains the default for every other shape.
//
// Why reusing the native mainloop can still be slightly faster than native:
//
// 1. The FP4 mainloop and the LoRA epilogue explain different parts of the measured win. Reusing
//    native's G2R A/B/SF loads and K-loop removes the repeated TMA scheduling cost that made the
//    original CuTe TMA path 13-14% slower. By itself, an identical mainloop would only promise
//    parity with native; the additional few-percent advantage comes from the replacement epilogue
//    and from the compiler's resulting whole-kernel schedule.
//
// 2. RowMajorLoraUp is a replacement for native's LoRA-up epilogue, not an extra pass after it.
//    Native reads fragment-major FP32 D and applies per-rank scales while forming BF16 MMA inputs.
//    Here the CuTe producer has already folded those scales into row-major BF16 D. The inverse
//    coordinate map below turns aligned 32-bit pair loads directly into native A fragments. Thus D
//    read traffic is halved, and the separate rank-scale multiply and FP32-to-BF16 input conversion
//    disappear. L1 loads, the low-rank MMA accumulation, residual bias/scale, and the final BF16
//    store retain native's implementation and accumulation order.
//
// 3. The LoRA term is small but its non-tensor tail is visible. For K=12288 and R=32, its MMA work
//    relative to the main product is R/K, about 0.26%; rank 32 needs only two R16 slices. The new
//    epilogue is therefore not free, but it can cost less than the native FP32-D epilogue even
//    without native's deeper prefetch machinery. Removing loads, scaling, conversion, and dependent
//    stalls from this tail is enough to matter after the large main loop has been made identical.
//
// 4. Higher tensor-pipe utilization does not contradict the extra row-major remapping. That metric
//    is tensor-active cycles divided by total cycles, not a count of useful tensor instructions.
//    The main FP4 math is unchanged; reducing non-tensor work and stalls shrinks the denominator,
//    so the tensor-active fraction rises. A shorter total duration with the same main MMA work is
//    exactly the expected signature of a cheaper epilogue, not evidence of extra FP4 throughput.
//
// 5. Reusing a mainloop template does not imply bit-identical SASS for the complete kernel. The
//    epilogue is inlined into the same compilation unit and affects instruction scheduling and
//    register allocation. In the validating SM120 build this specialization used 238 registers per
//    thread versus 252 for native. Consequently the two whole-kernel SASS schedules can differ;
//    the measured small edge is the combined epilogue/codegen result, not a claim that the shared
//    G2R K-loop executes different or fewer FP4 MMAs.
using Config = native_kernels::GEMMConfig_W4A4_BF16;
using Base = native_kernels::GEMMBase<Config>;
using NativeGemm = native_kernels::GEMM_W4A4<Config>;
using NativeLora = native_kernels::Lora<Config>;

// Native's mainloop consumes fragment-packed A/B/SF and needs no TMA or pipeline smem. Only the
// LoRA-up A operand differs here: the CuTe producer supplies ordinary row-major BF16 D[M,R].
struct RowMajorLoraUp {
  struct Arguments {
    Base::half_t const *lora_act;
    Base::packed_fpsum_t const *lora_wgt_up;
    int rank;
  };

  // Epilogue math: accum = FP32(fpsum) + D @ L1^T. fpsum already contains the scaled/biased
  // main NVFP4 product. D is [M,R] row-major BF16 and has already absorbed the per-rank scales in
  // the CuTe producer; L1 is the native fragment-packed proj_up [N,R]. alpha/wcscales therefore
  // must not be applied to this LoRA term a second time.
  //
  // Precision deliberately follows the native epilogue rather than the regular CuTe TMA path:
  // alpha*P is rounded to BF16 before the native packed-hfma2 residual scale/bias, then widened
  // for the FP32 LoRA accumulation and rounded once more after all rank tiles. The TMA path keeps
  // the residual epilogue in FP32 and rounds only its final result, so the two implementations are
  // mathematically equivalent but not bitwise identical. Cross-path tests therefore use a small
  // relative-L2 tolerance instead of requiring equal output bits.
  //
  // Each warp owns 32 output rows and all 128 output columns of its CTA. For every 16-rank slice
  // rt, the loops below construct two m16k16 A fragments (mt=0/1), load the matching n16k16 L1
  // fragments, and accumulate two m16n16 output tiles per n tile in FP32. binfo.bm/bn are logical
  // CTA coordinates after native's optional grid swap, so m0 and the L1 block offset must be
  // derived from binfo rather than directly from blockIdx.
  __device__ __forceinline__ void operator()(Base::BlockInfo binfo, Base::fpsum_warp &fpsum, int M,
                                             int N, int K, Arguments const &args) {
    if (args.rank == 0) return;
    const int lane = threadIdx.x % Base::WARP_SIZE;
    const int warp = threadIdx.x / Base::WARP_SIZE;
    // Native packs each 32-row warp tile as two 16-row MMA fragments. Widen the incoming BF16
    // residual once before the BF16xBF16->FP32 LoRA MMAs, so the low-rank sum stays FP32.
    const int m0 = binfo.bm * Base::BLOCK_M + warp * Base::WARP_M;
    auto accum = Base::packed_fp16_to_fp32(fpsum);
    // L1 pack layout: [N/16, R/16, 32 lanes] of m16n16k16 B fragments. Select this CTA's
    // 128-column / 16-row-group slab once; load_lora_wgt below advances only the rank tile.
    auto const *l1 =
      args.lora_wgt_up + binfo.bn * (Base::BLOCK_N / 16) * (args.rank / 16) * Base::WARP_SIZE;

    // Each iteration reduces one K=16 slice of D@L1^T. Rank tiles are accumulated into the same
    // FP32 output registers, preserving the native m,n,r accumulation order.
#pragma unroll 1
    for (int rt = 0; rt < args.rank / NativeLora::WARP_R; ++rt) {
      Base::packed_fpsum_t a[Base::WARP_M_TILES];
#pragma unroll
      for (int mt = 0; mt < Base::WARP_M_TILES; ++mt) {
#pragma unroll
        for (int p = 0; p < 4; ++p) {
          // Convert one packed A half2 (two BF16 values) from logical D(row,col) to native MMA
          // fragment slot p. lane/4 chooses the base row; lane%4 chooses a pair of rank columns.
          // p%2 selects row or row+8, while p/2 selects rank columns 0..7 or 8..15. mt selects
          // the first or second 16-row fragment of this warp's 32-row tile:
          //   row = warp_row + 16*mt + lane/4 + 8*(p%2)
          //   col = 16*rt + 2*(lane%4) + 8*(p/2) + {0,1}
          // The two values in half2 are adjacent columns, so one aligned 32-bit load fills p.
          // This is exactly the inverse of native's FP32 fragment-D coordinate map, but reads the
          // ordinary row-major BF16 D produced by CuTe instead of native's fragment-major buffer.
          const int row = m0 + mt * 16 + lane / 4 + 8 * (p % 2);
          const int col = rt * 16 + 8 * (p / 2) + 2 * (lane % 4);
          a[mt].data[p] = *reinterpret_cast<Base::half2_t const *>(
            args.lora_act + static_cast<int64_t>(row) * args.rank + col);
        }
      }
      // Load the matching native-packed L1 fragment for this 16-rank slice, then update every
      // 16x16 output tile owned by the warp. a[m] supplies the two row fragments and w[n] the
      // corresponding 16 output columns; out is the FP32 accumulator shared with the main GEMM.
      NativeLora::lora_wgt_warp w;
      NativeLora::load_lora_wgt(l1, rt, args.rank, w, true);
#pragma unroll
      for (int m = 0; m < Base::WARP_M_TILES; ++m) {
#pragma unroll
        for (int n = 0; n < Base::WARP_N_TILES; ++n) {
          auto &out = accum[m * Base::WARP_N_TILES + n];
          out = Base::mma_f16xf16_f32(a[m], w[n], out);
        }
      }
    }
    // Match native's epilogue contract: round residual+LoRA to BF16 only after all rank tiles.
    // EpilogueDefault then performs the ordinary dense store; no TMA or extra conversion pass is
    // introduced for the row-major D path.
    fpsum = Base::packed_fp32_to_fp16(accum);
  }
};

using Bias = Base::EpilogueBias<true, true>;
using Store = Base::EpilogueDefault;
using Epilogue = Base::EpilogueCombination<Bias, RowMajorLoraUp, Store>;

__global__ void __launch_bounds__(Base::NUM_WARPS *Base::WARP_SIZE, 1)
  gemm_w4a4_large_k_kernel(Base::packed_act_t const *act, Base::packed_wgt_t const *wgt,
                           NativeGemm::packed_amscale_t const *ascales,
                           NativeGemm::packed_wmscale_t const *wscales, float alpha, int M, int N,
                           int K, Epilogue::Arguments epilogue_args, bool swap_block_xy) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  // Instantiate the native runtime's pure-G2R FP4 mainloop with the row-major-D epilogue above.
  // The main A/B/SF loads and K-loop are native's implementation, not the CuTe TMA pipeline.
  typename NativeGemm::template gemm_w4a4_fp4_kernel<Epilogue, true> impl;
  impl(act, wgt, ascales, wscales, alpha, M, N, K, epilogue_args, swap_block_xy, false);
#else
  asm volatile("trap;");
#endif
}

}  // namespace

void launch_gemm_w4a4_large_k(void const *act, void const *wgt, void *out, void const *act_sf,
                              void const *wgt_sf, void const *lora_act, void const *lora_up,
                              void const *bias, void const *wcscales, float alpha, int M, int N,
                              int K, int rank, cudaStream_t stream) {
  dim3 grid(M / Base::BLOCK_M, N / Base::BLOCK_N);
  const bool swap_block_xy = M > N * 2;
  if (swap_block_xy) std::swap(grid.x, grid.y);
  Epilogue::Arguments args = std::make_tuple(
    Bias::Arguments{static_cast<Base::packed_wscale_t const *>(bias),
                    static_cast<Base::packed_wscale_t const *>(wcscales)},
    RowMajorLoraUp::Arguments{static_cast<Base::half_t const *>(lora_act),
                              static_cast<Base::packed_fpsum_t const *>(lora_up), rank},
    Store::Arguments{static_cast<Base::half_t *>(out), M, N});
  gemm_w4a4_large_k_kernel<<<grid, Base::NUM_WARPS * Base::WARP_SIZE, 0, stream>>>(
    static_cast<Base::packed_act_t const *>(act), static_cast<Base::packed_wgt_t const *>(wgt),
    static_cast<NativeGemm::packed_amscale_t const *>(act_sf),
    static_cast<NativeGemm::packed_wmscale_t const *>(wgt_sf), alpha, M, N, K, args, swap_block_xy);
}

}  // namespace svdq::cute_kernels
