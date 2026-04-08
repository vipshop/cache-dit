# Blackwell Desktop/Workstation (sm120) GPU Optimization Guide

Deep dive into RTX 5090 and RTX PRO 6000-specific optimizations for CUDA kernels.

## Blackwell Desktop Architecture Overview

### RTX PRO 6000

| Spec          | Value      | Optimization Impact                   |
| ------------- | ---------- | ------------------------------------- |
| SMs           | 188        | Full GB202 die, grid multiples of 188 |
| Threads/SM    | 1536       | 48 warps/SM (fewer than datacenter)   |
| Shared Memory | 128 KB/SM  | Smaller than sm100, adjust tile sizes |
| L2 Cache      | 128 MB     | Large L2 for workstation workloads    |
| Memory BW     | 1.8 TB/s   | 96 GB GDDR7, 512-bit bus              |
| Warp Size     | 32         | All reductions use warp shuffles      |

### RTX 5090

| Spec          | Value      | Optimization Impact                 |
| ------------- | ---------- | ----------------------------------- |
| SMs           | 170        | 170/192 SMs enabled on GB202        |
| Threads/SM    | 1536       | 48 warps/SM (fewer than datacenter) |
| Shared Memory | 128 KB/SM  | Same SM as RTX PRO 6000             |
| L2 Cache      | 96 MB      | Reduced vs full GB202               |
| Memory BW     | 1.79 TB/s  | 32 GB GDDR7, 512-bit bus            |
| Warp Size     | 32         | All reductions use warp shuffles    |

### Key sm120 Features

1. **GB202 Die** - Up to 192 SMs (188 enabled on PRO 6000, 170 on RTX 5090)
2. **GDDR7 Memory** - First generation GDDR7 with significantly higher bandwidth than GDDR6X
3. **5th-Gen Tensor Cores** - FP4/FP8/FP16/BF16/TF32 support (same gen as datacenter Blackwell)
4. **128 MB L2 Cache** - Massive L2 on full GB202 (PRO 6000), 96 MB on RTX 5090
5. **PCIe Gen5 x16** - 64 GB/s bidirectional host-device bandwidth
6. **No TMA** - Unlike datacenter sm100, sm120 lacks Tensor Memory Accelerator
7. **No Thread Block Clusters** - No cluster or distributed shared memory support
8. **No TMEM** - No dedicated tensor memory; tensor operands use register file
9. **No WGMMA/tcgen05** - Uses traditional WMMA or cuBLAS for tensor core access

### Key Differences from Datacenter Blackwell (sm100)

| Feature                   | sm120 (Desktop)      | sm100 (Datacenter)       |
| ------------------------- | -------------------- | ------------------------ |
| Max warps/SM              | 48                   | 64                       |
| Max threads/SM            | 1536                 | 2048                     |
| Max blocks/SM             | 24                   | 32                       |
| Shared memory/SM          | 128 KB               | 228 KB                   |
| L1+Shared combined        | 128 KB               | 256 KB                   |
| L2 Cache                  | 96-128 MB            | 126 MB                   |
| Registers/SM              | 65536                | 65536                    |
| Memory type               | GDDR7                | HBM3e                    |
| Memory bandwidth          | 1.79-1.8 TB/s        | 8 TB/s                   |
| Thread Block Clusters     | No                   | Yes                      |
| Distributed Shared Memory | No                   | Yes                      |
| TMA                       | No                   | Yes (v2, multicast)      |
| TMEM                      | No                   | Yes                      |
| WGMMA / tcgen05           | No                   | Yes                      |
| cp.async                  | Yes                  | Yes                      |
| Tensor Core gen           | 5th gen              | 5th gen                  |
| FP4 support               | Yes                  | Yes                      |
| NVLink                    | No                   | 5.0 (1.8 TB/s)          |

**sm120 is architecturally closer to Ada (sm89) than to datacenter Blackwell (sm100)** in terms of SM-level features. It has the same warp count, block limits, and lacks TMA/clusters/TMEM. The tensor cores are 5th-gen (same as sm100), and it gains GDDR7 and larger L2 over Ada.

### Key Differences from Ada (sm89)

| Feature                   | sm120 (Blackwell DT) | sm89 (Ada)               |
| ------------------------- | -------------------- | ------------------------ |
| Max warps/SM              | 48                   | 48                       |
| Max threads/SM            | 1536                 | 1536                     |
| Max blocks/SM             | 24                   | 24                       |
| Shared memory/SM          | 128 KB               | 100 KB                   |
| L1+Shared combined        | 128 KB               | 128 KB                   |
| L2 Cache                  | 96-128 MB            | 36-96 MB                 |
| Registers/SM              | 65536                | 65536                    |
| Memory type               | GDDR7                | GDDR6/GDDR6X             |
| Memory bandwidth          | 1.79-1.8 TB/s        | 504 GB/s - 1.01 TB/s     |
| Tensor Core gen           | 5th gen              | 4th gen                  |
| FP4 support               | Yes                  | No                       |
| FP8 support               | Yes                  | Yes                      |
| SMs (max)                 | 188                  | 142                      |

**The upgrade path from sm89 is clear**: 28% more shared memory (128 KB vs 100 KB), ~2x memory bandwidth via GDDR7, more SMs, larger L2, and 5th-gen tensor cores with FP4 support.

## Memory Hierarchy Optimization

### Global Memory Access Patterns

GDDR7 provides significantly higher bandwidth than GDDR6X but still far less than HBM3e. Coalescing remains paramount:

```cuda
// GOOD: Coalesced access (threads access consecutive addresses)
int idx = blockIdx.x * blockDim.x + threadIdx.x;
float val = input[idx];

// BAD: Strided access (wastes GDDR7 burst bandwidth)
int idx = threadIdx.x * stride;
float val = input[idx];
```

**Transaction sizes:**

- 32 bytes minimum
- 128 bytes optimal (full warp, FP32)
- Align to 128-byte boundaries when possible
- GDDR7 has higher burst granularity than GDDR6X — coalescing yields bigger gains

### Vectorized Memory Access (Critical for Bandwidth)

With 1.79-1.8 TB/s, sm120 has roughly half the bandwidth of sm90 (H100) and a quarter of sm100 (B200). Vectorized access is essential:

**BFloat16 vectorization (2x elements per load):**

```cuda
const __nv_bfloat162* vec_input = reinterpret_cast<const __nv_bfloat162*>(row_input);

#pragma unroll 4
for (int i = tid; i < hidden_size / 2; i += stride) {
    __nv_bfloat162 v = vec_input[i];
    float v0 = __bfloat162float(v.x);
    float v1 = __bfloat162float(v.y);
    // Process v0, v1...
}

// Write back vectorized
__nv_bfloat162* vec_output = reinterpret_cast<__nv_bfloat162*>(row_output);
__nv_bfloat162 result;
result.x = __float2bfloat16(val0);
result.y = __float2bfloat16(val1);
vec_output[i] = result;
```

**FP16 vectorization:**

```cuda
const __half2* vec_input = reinterpret_cast<const __half2*>(row_input);
__half2 v = vec_input[i];
float v0 = __half2float(v.x);
float v1 = __half2float(v.y);
```

**FP32 vectorization (4x elements per load):**

```cuda
const float4* vec_input = reinterpret_cast<const float4*>(row_input);
float4 v = vec_input[i];
// v.x, v.y, v.z, v.w are 4 consecutive floats
```

**Bandwidth context:**

| GPU           | Memory BW  | Expected utilization |
| ------------- | ---------- | -------------------- |
| RTX 5090      | 1.79 TB/s  | 40-55%               |
| RTX PRO 6000  | 1.8 TB/s   | 40-55%               |
| RTX 4090      | 1.01 TB/s  | 35-45%               |
| H100          | 3.35 TB/s  | 35-45%               |
| B200          | 8 TB/s     | 50-60%               |

sm120's GDDR7 achieves higher utilization than GDDR6X due to improved protocol efficiency.

### L2 Cache Utilization

sm120 has excellent L2 caches — 128 MB on PRO 6000, 96 MB on RTX 5090:

```cuda
// L2 is the critical performance lever on sm120
// With lower GDDR7 BW vs HBM, keeping data in L2 matters even more

// For attention: Process Q blocks to maximize K,V cache reuse
// BLOCK_SIZE_M = 64   (Q block — fits in 128 KB shared)
// BLOCK_SIZE_N = 64   (K,V block)
// head_dim = 128, FP16
// K tile: 64*128*2 = 16 KB
// V tile: 64*128*2 = 16 KB
// Multiple tiles fit in L2 for reuse

// L2 persistence hints
cudaAccessPolicyWindow policy;
policy.base_ptr = kv_cache;
policy.num_bytes = kv_cache_size;
policy.hitRatio = 1.0f;
policy.hitProp = cudaAccessPropertyPersisting;
policy.missProp = cudaAccessPropertyStreaming;
cudaStreamSetAccessPolicyWindow(stream, &policy);
```

**L2 sizing for inference:**

```cuda
// RTX 5090 (32 GB VRAM, 96 MB L2):
// Llama 7B FP16: ~14 GB weights
// KV cache per token (32 layers, 32 heads, head_dim=128, FP16):
//   2 * 32 * 32 * 128 * 2 = 0.5 MB/token
// 96 MB L2 holds ~192 tokens of KV cache
// For longer sequences, rotate L2 persistence per layer

// RTX PRO 6000 (96 GB VRAM, 128 MB L2):
// Llama 70B FP8: ~70 GB weights
// More L2 headroom for KV cache residency
```

### Shared Memory Configuration

sm120 has a combined 128 KB L1/shared memory pool per SM — same total as Ada but with more shared memory available:

- Max shared memory: **128 KB** per SM (vs 100 KB on Ada, 228 KB on sm100)
- Configurable splits similar to Ada

```cuda
// Request max shared memory for sm120
cudaFuncSetAttribute(
    kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    128 * 1024  // 128 KB max
);
```

**Tile sizing for 128 KB shared memory:**

```cuda
// 128 KB is 28% more than Ada's 100 KB, but 44% less than sm100's 228 KB
// Kernels ported from sm100 MUST reduce tile sizes
// Kernels from sm89 can slightly increase tile sizes

// For attention on sm120:
// Q tile: 64x64 * 2 bytes = 8 KB
// K tile: 64x64 * 2 bytes = 8 KB
// V tile: 64x64 * 2 bytes = 8 KB
// Total: ~24 KB, fits with double-buffering (48 KB)

// For GEMM:
// BLOCK_M=64, BLOCK_N=64, BLOCK_K=32 in FP16
// A tile: 64*32*2 = 4 KB
// B tile: 32*64*2 = 4 KB
// Double-buffered: 16 KB
// Plenty of room for larger tiles or triple-buffering
```

### Bank Conflicts

Shared memory has 32 banks (4 bytes per bank), same as all NVIDIA architectures:

```cuda
// Bank conflict example
__shared__ float data[1024];
float val = data[threadIdx.x * 32];  // BAD: 32-stride = same bank

// No bank conflict
float val = data[threadIdx.x];  // GOOD: consecutive access

// Padding to avoid bank conflicts
__shared__ float data[32][33];  // 33 instead of 32
float val = data[threadIdx.y][threadIdx.x];
```

### Asynchronous Memory Copy (cp.async)

sm120 supports `cp.async` (like Ada) but **not TMA**. Use `cp.async` for all async global-to-shared transfers:

```cuda
// cp.async: bypass register file for global→shared copies
#include <cuda_pipeline.h>

__shared__ float tile[TILE_M][TILE_N];

// Initiate async copy
__pipeline_memcpy_async(&tile[ty][tx], &global_data[gy * N + gx], sizeof(float));
__pipeline_commit();

// Do other work while copy is in flight...

// Wait for copy to complete
__pipeline_wait_prior(0);
__syncthreads();
```

**Double-buffering with cp.async:**

```cuda
__shared__ float buf[2][TILE_K][TILE_N];
int stage = 0;

// Prefetch first tile
__pipeline_memcpy_async(&buf[0][ty][tx], &A[...], sizeof(float));
__pipeline_commit();

for (int k = 0; k < K; k += TILE_K) {
    // Prefetch next tile
    if (k + TILE_K < K) {
        __pipeline_memcpy_async(&buf[1 - stage][ty][tx], &A[...], sizeof(float));
        __pipeline_commit();
    }

    __pipeline_wait_prior(1);
    __syncthreads();

    compute(buf[stage]);
    stage = 1 - stage;
}
```

**No TMA means:**

- No hardware tensor descriptors — all addressing is manual
- No multicast — each block must load its own copy of shared data
- No bulk async operations — cp.async operates per-thread
- Kernel code is simpler but less bandwidth-efficient than sm100

## Warp-Level Optimizations

### Shuffle Instructions

Identical to all other architectures:

```cuda
template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Broadcast from lane 0
float broadcast = __shfl_sync(0xffffffff, val, 0);
```

### Tensor Core Access (WMMA, not WGMMA)

sm120 uses WMMA (Warp Matrix Multiply-Accumulate) instead of WGMMA/tcgen05:

```cuda
#include <mma.h>
using namespace nvcuda;

// WMMA shapes for sm120 (5th-gen tensor cores via WMMA API):
// FP16:  16x16x16
// BF16:  16x16x16
// FP8:   16x16x32
// TF32:  16x16x8

// WMMA example: 16x16x16 FP16 multiply
wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

// Initialize accumulator
wmma::fill_fragment(c_frag, 0.0f);

// Load from shared memory
wmma::load_matrix_sync(a_frag, &smem_a[warp_row * 16], lda);
wmma::load_matrix_sync(b_frag, &smem_b[warp_col * 16], ldb);

// Multiply-accumulate
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

// Store result
wmma::store_matrix_sync(&smem_c[warp_row * 16 + warp_col * 16 * ldc],
                         c_frag, ldc, wmma::mem_row_major);
```

**Key differences from sm100 WGMMA:**

```cuda
// WMMA (sm120):
// - Operates on a single warp (32 threads)
// - Fragments stored in register file (consumes registers)
// - Smaller tile sizes (16x16x16)
// - Simpler programming model
// - No TMEM — tensor operands use registers

// WGMMA (sm100):
// - Operates on a warp group (128 threads / 4 warps)
// - Operands in TMEM (doesn't consume registers)
// - Larger tile sizes (64x256x16)
// - Higher throughput per instruction
// - Requires TMA for optimal data staging

// Practical impact:
// sm120 GEMM kernels have more register pressure
// sm120 tiles are smaller → more iterations → more overhead
// But sm120 code is simpler and more portable
```

### FP4 on sm120 Tensor Cores

sm120 supports NVFP4 via tensor cores, primarily through cuBLAS:

```cuda
// FP4 inference on sm120:
// - cuBLAS handles FP4 GEMM automatically
// - FP4 weights with per-block scaling
// - Accumulation in FP32

// For custom kernels, FP4 is typically accessed via:
// 1. cuBLASLt with FP4 data type
// 2. CUTLASS with sm120 FP4 configs
// 3. TensorRT for inference graphs

// Direct WMMA FP4 is limited — prefer library paths
```

## Register Optimization

### Register Pressure

65536 registers per SM, 255 max per thread. With only 48 warps/SM and **no TMEM**, register pressure is the primary concern for tensor core kernels:

```
Max registers per thread before reducing occupancy:
  128 threads/block (4 warps):  65536 / 128 = 512 → capped at 255
  256 threads/block (8 warps):  65536 / 256 = 256 → capped at 255
  512 threads/block (16 warps): 65536 / 512 = 128
  1024 threads/block (32 warps): 65536 / 1024 = 64
```

**Unlike sm100**, tensor core operands consume registers on sm120. WMMA fragments add to register pressure:

```cuda
// WMMA fragment register usage (approximate):
// matrix_a 16x16x16 FP16: 8 registers
// matrix_b 16x16x16 FP16: 8 registers
// accumulator 16x16x16 FP32: 8 registers
// Total per MMA: ~24 registers
//
// Multiple fragments for tiling: 3-4x more
// This can push total kernel registers above 128,
// limiting occupancy at 512-thread blocks

// Monitor register usage:
nvcc --ptxas-options=-v your_kernel.cu
```

### Register Tiling

```cuda
// Attention output accumulator in registers
float row_max = -INFINITY;
float row_sum = 0.0f;
float out_acc[HEAD_DIM];  // head_dim <= 64 is comfortable

// On sm120 with no TMEM, be careful about register budget:
// 64 floats for accumulator = 64 registers
// + WMMA fragments = ~24 registers
// + loop variables, addresses = ~20 registers
// Total: ~108 registers — still fits at 256 threads/block
```

## Occupancy Tuning

### Calculating Occupancy

```
Occupancy = Active Warps per SM / Max Warps per SM (48)

Limiting factors:
1. Registers: 65536 / (threads_per_block * regs_per_thread)
2. Shared Memory: 128KB / smem_per_block
3. Threads: 1536 / threads_per_block
4. Blocks: max 24 per SM
```

Same occupancy profile as Ada (sm89):

```
64 threads/block (2 warps):   24 blocks → 48 warps → 100%
128 threads/block (4 warps):  12 blocks → 48 warps → 100%
256 threads/block (8 warps):  6 blocks  → 48 warps → 100%
512 threads/block (16 warps): 3 blocks  → 48 warps → 100%
1024 threads/block (32 warps): 1 block  → 32 warps → 67%  ← avoid
```

**Avoid 1024-thread blocks** — they cap occupancy at 67% (same limitation as sm89).

### Block Size Selection

| Kernel Type  | Threads/Block | Warps | Reasoning                                     |
| ------------ | ------------- | ----- | --------------------------------------------- |
| Element-wise | 256           | 8     | High occupancy, simple workloads              |
| Reduction    | 256-512       | 8-16  | Enough threads for reduction, fits in 128 KB  |
| Attention    | 128-256       | 4-8   | Balance shared mem (128 KB) and registers     |
| GEMM (WMMA)  | 256           | 8     | Good register budget for WMMA fragments       |

### Cross-Device Grid Sizing

```cuda
// Query SM count at runtime
int sm_count;
cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
// RTX PRO 6000: 188, RTX 5090: 170

// Full wave sizing
int blocks_per_sm = 6;  // 6 blocks of 256 threads = 100% occupancy
int total_blocks = sm_count * blocks_per_sm;

// Wave quantization:
// RTX 5090 (170 SMs):
//   170 blocks → 1 wave → 100%
//   171 blocks → 2 waves → 50.3% avg (bad)
//   340 blocks → 2 waves → 100%
//
// RTX PRO 6000 (188 SMs):
//   188 blocks → 1 wave → 100%
//   376 blocks → 2 waves → 100%
```

## sm120-Specific Optimization Strategies

### Porting from sm89 (Ada)

sm120 is the natural upgrade path from sm89. Key improvements to leverage:

```cuda
// 1. Shared memory: 128 KB vs 100 KB (+28%)
//    Increase tile sizes slightly
//    Ada:     BLOCK_M=64, BLOCK_N=64, BLOCK_K=16
//    sm120:   BLOCK_M=64, BLOCK_N=64, BLOCK_K=32 (double K-tiles)

// 2. Memory bandwidth: 1.79 TB/s vs 1.01 TB/s (+77% on RTX 5090)
//    Memory-bound kernels see direct speedup
//    Compute-bound kernels: no change needed

// 3. More SMs: 170/188 vs 128/142
//    Increase grid dimensions proportionally

// 4. FP4 tensor cores: new on sm120
//    Inference kernels can use FP4 quantization
//    ~2x throughput over FP8 for weight-only quantization

// 5. Same SM-level architecture:
//    All sm89 kernel code runs on sm120
//    Just recompile with -arch=sm_120
```

### Porting from sm100 (Datacenter Blackwell)

Datacenter kernels need significant adaptation:

```cuda
// 1. Replace TMA with cp.async
// sm100 (TMA):
//   cuTensorMapEncodeTiled(...);  // Hardware descriptor
//   cp.async.bulk.tensor...       // Single instruction loads tile
//
// sm120 (cp.async):
//   Manual address calculation per thread
//   __pipeline_memcpy_async per element
//   More code, fewer features

// 2. Replace WGMMA with WMMA
// sm100: warp group (128 threads), 64x256x16 tiles, TMEM operands
// sm120: single warp (32 threads), 16x16x16 tiles, register operands
//
// Must restructure tile loops entirely

// 3. Remove cluster/DSMEM code
// sm100: __cluster_dims__, cluster.map_shared_rank()
// sm120: not available — each block is independent

// 4. Reduce tile sizes
// sm100: 228 KB shared memory
// sm120: 128 KB shared memory
// All tiles must shrink by ~44%

// 5. Account for lower memory bandwidth
// sm100: 8 TB/s HBM3e
// sm120: 1.8 TB/s GDDR7
// Many compute-bound sm100 kernels become memory-bound on sm120
// → Fusion and L2 reuse become critical
```

### Maximizing GDDR7 Bandwidth

GDDR7 is a new memory technology with different characteristics than GDDR6X:

```cuda
// GDDR7 characteristics:
// - PAM4 signaling (4 levels vs 2 for GDDR6X) → higher raw bandwidth
// - Higher burst lengths → coalescing matters more
// - 512-bit bus on both RTX 5090 and PRO 6000

// Bandwidth optimization strategies:
// 1. Always vectorize (float4, __half2, __nv_bfloat162)
// 2. Minimize random access patterns
// 3. Use L2 persistence for hot data
// 4. Fuse kernels aggressively — fewer memory round-trips

// Expected achievable bandwidth:
// RTX 5090:     ~800-1000 GB/s (45-56% of 1.79 TB/s)
// RTX PRO 6000: ~800-1000 GB/s (44-56% of 1.8 TB/s)
```

### Kernel Fusion (Critical on sm120)

With lower bandwidth than datacenter GPUs, fusion is the most impactful optimization:

```cuda
// BAD: Separate kernels (3 global memory round-trips)
rmsnorm_kernel<<<...>>>(input, normed);
geglu_kernel<<<...>>>(normed, activated);
residual_kernel<<<...>>>(activated, residual, output);

// GOOD: Fused kernel (1 global memory round-trip)
__global__ void fused_rmsnorm_geglu_residual(
    const scalar_t* input,
    const scalar_t* residual,
    const scalar_t* weight,
    scalar_t* output,
    int hidden_size,
    float epsilon
) {
    // Each block handles one row
    int row = blockIdx.x;
    const scalar_t* row_input = input + row * hidden_size;

    // Step 1: RMSNorm (reduction + normalize)
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = float(row_input[i]);
        sum_sq += v * v;
    }
    sum_sq = block_reduce_sum(sum_sq);
    float rms = rsqrtf(sum_sq / hidden_size + epsilon);

    // Step 2: GEGLU activation + residual (element-wise)
    int half_hidden = hidden_size / 2;
    for (int i = threadIdx.x; i < half_hidden; i += blockDim.x) {
        float gate = float(row_input[i]) * rms * float(weight[i]);
        float up = float(row_input[i + half_hidden]) * rms * float(weight[i + half_hidden]);
        float activated = gate * up * (1.0f / (1.0f + expf(-gate)));  // SiLU gate

        output[row * half_hidden + i] = scalar_t(activated + float(residual[row * half_hidden + i]));
    }
}
```

### VRAM Capacity Planning

RTX 5090 has 32 GB, RTX PRO 6000 has 96 GB — plan model deployment accordingly:

```cuda
// RTX 5090 (32 GB GDDR7):
// Llama 7B FP16:  ~14 GB → fits, room for KV cache
// Llama 7B FP8:   ~7 GB  → fits easily
// Llama 13B FP8:  ~13 GB → fits
// Llama 70B FP4:  ~35 GB → does NOT fit
// Mistral 7B FP4: ~3.5 GB → fits easily

// RTX PRO 6000 (96 GB GDDR7):
// Llama 70B FP16:  ~140 GB → does NOT fit
// Llama 70B FP8:   ~70 GB  → fits
// Llama 70B FP4:   ~35 GB  → fits with room for large KV cache
// Mixtral 8x7B FP8: ~48 GB → fits
```

## Precision and Numerical Stability

### Online Softmax for Attention

Same algorithm as other architectures, with smaller tiles:

```cuda
float row_max = -INFINITY;
float row_sum = 0.0f;

for each K block:  // Smaller blocks due to 128 KB shared memory
    compute local_scores
    local_max = max(local_scores)

    new_max = max(row_max, local_max)
    rescale = exp(row_max - new_max)

    row_sum = row_sum * rescale + sum(exp(local_scores - new_max))
    row_max = new_max

    out_acc = out_acc * rescale + softmax_scores @ V_block
```

### Mixed Precision Pattern

```cuda
// Input in FP16/BF16
float sum = 0.0f;  // Accumulate in FP32
for (int i = tid; i < hidden_size; i += blockDim.x) {
    float val = float(input[i]);
    sum += val * val;
}
sum = block_reduce_sum(sum);
output[i] = scalar_t(result);  // Cast back
```

## Profiling and Debugging

### NVIDIA Nsight Compute (ncu)

```bash
# Full metrics
ncu --set full -o metrics.ncu-rep python your_script.py

# Key metrics for sm120:
ncu --metrics \
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  dram__throughput.avg.pct_of_peak_sustained_elapsed,\
  lts__t_sectors_op_read_hit_rate.pct \
  python your_script.py

# What to watch:
# - Memory throughput (% of 1.79-1.8 TB/s GDDR7 peak)
# - L2 hit rate (96-128 MB L2 should yield high hit rates)
# - Achieved occupancy (max 48 warps/SM)
# - Register usage (no TMEM — watch for spilling)
# - Warp stall reasons (likely memory latency)
```

### NVIDIA Nsight Systems (nsys)

```bash
nsys profile -o profile_report python your_script.py

# Watch for:
# - Kernel fusion opportunities (many small kernels)
# - PCIe Gen5 transfer rates
# - GPU idle time between kernels
```

### Common Performance Issues

1. **Memory bandwidth bottleneck**: 1.8 TB/s is the primary limiter

   - Solution: Fuse kernels aggressively, maximize L2 hits, vectorize everything

2. **Shared memory too small for sm100 tiles**: Porting datacenter kernels directly

   - Solution: Reduce tile sizes from 228 KB budget to 128 KB budget

3. **Register spilling from WMMA**: Too many fragments in registers

   - Solution: Reduce tile dimensions, limit concurrent WMMA fragments, use `__launch_bounds__`

4. **1024-thread block occupancy**: Capped at 67%

   - Solution: Use 256 or 512 thread blocks instead

5. **No TMA overhead**: Manual address calculation per thread

   - Solution: Use cp.async with double-buffering, precompute addresses

6. **VRAM limit on RTX 5090**: 32 GB constrains model size

   - Solution: Use FP4/FP8 quantization, offload KV cache to host with pinned memory

## CUDA Compilation Flags

```bash
# For Blackwell desktop specifically
nvcc -arch=sm_120 -O3 your_kernel.cu

# Multi-target (Ada + Blackwell desktop)
nvcc -gencode arch=compute_89,code=sm_89 \
     -gencode arch=compute_120,code=sm_120 \
     -O3 your_kernel.cu

# Full consumer lineup
nvcc -gencode arch=compute_89,code=sm_89 \
     -gencode arch=compute_120,code=sm_120 \
     -gencode arch=compute_121,code=sm_121 \
     -O3 your_kernel.cu

# Useful flags:
# -maxrregcount=N    Limit registers per thread (critical for WMMA kernels)
# --ptxas-options=-v Print register/smem usage
# -lineinfo          Add debug line info
# --use_fast_math    Fast but less precise math
# -Xptxas -dlcm=ca   Cache global loads in L1
```

## Best Practices Summary

1. **Memory Access**: Coalesce always. GDDR7 is less forgiving of random access than HBM
2. **Shared Memory**: 128 KB max — larger than Ada (100 KB), much smaller than sm100 (228 KB)
3. **L2 Cache**: 96-128 MB — exploit aggressively, pin KV caches with persistence hints
4. **Kernel Fusion**: The single most important optimization on sm120 due to limited GDDR7 BW
5. **Tensor Cores**: Use WMMA (not WGMMA). Watch register pressure from fragments
6. **Block Size**: Prefer 256 threads. Avoid 1024 (caps at 67% occupancy)
7. **No TMA/Clusters**: Use cp.async for async loads. Each block is independent
8. **Precision**: FP4 for inference weights (new on sm120), FP8 for training, BF16 for general use
9. **Profile**: Watch memory throughput and L2 hit rate — these define the performance ceiling
10. **Type Conversions**: Always use explicit `to_float()`/`from_float()` helpers (PyTorch disables implicit FP16/BF16 conversions)
