# Ada Lovelace (sm89) GPU Optimization Guide

Deep dive into Ada Lovelace-specific optimizations for CUDA kernels.

## Ada Architecture Overview

### L40S

| Spec          | Value      | Optimization Impact                      |
| ------------- | ---------- | ---------------------------------------- |
| SMs           | 142        | Near-full AD102, grid multiples of 142   |
| Threads/SM    | 1536       | 48 warps/SM, max 24 blocks per SM        |
| Shared Memory | 100 KB/SM  | Moderate tile sizes                      |
| L2 Cache      | 96 MB      | Large L2 for datacenter workloads        |
| Memory BW     | 864 GB/s   | 48 GB GDDR6 ECC, 384-bit bus             |
| Warp Size     | 32         | All reductions use warp shuffles         |

### RTX 4070

| Spec          | Value      | Optimization Impact                      |
| ------------- | ---------- | ---------------------------------------- |
| SMs           | 46         | AD104 die, grid multiples of 46          |
| Threads/SM    | 1536       | 48 warps/SM, max 24 blocks per SM        |
| Shared Memory | 100 KB/SM  | Same SM architecture as all sm89         |
| L2 Cache      | 36 MB      | Smallest L2 of sm89 targets              |
| Memory BW     | 504 GB/s   | 12 GB GDDR6X, 192-bit bus                |
| Warp Size     | 32         | All reductions use warp shuffles         |

### RTX 4090

| Spec          | Value      | Optimization Impact                      |
| ------------- | ---------- | ---------------------------------------- |
| SMs           | 128        | AD102 die, grid multiples of 128         |
| Threads/SM    | 1536       | 48 warps/SM, max 24 blocks per SM        |
| Shared Memory | 100 KB/SM  | Same SM architecture as all sm89         |
| L2 Cache      | 72 MB      | Good reuse across blocks                 |
| Memory BW     | 1.01 TB/s  | 24 GB GDDR6X, 384-bit bus                |
| Warp Size     | 32         | All reductions use warp shuffles         |

### RTX 6000 Ada

| Spec          | Value      | Optimization Impact                      |
| ------------- | ---------- | ---------------------------------------- |
| SMs           | 142        | Near-full AD102, grid multiples of 142   |
| Threads/SM    | 1536       | 48 warps/SM, max 24 blocks per SM        |
| Shared Memory | 100 KB/SM  | Same SM architecture as all sm89         |
| L2 Cache      | 96 MB      | Large L2 for workstation workloads       |
| Memory BW     | 960 GB/s   | 48 GB GDDR6, 384-bit bus                 |
| Warp Size     | 32         | All reductions use warp shuffles         |

### Key Ada Features

1. **Dual-Issue FP32** - Each SM has 2x FP32 throughput (128 FP32 CUDA cores/SM via dual datapath)
2. **4th-Gen Tensor Cores** - FP8 (E4M3/E5M2) support alongside FP16/BF16/TF32
3. **Shader Execution Reordering (SER)** - Hardware thread scheduling optimization
4. **Large L2 Caches** - Up to 96 MB (L40S/RTX 6000), a major advantage over Hopper's 50 MB
5. **cp.async** - Asynchronous global-to-shared memory copies (inherited from Ampere)

### Key Differences from Hopper (sm90)

| Feature                   | sm89 (Ada)          | sm90 (Hopper)            |
| ------------------------- | ------------------- | ------------------------ |
| Max warps/SM              | 48                  | 64                       |
| Max threads/SM            | 1536                | 2048                     |
| Max blocks/SM             | 24                  | 16                       |
| Shared memory/SM          | 100 KB              | 192 KB                   |
| L1+Shared combined        | 128 KB              | 256 KB                   |
| Registers/SM              | 65536               | 65536                    |
| Thread Block Clusters     | No                  | Yes                      |
| Distributed Shared Memory | No                  | Yes                      |
| TMA (Tensor Memory Accel) | No                  | Yes                      |
| FP8 Tensor Cores          | Yes (4th gen)       | Yes (4th gen)            |
| Memory type               | GDDR6/GDDR6X        | HBM3/HBM3e              |

## Memory Hierarchy Optimization

### Global Memory Access Patterns

GDDR6/GDDR6X has higher latency but narrower bus than HBM. Coalescing is even more critical:

```cuda
// GOOD: Coalesced access (threads access consecutive addresses)
// Each thread reads 4 bytes, warp reads 128 bytes (one transaction)
int idx = blockIdx.x * blockDim.x + threadIdx.x;
float val = input[idx];

// BAD: Strided access (wastes bus bandwidth on GDDR)
int idx = threadIdx.x * stride;  // Avoid stride > 1
float val = input[idx];
```

**Transaction sizes:**

- 32 bytes minimum
- 128 bytes optimal (full warp, FP32)
- Align to 128-byte boundaries when possible
- GDDR6X transactions are burstier than HBM — coalescing matters more

### Vectorized Memory Access (Critical for Bandwidth)

Vectorized loads/stores are essential on sm89 due to lower memory bandwidth vs datacenter GPUs:

**BFloat16 vectorization (2x elements per load):**

```cuda
// Load 2 bfloat16 elements at once (32-bit transaction)
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

**Bandwidth context (L40S):**

With 864 GB/s theoretical bandwidth, expect 30-40% utilization for well-optimized memory-bound kernels. Vectorized loads are the single biggest optimization for bandwidth-limited kernels on GDDR.

### L2 Cache Utilization

Ada's large L2 caches are a major strength. The 96 MB L2 on L40S/RTX 6000 is nearly 2x the H100's 50 MB:

```cuda
// For attention: Process Q blocks to maximize K,V cache reuse
// K,V tiles stay in L2 while Q block iterates

// Block size tuning for L2 on L40S (96 MB L2):
// BLOCK_SIZE_M = 64   (Q block — smaller than H100 due to less shared mem)
// BLOCK_SIZE_N = 64   (K,V block)
// With head_dim=64, each tile = 64*64*2 = 8KB (FP16)
// Many tiles fit in 96 MB L2 — excellent reuse

// For RTX 4070 (36 MB L2):
// Keep working set under 36 MB for L2 residency
// Smaller batch sizes benefit more from L2 reuse
```

**L2 persistence hints (Ada supports these):**

```cuda
// Mark data as streaming (don't pollute L2)
asm volatile("ld.global.cs.b32 %0, [%1];" : "=r"(val) : "l"(ptr));

// Mark data as persistent (keep in L2)
asm volatile("ld.global.ca.b32 %0, [%1];" : "=r"(val) : "l"(ptr));
```

### Shared Memory Configuration

Ada has a combined 128 KB L1/shared memory pool per SM. Configurable splits:

- 0 KB shared + 128 KB L1
- 8 KB shared + 120 KB L1
- 16 KB shared + 112 KB L1
- 32 KB shared + 96 KB L1
- 64 KB shared + 64 KB L1
- 100 KB shared + 28 KB L1

**Max usable shared memory is 100 KB** (vs 192 KB on Hopper). This fundamentally limits tile sizes:

```cuda
// Request max shared memory for sm89
cudaFuncSetAttribute(
    attention_forward_kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    100 * 1024  // 100 KB max (not 192 KB like Hopper)
);
```

**Impact on tile sizing:**

```cuda
// Hopper tile: 128x64 in FP16 = 16 KB per tile, fits easily in 192 KB
// Ada tile: must fit in 100 KB total

// For attention on sm89, reduce tile sizes:
// Q tile: 64x64 * 2 bytes = 8 KB
// K tile: 64x64 * 2 bytes = 8 KB
// V tile: 64x64 * 2 bytes = 8 KB
// Total: ~24 KB — leaves room for output accumulators

// Alternatively for GEMM-like ops:
// BLOCK_M=64, BLOCK_N=64, BLOCK_K=32 in FP16
// A tile: 64*32*2 = 4 KB
// B tile: 32*64*2 = 4 KB
// Fits comfortably with double-buffering
```

### Bank Conflicts

Shared memory has 32 banks (4 bytes per bank), same as Hopper:

```cuda
// Bank conflict example (all threads hit same bank)
__shared__ float data[1024];
float val = data[threadIdx.x * 32];  // BAD: 32-stride = same bank

// No bank conflict
float val = data[threadIdx.x];  // GOOD: consecutive access

// Bank conflict avoidance with padding
__shared__ float data[32][33];  // 33 instead of 32
float val = data[threadIdx.y][threadIdx.x];  // Different banks
```

### Asynchronous Memory Copy (cp.async)

Ada supports `cp.async` for overlapping global-to-shared memory copies with computation. Unlike Hopper, there is **no TMA** — you must use `cp.async` directly:

```cuda
// Asynchronous copy from global to shared memory
// Bypasses register file, reducing register pressure
#include <cuda_pipeline.h>

__shared__ float tile[BLOCK_SIZE][BLOCK_SIZE];

// Initiate async copy
__pipeline_memcpy_async(&tile[ty][tx], &global_data[gy * N + gx], sizeof(float));
__pipeline_commit();

// Do other work here while copy is in flight...

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
    // Prefetch next tile into other buffer
    if (k + TILE_K < K) {
        __pipeline_memcpy_async(&buf[1 - stage][ty][tx], &A[...], sizeof(float));
        __pipeline_commit();
    }

    // Wait for current tile
    __pipeline_wait_prior(1);
    __syncthreads();

    // Compute on current tile
    compute(buf[stage]);

    stage = 1 - stage;
}
```

## Warp-Level Optimizations

### Shuffle Instructions

Fastest way to share data within a warp (identical to Hopper):

```cuda
// Reduction using shuffles (no shared memory needed)
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

// Butterfly shuffle for max
float max_val = __shfl_xor_sync(0xffffffff, val, 16);
val = max(val, max_val);
// ... repeat for 8, 4, 2, 1
```

### Block-Level Reduction on sm89

With 48 warps max per SM (vs 64 on Hopper), block reductions with fewer warps are common:

```cuda
// Block reduction for sm89: typically 8-16 warps per block
template <typename T, int NUM_WARPS>
__device__ T block_reduce_sum(T val) {
    __shared__ T warp_sums[NUM_WARPS];

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // Warp-level reduction
    val = warp_reduce_sum(val);

    // Write warp result
    if (lane_id == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    // Final reduction in first warp
    if (warp_id == 0) {
        val = (lane_id < NUM_WARPS) ? warp_sums[lane_id] : T(0);
        val = warp_reduce_sum(val);
    }

    return val;
}
```

## Register Optimization

### Register Pressure

Ada allows 255 registers per thread, with 65536 registers per SM (same as Hopper). However, with only 48 warps max, register pressure dynamics differ:

```
Max registers per thread before spilling:
  256 threads/block (8 warps):  65536 / 256 = 256 → capped at 255
  512 threads/block (16 warps): 65536 / 512 = 128
  1024 threads/block (32 warps): 65536 / 1024 = 64
```

With fewer warps/SM than Hopper, each warp can use more registers without reducing occupancy:

```bash
nvcc --ptxas-options=-v your_kernel.cu
# Shows: "Used X registers, Y bytes smem"
```

### Register Tiling

For attention, keep partial results in registers:

```cuda
// Each thread maintains its own row_max and row_sum
float row_max = -INFINITY;
float row_sum = 0.0f;

// Output accumulator in registers (fits for head_dim <= ~64)
float out_acc[HEAD_DIM];  // Works for head_dim <= ~64

// On sm89, with 255 regs/thread and 8-warp blocks,
// you have generous register budgets per thread
```

## Occupancy Tuning

### Calculating Occupancy

```
Occupancy = Active Warps per SM / Max Warps per SM (48)

Limiting factors:
1. Registers: 65536 registers / (threads_per_block * regs_per_thread)
2. Shared Memory: 100KB / smem_per_block
3. Threads: 1536 / threads_per_block
4. Blocks: max 24 blocks per SM
```

**Key difference from Hopper:** Ada allows up to 24 blocks per SM (vs 16 on Hopper). This means small blocks (e.g., 64 threads) can still achieve high occupancy:

```
64 threads/block (2 warps):  24 blocks → 48 warps → 100% occupancy
128 threads/block (4 warps): 12 blocks → 48 warps → 100% occupancy
256 threads/block (8 warps): 6 blocks  → 48 warps → 100% occupancy
512 threads/block (16 warps): 3 blocks → 48 warps → 100% occupancy
1024 threads/block (32 warps): 1 block → 32 warps → 67% occupancy
```

### Block Size Selection

For sm89 kernels:

| Kernel Type  | Threads/Block | Warps | Reasoning                                      |
| ------------ | ------------- | ----- | ---------------------------------------------- |
| Element-wise | 256           | 8     | High occupancy, simple workloads               |
| Reduction    | 256-512       | 8-16  | Enough threads for reduction, fit in 100 KB    |
| Attention    | 128-256       | 4-8   | Balance shared mem (100 KB) and registers       |
| GEMM tiles   | 256           | 8     | Good register budget per thread                |

**Avoid 1024-thread blocks on sm89** — they cap occupancy at 67% (only 1 block fits per SM due to the 1536 thread limit).

### Cross-Device Grid Sizing

When targeting multiple sm89 devices, choose grid dimensions carefully:

```cuda
// Portable grid sizing across sm89 devices
int sm_count;
cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);

// L40S: 142 SMs, RTX 4090: 128 SMs, RTX 4070: 46 SMs
// Use waves: total_blocks should be a multiple of sm_count
int blocks_per_sm = 6;  // 6 blocks of 256 threads = 48 warps = 100%
int total_blocks = sm_count * blocks_per_sm;

// For variable workloads, ensure at least 1 full wave
int min_blocks = sm_count;
int num_blocks = max(min_blocks, (work_items + BLOCK_SIZE - 1) / BLOCK_SIZE);
```

## Precision and Numerical Stability

### FP8 on Ada Tensor Cores

Ada's 4th-gen tensor cores support FP8 (E4M3 and E5M2):

```cuda
// FP8 is primarily used via cuBLAS/cuDNN, not direct WMMA on Ada
// For custom kernels, prefer FP16/BF16 with FP32 accumulation

// E4M3: 4-bit exponent, 3-bit mantissa — more range, less precision
// E5M2: 5-bit exponent, 2-bit mantissa — wider range, least precision
// Use E4M3 for weights, E5M2 for gradients (standard convention)
```

### Online Softmax for Attention

Numerically stable softmax without materializing full attention matrix:

```cuda
// Traditional (bad for memory)
// scores = Q @ K^T  // [seq, seq] - huge!
// softmax(scores)
// output = scores @ V

// Online softmax (good)
float row_max = -INFINITY;
float row_sum = 0.0f;

for each K block:
    compute local_scores
    local_max = max(local_scores)

    // Update running statistics
    new_max = max(row_max, local_max)
    rescale = exp(row_max - new_max)

    row_sum = row_sum * rescale + sum(exp(local_scores - new_max))
    row_max = new_max

    // Update output accumulator with rescaling
    out_acc = out_acc * rescale + softmax_scores @ V_block
```

### Mixed Precision Pattern

Use FP32 for reductions, low precision for memory:

```cuda
// Input in FP16/BF16
float sum = 0.0f;  // Accumulate in FP32
for (int i = tid; i < hidden_size; i += blockDim.x) {
    float val = float(input[i]);  // Cast to FP32
    sum += val * val;
}
// Reduction in FP32
sum = block_reduce_sum(sum);

// Output in FP16/BF16
output[i] = scalar_t(result);  // Cast back
```

## Ada-Specific Optimization Strategies

### Leveraging Large L2 Over Small Shared Memory

Ada's L2 (72-96 MB) is much larger relative to shared memory (100 KB) compared to Hopper. This changes the optimization strategy:

```cuda
// Strategy: Use L2 as a "second level" tile buffer
// Instead of large shared memory tiles, use smaller tiles
// and rely on L2 for inter-tile reuse

// On Hopper: large 192 KB smem tiles, moderate L2
// On Ada: small 100 KB smem tiles, massive L2

// Practical impact for GEMM:
// - Use smaller BLOCK_K to fit in shared memory
// - Iterate more K-steps (data stays in L2 between iterations)
// - Double-buffer to overlap loads with compute
```

### Dual-Issue FP32 Advantage

Ada SMs have 128 FP32 cores (dual datapath), giving 2x FP32 throughput per SM compared to Ampere. For FP32-heavy kernels:

```cuda
// FP32 operations are "free" relative to shared memory bandwidth
// Use this to do more compute per memory access

// Example: fused RMSNorm + activation in FP32
// The extra FP32 ops for activation cost almost nothing
float val = float(input[i]);
float normalized = val * rsqrt_var;
float activated = normalized * (1.0f / (1.0f + expf(-normalized)));  // SiLU
output[i] = scalar_t(activated);
```

### Memory Bandwidth Compensation

sm89 devices have 2-4x less memory bandwidth than H100 (864 GB/s vs 3.35 TB/s). Compensate with:

1. **Aggressive kernel fusion** — reduce global memory round-trips
2. **Higher arithmetic intensity** — do more compute per byte loaded
3. **L2 cache exploitation** — keep working sets in L2
4. **Vectorized access** — maximize bytes per transaction

```cuda
// BAD: Separate kernels (3 global memory round-trips)
rmsnorm_kernel<<<...>>>(input, normed);
geglu_kernel<<<...>>>(normed, activated);
residual_kernel<<<...>>>(activated, residual, output);

// GOOD: Fused kernel (1 global memory round-trip)
fused_rmsnorm_geglu_residual_kernel<<<...>>>(input, residual, output);
```

## Profiling and Debugging

### NVIDIA Nsight Systems (nsys)

System-wide profiling:

```bash
nsys profile -o profile_report python your_script.py

# Key metrics to watch:
# - Kernel duration
# - Memory transfer time (PCIe Gen4 for L40S)
# - GPU idle time
# - Stream utilization
```

### NVIDIA Nsight Compute (ncu)

Detailed kernel analysis:

```bash
# Full metrics
ncu --set full -o metrics.ncu-rep python your_script.py

# Specific metrics
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed \
python your_script.py

# Key metrics for sm89 kernels:
# - Achieved occupancy (max 48 warps/SM)
# - Memory throughput (% of GDDR peak)
# - Compute throughput (watch FP32 dual-issue utilization)
# - L2 hit rate (critical for Ada)
# - Warp stall reasons
```

### Common Performance Issues

1. **Shared memory bottleneck**: 100 KB limit forces smaller tiles

   - Solution: Reduce tile sizes, rely on L2 for reuse, double-buffer

2. **Memory bound** (most common on sm89): Low BW vs datacenter GPUs

   - Solution: Fuse kernels aggressively, vectorize all loads/stores, maximize L2 hits

3. **Bank conflicts**: Shared memory access pattern issues

   - Solution: Add padding, change access pattern

4. **Warp divergence**: Conditional branches within warp

   - Solution: Restructure to process similar elements together

5. **Low occupancy from large blocks**: 1024-thread blocks cap at 67%

   - Solution: Prefer 256-512 thread blocks on sm89

6. **PCIe bottleneck** (L40S/RTX cards): Host-device transfers over PCIe Gen4

   - Solution: Overlap transfers with compute, use pinned memory, minimize transfers

## CUDA Compilation Flags

```bash
# For Ada Lovelace specifically
nvcc -arch=sm_89 -O3 your_kernel.cu

# Multi-target build (Ada + Hopper)
nvcc -gencode arch=compute_89,code=sm_89 \
     -gencode arch=compute_90,code=sm_90 \
     -O3 your_kernel.cu

# Useful flags:
# -maxrregcount=N    Limit registers per thread
# --ptxas-options=-v Print register/smem usage
# -lineinfo          Add debug line info
# --use_fast_math    Fast but less precise math
# -Xptxas -dlcm=ca   Cache global loads in L1
```

## Best Practices Summary

1. **Memory Access**: Always coalesce, align to 128 bytes. GDDR is less forgiving than HBM
2. **Shared Memory**: 100 KB max — use smaller tiles than Hopper, double-buffer with cp.async
3. **L2 Cache**: Exploit the large L2 (72-96 MB) as a major caching layer
4. **Registers**: 255/thread with generous per-warp budgets due to fewer warps/SM
5. **Reductions**: Use warp shuffles, avoid atomics when possible
6. **Precision**: BF16 for training, FP16 for inference, FP32 for accumulation
7. **Block Size**: Prefer 256 threads. Avoid 1024 (caps occupancy at 67%)
8. **Fusion**: Critical on sm89 — lower BW means every kernel launch costs more
9. **Profile**: Watch L2 hit rate and memory throughput — these are the usual bottlenecks
10. **Type Conversions**: Always use explicit `to_float()`/`from_float()` helpers (PyTorch disables implicit FP16/BF16 conversions)
