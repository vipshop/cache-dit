# Blackwell Datacenter (sm100) GPU Optimization Guide

Deep dive into B200/GB200-specific optimizations for CUDA kernels.

## Blackwell Architecture Overview

### B200

| Spec          | Value     | Optimization Impact                        |
| ------------- | --------- | ------------------------------------------ |
| SMs           | 148       | Dual-die (74/die), grid multiples of 148   |
| Threads/SM    | 2048      | 64 warps/SM (up from 48 on Hopper)         |
| Shared Memory | 228 KB/SM | Larger tiles than Hopper                   |
| L2 Cache      | 126 MB    | 2.5x H100, significant reuse potential     |
| Memory BW     | 8 TB/s    | 2.4x H100, coalesced access still critical |
| Warp Size     | 32        | All reductions use warp shuffles           |

### GB200

| Spec          | Value     | Optimization Impact              |
| ------------- | --------- | -------------------------------- |
| SMs           | 148       | Same B200 die + Grace CPU        |
| Threads/SM    | 2048      | 64 warps/SM                      |
| Shared Memory | 228 KB/SM | Same SM architecture as B200     |
| L2 Cache      | 126 MB    | Same as B200                     |
| Memory BW     | 8 TB/s    | NVLink 5.0: 1.8 TB/s inter-GPU  |
| Warp Size     | 32        | All reductions use warp shuffles |

### Key Blackwell Features

1. **Dual-Die Design** - Two GPU dies on one package (74 SMs each), connected via 10 TB/s NVLink-C2C
2. **5th-Gen Tensor Cores** - FP4/FP6/FP8/FP16/BF16/TF32 support, 2x throughput vs Hopper
3. **Tensor Memory Accelerator (TMA)** - Enhanced over Hopper with multicast and bulk async operations
4. **Thread Block Clusters** - Groups of thread blocks that cooperate via distributed shared memory
5. **WGMMA (Warp Group Matrix Multiply-Accumulate)** - New instruction replacing Hopper's GMMA
6. **Tensor Memory (TMEM)** - Dedicated memory for tensor core operands, separate from register file
7. **tcgen05** - New tensor core generation instruction set
8. **NVFP4** - Native 4-bit floating point in tensor cores
9. **64 Warps/SM** - Up from 48 (Ada) and 48 (Hopper), improving latency hiding

### Key Differences from Hopper (sm90)

| Feature                   | sm100 (Blackwell DC) | sm90 (Hopper)            |
| ------------------------- | -------------------- | ------------------------ |
| Max warps/SM              | 64                   | 64                       |
| Max threads/SM            | 2048                 | 2048                     |
| Max blocks/SM             | 32                   | 16                       |
| Shared memory/SM          | 228 KB               | 192 KB                   |
| L1+Shared combined        | 256 KB               | 256 KB                   |
| L2 Cache                  | 126 MB               | 50 MB                    |
| Registers/SM              | 65536                | 65536                    |
| SMs (full die)            | 148 (dual-die)       | 132                      |
| Memory bandwidth          | 8 TB/s               | 3.35 TB/s                |
| Thread Block Clusters     | Yes (enhanced)       | Yes                      |
| Distributed Shared Memory | Yes (enhanced)       | Yes                      |
| TMA                       | Yes (v2, multicast)  | Yes (v1)                 |
| Tensor Core gen           | 5th gen              | 4th gen                  |
| FP4 support               | Yes (NVFP4)          | No                       |
| FP6 support               | Yes                  | No                       |
| WGMMA / tcgen05           | Yes                  | No (uses wgmma.mma_async)|
| Tensor Memory (TMEM)      | Yes                  | No                       |
| NVLink                    | 5.0 (1.8 TB/s)      | 4.0 (900 GB/s)           |

## Memory Hierarchy Optimization

### Global Memory Access Patterns

With 8 TB/s HBM3e bandwidth, Blackwell can tolerate more memory traffic — but coalescing remains critical:

```cuda
// GOOD: Coalesced access (threads access consecutive addresses)
// Each thread reads 4 bytes, warp reads 128 bytes (one transaction)
int idx = blockIdx.x * blockDim.x + threadIdx.x;
float val = input[idx];

// BAD: Strided access (wastes HBM bandwidth)
int idx = threadIdx.x * stride;  // Avoid stride > 1
float val = input[idx];
```

**Transaction sizes:**

- 32 bytes minimum
- 128 bytes optimal (full warp, FP32)
- Align to 128-byte boundaries when possible
- HBM3e has 16x 512-bit controllers — coalescing maps to controller utilization

### Vectorized Memory Access (Critical for Bandwidth)

Even with 8 TB/s, vectorized loads maximize bandwidth utilization:

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

**Bandwidth context (B200):**

With 8 TB/s theoretical bandwidth, well-optimized memory-bound kernels should target 50-60% utilization. The 2.4x bandwidth increase over H100 means many previously compute-bound kernels become memory-bound on Blackwell at smaller problem sizes.

### L2 Cache Utilization

Blackwell's 126 MB L2 cache is 2.5x H100's 50 MB:

```cuda
// For attention: Process Q blocks to maximize K,V cache reuse
// K,V tiles stay in L2 while Q block iterates

// Block size tuning for L2 on B200 (126 MB L2):
// BLOCK_SIZE_M = 128  (Q block)
// BLOCK_SIZE_N = 128  (K,V block — larger than H100 thanks to more smem + L2)
// With head_dim=128, each tile = 128*128*2 = 32KB (FP16)
// Hundreds of tiles fit in 126 MB L2

// L2 residency control via access properties
cudaAccessPolicyWindow policy;
policy.base_ptr = kv_cache;
policy.num_bytes = kv_cache_size;
policy.hitRatio = 1.0f;
policy.hitProp = cudaAccessPropertyPersisting;
policy.missProp = cudaAccessPropertyStreaming;

cudaStreamSetAccessPolicyWindow(stream, &policy);
```

### Shared Memory Configuration

Blackwell has a 256 KB combined L1/shared memory pool per SM. Configurable in finer granularity than Hopper:

- Supported shared memory sizes: 0, 8, 16, 32, 64, 100, 132, 164, 196, 228 KB
- Remaining capacity goes to L1

**Max usable shared memory is 228 KB** (vs 192 KB on Hopper). This enables larger tiles:

```cuda
// Request max shared memory for sm100
cudaFuncSetAttribute(
    attention_forward_kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    228 * 1024  // 228 KB max
);
```

**Impact on tile sizing vs Hopper:**

```cuda
// Hopper: 192 KB shared → limited tile sizes
// Blackwell: 228 KB shared → 19% more tile data

// For attention on sm100:
// Q tile: 128x128 * 2 bytes = 32 KB
// K tile: 128x128 * 2 bytes = 32 KB
// V tile: 128x128 * 2 bytes = 32 KB
// O accumulator: 128x128 * 4 bytes = 64 KB (FP32 accumulation)
// Total: 160 KB — fits in 228 KB with headroom for bookkeeping

// For GEMM:
// BLOCK_M=128, BLOCK_N=128, BLOCK_K=64 in FP16
// A tile: 128*64*2 = 16 KB
// B tile: 64*128*2 = 16 KB
// With double-buffering: 64 KB total — plenty of room
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

### Tensor Memory Accelerator (TMA) v2

Blackwell enhances Hopper's TMA with multicast support and improved async semantics:

```cuda
// TMA: Hardware-accelerated bulk copy from global to shared memory
// Bypasses register file entirely, reducing pressure
// Supports 1D-5D tensor descriptors with striding

// Create TMA descriptor (host-side)
CUtensorMap tensorMap;
cuTensorMapEncodeTiled(
    &tensorMap,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
    2,                    // 2D tensor
    (void*)global_ptr,
    sizes,                // {M, K}
    strides,              // {K * sizeof(half), sizeof(half)}
    boxSizes,             // {TILE_M, TILE_K}
    elementStrides,
    CU_TENSOR_MAP_INTERLEAVE_NONE,
    CU_TENSOR_MAP_SWIZZLE_128B,
    CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
);
```

**TMA multicast (new in Blackwell):**

```cuda
// Multicast TMA: Broadcast a single global memory tile to
// shared memory of multiple thread blocks in a cluster simultaneously
// Reduces global memory traffic by cluster_size factor

// In kernel, use cp.async.bulk.tensor with multicast
// Only 1 block in the cluster issues the load, all receive the data
asm volatile(
    "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
    " [%0], [%1, {%2, %3}], [%4], %5;"
    :
    : "r"(smem_addr), "l"(tensorMap), "r"(x), "r"(y),
      "r"(mbar_addr), "h"(multicast_mask)
);
```

### Distributed Shared Memory (DSMEM)

Thread blocks within a cluster can access each other's shared memory:

```cuda
// Access another block's shared memory within the cluster
// Useful for producer-consumer patterns and reducing redundant loads

// Get shared memory address from another block in the cluster
__shared__ float my_tile[TILE_SIZE];

// Block 0 loads K tile, Block 1 loads V tile
// Both can access each other's data without going through global memory
extern __shared__ float smem[];
float* remote_smem = cluster.map_shared_rank(smem, target_rank);
```

## Warp-Level Optimizations

### Shuffle Instructions

Fastest way to share data within a warp:

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

### WGMMA (Warp Group Matrix Multiply-Accumulate)

Blackwell introduces `tcgen05` instructions that supersede Hopper's GMMA. WGMMA operates on warp groups (4 warps = 128 threads):

```cuda
// WGMMA: Matrix multiply using Tensor Memory (TMEM)
// Operand A comes from TMEM (dedicated tensor memory, not registers)
// Operand B comes from shared memory or TMEM
// Result accumulates in TMEM

// Key advantage: TMEM is separate from the 64K register file
// This means WGMMA doesn't consume registers for tensor operands

// Typical WGMMA shapes (M x N x K):
// FP16/BF16: 64x256x16, 64x128x16, 64x64x16
// FP8:       64x256x32, 64x128x32, 64x64x32
// FP4:       64x256x64, 64x128x64, 64x64x64

// WGMMA is accessed via CUTLASS or inline PTX:
asm volatile(
    "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
    "{%0, %1, %2, %3, %4, %5, %6, %7, "
    " %8, %9, %10, %11, %12, %13, %14, %15, "
    " %16, %17, %18, %19, %20, %21, %22, %23, "
    " %24, %25, %26, %27, %28, %29, %30, %31}, "
    " %32, %33, p;"
    : /* outputs (accumulator in TMEM) */
    : /* inputs (desc_a from TMEM, desc_b from smem) */
);
```

### Tensor Memory (TMEM)

TMEM is a new memory space dedicated to tensor core operands, separate from registers:

```cuda
// TMEM key properties:
// - Not part of the 64K register file per SM
// - Accessed exclusively by tensor cores via tcgen05/WGMMA
// - Reduces register pressure for GEMM/attention kernels
// - Operands must be staged into TMEM before WGMMA

// Practical impact:
// - On Hopper: tensor operands consumed registers, limiting occupancy
// - On Blackwell: tensor operands live in TMEM, freeing registers
// - Allows higher occupancy for GEMM-heavy kernels
```

## Register Optimization

### Register Pressure

Blackwell allows 255 registers per thread, with 65536 registers per SM. With 64 warps max and TMEM offloading tensor operands, register pressure is significantly reduced:

```
Max registers per thread before spilling:
  128 threads/block (4 warps):  65536 / 128 = 512 → capped at 255
  256 threads/block (8 warps):  65536 / 256 = 256 → capped at 255
  512 threads/block (16 warps): 65536 / 512 = 128
  1024 threads/block (32 warps): 65536 / 1024 = 64
```

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

// Output accumulator in registers
// On Blackwell, TMEM handles tensor core operands,
// leaving more registers free for accumulators
float out_acc[HEAD_DIM];  // Works for head_dim <= ~128
```

## Occupancy Tuning

### Calculating Occupancy

```
Occupancy = Active Warps per SM / Max Warps per SM (64)

Limiting factors:
1. Registers: 65536 registers / (threads_per_block * regs_per_thread)
2. Shared Memory: 228KB / smem_per_block
3. Threads: 2048 / threads_per_block
4. Blocks: max 32 blocks per SM
```

**Blackwell allows up to 32 blocks per SM** (vs 16 on Hopper, 24 on Ada). This is ideal for small-block kernels:

```
64 threads/block (2 warps):   32 blocks → 64 warps → 100% occupancy
128 threads/block (4 warps):  16 blocks → 64 warps → 100% occupancy
256 threads/block (8 warps):  8 blocks  → 64 warps → 100% occupancy
512 threads/block (16 warps): 4 blocks  → 64 warps → 100% occupancy
1024 threads/block (32 warps): 2 blocks → 64 warps → 100% occupancy
```

All standard block sizes can achieve 100% occupancy on Blackwell (register and shared memory permitting).

### Block Size Selection

For B200 kernels:

| Kernel Type  | Threads/Block | Warps | Reasoning                                   |
| ------------ | ------------- | ----- | ------------------------------------------- |
| Element-wise | 256           | 8     | High occupancy, simple workloads            |
| Reduction    | 512-1024      | 16-32 | Enough threads for full reduction           |
| Attention    | 128-256       | 4-8   | Balance shared mem tiles and WGMMA usage    |
| GEMM (WGMMA) | 128           | 4     | Warp group (4 warps) is the WGMMA unit      |

### Thread Block Clusters

Clusters group thread blocks for cooperative execution. Blackwell improves on Hopper's cluster support:

```cuda
// Launch with clusters (CUDA 12+)
__global__ void __cluster_dims__(2, 1, 1)
attention_kernel(float* Q, float* K, float* V, float* O) {
    // 2 blocks per cluster
    // Block 0: loads and processes Q, K
    // Block 1: loads and processes V
    // Both share data via distributed shared memory

    cluster_group cluster = this_cluster();
    unsigned int block_rank = cluster.block_rank();

    // Synchronize within cluster (cheaper than global sync)
    cluster.sync();
}

// Or configure at launch time
cudaLaunchConfig_t config;
cudaLaunchAttribute attrs[1];
attrs[0].id = cudaLaunchAttributeClusterDimension;
attrs[0].val.clusterDim = {2, 1, 1};
config.attrs = attrs;
config.numAttrs = 1;
cudaLaunchKernelEx(&config, kernel, grid, block, smem, stream);
```

**When to use clusters:**

- Attention: One block loads K, another loads V — share via DSMEM
- Reduction across blocks without atomics
- Stencil patterns with halo exchange
- Any pattern where adjacent blocks need each other's data

### Dual-Die Considerations

The B200 has two GPU dies (74 SMs each) connected via 10 TB/s NVLink-C2C. This is mostly transparent but affects optimization:

```cuda
// Cross-die communication is handled by hardware, but:
// - L2 cache is split across dies (63 MB per die)
// - Thread block clusters should ideally stay within one die
// - Grid scheduling distributes blocks across dies

// For best performance:
// - Keep cluster_size small enough to fit on one die
// - Avoid access patterns that frequently cross die boundary
// - Let the hardware handle most cross-die scheduling
```

## Precision and Numerical Stability

### FP4/FP6 on Blackwell Tensor Cores

Blackwell introduces native FP4 (NVFP4) and FP6 support:

```cuda
// NVFP4: 1-bit sign, 2-bit exponent, 1-bit mantissa + block scaling
// FP6: 1-bit sign, 3-bit exponent, 2-bit mantissa
// Both use per-block scaling factors for range preservation

// FP4 is primarily used via:
// - cuBLAS with CUBLASLT_ORDER_COL32_2R_4R4 format
// - CUTLASS with SM100 FP4 kernel configs
// - TensorRT-LLM quantized inference

// FP4 WGMMA shape: 64x256x64 — double the K-dimension of FP8
// This means 2x more elements processed per instruction
```

### FP8 on Blackwell

Enhanced FP8 with higher tensor core throughput:

```cuda
// FP8 E4M3 (weights) and E5M2 (gradients)
// 2x throughput vs Hopper FP8 tensor cores
// Use per-tensor or per-channel scaling

// Scaling pattern for FP8 GEMM:
// C = scale_a * scale_b * (fp8_A @ fp8_B) + bias
// Where scale_a and scale_b are FP32 scaling factors
```

### Online Softmax for Attention

Numerically stable softmax without materializing full attention matrix:

```cuda
// Online softmax (same algorithm, larger tiles on Blackwell)
float row_max = -INFINITY;
float row_sum = 0.0f;

for each K block:  // Larger blocks fit in 228 KB shared memory
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

## Blackwell-Specific Optimization Strategies

### Porting from Hopper (sm90 → sm100)

Key changes when porting kernels from H100 to B200:

```cuda
// 1. Increase tile sizes (228 KB vs 192 KB shared memory)
// Hopper: BLOCK_M=128, BLOCK_N=64
// Blackwell: BLOCK_M=128, BLOCK_N=128 (more smem available)

// 2. Use larger clusters (improved DSMEM)
// Hopper: cluster_size = 2
// Blackwell: cluster_size = 2-4 (better cross-block communication)

// 3. Replace wgmma.mma_async with tcgen05 WGMMA
// Hopper: wgmma.mma_async (register → smem)
// Blackwell: tcgen05 WGMMA (TMEM → smem), frees registers

// 4. Use TMA multicast for shared loads across clusters
// Hopper: each block loads its own copy
// Blackwell: one block loads, multicast to all in cluster

// 5. Consider FP4/FP6 for inference kernels
// Not available on Hopper, 2-4x throughput improvement
```

### Maximizing Tensor Core Utilization

Tensor cores are the primary compute engine on Blackwell:

```cuda
// Tensor core utilization checklist:
// 1. Tile dimensions must be multiples of WGMMA shapes
//    FP16: M=64, N=64/128/256, K=16
//    FP8:  M=64, N=64/128/256, K=32
//    FP4:  M=64, N=64/128/256, K=64

// 2. Data must be in shared memory (for B operand) or TMEM (for A operand)
// 3. Accumulation is always in FP32 (output is in TMEM)
// 4. Software pipeline: overlap WGMMA with TMA loads

// Pipelining pattern:
// Stage 0: TMA load tile[0] → smem
// Stage 1: TMA load tile[1] → smem, WGMMA on tile[0]
// Stage 2: TMA load tile[2] → smem, WGMMA on tile[1]
// ...
```

### NVLink 5.0 Multi-GPU Optimization

GB200 systems use NVLink 5.0 with 1.8 TB/s bidirectional bandwidth:

```cuda
// Multi-GPU patterns for GB200 NVL72:
// - All-reduce: 1.8 TB/s per GPU link
// - Ring all-reduce across 72 GPUs
// - Tensor parallelism across NVLink-connected GPUs

// For custom multi-GPU kernels:
// Use NVSHMEM for direct GPU-to-GPU shared memory access
// Or cuBLAS/NCCL for standard collective operations
```

## Profiling and Debugging

### NVIDIA Nsight Systems (nsys)

System-wide profiling:

```bash
nsys profile -o profile_report python your_script.py

# Key metrics to watch:
# - Kernel duration
# - TMA utilization
# - NVLink traffic (for multi-die and multi-GPU)
# - Stream utilization
# - Cluster efficiency
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

# Key metrics for sm100 kernels:
# - Achieved occupancy (max 64 warps/SM)
# - Memory throughput (% of 8 TB/s peak)
# - Tensor core utilization (WGMMA throughput)
# - TMA throughput
# - L2 hit rate (126 MB L2)
# - Cross-die traffic
# - Warp stall reasons
```

### Common Performance Issues

1. **Underutilizing tensor cores**: Not using WGMMA/tcgen05

   - Solution: Use CUTLASS SM100 kernels or write WGMMA PTX

2. **Cross-die thrashing**: Working set spans both dies inefficiently

   - Solution: Keep clusters on one die, partition work by die

3. **TMA underuse**: Manual loads instead of hardware TMA

   - Solution: Use TMA for all bulk global→shared transfers

4. **Small tiles**: Porting H100 tile sizes without increasing for 228 KB smem

   - Solution: Increase tile dimensions to fill available shared memory

5. **FP8/FP4 scaling errors**: Incorrect scaling factors causing numerical issues

   - Solution: Calibrate scaling per-tensor or per-channel, validate against FP16 baseline

6. **Cluster overhead**: Clusters too large or poorly matched to problem

   - Solution: Profile cluster_size=1 vs 2 vs 4, pick based on actual DSMEM usage

## CUDA Compilation Flags

```bash
# For Blackwell datacenter specifically
nvcc -arch=sm_100 -O3 your_kernel.cu

# With TMA and cluster support
nvcc -arch=sm_100a -O3 your_kernel.cu  # 'a' suffix enables arch-specific features

# Multi-target build (Hopper + Blackwell)
nvcc -gencode arch=compute_90,code=sm_90 \
     -gencode arch=compute_100,code=sm_100 \
     -O3 your_kernel.cu

# Useful flags:
# -maxrregcount=N    Limit registers per thread
# --ptxas-options=-v Print register/smem usage
# -lineinfo          Add debug line info
# --use_fast_math    Fast but less precise math
# -Xptxas -dlcm=ca   Cache global loads in L1
```

## Best Practices Summary

1. **Memory Access**: Coalesce always, align to 128 bytes. 8 TB/s is forgiving but not unlimited
2. **Shared Memory**: 228 KB max — increase tile sizes vs Hopper, take advantage of the extra 36 KB
3. **L2 Cache**: 126 MB — 2.5x Hopper, use persistence hints for KV caches
4. **TMA**: Use for all bulk global→shared transfers; multicast across clusters to reduce BW
5. **WGMMA/tcgen05**: Use warp group MMA for all GEMM/attention — TMEM frees registers
6. **Clusters**: Use 2-4 block clusters for attention and stencil patterns with DSMEM
7. **Precision**: FP4 for inference weights, FP8 for training, BF16 fallback, FP32 accumulation
8. **Block Size**: 128-256 threads for WGMMA kernels. All sizes achieve 100% occupancy
9. **Profile**: Watch tensor core utilization and TMA throughput — these are the new bottlenecks
10. **Type Conversions**: Always use explicit `to_float()`/`from_float()` helpers (PyTorch disables implicit FP16/BF16 conversions)
