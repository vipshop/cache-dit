# Blackwell Ultra (sm103) GPU Optimization Guide

Deep dive into B300/GB300-specific optimizations for CUDA kernels.

## Blackwell Ultra Architecture Overview

### B300

| Spec          | Value     | Optimization Impact                   |
| ------------- | --------- | ------------------------------------- |
| SMs           | 160       | Grid sizing: aim for multiples of 160 |
| Threads/SM    | 2048      | 64 warps/SM, same as sm100            |
| Shared Memory | 228 KB/SM | Same SM architecture as B200          |
| L2 Cache      | 192 MB    | 1.5x B200, massive reuse potential    |
| Memory BW     | 8 TB/s    | 288 GB HBM3e (12-high stacks)        |
| Warp Size     | 32        | All reductions use warp shuffles      |

### GB300

| Spec          | Value     | Optimization Impact              |
| ------------- | --------- | -------------------------------- |
| SMs           | 160       | Same B300 die + Grace CPU        |
| Threads/SM    | 2048      | 64 warps/SM                      |
| Shared Memory | 228 KB/SM | Same SM architecture as B300     |
| L2 Cache      | 192 MB    | Same as B300                     |
| Memory BW     | 8 TB/s    | NVLink 5.0: 1.8 TB/s inter-GPU  |
| Warp Size     | 32        | All reductions use warp shuffles |

### Key Blackwell Ultra Features

1. **160 SMs** - 8% more SMs than B200 (148), organized in 8 GPCs with 20 SMs each
2. **192 MB L2 Cache** - 1.5x B200 (126 MB), 3.8x H100 (50 MB)
3. **288 GB HBM3e** - 12-high stacks, 50% more capacity than B200 (192 GB)
4. **Enhanced NVFP4** - Improved FP4 tensor core throughput over sm100
5. **208B Transistors** - Dual-die design, same NVLink-C2C interconnect as B200
6. **PCIe Gen6** - 2x PCIe Gen5 bandwidth for host-device transfers
7. **1,400W TDP** - Higher power budget enables sustained boost clocks
8. **Same SM microarchitecture as sm100** - WGMMA, TMEM, tcgen05, TMA v2 all carry over

### Key Differences from Blackwell (sm100)

| Feature                   | sm103 (Blackwell Ultra) | sm100 (Blackwell)        |
| ------------------------- | ----------------------- | ------------------------ |
| SMs                       | 160                     | 148                      |
| L2 Cache                  | 192 MB                  | 126 MB                   |
| HBM3e capacity            | 288 GB                  | 192 GB                   |
| Max warps/SM              | 64                      | 64                       |
| Max threads/SM            | 2048                    | 2048                     |
| Max blocks/SM             | 32                      | 32                       |
| Shared memory/SM          | 228 KB                  | 228 KB                   |
| Registers/SM              | 65536                   | 65536                    |
| Memory bandwidth          | 8 TB/s                  | 8 TB/s                   |
| Tensor cores per SM       | Same (5th gen)          | 5th gen                  |
| WGMMA / tcgen05 / TMEM    | Yes                     | Yes                      |
| TMA v2 (multicast)        | Yes                     | Yes                      |
| Thread Block Clusters     | Yes                     | Yes                      |
| NVLink                    | 5.0 (1.8 TB/s)         | 5.0 (1.8 TB/s)          |
| PCIe                      | Gen6                    | Gen5                     |
| TDP                       | 1,400W                  | 1,000W                   |

**The SM microarchitecture is identical to sm100.** The gains come from more SMs, more L2, more HBM capacity, and PCIe Gen6. All sm100 kernel code runs on sm103 without modification — the optimization focus is on scaling and cache behavior.

## Porting from sm100 to sm103

### What Changes

Most sm100 kernels run optimally on sm103 with zero code changes. The key tuning opportunities are:

```cuda
// 1. Grid sizing: 160 SMs instead of 148
//    Adjust grid dimensions for full GPU utilization
int sm_count;  // 160 on B300, 148 on B200
cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
int total_blocks = sm_count * blocks_per_sm;

// 2. L2 residency: 192 MB instead of 126 MB
//    Larger working sets can stay resident in L2
//    KV caches for longer sequences fit entirely in L2

// 3. HBM capacity: 288 GB instead of 192 GB
//    Larger models fit without offloading
//    Bigger batch sizes possible
```

### What Stays the Same

Everything at the SM level is identical to sm100:

- 228 KB shared memory per SM
- 64 warps/SM, 32 max blocks/SM
- 65536 registers per SM
- WGMMA shapes and throughput
- TMEM size and behavior
- TMA descriptor format and multicast
- Bank conflict rules
- Warp shuffle semantics

## Memory Hierarchy Optimization

### Global Memory Access Patterns

Same 8 TB/s HBM3e bandwidth as B200. Coalescing rules are identical:

```cuda
// GOOD: Coalesced access
int idx = blockIdx.x * blockDim.x + threadIdx.x;
float val = input[idx];

// BAD: Strided access
int idx = threadIdx.x * stride;
float val = input[idx];
```

**Transaction sizes (same as sm100):**

- 32 bytes minimum
- 128 bytes optimal (full warp, FP32)
- Align to 128-byte boundaries

### L2 Cache: The Key sm103 Advantage

The 192 MB L2 is the primary differentiator. This impacts kernel design significantly:

```cuda
// Working set sizing for L2 residency:
//
// B200 (126 MB L2):
//   KV cache for 1 layer, head_dim=128, seq_len=8K, batch=32, FP16:
//   2 * 32 * 8192 * 128 * 2 bytes = 128 MB → barely fits
//
// B300 (192 MB L2):
//   Same config = 128 MB → fits with 64 MB headroom
//   Can increase to seq_len=12K or batch=48 and still fit

// L2 persistence control (same API as sm100)
cudaAccessPolicyWindow policy;
policy.base_ptr = kv_cache;
policy.num_bytes = kv_cache_size;  // Can be up to ~180 MB effectively
policy.hitRatio = 1.0f;
policy.hitProp = cudaAccessPropertyPersisting;
policy.missProp = cudaAccessPropertyStreaming;
cudaStreamSetAccessPolicyWindow(stream, &policy);
```

**L2 partitioning across dies:**

```cuda
// B300 dual-die: 192 MB total, ~96 MB per die
// vs B200: 126 MB total, ~63 MB per die
//
// Each die's L2 partition is now comparable to an entire H100's L2 (50 MB)
// Cross-die L2 access has higher latency — keep hot data on the local die

// Strategy: Partition KV cache across dies
// Die 0: layers 0..N/2 KV cache → local L2
// Die 1: layers N/2..N KV cache → local L2
// Pipeline execution to keep each die working on its local data
```

**Attention kernel tile sizing for L2:**

```cuda
// With 192 MB L2, attention can use larger K,V working sets
// before eviction occurs

// BLOCK_SIZE_M = 128  (Q block)
// BLOCK_SIZE_N = 128  (K,V block)
// head_dim = 128, FP16
// K tile: 128 * 128 * 2 = 32 KB
// V tile: 128 * 128 * 2 = 32 KB
//
// At 192 MB L2, ~3000 K+V tile pairs can be resident
// This covers most practical sequence lengths without eviction
```

### Shared Memory (Same as sm100)

228 KB max per SM, same configurable splits:

```cuda
// Supported sizes: 0, 8, 16, 32, 64, 100, 132, 164, 196, 228 KB
cudaFuncSetAttribute(
    kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    228 * 1024
);
```

### Vectorized Memory Access

Identical patterns to sm100 — vectorization remains critical:

```cuda
// BFloat16 vectorization
const __nv_bfloat162* vec_input = reinterpret_cast<const __nv_bfloat162*>(row_input);
#pragma unroll 4
for (int i = tid; i < hidden_size / 2; i += stride) {
    __nv_bfloat162 v = vec_input[i];
    float v0 = __bfloat162float(v.x);
    float v1 = __bfloat162float(v.y);
}

// FP16 vectorization
const __half2* vec_input = reinterpret_cast<const __half2*>(row_input);
__half2 v = vec_input[i];

// FP32 vectorization
const float4* vec_input = reinterpret_cast<const float4*>(row_input);
float4 v = vec_input[i];
```

### TMA v2 and Multicast

Same TMA as sm100, but more SMs means more concurrent TMA operations:

```cuda
// TMA multicast: broadcast tile to cluster members
// With 160 SMs, more clusters can be active simultaneously
// This increases aggregate TMA throughput

// TMA descriptor creation (identical to sm100)
CUtensorMap tensorMap;
cuTensorMapEncodeTiled(
    &tensorMap,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
    2, (void*)global_ptr,
    sizes, strides, boxSizes, elementStrides,
    CU_TENSOR_MAP_INTERLEAVE_NONE,
    CU_TENSOR_MAP_SWIZZLE_128B,
    CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
);
```

## Warp-Level and Tensor Core Optimizations

All warp-level features are identical to sm100. See the [sm100 guide](sm100-optimization-guide.md) for:

- Warp shuffle reductions
- WGMMA / tcgen05 instructions
- Tensor Memory (TMEM) usage
- Distributed Shared Memory patterns

### WGMMA Quick Reference

```cuda
// WGMMA shapes (same as sm100):
// FP16/BF16: 64x256x16, 64x128x16, 64x64x16
// FP8:       64x256x32, 64x128x32, 64x64x32
// FP4:       64x256x64, 64x128x64, 64x64x64

// Warp group = 4 warps = 128 threads
// Operand A: from TMEM
// Operand B: from shared memory or TMEM
// Accumulator: in TMEM (FP32)
```

## Occupancy Tuning

### Calculating Occupancy (Same as sm100)

```
Occupancy = Active Warps per SM / Max Warps per SM (64)

Limiting factors:
1. Registers: 65536 / (threads_per_block * regs_per_thread)
2. Shared Memory: 228KB / smem_per_block
3. Threads: 2048 / threads_per_block
4. Blocks: max 32 per SM
```

All standard block sizes achieve 100% occupancy (same as sm100):

```
64 threads/block:   32 blocks → 64 warps → 100%
128 threads/block:  16 blocks → 64 warps → 100%
256 threads/block:  8 blocks  → 64 warps → 100%
512 threads/block:  4 blocks  → 64 warps → 100%
1024 threads/block: 2 blocks  → 64 warps → 100%
```

### Grid Sizing for 160 SMs

The primary tuning point vs sm100:

```cuda
// B300 has 160 SMs (vs B200's 148)
// For full utilization, grid should be a multiple of 160

// Example: attention across 32 heads
// 32 heads → 32 blocks (1 per head) → only 20% of SMs active
// Better: 32 heads * 5 Q-blocks = 160 blocks → 100% utilization

// General formula:
int waves = (total_blocks + sm_count - 1) / sm_count;
int padded_blocks = waves * sm_count;  // Round up to full wave

// For small problems, ensure at least 1 full wave:
int min_blocks_per_sm = 2;  // At least 2 blocks per SM for latency hiding
int min_total_blocks = sm_count * min_blocks_per_sm;  // 320 blocks minimum
```

**Wave quantization effect:**

```
160 SMs, varying grid sizes:
  160 blocks → 1 wave  → 100% utilization
  161 blocks → 2 waves → 50.3% avg utilization (wave 2 has 1 block)
  240 blocks → 2 waves → 75% avg utilization
  320 blocks → 2 waves → 100% utilization

// Rule: Keep total_blocks as a multiple of 160
// Or use enough blocks that tail effects are negligible (>= 5 waves)
```

### Thread Block Clusters on 160 SMs

More SMs means more clusters can be active:

```cuda
// B200: 148 SMs, cluster_size=2 → 74 clusters max
// B300: 160 SMs, cluster_size=2 → 80 clusters max
// B300: cluster_size=4 → 40 clusters max

// More concurrent clusters → better for attention with many heads
// 80 heads with cluster_size=2 → exactly 1 wave on B300

__global__ void __cluster_dims__(2, 1, 1)
attention_kernel(/* ... */) {
    // 2 blocks per cluster
    // With 160 SMs: 80 clusters per wave
}
```

## sm103-Specific Optimization Strategies

### Large Model Inference (288 GB HBM)

The 288 GB HBM3e enables running larger models without tensor parallelism:

```cuda
// Model memory estimation:
// Llama 70B in FP16:  ~140 GB → fits on single B300
// Llama 70B in FP8:   ~70 GB  → fits with room for KV cache
// Llama 405B in FP4:  ~100 GB → fits on single B300
// Mixtral 8x22B FP16: ~280 GB → barely fits on single B300

// KV cache sizing with 288 GB:
// After model weights, remaining memory for KV cache:
// 288 GB - 140 GB (70B FP16) = 148 GB for KV cache
// Per-token KV: 2 * n_layers * n_heads * head_dim * 2 bytes (FP16)
// Llama 70B: 2 * 80 * 64 * 128 * 2 = 2.5 MB per token
// 148 GB / 2.5 MB = ~59,200 tokens of context
```

### Exploiting 192 MB L2 for Decoding

During autoregressive decoding, the KV cache dominates memory access:

```cuda
// Strategy: Keep frequently accessed KV cache layers in L2
// 192 MB L2 can hold:
//   - ~6 full attention layers of KV cache (32 heads, 8K seq, FP16)
//   - Or partial KV for all layers if seq is short

// Prefetching pattern for decode:
// 1. Pin current + next layer KV in L2 (persistence hints)
// 2. Stream Q vectors (small, don't pollute L2)
// 3. Attention output goes to streaming (will be consumed by next layer)

// Per-layer L2 budget:
// 192 MB / 80 layers ≈ 2.4 MB per layer
// This holds ~960 tokens of KV per layer in FP16 (head_dim=128, 64 heads)
// For longer sequences, rotate which layers are L2-resident
```

### PCIe Gen6 Host-Device Transfers

B300 introduces PCIe Gen6 (128 GB/s vs Gen5's 64 GB/s):

```cuda
// PCIe Gen6 doubles host-device transfer bandwidth
// Critical for:
// - Prefilling long prompts from CPU memory
// - Offloading KV cache to host memory
// - Multi-node inference with CPU-mediated communication

// Use pinned memory for maximum PCIe throughput
float* h_pinned;
cudaMallocHost(&h_pinned, size);  // Pinned (page-locked) memory

// Overlap transfers with compute using streams
cudaStream_t compute_stream, transfer_stream;
cudaStreamCreate(&compute_stream);
cudaStreamCreate(&transfer_stream);

// Transfer next batch while computing current batch
cudaMemcpyAsync(d_next, h_next, size, cudaMemcpyHostToDevice, transfer_stream);
compute_kernel<<<grid, block, 0, compute_stream>>>(d_current, ...);
```

### Power-Aware Optimization (1,400W TDP)

The B300's 1,400W TDP (vs B200's 1,000W) allows sustained high-frequency operation:

```cuda
// Higher power budget means:
// 1. Less frequency throttling under sustained load
// 2. Both dies can run at full speed simultaneously
// 3. More headroom for power-hungry FP4/FP8 tensor core operations

// Practical impact:
// - Sustained WGMMA throughput is higher (less thermal throttling)
// - Memory-bound kernels benefit less (HBM BW is the same)
// - Compute-bound kernels see the biggest gains

// Monitor power and clocks:
// nvidia-smi dmon -s puc  (power, utilization, clocks)
```

## Precision and Numerical Stability

All precision features are identical to sm100:

- **NVFP4**: Native 4-bit with block scaling (inference weights)
- **FP6**: 6-bit floating point (inference weights)
- **FP8 E4M3/E5M2**: 8-bit for training and inference
- **FP16/BF16**: Standard training precision
- **TF32**: Transparent FP32 acceleration via tensor cores
- **FP32**: Accumulation and reductions

See the [sm100 guide](sm100-optimization-guide.md) for detailed precision patterns.

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

# sm103-specific metrics to watch:
ncu --metrics \
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  dram__throughput.avg.pct_of_peak_sustained_elapsed,\
  lts__t_sectors_op_read_hit_rate.pct,\
  lts__t_sectors_op_write_hit_rate.pct \
  python your_script.py

# Key metrics for sm103:
# - L2 hit rate (critical — 192 MB should yield high hit rates)
# - Memory throughput (% of 8 TB/s)
# - Tensor core utilization
# - Wave efficiency (are you filling 160 SMs?)
# - Cross-die traffic
```

### NVIDIA Nsight Systems (nsys)

```bash
nsys profile -o profile_report python your_script.py

# Watch for:
# - Kernel duration (should scale ~8% faster than B200 due to more SMs)
# - PCIe Gen6 transfer rates
# - NVLink utilization (GB300 NVL72 systems)
# - Tail effects from wave quantization at 160 SMs
```

### Common Performance Issues

1. **Wave quantization at 160 SMs**: Grid size not a multiple of 160

   - Solution: Pad grid to multiples of 160, or use enough blocks (>= 800) to amortize

2. **L2 underutilization**: Not using persistence hints for hot data

   - Solution: Pin KV cache in L2 with `cudaAccessPropertyPersisting`

3. **Porting sm100 grids without adjustment**: Grid tuned for 148 SMs leaves 12 SMs idle

   - Solution: Query SM count at runtime, compute grid dynamically

4. **Cross-die L2 misses**: Working set split across dies

   - Solution: Partition data to keep hot sets on one die, use cluster locality

5. **PCIe Gen6 not saturated**: Using unpinned memory or small transfers

   - Solution: Use `cudaMallocHost` for pinned memory, batch transfers

6. **Power throttling on sustained FP4**: Hitting 1,400W limit

   - Solution: Monitor with `nvidia-smi`, consider pipelining compute with idle periods

## CUDA Compilation Flags

```bash
# For Blackwell Ultra specifically
nvcc -arch=sm_103 -O3 your_kernel.cu

# With arch-specific features (TMA, clusters, tcgen05)
nvcc -arch=sm_103a -O3 your_kernel.cu

# Multi-target build (Blackwell + Blackwell Ultra)
nvcc -gencode arch=compute_100,code=sm_100 \
     -gencode arch=compute_103,code=sm_103 \
     -O3 your_kernel.cu

# Full multi-target (Ada + Hopper + Blackwell + Blackwell Ultra)
nvcc -gencode arch=compute_89,code=sm_89 \
     -gencode arch=compute_90,code=sm_90 \
     -gencode arch=compute_100,code=sm_100 \
     -gencode arch=compute_103,code=sm_103 \
     -O3 your_kernel.cu

# Useful flags:
# -maxrregcount=N    Limit registers per thread
# --ptxas-options=-v Print register/smem usage
# -lineinfo          Add debug line info
# --use_fast_math    Fast but less precise math
```

## Best Practices Summary

1. **Grid Sizing**: Target multiples of 160 SMs. Query SM count at runtime for portability
2. **L2 Cache**: 192 MB is the key advantage — pin KV caches, use persistence hints aggressively
3. **Shared Memory**: 228 KB max, identical to sm100 — no tile size changes needed
4. **HBM Capacity**: 288 GB enables larger models; reduce tensor parallelism where possible
5. **SM Architecture**: Identical to sm100 — all WGMMA, TMEM, TMA patterns carry over directly
6. **PCIe Gen6**: Use pinned memory and async transfers to exploit 128 GB/s host bandwidth
7. **Clusters**: More SMs = more concurrent clusters. Scale cluster count with problem size
8. **Precision**: Same FP4/FP6/FP8/FP16/BF16 support as sm100
9. **Profile**: Focus on L2 hit rate and wave efficiency at 160 SMs
10. **Portability**: Write kernels for sm100, they run on sm103. Only grid sizing and L2 budgets need adjustment
