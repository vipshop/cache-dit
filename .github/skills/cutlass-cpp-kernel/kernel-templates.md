# CUDA Kernel Templates for H100

Complete, copy-paste ready templates for implementing new kernels.

## Template 1: Element-wise Operation (RoPE style)
Use this pattern for operations that process elements independently.

```cuda
/*
 * Element-wise kernel template for H100 (sm_90)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

constexpr int BLOCK_SIZE = 256;

// Type conversion helpers (include in every .cu file)
__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

__device__ __forceinline__ float from_float(float x, float*) { return x; }
__device__ __forceinline__ __half from_float(float x, __half*) { return __float2half(x); }
__device__ __forceinline__ __nv_bfloat16 from_float(float x, __nv_bfloat16*) { return __float2bfloat16(x); }

template <typename scalar_t>
__global__ void your_elementwise_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int total_elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        float val = to_float(input[idx]);

        // Your computation here
        float result = val;  // Replace with actual operation

        output[idx] = from_float(result, (scalar_t*)nullptr);
    }
}

// C++ entry points
extern "C" {

void your_kernel_forward_fp16(
    __half* output,
    const __half* input,
    int total_elements,
    cudaStream_t stream
) {
    const int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    your_elementwise_kernel<__half><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        output, input, total_elements
    );
}

void your_kernel_forward_bf16(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    int total_elements,
    cudaStream_t stream
) {
    const int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    your_elementwise_kernel<__nv_bfloat16><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        output, input, total_elements
    );
}

void your_kernel_forward_fp32(
    float* output,
    const float* input,
    int total_elements,
    cudaStream_t stream
) {
    const int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    your_elementwise_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        output, input, total_elements
    );
}

}
```

## Template 2: Row-wise Reduction (LayerNorm style)
Use for operations requiring reduction across a dimension (normalization, softmax).

### Basic Version (Scalar Loads)
```cuda
/*
 * Row-wise reduction kernel template for H100 (sm_90)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS = 1024;

// Type conversion helpers
__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

__device__ __forceinline__ float from_float(float x, float*) { return x; }
__device__ __forceinline__ __half from_float(float x, __half*) { return __float2half(x); }
__device__ __forceinline__ __nv_bfloat16 from_float(float x, __nv_bfloat16*) { return __float2bfloat16(x); }

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename T>
__device__ __forceinline__ T block_reduce_sum(T val) {
    __shared__ T shared[32];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : T(0);
    if (wid == 0) val = warp_reduce_sum(val);

    return val;
}

template <typename scalar_t>
__global__ void your_reduction_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    const int hidden_size,
    const float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const scalar_t* row_input = input + row * hidden_size;
    scalar_t* row_output = output + row * hidden_size;

    // Step 1: Compute reduction (e.g., sum of squares for RMSNorm)
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = to_float(row_input[i]);
        sum_sq += val * val;
    }
    sum_sq = block_reduce_sum(sum_sq);

    // Step 2: Compute normalization factor
    __shared__ float s_factor;
    if (tid == 0) {
        s_factor = rsqrtf(sum_sq / hidden_size + eps);
    }
    __syncthreads();
    float factor = s_factor;

    // Step 3: Apply normalization
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float normalized = to_float(row_input[i]) * factor;
        row_output[i] = from_float(normalized * to_float(weight[i]), (scalar_t*)nullptr);
    }
}
```

### Optimized Version: Vectorized BF16 RMSNorm (2.67x faster)

```cuda
/*
 * Vectorized RMSNorm kernel for BF16 - H100 optimized
 * Uses __nv_bfloat162 for 2-element vectorized memory access
 * Achieves 2.67x speedup over scalar version
 */
__global__ void rmsnorm_kernel_bf16_vectorized(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const int hidden_size,
    const float eps
) {
    extern __shared__ char smem[];
    float* shared = reinterpret_cast<float*>(smem);

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const __nv_bfloat16* row_input = input + row * hidden_size;
    __nv_bfloat16* row_output = output + row * hidden_size;

    // Phase 1: Compute sum of squares with bf16x2 vectorized loads
    float sum_sq = 0.0f;
    const int vec_hidden = hidden_size / 2;
    const __nv_bfloat162* vec_input = reinterpret_cast<const __nv_bfloat162*>(row_input);

    #pragma unroll 4
    for (int i = tid; i < vec_hidden; i += stride) {
        __nv_bfloat162 v = vec_input[i];
        float v0 = __bfloat162float(v.x);
        float v1 = __bfloat162float(v.y);
        sum_sq += v0 * v0 + v1 * v1;
    }

    // Handle odd element if hidden_size is odd
    if (hidden_size % 2 == 1 && tid == 0) {
        float v = __bfloat162float(row_input[hidden_size - 1]);
        sum_sq += v * v;
    }

    // Reduce across block
    sum_sq = block_reduce_sum(sum_sq, shared);

    // Compute RMS inverse
    __shared__ float rms_inv;
    if (tid == 0) {
        float mean_sq = sum_sq / static_cast<float>(hidden_size);
        rms_inv = rsqrtf(mean_sq + eps);
    }
    __syncthreads();

    const float factor = rms_inv;

    // Phase 2: Apply normalization with bf16x2 vectorized stores
    const __nv_bfloat162* vec_weight = reinterpret_cast<const __nv_bfloat162*>(weight);
    __nv_bfloat162* vec_output = reinterpret_cast<__nv_bfloat162*>(row_output);

    #pragma unroll 4
    for (int i = tid; i < vec_hidden; i += stride) {
        __nv_bfloat162 v_in = vec_input[i];
        __nv_bfloat162 v_w = vec_weight[i];

        float v0 = __bfloat162float(v_in.x);
        float v1 = __bfloat162float(v_in.y);
        float w0 = __bfloat162float(v_w.x);
        float w1 = __bfloat162float(v_w.y);

        __nv_bfloat162 result;
        result.x = __float2bfloat16(v0 * factor * w0);
        result.y = __float2bfloat16(v1 * factor * w1);
        vec_output[i] = result;
    }

    // Handle odd element
    if (hidden_size % 2 == 1 && tid == 0) {
        float v = __bfloat162float(row_input[hidden_size - 1]);
        float w = __bfloat162float(weight[hidden_size - 1]);
        row_output[hidden_size - 1] = __float2bfloat16(v * factor * w);
    }
}

// Launch configuration for vectorized kernel
void rmsnorm_forward_bf16(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    const int batch_size,
    const int seq_len,
    const int hidden_size,
    const float eps,
    cudaStream_t stream
) {
    const int num_rows = batch_size * seq_len;
    // Divide by 2 for vectorized access
    int threads = min(hidden_size / 2, MAX_THREADS);
    threads = max(threads, WARP_SIZE);
    threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

    size_t smem_size = ((threads + WARP_SIZE - 1) / WARP_SIZE) * sizeof(float);

    if (hidden_size % 2 == 0 && hidden_size >= 64) {
        rmsnorm_kernel_bf16_vectorized<<<num_rows, threads, smem_size, stream>>>(
            output, input, weight, hidden_size, eps
        );
    } else {
        // Fallback to scalar kernel for small/odd sizes
        rmsnorm_kernel<__nv_bfloat16><<<num_rows, threads, smem_size, stream>>>(
            output, input, weight, hidden_size, eps
        );
    }
}

// C++ entry points
extern "C" {

void your_reduction_forward_fp16(
    const __half* input,
    const __half* weight,
    __half* output,
    int batch_size,
    int hidden_size,
    float eps,
    cudaStream_t stream
) {
    int threads = min(hidden_size, MAX_THREADS);
    threads = (threads + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;

    your_reduction_kernel<__half><<<batch_size, threads, 0, stream>>>(
        input, weight, output, hidden_size, eps
    );
}

void your_reduction_forward_bf16(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    __nv_bfloat16* output,
    int batch_size,
    int hidden_size,
    float eps,
    cudaStream_t stream
) {
    int threads = min(hidden_size, MAX_THREADS);
    threads = (threads + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;

    your_reduction_kernel<__nv_bfloat16><<<batch_size, threads, 0, stream>>>(
        input, weight, output, hidden_size, eps
    );
}

void your_reduction_forward_fp32(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int hidden_size,
    float eps,
    cudaStream_t stream
) {
    int threads = min(hidden_size, MAX_THREADS);
    threads = (threads + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;

    your_reduction_kernel<float><<<batch_size, threads, 0, stream>>>(
        input, weight, output, hidden_size, eps
    );
}

}
```

## Template 3: Tiled Matrix Operation (Attention style)

Use for operations requiring shared memory tiling (matmul, attention).

```cuda
/*
 * Tiled matrix operation template for H100 (sm_90)
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

// Block sizes optimized for H100 L2 cache
constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 64;
constexpr int NUM_WARPS = 8;

template <typename T>
__device__ __forceinline__ T warp_reduce_max(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = max(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename scalar_t>
__global__ void your_tiled_kernel(
    const scalar_t* __restrict__ A,  // [batch, M, K]
    const scalar_t* __restrict__ B,  // [batch, K, N]
    scalar_t* __restrict__ C,        // [batch, M, N]
    const int batch_size,
    const int M,
    const int N,
    const int K
) {
    // Shared memory for tiles
    extern __shared__ char shared_mem[];
    scalar_t* tile_A = reinterpret_cast<scalar_t*>(shared_mem);
    scalar_t* tile_B = tile_A + BLOCK_M * BLOCK_K;

    const int batch_idx = blockIdx.z;
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;

    const int tid = threadIdx.x;

    // Base offsets for this batch
    const scalar_t* batch_A = A + batch_idx * M * K;
    const scalar_t* batch_B = B + batch_idx * K * N;
    scalar_t* batch_C = C + batch_idx * M * N;

    // Initialize accumulator
    float acc[BLOCK_M / (NUM_WARPS * 32)][BLOCK_N / 32] = {0};

    // Iterate over K dimension tiles
    for (int k_tile = 0; k_tile < (K + BLOCK_K - 1) / BLOCK_K; k_tile++) {
        // Cooperative loading of tiles to shared memory
        for (int i = tid; i < BLOCK_M * BLOCK_K; i += blockDim.x) {
            int row = i / BLOCK_K;
            int col = i % BLOCK_K;
            int global_row = block_row * BLOCK_M + row;
            int global_col = k_tile * BLOCK_K + col;

            if (global_row < M && global_col < K) {
                tile_A[i] = batch_A[global_row * K + global_col];
            } else {
                tile_A[i] = scalar_t(0);
            }
        }

        for (int i = tid; i < BLOCK_K * BLOCK_N; i += blockDim.x) {
            int row = i / BLOCK_N;
            int col = i % BLOCK_N;
            int global_row = k_tile * BLOCK_K + row;
            int global_col = block_col * BLOCK_N + col;

            if (global_row < K && global_col < N) {
                tile_B[i] = batch_B[global_row * N + global_col];
            } else {
                tile_B[i] = scalar_t(0);
            }
        }
        __syncthreads();

        // Compute partial results
        // (Simplified - real implementation would use register tiling)
        #pragma unroll
        for (int k = 0; k < BLOCK_K; k++) {
            // Your tiled computation here
        }
        __syncthreads();
    }

    // Write results
    // (Implementation depends on your specific needs)
}

// C++ entry points follow same pattern as above
```

## Tiled Matrix Multiplication

Computes C = A * B where A is MxK, B is KxN, C is MxN.
Each block loads a TILE_SIZE x TILE_SIZE sub-matrix into shared memory,
reducing global memory accesses from O(K) per element to O(K / TILE_SIZE).

```cuda
constexpr int TILE = 32;

__global__ void matmul_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE + 1]; // +1 padding avoids bank conflicts

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    int num_tiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        // Coalesced load from global to shared (row-major layout)
        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        // Compute partial dot product from the tile
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// Launch: one block per TILE x TILE output tile
void launch_matmul(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    matmul_tiled<<<grid, block>>>(A, B, C, M, N, K);
    CUDA_CHECK_KERNEL();
}
```

**Why it works:** Each element of C requires K multiply-adds. Without tiling,
every thread reads K floats from global memory. With TILE=32, each global
load is reused 32 times from shared memory, yielding a 32x reduction in
global memory traffic. The `+1` padding on `Bs` prevents shared memory
bank conflicts when threads in a warp access the same column.


## Parallel Reduction Tree

Reduces an array of N floats to a single sum. Uses a two-phase approach:
per-block reduction into partial sums, then a final reduction of partials.

```cuda
// Warp-level reduction using shuffle -- no shared memory, no __syncthreads
__device__ float warp_reduce(float val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Block-level reduction: warp shuffle + shared memory for cross-warp
__global__ void reduce_sum(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    __shared__ float warp_results[32];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Each thread loads two elements (reduces idle threads in first step)
    float sum = 0.0f;
    if (idx < n)               sum += input[idx];
    if (idx + blockDim.x < n)  sum += input[idx + blockDim.x];

    // Intra-warp reduction
    sum = warp_reduce(sum);

    int lane = tid % warpSize;
    int warp_id = tid / warpSize;

    if (lane == 0) {
        warp_results[warp_id] = sum;
    }
    __syncthreads();

    // First warp reduces the per-warp results
    int num_warps = blockDim.x / warpSize;
    sum = (tid < num_warps) ? warp_results[tid] : 0.0f;
    if (warp_id == 0) {
        sum = warp_reduce(sum);
    }

    // Block result
    if (tid == 0) {
        output[blockIdx.x] = sum;
    }
}

// Host-side: two-pass reduction
float reduce(const float* d_input, int n) {
    int block_size = 256;
    int grid_size = (n + block_size * 2 - 1) / (block_size * 2);

    float* d_partial;
    CUDA_CHECK(cudaMalloc(&d_partial, grid_size * sizeof(float)));

    // Pass 1: N elements -> grid_size partial sums
    reduce_sum<<<grid_size, block_size>>>(d_input, d_partial, n);
    CUDA_CHECK_KERNEL();

    // Pass 2: grid_size partial sums -> 1 final sum
    // For small grid_size, a single block suffices
    float* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
    reduce_sum<<<1, block_size>>>(d_partial, d_result, grid_size);
    CUDA_CHECK_KERNEL();

    float result;
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(float),
                           cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_partial));
    CUDA_CHECK(cudaFree(d_result));
    return result;
}
```

**Performance notes:**
- The "load two elements" trick doubles useful work per thread and
  eliminates the idle-thread waste in the first reduction step.
- `__shfl_down_sync` is faster than shared memory for intra-warp
  communication (no memory access, no bank conflicts).
- For very large arrays (>100M elements), add a grid-stride accumulation
  loop before the warp reduction to reduce grid_size further.

## Stream Overlap Pipeline

Overlaps host-to-device transfer, kernel execution, and device-to-host
transfer using multiple CUDA streams. Requires pinned host memory.

```cuda
// Pipeline: divide work into chunks, overlap transfer + compute
void stream_pipeline(
    const float* h_input,    // Pinned host memory (cudaMallocHost)
    float* h_output,         // Pinned host memory
    int total_elements
) {
    constexpr int NUM_STREAMS = 4;
    int chunk_size = (total_elements + NUM_STREAMS - 1) / NUM_STREAMS;
    size_t chunk_bytes = chunk_size * sizeof(float);

    // Create streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    // Allocate device memory for all chunks
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input,  total_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, total_elements * sizeof(float)));

    // Issue async operations per stream
    for (int i = 0; i < NUM_STREAMS; ++i) {
        int offset = i * chunk_size;
        int count = min(chunk_size, total_elements - offset);
        size_t bytes = count * sizeof(float);

        // Stage 1: Async copy host -> device
        CUDA_CHECK(cudaMemcpyAsync(
            d_input + offset, h_input + offset,
            bytes, cudaMemcpyHostToDevice, streams[i]));

        // Stage 2: Kernel execution
        int block_size = 256;
        int grid_size = (count + block_size - 1) / block_size;
        process_kernel<<<grid_size, block_size, 0, streams[i]>>>(
            d_input + offset, d_output + offset, count);

        // Stage 3: Async copy device -> host
        CUDA_CHECK(cudaMemcpyAsync(
            h_output + offset, d_output + offset,
            bytes, cudaMemcpyDeviceToHost, streams[i]));
    }

    // Wait for all streams to complete
    CUDA_CHECK(cudaDeviceSynchronize());

    // Cleanup
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

// Caller must use pinned memory for async transfers to actually overlap
void run_pipeline() {
    int n = 10000000;
    size_t bytes = n * sizeof(float);

    float *h_in, *h_out;
    CUDA_CHECK(cudaMallocHost(&h_in,  bytes));  // Pinned
    CUDA_CHECK(cudaMallocHost(&h_out, bytes));   // Pinned

    // Initialize input ...
    for (int i = 0; i < n; ++i) h_in[i] = (float)i;

    stream_pipeline(h_in, h_out, n);

    CUDA_CHECK(cudaFreeHost(h_in));
    CUDA_CHECK(cudaFreeHost(h_out));
}
```

**Timeline with 4 streams (ideal overlap):**

```
Stream 0: [H2D][Kernel][D2H]
Stream 1:      [H2D][Kernel][D2H]
Stream 2:           [H2D][Kernel][D2H]
Stream 3:                [H2D][Kernel][D2H]
```

**Key requirements for overlap:**
- Host memory MUST be pinned (`cudaMallocHost`); pageable memory forces synchronous transfers
- The GPU must have a copy engine separate from the compute engine (all modern GPUs do)
- Use `cudaMemcpyAsync` -- synchronous `cudaMemcpy` blocks the host thread
- Chunk sizes should be large enough to saturate PCIe bandwidth (~64 KB minimum, 1-4 MB recommended)
- Verify overlap with `nsys profile` -- look for concurrent copy and compute on the timeline
- 
