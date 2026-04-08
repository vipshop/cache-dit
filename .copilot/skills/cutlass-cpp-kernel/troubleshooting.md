# Troubleshooting Guide
Common issues and solutions when working with H100 CUDA kernels

## Memory Error Detection (Memcheck)
Detect memory access errors and leaks:

```bash
# Basic memory check
compute-sanitizer --tool memcheck ./cuda_program

# With detailed error reporting
compute-sanitizer --tool memcheck --report-api-errors all ./cuda_program

# Log errors to file
compute-sanitizer --tool memcheck --log-file memcheck.log ./cuda_program

# Check for memory leaks
compute-sanitizer --tool memcheck --leak-check full ./cuda_program

# Track allocations
compute-sanitizer --tool memcheck --track-alloc-dealloc yes ./cuda_program
```

Common memory errors detected:

- Out-of-bounds global memory access
- Misaligned memory access
- Invalid global memory access
- Memory leaks (device allocations not freed)
- Double free errors
- Invalid device pointer operations

## Race Condition Detection (Racecheck)
Detect shared memory data access hazards:

```bash
# Basic race check
compute-sanitizer --tool racecheck ./cuda_program

# With detailed analysis
compute-sanitizer --tool racecheck --racecheck-report all ./cuda_program

# Save analysis to file
compute-sanitizer --tool racecheck --save racecheck.nvsanreport ./cuda_program

# Analyze previous run
compute-sanitizer --tool racecheck --import racecheck.nvsanreport --print-analysis ./cuda_program
```
Race condition types detected:

- Write-after-read (WAR) hazards
- Write-after-write (WAW) hazards
- Read-after-write (RAW) hazards
- Bank conflicts in shared memory
- Synchronization-related races

## Uninitialized Memory Detection (Initcheck)
Detect uninitialized global memory access:

```bash
# Basic initcheck
compute-sanitizer --tool initcheck ./cuda_program

# Track all memory accesses
compute-sanitizer --tool initcheck --track-unused-memory yes ./cuda_program

# With error details
compute-sanitizer --tool initcheck --show-backtrace yes ./cuda_program
```

## Synchronization Validation (Synccheck)
Detect illegal synchronization in CUDA code:

```bash
# Basic synccheck
compute-sanitizer --tool synccheck ./cuda_program

# With detailed reporting
compute-sanitizer --tool synccheck --show-backtrace all ./cuda_program
```
Synchronization issues detected:

- Divergent __syncthreads() calls
- Invalid thread block synchronization
- Illegal cooperative groups usage
- Missing synchronization barriers

## CUDA-GDB Debugging Commands
Interactive debugging with CUDA-GDB:

```bash
# Launch CUDA-GDB
cuda-gdb ./cuda_program

# Common debugging commands
(cuda-gdb) set cuda memcheck on        # Enable memory checking
(cuda-gdb) set cuda break_on_launch    # Break at kernel launch
(cuda-gdb) break kernel_name           # Set breakpoint at kernel
(cuda-gdb) run                         # Start execution

# Thread navigation
(cuda-gdb) info cuda threads           # List all GPU threads
(cuda-gdb) cuda thread (0,0,0) (0,0,0) # Switch to specific thread
(cuda-gdb) cuda block                  # Show current block
(cuda-gdb) cuda kernel                 # Show current kernel

# Memory inspection
(cuda-gdb) print *d_array@10           # Print device array
(cuda-gdb) print __shared_memory__     # Inspect shared memory
(cuda-gdb) info cuda devices           # List CUDA devices

# Stepping through code
(cuda-gdb) cuda step                   # Step one warp instruction
(cuda-gdb) cuda next                   # Step over function calls
(cuda-gdb) continue                    # Continue execution
```

## Common Debugging Patterns
### Pattern 1: Memory Bounds Checking

```cpp
// Add bounds checking to kernel
__global__ void safeKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check
    if (idx >= n) return;

    // Safe access
    data[idx] = data[idx] * 2.0f;
}
```

### Pattern 2: Shared Memory Synchronization
```cpp
__global__ void reductionKernel(float* input, float* output, int n) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load to shared memory
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();  // Required before reading shared memory

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();  // Required after each reduction step
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
```

### Pattern 3: Atomic Operation Validation
```cpp
// Validate atomic operations
__global__ void atomicTest(int* counter, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Use atomicAdd for thread-safe increment
        atomicAdd(counter, 1);
    }
}

// Verify result on host
int h_counter;
cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
assert(h_counter == n);  // Should equal number of threads
```

##  Debugging Report Generation
Generate comprehensive debugging reports:

```bash
# Full debugging session
compute-sanitizer --tool memcheck \
    --report-api-errors all \
    --show-backtrace yes \
    --log-file debug_report.txt \
    ./cuda_program 2>&1 | tee debug_output.log

# Summary report generation
echo "=== CUDA Debugging Report ===" > debug_summary.md
echo "Date: $(date)" >> debug_summary.md
echo "" >> debug_summary.md
echo "## Memory Check Results" >> debug_summary.md
compute-sanitizer --tool memcheck ./cuda_program 2>&1 >> debug_summary.md
echo "" >> debug_summary.md
echo "## Race Check Results" >> debug_summary.md
compute-sanitizer --tool racecheck ./cuda_program 2>&1 >> debug_summary.md
```

## Debugging Build Configuration
```bash
# Debug build flags
DEBUG_FLAGS = -G -lineinfo -Xcompiler -rdynamic -O0

# Release build with symbols
RELEASE_FLAGS = -O3 -lineinfo

# Compile for debugging
nvcc $(DEBUG_FLAGS) -o program_debug program.cu

# Compile for profiling (with symbols)
nvcc $(RELEASE_FLAGS) -o program_release program.cu
```

## Performance Issues

### Bank Conflicts in Shared Memory
**Problem:** Poor performance due to shared memory bank conflicts.

**Solution:** Add padding for 32-bank conflict avoidance:
```cuda
__shared__ float data[32][33];  // 33 instead of 32
```

### Poor Occupancy
**Problem:** Low SM utilization.

**Solution:** Check register usage:
```bash
nvcc --ptxas-options=-v your_kernel.cu
```

### Memory Coalescing
**Problem:** Poor memory bandwidth utilization.

**Solution:** Ensure 128-byte aligned accesses for optimal bandwidth.
