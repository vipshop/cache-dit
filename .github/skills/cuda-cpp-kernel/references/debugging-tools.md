# CUDA Debugging Tools Reference

## Table of Contents

- [compute-sanitizer](#compute-sanitizer) — Memory checking and race detection (memcheck, racecheck, initcheck, synccheck)
- [cuda-gdb](#cuda-gdb) — CUDA-aware debugger batch mode
- [cuobjdump](#cuobjdump) — Binary inspection (PTX, SASS, resources)
- [Debugging Strategy](#debugging-strategy) — Workflow for systematic bug isolation

## compute-sanitizer

Memory checking and race detection for CUDA applications.

### Tools

```bash
compute-sanitizer --tool memcheck ./program    # Memory errors (default)
compute-sanitizer --tool racecheck ./program   # Race conditions
compute-sanitizer --tool initcheck ./program   # Uninitialized memory
compute-sanitizer --tool synccheck ./program   # Synchronization errors
```

### memcheck (Default)

Detects:
- Out-of-bounds global memory access
- Misaligned memory access
- Invalid memory access (freed, unallocated)
- Memory leaks

```bash
# Basic
compute-sanitizer ./program

# With backtraces (requires -lineinfo)
compute-sanitizer --show-backtrace yes ./program

# Log to file
compute-sanitizer --log-file errors.txt ./program

# Continue on error (collect all errors)
compute-sanitizer --error-exitcode 0 ./program

# Specific checks
compute-sanitizer --check-device-heap yes ./program  # Dynamic allocation
```

### racecheck

Detects data races in shared memory and global memory.

```bash
compute-sanitizer --tool racecheck ./program

# Hazard types reported:
# - WAW (Write-After-Write)
# - WAR (Write-After-Read)  
# - RAW (Read-After-Write)
```

**Note:** Significant overhead. Use on specific kernels when race is suspected.

### initcheck

Detects reads of uninitialized device memory.

```bash
compute-sanitizer --tool initcheck ./program

# Track through memory copies
compute-sanitizer --tool initcheck --track-unused-memory yes ./program
```

### synccheck

Detects invalid synchronization:
- `__syncthreads()` in divergent code
- Invalid `__syncwarp()` usage

```bash
compute-sanitizer --tool synccheck ./program
```

### Common Options

```bash
--show-backtrace yes          # Show source locations (needs -lineinfo)
--log-file FILE               # Output to file
--error-exitcode N            # Exit code on error (default 1)
--destroy-on-device-error all # Clean up on error
--print-limit N               # Max errors to print (default 100)
```

**Note:** Compile with `-g -G -lineinfo` for best debugging experience (see SKILL.md).

## cuda-gdb

CUDA-aware debugger. For non-interactive batch usage:

### Batch Mode Commands

```bash
# Basic: run and get backtrace on crash
cuda-gdb -batch -ex "run" -ex "bt" ./program

# With program arguments
cuda-gdb -batch -ex "run arg1 arg2" -ex "bt" ./program

# Multiple commands
cuda-gdb -batch \
  -ex "run" \
  -ex "bt" \
  -ex "info cuda threads" \
  ./program
```

### Common -ex Commands

```bash
# Execution
run [args]              # Start program
continue                # Continue after stop
next                    # Step over
step                    # Step into

# Breakpoints
break function_name     # Break at function
break file.cu:123       # Break at line
break myKernel          # Break at kernel entry

# Backtraces
bt                      # Host backtrace
info cuda threads       # CUDA thread states

# Examination
print variable          # Print value
print *array@10         # Print 10 elements
info locals             # Local variables

# CUDA specific
cuda thread             # Current CUDA thread
cuda block              # Current block coordinates
cuda kernel             # Current kernel info
```

### Batch Patterns

**Get backtrace on any crash:**
```bash
cuda-gdb -batch -ex "run" -ex "bt" -ex "info cuda threads" ./program
```

**Break at kernel, examine state:**
```bash
cuda-gdb -batch \
  -ex "break myKernel" \
  -ex "run" \
  -ex "info cuda threads" \
  -ex "cuda thread (0,0,0) (0,0,0)" \
  -ex "print idx" \
  -ex "continue" \
  ./program
```

**Run until error, collect info:**
```bash
cuda-gdb -batch \
  -ex "set cuda memcheck on" \
  -ex "run" \
  -ex "bt" \
  -ex "info cuda threads" \
  -ex "info locals" \
  ./program
```

### Commands File

For complex debugging, use a commands file:

```bash
# debug_commands.txt
set cuda memcheck on
break myKernel
run
info cuda threads
bt
quit
```

```bash
cuda-gdb -batch -x debug_commands.txt ./program
```

## cuobjdump

Inspect compiled CUDA binaries.

### PTX Dump

```bash
# Dump all PTX
cuobjdump -ptx ./program

# Specific kernel (grep)
cuobjdump -ptx ./program | grep -A 100 "myKernel"
```

### SASS Dump (GPU Assembly)

```bash
# All SASS
cuobjdump -sass ./program

# Specific architecture
cuobjdump -sass -arch sm_80 ./program
```

### Resource Usage

```bash
# Per-kernel resource summary
cuobjdump -res-usage ./program
```

Shows:
- Registers per thread
- Shared memory per block
- Constant memory usage
- Stack frame size

### Symbols

```bash
# List all symbols
cuobjdump -symbols ./program

# Find kernels
cuobjdump -symbols ./program | grep -i function
```

### ELF Information

```bash
# All ELF sections
cuobjdump -elf ./program

# Specific to GPU code
cuobjdump -elf -arch sm_80 ./program
```

### Common Patterns

**Check register usage for occupancy:**
```bash
cuobjdump -res-usage ./program 2>&1 | grep -A 5 "Function"
```

**Verify correct architecture:**
```bash
cuobjdump -lelf ./program  # List embedded architectures
```

**Extract PTX for inspection:**
```bash
cuobjdump -ptx ./program > kernels.ptx
```

## Debugging Strategy

### 1. Reproduce Minimally

Before any tool, create minimal reproduction:
- Smallest input that triggers bug
- Single kernel if possible
- Deterministic if possible

### 2. Use printf First

Add strategic printf statements to device code (see printf patterns in SKILL.md):
- Print at kernel entry to confirm launch
- Print intermediate values at suspected failure points
- Guard with `if (idx == 0)` or `if (idx < N)` to limit output

Printf works when all else fails. Don't underestimate it.

### 3. compute-sanitizer

```bash
compute-sanitizer --show-backtrace yes ./program
```

Catches most memory errors quickly.

**Interpreting compute-sanitizer output:**

- Thread IDs in errors reveal patterns: errors only for threads 64+ might indicate a warp-boundary issue
- "Invalid __shared__ write... out of bounds" often means **insufficient shared memory allocation**, not wrong indexing
- Multiple identical errors at same address = systematic bug; scattered errors = data-dependent bug

### 4. cuda-gdb (if needed)

When compute-sanitizer doesn't help:
```bash
cuda-gdb -batch -ex "run" -ex "bt" -ex "info cuda threads" ./program
```

### 5. Stare at the Diff

When tools fail (they often do with GPU code):
1. Find last known working state
2. Minimize changes between working and broken
3. Read the diff carefully
4. The bug is in the diff

This is not a fallback — it's often the fastest path.

### 6. Incremental Testing

For complex GPU operations, build up from verified simple cases:

1. **Test components in isolation** — Create minimal kernels that test one thing
2. **Verify each layer** — If building A → B → C pipeline, test A alone, then A→B, then A→B→C
3. **Use identity operations** — Test with identity matrices, trivial inputs where output is predictable
4. **Compare against reference** — Before optimizing, have a slow-but-correct reference implementation

```cuda
// Test pattern: use identity to verify pipeline
// If W=identity, C = W @ X should equal X
for (int i = 0; i < N; i++)
    W[i*N + i] = 1.0f;  // Identity matrix
// Run kernel, verify C == X
```

### 7. Inline Assembly and Hardware-Specific Instructions

Some GPU instructions (ldmatrix, tensor core ops) have non-intuitive register layouts:

- **Don't assume** that loading data via special instructions produces the same register layout as scalar loads
- **Verify experimentally** with printf or store-back-and-compare patterns
- **Check PTX documentation** for exact semantics of special instructions
- **Common approach**: Load with special instruction, store results to global memory, inspect on CPU
