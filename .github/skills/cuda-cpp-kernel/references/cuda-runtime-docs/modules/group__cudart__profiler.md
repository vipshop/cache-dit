# 6.36. Profiler Control

**Source:** group__CUDART__PROFILER.html#group__CUDART__PROFILER


### Functions

__host__ cudaError_t cudaProfilerStart ( void )


Enable profiling.

###### Returns

cudaSuccess

###### Description

Enables profile collection by the active profiling tool for the current context. If profiling is already enabled, then cudaProfilerStart() has no effect.

cudaProfilerStart and cudaProfilerStop APIs are used to programmatically control the profiling granularity by allowing profiling to be done only on selective pieces of code.

###### Returns

cudaSuccess

###### Description

Disables profile collection by the active profiling tool for the current context. If profiling is already disabled, then cudaProfilerStop() has no effect.

cudaProfilerStart and cudaProfilerStop APIs are used to programmatically control the profiling granularity by allowing profiling to be done only on selective pieces of code.
