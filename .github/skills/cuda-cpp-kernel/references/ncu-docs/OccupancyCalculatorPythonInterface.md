---
url: https://docs.nvidia.com/nsight-compute/OccupancyCalculatorPythonInterface/index.html
---

[ ![Logo](https://docs.nvidia.com/nsight-compute/_static/nsight-compute.png) ](../index.html)

Nsight Compute

  * [1\. Release Notes](../ReleaseNotes/index.html)
  * [2\. Profiling Guide](../ProfilingGuide/index.html)
  * [3\. Nsight Compute](../NsightCompute/index.html)
  * [4\. Nsight Compute CLI](../NsightComputeCli/index.html)


Developer Interfaces

  * [1\. Customization Guide](../CustomizationGuide/index.html)
  * [2\. Python Report Interface](../PythonReportInterface/index.html)
  * [3\. NvRules API](../NvRulesAPI/index.html)
  * [4\. Occupancy Calculator Python Interface](#)
    * [4.1. Introduction](#introduction)
    * [4.2. API Reference](#api-reference)


Training

  * [Training](../Training/index.html)


Release Information

  * [Archives](../Archives/index.html)


Copyright and Licenses

  * [Copyright and Licenses](../CopyrightAndLicenses/index.html)


__[NsightCompute](../index.html)

  * [](../index.html) »
  * 4\. Occupancy Calculator Python Interface
  *   * v2026.1.0 | [Archive](https://developer.nvidia.com/nsight-compute-history)


* * *

# 4\. Occupancy Calculator Python Interface

## 4.1. Introduction

NVIDIA Nsight Compute features a Python-based interface for performing occupancy calculations and analysis for kernels on NVIDIA GPUs. These APIs are designed to help developers understand and optimize the utilization of GPU resources to achieve better performance for their kernel.

The module is called [`ncu_occupancy`](#module-ncu_occupancy "ncu_occupancy") and works on any Python version from 3.7 [1](#fn1). It can be found in the `extras/python` directory of your NVIDIA Nsight Compute package.

[1](#id1)
    

On Linux machines you will also need a GNU-compatible libc and `libgcc_s.so`.

## 4.2. API Reference

This documents the content of the [`ncu_occupancy`](#module-ncu_occupancy "ncu_occupancy") package which can be found in the `extras/python` directory of your NVIDIA Nsight Compute installation.

_class _ncu_occupancy.OccupancyCalculator
    

Provide methods to calculate occupancy and analyze ways to improve it, for a given GPU.

Parameters
    

  * **computeCapabilityMajor** ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) – The major compute capability version of the GPU.

  * **computeCapabilityMinor** ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) – The minor compute capability version of the GPU.


Returns
    

An instance of the occupancy calculator.

Return type
    

[`OccupancyCalculator`](#ncu_occupancy.OccupancyCalculator "ncu_occupancy.OccupancyCalculator")

get_occupancy_limiters(_occupancy_parameters : [ncu_occupancy.OccupancyParameters](#ncu_occupancy.OccupancyParameters "ncu_occupancy.OccupancyParameters")_) → [list](https://docs.python.org/3/library/stdtypes.html#list "\(in Python v3.14\)")
    

Get the occupancy limiters for the given occupancy parameters.

Parameters
    

**occupancy_parameters** ([`OccupancyParameters`](#ncu_occupancy.OccupancyParameters "ncu_occupancy.OccupancyParameters")) – The input parameters for the occupancy calculation.

Returns
    

[`list`](https://docs.python.org/3/library/stdtypes.html#list "\(in Python v3.14\)") of [`OccupancyLimiter`](#ncu_occupancy.OccupancyLimiter "ncu_occupancy.OccupancyLimiter")

get_optimal_occupancy(_occupancy_parameters: ncu_occupancy.OccupancyParameters, occupancy_variable_list: list = [<OccupancyVariable.THREADS_PER_BLOCK: 0>]_) → [dict](https://docs.python.org/3/library/stdtypes.html#dict "\(in Python v3.14\)")
    

Get the optimal occupancy configuration.

Optimal occupancy is calculated by varying input occupancy variable values while keeping other occupancy variable values constant. If no input occupancy variable list provided then [`OccupancyVariable.THREADS_PER_BLOCK`](#ncu_occupancy.OccupancyVariable.THREADS_PER_BLOCK "ncu_occupancy.OccupancyVariable.THREADS_PER_BLOCK") will be considered by default.

Parameters
    

  * **occupancy_parameters** ([`OccupancyParameters`](#ncu_occupancy.OccupancyParameters "ncu_occupancy.OccupancyParameters")) – The input parameters for the occupancy calculation.

  * **occupancy_variable_list** ([`list`](https://docs.python.org/3/library/stdtypes.html#list "\(in Python v3.14\)") of [`OccupancyVariable`](#ncu_occupancy.OccupancyVariable "ncu_occupancy.OccupancyVariable"), optional) – The list of occupancy variables to consider for optimal occupancy calculation. Only up to two occupancy variables can be specified. (default: [`OccupancyVariable.THREADS_PER_BLOCK`](#ncu_occupancy.OccupancyVariable.THREADS_PER_BLOCK "ncu_occupancy.OccupancyVariable.THREADS_PER_BLOCK"))


Returns
    

The optimal occupancy configuration. The dictionary contains the following key-value pairs:
    

  * ’optimal_occupancy’: ([`float`](https://docs.python.org/3/library/functions.html#float "\(in Python v3.14\)")) The optimal occupancy.

  * ’occupancy_variable_config’:
    
    * For single occupancy variable input: [`list`](https://docs.python.org/3/library/stdtypes.html#list "\(in Python v3.14\)") of [`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.14\)") of ranges.

    * For two occupancy variables input: [`list`](https://docs.python.org/3/library/stdtypes.html#list "\(in Python v3.14\)") of [`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.14\)") of value combinations. The [`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.14\)") contains the values in the same order as the input occupancy variable list.


Return type
    

[`dict`](https://docs.python.org/3/library/stdtypes.html#dict "\(in Python v3.14\)")

get_resource_utilization(_occupancy_parameters : [ncu_occupancy.OccupancyParameters](#ncu_occupancy.OccupancyParameters "ncu_occupancy.OccupancyParameters")_) → [dict](https://docs.python.org/3/library/stdtypes.html#dict "\(in Python v3.14\)")
    

Get the resource utilization for the given occupancy parameters.

Parameters
    

**occupancy_parameters** ([`OccupancyParameters`](#ncu_occupancy.OccupancyParameters "ncu_occupancy.OccupancyParameters")) – The input parameters for the occupancy calculation.

Returns
    

Resource utilization. The dictionary contains the following key-value pairs:
    

  * ’sm_occupancy’ : ([`float`](https://docs.python.org/3/library/functions.html#float "\(in Python v3.14\)")) The occupancy of the SMs.

  * ’allocated_blocks’ : ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) The number of allocated blocks out of the total possible blocks per SM.

  * ’resource_utilization’([`dict`](https://docs.python.org/3/library/stdtypes.html#dict "\(in Python v3.14\)")) The resource utilization for each resource i.e. threads, registers, shared memory. The resource utilization dictionary contains the following key-value pairs:
    
    * ’<resource name>’: ([`dict`](https://docs.python.org/3/library/stdtypes.html#dict "\(in Python v3.14\)")) The resource utilization for the resource. The resource utilization dictionary contains the following key-value pairs:
    
      * ’resource_per_block’: ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) The resource utilized per block.

      * ’unused_resource_count’: ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) The unused resource count per SM.

      * ’unallocated_blocks’: ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) The number of unallocated blocks per SM.


Return type
    

[`dict`](https://docs.python.org/3/library/stdtypes.html#dict "\(in Python v3.14\)")

get_sm_occupancy(_occupancy_parameters : [ncu_occupancy.OccupancyParameters](#ncu_occupancy.OccupancyParameters "ncu_occupancy.OccupancyParameters")_) → [float](https://docs.python.org/3/library/functions.html#float "\(in Python v3.14\)")
    

Calculate the occupancy of the SMs for the given occupancy parameters.

Parameters
    

**occupancy_parameters** ([`OccupancyParameters`](#ncu_occupancy.OccupancyParameters "ncu_occupancy.OccupancyParameters")) – The input parameters for the occupancy calculation.

Returns
    

The occupancy of the SMs.

Return type
    

[`float`](https://docs.python.org/3/library/functions.html#float "\(in Python v3.14\)")

_class _ncu_occupancy.OccupancyLimiter
    

Enum representing the occupancy limiters.

REGISTERS
    

Register usage is the occupancy limiter.

SHARED_MEMORY
    

Shared memory usage is the occupancy limiter.

BLOCKS
    

Block size is the occupancy limiter.

BARRIERS
    

Barrier usage is the occupancy limiter.

_class _ncu_occupancy.OccupancyParameters
    

OccupancyParameters is a `dataclass` that holds configuration parameters for occupancy calculations.

shared_mem_size
    

Shared memory size configuration (bytes). (default: 0)

Type
    

[int](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

threads_per_block
    

Number of threads per block. (default: 256)

Type
    

[int](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

registers_per_thread
    

Number of registers per thread. (default: 32)

Type
    

[int](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

shared_mem_per_block
    

Shared memory (bytes) per block. (default: 2048)

Type
    

[int](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

num_block_barriers
    

Number of block barriers. (default: 1)

Type
    

[int](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

_class _ncu_occupancy.OccupancyVariable
    

Enum representing the occupancy variables.

THREADS_PER_BLOCK
    

Threads per block.

REGISTERS_PER_THREAD
    

Registers per thread.

SHARED_MEMORY_PER_BLOCK
    

Shared memory per block.

BLOCK_BARRIERS
    

Block barriers.

ncu_occupancy.get_gpu_data(_major : [int](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")_, _minor : [int](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")_) → [dict](https://docs.python.org/3/library/stdtypes.html#dict "\(in Python v3.14\)")
    

Get the GPU data for the given compute capability version.

Parameters
    

  * **major** ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) – The major compute capability version of the GPU.

  * **minor** ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) – The minor compute capability version of the GPU.


Returns
    

The GPU data for the given compute capability version. The dictionary contains the following key-value pairs:
    

  * ’cc_major’: ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) The major compute capability version of the GPU.

  * ’cc_minor’: ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) The minor compute capability version of the GPU.

  * ’sm_version’: ([`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)")) The SM version of the GPU.

  * ’chip_family’: ([`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)")) The chip family of the GPU.

  * ’threads_per_warp’: ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) The number of threads per warp.

  * ’max_warps_per_sm’: ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) The number of warps per SM.

  * ’max_threads_per_sm’: ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) The number of threads per SM.

  * ’max_thread_blocks_per_sm’: ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) The number of thread blocks per SM.

  * ’block_barriers_per_sm’: ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) The number of block barriers per SM.

  * ’smem_per_sm’: ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) The shared memory (bytes) per SM.

  * ’max_shared_mem_per_block’: ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) The maximum shared memory (bytes) per block.

  * ’registers_per_sm’: ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) The registers per SM.

  * ’max_regs_per_block’: ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) The maximum registers per block.

  * ’max_regs_per_thread’: ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) The maximum registers per thread.

  * ’reg_allocation_unit_size’: ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) The register allocation unit size.

  * ’reg_allocation_granularity’: ([`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)")) The register allocation granularity.

  * ’shared_mem_allocation_unit_size’: ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) The shared memory allocation unit size.

  * ’warps_allocation_granularity’: ([`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)")) The warp allocation granularity.

  * ’max_thread_block_size’: ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) The maximum thread block size.

  * ’shared_mem_size_configs’: ([`list`](https://docs.python.org/3/library/stdtypes.html#list "\(in Python v3.14\)") of [`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) The shared memory size configurations (bytes).

  * ’warp_reg_allocation_granularities’: ([`list`](https://docs.python.org/3/library/stdtypes.html#list "\(in Python v3.14\)") of [`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) The warp register allocation granularities.


Return type
    

[`dict`](https://docs.python.org/3/library/stdtypes.html#dict "\(in Python v3.14\)")

* * *
