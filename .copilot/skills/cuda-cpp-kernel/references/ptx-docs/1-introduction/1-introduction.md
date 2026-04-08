# 1. Introduction


This document describes PTX, a low-level _parallel thread execution_ virtual machine and instruction set architecture (ISA). PTX exposes the GPU as a data-parallel computing _device_.


##  1.1. [Scalable Data-Parallel Computing using GPUs](#scalable-data-parallel-computing-using-gpus)

Driven by the insatiable market demand for real-time, high-definition 3D graphics, the programmable GPU has evolved into a highly parallel, multithreaded, many-core processor with tremendous computational horsepower and very high memory bandwidth. The GPU is especially well-suited to address problems that can be expressed as data-parallel computations - the same program is executed on many data elements in parallel - with high arithmetic intensity - the ratio of arithmetic operations to memory operations. Because the same program is executed for each data element, there is a lower requirement for sophisticated flow control; and because it is executed on many data elements and has high arithmetic intensity, the memory access latency can be hidden with calculations instead of big data caches.

Data-parallel processing maps data elements to parallel processing threads. Many applications that process large data sets can use a data-parallel programming model to speed up the computations. In 3D rendering large sets of pixels and vertices are mapped to parallel threads. Similarly, image and media processing applications such as post-processing of rendered images, video encoding and decoding, image scaling, stereo vision, and pattern recognition can map image blocks and pixels to parallel processing threads. In fact, many algorithms outside the field of image rendering and processing are accelerated by data-parallel processing, from general signal processing or physics simulation to computational finance or computational biology.

_PTX_ defines a virtual machine and ISA for general purpose parallel thread execution. PTX programs are translated at install time to the target hardware instruction set. The PTX-to-GPU translator and driver enable NVIDIA GPUs to be used as programmable parallel computers.


##  1.2. [Goals of PTX](#goals-of-ptx)

_PTX_ provides a stable programming model and instruction set for general purpose parallel programming. It is designed to be efficient on NVIDIA GPUs supporting the computation features defined by the NVIDIA Tesla architecture. High level language compilers for languages such as CUDA and C/C++ generate PTX instructions, which are optimized for and translated to native target-architecture instructions.

The goals for PTX include the following:

  * Provide a stable ISA that spans multiple GPU generations.

  * Achieve performance in compiled applications comparable to native GPU performance.

  * Provide a machine-independent ISA for C/C++ and other compilers to target.

  * Provide a code distribution ISA for application and middleware developers.

  * Provide a common source-level ISA for optimizing code generators and translators, which map PTX to specific target machines.

  * Facilitate hand-coding of libraries, performance kernels, and architecture tests.

  * Provide a scalable programming model that spans GPU sizes from a single unit to many parallel units.


##  1.3. [PTX ISA Version 9.2](#ptx-isa-version-9-2)

PTX ISA version 9.2 introduces the following new features:

  * Adds support for `.u8x4` and `.s8x4` instruction types for `add`, `sub`, `min`, `max`, `neg` instructions.

  * Adds support for `add.sat.{u16x2/s16x2/u32}` instruction.

  * Adds support for `.b128` type for `st.async` instruction.

  * Adds support for `.ignore_oob` qualifier for `cp.async.bulk` instruction.

  * Adds support for `.bf16x2` destination type for `cvt` instruction with `.e4m3x2`, `.e5m2x2`, `.e3m2x2`, `.e2m3x2`, `.e2m1x2` source types.


##  1.4. [Document Structure](#document-structure)

The information in this document is organized into the following Chapters:

  * [Programming Model](#programming-model) outlines the programming model.

  * [PTX Machine Model](#ptx-machine-model) gives an overview of the PTX virtual machine model.

  * [Syntax](#syntax) describes the basic syntax of the PTX language.

  * [State Spaces, Types, and Variables](#state-spaces-types-and-variables) describes state spaces, types, and variable declarations.

  * [Instruction Operands](#instruction-operands) describes instruction operands.

  * [Abstracting the ABI](#abstracting-abi) describes the function and call syntax, calling convention, and PTX support for abstracting the _Application Binary Interface (ABI)_.

  * [Instruction Set](#instruction-set) describes the instruction set.

  * [Special Registers](#special-registers) lists special registers.

  * [Directives](#directives) lists the assembly directives supported in PTX.

  * [Release Notes](#release-notes) provides release notes for PTX ISA versions 2.x and beyond.


References

  * 754-2008 IEEE Standard for Floating-Point Arithmetic. ISBN 978-0-7381-5752-8, 2008.

<http://ieeexplore.ieee.org/servlet/opac?punumber=4610933>

  * The OpenCL Specification, Version: 1.1, Document Revision: 44, June 1, 2011.

<http://www.khronos.org/registry/cl/specs/opencl-1.1.pdf>

  * CUDA Programming Guide.

<https://docs.nvidia.com/cuda/cuda-programming-guide/index.html>

  * CUDA Dynamic Parallelism Programming Guide.

<https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html>

  * CUDA Atomicity Requirements.

<https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#atomicity>

  * PTX Writers Guide to Interoperability.

<https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/index.html>
