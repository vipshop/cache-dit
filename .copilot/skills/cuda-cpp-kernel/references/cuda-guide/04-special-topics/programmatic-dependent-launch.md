---
url: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html
---

# 4.5. Programmatic Dependent Launch and Synchronization

The _Programmatic Dependent Launch_ mechanism allows for a dependent _secondary_ kernel to launch before the _primary_ kernel it depends on in the same CUDA stream has finished executing. Available starting with devices of compute capability 9.0, this technique can provide performance benefits when the _secondary_ kernel can complete significant work that does not depend on the results of the _primary_ kernel.

## 4.5.1. Background

A CUDA application utilizes the GPU by launching and executing multiple kernels on it. A typical GPU activity timeline is shown in [Figure 39](#gpu-activity).

[![GPU activity timeline](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/gpu-activity.png) ](../_images/gpu-activity.png)

Figure 39 GPU activity timeline

Here, `secondary_kernel` is launched after `primary_kernel` finishes its execution. Serialized execution is usually necessary because `secondary_kernel` depends on result data produced by `primary_kernel`. If `secondary_kernel` has no dependency on `primary_kernel`, both of them can be launched concurrently by using [CUDA Streams](../02-basics/asynchronous-execution.html#cuda-streams). Even if `secondary_kernel` is dependent on `primary_kernel`, there is some potential for concurrent execution. For example, almost all the kernels have some sort of _preamble_ section during which tasks such as zeroing buffers or loading constant values are performed.

[![Preamble section of ``secondary_kernel``](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/secondary-kernel-preamble.png) ](../_images/secondary-kernel-preamble.png)

Figure 40 Preamble section of `secondary_kernel`

[Figure 40](#secondary-kernel-preamble) demonstrates the portion of `secondary_kernel` that could be executed concurrently without impacting the application. Note that concurrent launch also allows us to hide the launch latency of `secondary_kernel` behind the execution of `primary_kernel`.

[![Concurrent execution of ``primary_kernel`` and ``secondary_kernel``](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/preamble-overlap.png) ](../_images/preamble-overlap.png)

Figure 41 Concurrent execution of `primary_kernel` and `secondary_kernel`

The concurrent launch and execution of `secondary_kernel` shown in [Figure 41](#preamble-overlap) is achievable using _Programmatic Dependent Launch_.

_Programmatic Dependent Launch_ introduces changes to the CUDA kernel launch APIs as explained in following section. These APIs require at least compute capability 9.0 to provide overlapping execution.

## 4.5.2. API Description

In Programmatic Dependent Launch, a primary and a secondary kernel are launched in the same CUDA stream. The primary kernel should execute `cudaTriggerProgrammaticLaunchCompletion` with all thread blocks when it’s ready for the secondary kernel to launch. The secondary kernel must be launched using the extensible launch API as shown.
    
    
    __global__ void primary_kernel() {
       // Initial work that should finish before starting secondary kernel
    
       // Trigger the secondary kernel
       cudaTriggerProgrammaticLaunchCompletion();
    
       // Work that can coincide with the secondary kernel
    }
    
    __global__ void secondary_kernel()
    {
       // Independent work
    
       // Will block until all primary kernels the secondary kernel is dependent on have completed and flushed results to global memory
       cudaGridDependencySynchronize();
    
       // Dependent work
    }
    
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = 1;
    configSecondary.attrs = attribute;
    configSecondary.numAttrs = 1;
    
    primary_kernel<<<grid_dim, block_dim, 0, stream>>>();
    cudaLaunchKernelEx(&configSecondary, secondary_kernel);
    

When the secondary kernel is launched using the `cudaLaunchAttributeProgrammaticStreamSerialization` attribute, the CUDA driver is safe to launch the secondary kernel early and not wait on the completion and memory flush of the primary before launching the secondary.

The CUDA driver can launch the secondary kernel when all primary thread blocks have launched and executed `cudaTriggerProgrammaticLaunchCompletion`. If the primary kernel doesn’t execute the trigger, it implicitly occurs after all thread blocks in the primary kernel exit.

In either case, the secondary thread blocks might launch before data written by the primary kernel is visible. As such, when the secondary kernel is configured with _Programmatic Dependent Launch_ , it must always use `cudaGridDependencySynchronize` or other means to verify that the result data from the primary is available.

Please note that these methods provide the opportunity for the primary and secondary kernels to execute concurrently, however this behavior is opportunistic and not guaranteed to lead to concurrent kernel execution. Reliance on concurrent execution in this manner is unsafe and can lead to deadlock.

## 4.5.3. Use in CUDA Graphs

Programmatic Dependent Launch can be used in [CUDA Graphs](cuda-graphs.html#cuda-graphs) via [stream capture](cuda-graphs.html#cuda-graphs-creating-a-graph-using-stream-capture) or directly via [edge data](cuda-graphs.html#cuda-graphs-edge-data). To program this feature in a CUDA Graph with edge data, use a `cudaGraphDependencyType` value of `cudaGraphDependencyTypeProgrammatic` on an edge connecting two kernel nodes. This edge type makes the upstream kernel visible to a `cudaGridDependencySynchronize()` in the downstream kernel. This type must be used with an outgoing port of either `cudaGraphKernelNodePortLaunchCompletion` or `cudaGraphKernelNodePortProgrammatic`.

The resulting graph equivalents for stream capture are as follows:

Stream code (abbreviated) | Resulting graph edge  
---|---  
      
    
    cudaLaunchAttribute attribute;
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;
    

| 
    
    
    cudaGraphEdgeData edgeData;
    edgeData.type = cudaGraphDependencyTypeProgrammatic;
    edgeData.from_port = cudaGraphKernelNodePortProgrammatic;
      
      
    
    cudaLaunchAttribute attribute;
    attribute.id = cudaLaunchAttributeProgrammaticEvent;
    attribute.val.programmaticEvent.triggerAtBlockStart = 0;
    

| 
    
    
    cudaGraphEdgeData edgeData;
    edgeData.type = cudaGraphDependencyTypeProgrammatic;
    edgeData.from_port = cudaGraphKernelNodePortProgrammatic;
      
      
    
    cudaLaunchAttribute attribute;
    attribute.id = cudaLaunchAttributeProgrammaticEvent;
    attribute.val.programmaticEvent.triggerAtBlockStart = 1;
    

| 
    
    
    cudaGraphEdgeData edgeData;
    edgeData.type = cudaGraphDependencyTypeProgrammatic;
    edgeData.from_port = cudaGraphKernelNodePortLaunchCompletion;
