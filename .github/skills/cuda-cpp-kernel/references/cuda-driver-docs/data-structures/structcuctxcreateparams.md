# 7.11. CUctxCreateParams

**Source:** structCUctxCreateParams.html#structCUctxCreateParams


### Public Variables

CUctxCigParam * cigParams

CUexecAffinityParam * execAffinityParams

int numExecAffinityParams


### Variables

CUctxCigParam * CUctxCreateParams::cigParams


CIG (CUDA in Graphics) parameters for sharing data from D3D12/Vulkan graphics clients. Mutually exclusive with execAffinityParams.

CUexecAffinityParam * CUctxCreateParams::execAffinityParams


Array of execution affinity parameters to limit context resources (e.g., SM count). Only supported Volta+ MPS. Mutually exclusive with cigParams.

int CUctxCreateParams::numExecAffinityParams


Number of elements in execAffinityParams array. Must be 0 if execAffinityParams is NULL.

* * *
