# 7.8. CUcheckpointRestoreArgs

**Source:** structCUcheckpointRestoreArgs.html#structCUcheckpointRestoreArgs


### Public Variables

CUcheckpointGpuPair * gpuPairs

unsigned int gpuPairsCount

char reserved[52-sizeof(CUcheckpointGpuPair *)]

cuuint64_t reserved1


### Variables

CUcheckpointGpuPair * CUcheckpointRestoreArgs::gpuPairs


Pointer to array of gpu pairs that indicate how to remap GPUs during restore

unsigned int CUcheckpointRestoreArgs::gpuPairsCount


Number of gpu pairs to remap

char CUcheckpointRestoreArgs::reserved[52-sizeof(CUcheckpointGpuPair *)]


Reserved for future use, must be zeroed

cuuint64_t CUcheckpointRestoreArgs::reserved1


Reserved for future use, must be zeroed

* * *
