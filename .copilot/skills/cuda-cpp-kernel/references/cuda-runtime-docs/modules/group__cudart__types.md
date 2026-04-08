# Data types used by CUDA Runtime

**Source:** group__CUDART__TYPES.html


### Classes

struct

CUuuid_st


struct

cudaAccessPolicyWindow


struct

cudaArrayMemoryRequirements


struct

cudaArraySparseProperties


struct

cudaAsyncNotificationInfo_t


struct

cudaChannelFormatDesc


struct

cudaChildGraphNodeParams


struct

cudaConditionalNodeParams


struct

cudaDevResource


struct

cudaDevSmResource


struct

cudaDevSmResourceGroupParams


struct

cudaDevWorkqueueConfigResource


struct

cudaDevWorkqueueResource


struct

cudaDeviceProp


struct

cudaEglFrame


struct

cudaEglPlaneDesc


struct

cudaEventRecordNodeParams


struct

cudaEventWaitNodeParams


struct

cudaExtent


struct

cudaExternalMemoryBufferDesc


struct

cudaExternalMemoryHandleDesc


struct

cudaExternalMemoryMipmappedArrayDesc


struct

cudaExternalSemaphoreHandleDesc


struct

cudaExternalSemaphoreSignalNodeParams


struct

cudaExternalSemaphoreSignalNodeParamsV2


struct

cudaExternalSemaphoreSignalParams


struct

cudaExternalSemaphoreWaitNodeParams


struct

cudaExternalSemaphoreWaitNodeParamsV2


struct

cudaExternalSemaphoreWaitParams


struct

cudaFuncAttributes


struct

cudaGraphEdgeData


struct

cudaGraphExecUpdateResultInfo


struct

cudaGraphInstantiateParams


struct

cudaGraphKernelNodeUpdate


struct

cudaGraphNodeParams


struct

cudaHostNodeParams


struct

cudaHostNodeParamsV2


struct

cudaIpcEventHandle_t


struct

cudaIpcMemHandle_t


struct

cudaKernelNodeParams


struct

cudaKernelNodeParamsV2


struct

cudaLaunchAttribute


union

cudaLaunchAttributeValue


struct

cudaLaunchConfig_t


struct

cudaLaunchMemSyncDomainMap


struct

cudaMemAccessDesc


struct

cudaMemAllocNodeParams


struct

cudaMemAllocNodeParamsV2


struct

cudaMemFreeNodeParams


struct

cudaMemLocation


struct

cudaMemPoolProps


struct

cudaMemPoolPtrExportData


struct

cudaMemcpy3DOperand


struct

cudaMemcpy3DParms


struct

cudaMemcpy3DPeerParms


struct

cudaMemcpyAttributes


struct

cudaMemcpyNodeParams


struct

cudaMemsetParams


struct

cudaMemsetParamsV2


struct

cudaOffset3D


struct

cudaPitchedPtr


struct

cudaPointerAttributes


struct

cudaPos


struct

cudaResourceDesc


struct

cudaResourceViewDesc


struct

cudaTextureDesc



### Defines

#define CUDA_EGL_MAX_PLANES 3

#define CUDA_IPC_HANDLE_SIZE 64

#define cudaArrayColorAttachment 0x20

#define cudaArrayCubemap 0x04

#define cudaArrayDefault 0x00

#define cudaArrayDeferredMapping 0x80

#define cudaArrayLayered 0x01

#define cudaArraySparse 0x40

#define cudaArraySparsePropertiesSingleMipTail 0x1

#define cudaArraySurfaceLoadStore 0x02

#define cudaArrayTextureGather 0x08

#define cudaCpuDeviceId ((int)-1)

#define cudaDeviceBlockingSync 0x04

#define cudaDeviceLmemResizeToMax 0x10

#define cudaDeviceMapHost 0x08

#define cudaDeviceMask 0xff

#define cudaDeviceScheduleAuto 0x00

#define cudaDeviceScheduleBlockingSync 0x04

#define cudaDeviceScheduleMask 0x07

#define cudaDeviceScheduleSpin 0x01

#define cudaDeviceScheduleYield 0x02

#define cudaDeviceSyncMemops 0x80

#define cudaEventBlockingSync 0x01

#define cudaEventDefault 0x00

#define cudaEventDisableTiming 0x02

#define cudaEventInterprocess 0x04

#define cudaEventRecordDefault 0x00

#define cudaEventRecordExternal 0x01

#define cudaEventWaitDefault 0x00

#define cudaEventWaitExternal 0x01

#define cudaExternalMemoryDedicated 0x1

#define cudaExternalSemaphoreSignalSkipNvSciBufMemSync 0x01

#define cudaExternalSemaphoreWaitSkipNvSciBufMemSync 0x02

#define cudaGraphKernelNodePortDefault 0

#define cudaGraphKernelNodePortLaunchCompletion 2

#define cudaGraphKernelNodePortProgrammatic 1

#define cudaHostAllocDefault 0x00

#define cudaHostAllocMapped 0x02

#define cudaHostAllocPortable 0x01

#define cudaHostAllocWriteCombined 0x04

#define cudaHostRegisterDefault 0x00

#define cudaHostRegisterIoMemory 0x04

#define cudaHostRegisterMapped 0x02

#define cudaHostRegisterPortable 0x01

#define cudaHostRegisterReadOnly 0x08

#define cudaInitDeviceFlagsAreValid 0x01

#define cudaInvalidDeviceId ((int)-2)

#define cudaIpcMemLazyEnablePeerAccess 0x01

#define cudaMemAttachGlobal 0x01

#define cudaMemAttachHost 0x02

#define cudaMemAttachSingle 0x04

#define cudaMemPoolCreateUsageHwDecompress 0x2

#define cudaNvSciSyncAttrSignal 0x1

#define cudaNvSciSyncAttrWait 0x2

#define cudaOccupancyDefault 0x00

#define cudaOccupancyDisableCachingOverride 0x01

#define cudaPeerAccessDefault 0x00

#define cudaStreamDefault 0x00

#define cudaStreamLegacy ((cudaStream_t)0x1)

#define cudaStreamNonBlocking 0x01

#define cudaStreamPerThread ((cudaStream_t)0x2)


### Typedefs

typedef cudaArray * cudaArray_const_t

typedef cudaArray * cudaArray_t

typedef cudaAsyncCallbackEntry * cudaAsyncCallbackHandle_t

typedef CUdevResourceDesc_st * cudaDevResourceDesc_t

typedef CUeglStreamConnection_st * cudaEglStreamConnection
