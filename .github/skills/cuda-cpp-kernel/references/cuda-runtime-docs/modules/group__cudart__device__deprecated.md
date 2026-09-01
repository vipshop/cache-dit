# 6.2. Device Management [DEPRECATED]

**Source:** group__CUDART__DEVICE__DEPRECATED.html#group__CUDART__DEVICE__DEPRECATED


### Functions

__host__  __device__ cudaError_t cudaDeviceGetSharedMemConfig ( cudaSharedMemConfig ** pConfig )


Returns the shared memory configuration for the current device.

######  Parameters

`pConfig`
    \- Returned cache configuration

###### Returns

cudaSuccess, cudaErrorInvalidValue

###### Deprecated

###### Description

This function will return in `pConfig` the current size of shared memory banks on the current device. On devices with configurable shared memory banks, cudaDeviceSetSharedMemConfig can be used to change this setting, so that all subsequent kernel launches will by default use the new bank size. When cudaDeviceGetSharedMemConfig is called on devices without configurable shared memory, it will return the fixed bank size of the hardware.

The returned bank configurations can be either:

  * cudaSharedMemBankSizeFourByte - shared memory bank width is four bytes.

  * cudaSharedMemBankSizeEightByte - shared memory bank width is eight bytes.


  *

  * Note that this function may also return cudaErrorInitializationError, cudaErrorInsufficientDriver or cudaErrorNoDevice if this call tries to initialize internal CUDA RT state.

  * Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cudaErrorNotPermitted may, but is not guaranteed to, be returned as a diagnostic in such case.


######  Parameters

`config`
    \- Requested cache configuration

###### Returns

cudaSuccess, cudaErrorInvalidValue

###### Deprecated

###### Description

On devices with configurable shared memory banks, this function will set the shared memory bank size which is used for all subsequent kernel launches. Any per-function setting of shared memory set via cudaFuncSetSharedMemConfig will override the device wide setting.

Changing the shared memory configuration between launches may introduce a device side synchronization point.

Changing the shared memory bank size will not increase shared memory usage or affect occupancy of kernels, but may have major effects on performance. Larger bank sizes will allow for greater potential bandwidth to shared memory, but will change what kinds of accesses to shared memory will result in bank conflicts.

This function will do nothing on devices with fixed shared memory bank size.

The supported bank configurations are:

  * cudaSharedMemBankSizeDefault: set bank width the device default (currently, four bytes)

  * cudaSharedMemBankSizeFourByte: set shared memory bank width to be four bytes natively.

  * cudaSharedMemBankSizeEightByte: set shared memory bank width to be eight bytes natively.


  *

  * Note that this function may also return cudaErrorInitializationError, cudaErrorInsufficientDriver or cudaErrorNoDevice if this call tries to initialize internal CUDA RT state.

  * Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cudaErrorNotPermitted may, but is not guaranteed to, be returned as a diagnostic in such case.
