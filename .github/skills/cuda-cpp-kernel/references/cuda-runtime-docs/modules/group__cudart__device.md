# Next >

**Source:** group__CUDART__DEVICE.html


### Functions

__host__ cudaError_t cudaChooseDevice ( int* device, const cudaDeviceProp* prop )


Select compute-device which best matches criteria.

######  Parameters

`device`
    \- Device with best match
`prop`
    \- Desired device properties

###### Returns

cudaSuccess, cudaErrorInvalidValue

###### Description

Returns in `*device` the device which has properties that best match `*prop`.

  *

  * Note that this function may also return cudaErrorInitializationError, cudaErrorInsufficientDriver or cudaErrorNoDevice if this call tries to initialize internal CUDA RT state.

  * Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cudaErrorNotPermitted may, but is not guaranteed to, be returned as a diagnostic in such case.


######  Parameters

`target`
    \- The target of the operation, see cudaFlushGPUDirectRDMAWritesTarget
`scope`
    \- The scope of the operation, see cudaFlushGPUDirectRDMAWritesScope

###### Returns

cudaSuccess, cudaErrorNotSupported

###### Description

Blocks until remote writes to the target context via mappings created through GPUDirect RDMA APIs, like nvidia_p2p_get_pages (see <https://docs.nvidia.com/cuda/gpudirect-rdma> for more information), are visible to the specified scope.

If the scope equals or lies within the scope indicated by cudaDevAttrGPUDirectRDMAWritesOrdering, the call will be a no-op and can be safely omitted for performance. This can be determined by comparing the numerical values between the two enums, with smaller scopes having smaller values.

Users may query support for this API via cudaDevAttrGPUDirectRDMAFlushWritesOptions.

  *

  * Note that this function may also return cudaErrorInitializationError, cudaErrorInsufficientDriver or cudaErrorNoDevice if this call tries to initialize internal CUDA RT state.

  * Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cudaErrorNotPermitted may, but is not guaranteed to, be returned as a diagnostic in such case.


######  Parameters

`value`
    \- Returned device attribute value
`attr`
    \- Device attribute to query
`device`
    \- Device number to query

###### Returns

cudaSuccess, cudaErrorInvalidDevice, cudaErrorInvalidValue

###### Description

Returns in `*value` the integer value of the attribute `attr` on device `device`.

  *

  * Note that this function may also return cudaErrorInitializationError, cudaErrorInsufficientDriver or cudaErrorNoDevice if this call tries to initialize internal CUDA RT state.

  * Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cudaErrorNotPermitted may, but is not guaranteed to, be returned as a diagnostic in such case.


######  Parameters

`device`
    \- Returned device ordinal
`pciBusId`
    \- String in one of the following forms: [domain]:[bus]:[device].[function] [domain]:[bus]:[device] [bus]:[device].[function] where `domain`, `bus`, `device`, and `function` are all hexadecimal values

###### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevice

###### Description

Returns in `*device` a device ordinal given a PCI bus ID string.

  *

  * Note that this function may also return cudaErrorInitializationError, cudaErrorInsufficientDriver or cudaErrorNoDevice if this call tries to initialize internal CUDA RT state.

  * Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cudaErrorNotPermitted may, but is not guaranteed to, be returned as a diagnostic in such case.


######  Parameters

`pCacheConfig`
    \- Returned cache configuration

###### Returns

cudaSuccess

###### Description

On devices where the L1 cache and shared memory use the same hardware resources, this returns through `pCacheConfig` the preferred cache configuration for the current device. This is only a preference. The runtime will use the requested configuration if possible, but it is free to choose a different configuration if required to execute functions.

This will return a `pCacheConfig` of cudaFuncCachePreferNone on devices where the size of the L1 cache and shared memory are fixed.

The supported cache configurations are:

  * cudaFuncCachePreferNone: no preference for shared memory or L1 (default)

  * cudaFuncCachePreferShared: prefer larger shared memory and smaller L1 cache

  * cudaFuncCachePreferL1: prefer larger L1 cache and smaller shared memory

  * cudaFuncCachePreferEqual: prefer equal size L1 cache and shared memory


  *

  * Note that this function may also return cudaErrorInitializationError, cudaErrorInsufficientDriver or cudaErrorNoDevice if this call tries to initialize internal CUDA RT state.

  * Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cudaErrorNotPermitted may, but is not guaranteed to, be returned as a diagnostic in such case.


###### Returns

cudaSuccess, cudaErrorInvalidDevice, cudaErrorInvalidValuecudaErrorNotSupported

###### Description

The default mempool of a device contains device memory from that device.

  *

  * Note that this function may also return cudaErrorInitializationError, cudaErrorInsufficientDriver or cudaErrorNoDevice if this call tries to initialize internal CUDA RT state.

  * Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cudaErrorNotPermitted may, but is not guaranteed to, be returned as a diagnostic in such case.


######  Parameters

`capabilities`
    \- Returned capability details of each requested operation
`operations`
    \- Requested operations
`count`
    \- Count of requested operations and size of capabilities
`device`


###### Returns

cudaSuccess, cudaErrorInvalidDevice, cudaErrorInvalidValue

###### Description

Returns in `*capabilities` the details about requested atomic `*operations` over the the link between `dev` and the host. The allocated size of `*operations` and `*capabilities` must be `count`.

For each cudaAtomicOperation in `*operations`, the corresponding result in `*capabilities` will be a bitmask indicating which of cudaAtomicOperationCapability the link supports natively.

Returns cudaErrorInvalidDevice if `dev` is not valid.

Returns cudaErrorInvalidValue if `*capabilities` or `*operations` is NULL, if `count` is 0, or if any of `*operations` is not valid.

######  Parameters

`pValue`
    \- Returned size of the limit
`limit`
    \- Limit to query

###### Returns

cudaSuccess, cudaErrorUnsupportedLimit, cudaErrorInvalidValue

###### Description

Returns in `*pValue` the current size of `limit`. The following cudaLimit values are supported.

  * cudaLimitStackSize is the stack size in bytes of each GPU thread.

  * cudaLimitPrintfFifoSize is the size in bytes of the shared FIFO used by the printf() device system call.

  * cudaLimitMallocHeapSize is the size in bytes of the heap used by the malloc() and free() device system calls.

  * cudaLimitDevRuntimeSyncDepth is the maximum grid depth at which a thread can isssue the device runtime call cudaDeviceSynchronize() to wait on child grid launches to complete. This functionality is removed for devices of compute capability >= 9.0, and hence will return error cudaErrorUnsupportedLimit on such devices.

  * cudaLimitDevRuntimePendingLaunchCount is the maximum number of outstanding device runtime launches.

  * cudaLimitMaxL2FetchGranularity is the L2 cache fetch granularity.

  * cudaLimitPersistingL2CacheSize is the persisting L2 cache size in bytes.


  *

  * Note that this function may also return cudaErrorInitializationError, cudaErrorInsufficientDriver or cudaErrorNoDevice if this call tries to initialize internal CUDA RT state.

  * Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cudaErrorNotPermitted may, but is not guaranteed to, be returned as a diagnostic in such case.


###### Returns

cudaSuccess, cudaErrorInvalidValuecudaErrorNotSupported

###### Description

Returns the last pool provided to cudaDeviceSetMemPool for this device or the device's default memory pool if cudaDeviceSetMemPool has never been called. By default the current mempool is the default mempool for a device, otherwise the returned pool must have been set with cuDeviceSetMemPool or cudaDeviceSetMemPool.

  *

  * Note that this function may also return cudaErrorInitializationError, cudaErrorInsufficientDriver or cudaErrorNoDevice if this call tries to initialize internal CUDA RT state.

  * Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cudaErrorNotPermitted may, but is not guaranteed to, be returned as a diagnostic in such case.


######  Parameters

`nvSciSyncAttrList`
    \- Return NvSciSync attributes supported.
`device`
    \- Valid Cuda Device to get NvSciSync attributes for.
`flags`
    \- flags describing NvSciSync usage.

###### Description

Returns in `nvSciSyncAttrList`, the properties of NvSciSync that this CUDA device, `dev` can support. The returned `nvSciSyncAttrList` can be used to create an NvSciSync that matches this device's capabilities.

If NvSciSyncAttrKey_RequiredPerm field in `nvSciSyncAttrList` is already set this API will return cudaErrorInvalidValue.

The applications should set `nvSciSyncAttrList` to a valid NvSciSyncAttrList failing which this API will return cudaErrorInvalidHandle.

The `flags` controls how applications intends to use the NvSciSync created from the `nvSciSyncAttrList`. The valid flags are:

  * cudaNvSciSyncAttrSignal, specifies that the applications intends to signal an NvSciSync on this CUDA device.

  * cudaNvSciSyncAttrWait, specifies that the applications intends to wait on an NvSciSync on this CUDA device.


At least one of these flags must be set, failing which the API returns cudaErrorInvalidValue. Both the flags are orthogonal to one another: a developer may set both these flags that allows to set both wait and signal specific attributes in the same `nvSciSyncAttrList`.

Note that this API updates the input `nvSciSyncAttrList` with values equivalent to the following public attribute key-values: NvSciSyncAttrKey_RequiredPerm is set to

  * NvSciSyncAccessPerm_SignalOnly if cudaNvSciSyncAttrSignal is set in `flags`.

  * NvSciSyncAccessPerm_WaitOnly if cudaNvSciSyncAttrWait is set in `flags`.

  * NvSciSyncAccessPerm_WaitSignal if both cudaNvSciSyncAttrWait and cudaNvSciSyncAttrSignal are set in `flags`. NvSciSyncAttrKey_PrimitiveInfo is set to

  * NvSciSyncAttrValPrimitiveType_SysmemSemaphore on any valid `device`.

  * NvSciSyncAttrValPrimitiveType_Syncpoint if `device` is a Tegra device.

  * NvSciSyncAttrValPrimitiveType_SysmemSemaphorePayload64b if `device` is GA10X+. NvSciSyncAttrKey_GpuId is set to the same UUID that is returned in `cudaDeviceProp.uuid` from cudaDeviceGetProperties for this `device`.


cudaSuccess, cudaErrorDeviceUninitialized, cudaErrorInvalidValue, cudaErrorInvalidHandle, cudaErrorInvalidDevice, cudaErrorNotSupported, cudaErrorMemoryAllocation

######  Parameters

`capabilities`
    \- Returned capability details of each requested operation
`operations`
    \- Requested operations
`count`
    \- Count of requested operations and size of capabilities
`srcDevice`
    \- The source device of the target link
`dstDevice`
    \- The destination device of the target link

###### Returns

cudaSuccess, cudaErrorInvalidDevice, cudaErrorInvalidValue

###### Description

Returns in `*capabilities` the details about requested atomic `*operations` over the the link between `srcDevice` and `dstDevice`. The allocated size of `*operations` and `*capabilities` must be `count`.

For each cudaAtomicOperation in `*operations`, the corresponding result in `*capabilities` will be a bitmask indicating which of cudaAtomicOperationCapability the link supports natively.

Returns cudaErrorInvalidDevice if `srcDevice` or `dstDevice` are not valid or if they represent the same device.

Returns cudaErrorInvalidValue if `*capabilities` or `*operations` is NULL, if `count` is 0, or if any of `*operations` is not valid.

######  Parameters

`value`
    \- Returned value of the requested attribute
`attr`

`srcDevice`
    \- The source device of the target link.
`dstDevice`
    \- The destination device of the target link.

###### Returns

cudaSuccess, cudaErrorInvalidDevice, cudaErrorInvalidValue

###### Description

Returns in `*value` the value of the requested attribute `attrib` of the link between `srcDevice` and `dstDevice`. The supported attributes are:

  * cudaDevP2PAttrPerformanceRank: A relative value indicating the performance of the link between two devices. Lower value means better performance (0 being the value used for most performant link).

  * cudaDevP2PAttrAccessSupported: 1 if peer access is enabled.

  * cudaDevP2PAttrNativeAtomicSupported: 1 if all native atomic operations over the link are supported.

  * cudaDevP2PAttrCudaArrayAccessSupported: 1 if accessing CUDA arrays over the link is supported.

  * cudaDevP2PAttrOnlyPartialNativeAtomicSupported: 1 if some CUDA-valid atomic operations over the link are supported. Information about specific operations can be retrieved with cudaDeviceGetP2PAtomicCapabilities.


Returns cudaErrorInvalidDevice if `srcDevice` or `dstDevice` are not valid or if they represent the same device.

Returns cudaErrorInvalidValue if `attrib` is not valid or if `value` is a null pointer.

  *

  * Note that this function may also return cudaErrorInitializationError, cudaErrorInsufficientDriver or cudaErrorNoDevice if this call tries to initialize internal CUDA RT state.

  * Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cudaErrorNotPermitted may, but is not guaranteed to, be returned as a diagnostic in such case.


######  Parameters

`pciBusId`
    \- Returned identifier string for the device in the following format [domain]:[bus]:[device].[function] where `domain`, `bus`, `device`, and `function` are all hexadecimal values. pciBusId should be large enough to store 13 characters including the NULL-terminator.
`len`
    \- Maximum length of string to store in `name`
`device`
    \- Device to get identifier string for

###### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevice

###### Description

Returns an ASCII string identifying the device `dev` in the NULL-terminated string pointed to by `pciBusId`. `len` specifies the maximum length of the string that may be returned.

  *

  * Note that this function may also return cudaErrorInitializationError, cudaErrorInsufficientDriver or cudaErrorNoDevice if this call tries to initialize internal CUDA RT state.

  * Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cudaErrorNotPermitted may, but is not guaranteed to, be returned as a diagnostic in such case.


######  Parameters

`leastPriority`
    \- Pointer to an int in which the numerical value for least stream priority is returned
`greatestPriority`
    \- Pointer to an int in which the numerical value for greatest stream priority is returned

###### Returns

cudaSuccess

###### Description

Returns in `*leastPriority` and `*greatestPriority` the numerical values that correspond to the least and greatest stream priorities respectively. Stream priorities follow a convention where lower numbers imply greater priorities. The range of meaningful stream priorities is given by [`*greatestPriority`, `*leastPriority`]. If the user attempts to create a stream with a priority value that is outside the the meaningful range as specified by this API, the priority is automatically clamped down or up to either `*leastPriority` or `*greatestPriority` respectively. See cudaStreamCreateWithPriority for details on creating a priority stream. A NULL may be passed in for `*leastPriority` or `*greatestPriority` if the value is not desired.

This function will return '0' in both `*leastPriority` and `*greatestPriority` if the current context's device does not support stream priorities (see cudaDeviceGetAttribute).

  *

  * Note that this function may also return cudaErrorInitializationError, cudaErrorInsufficientDriver or cudaErrorNoDevice if this call tries to initialize internal CUDA RT state.

  * Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cudaErrorNotPermitted may, but is not guaranteed to, be returned as a diagnostic in such case.


######  Parameters

`maxWidthInElements`
    \- Returns maximum number of texture elements allocatable for given `fmtDesc`.
`fmtDesc`
    \- Texture format description.
`device`


###### Returns

cudaSuccess, cudaErrorUnsupportedLimit, cudaErrorInvalidValue

###### Description

Returns in `maxWidthInElements` the maximum number of elements allocatable in a 1D linear texture for given format descriptor `fmtDesc`.

  *

  * Note that this function may also return cudaErrorInitializationError, cudaErrorInsufficientDriver or cudaErrorNoDevice if this call tries to initialize internal CUDA RT state.

  * Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cudaErrorNotPermitted may, but is not guaranteed to, be returned as a diagnostic in such case.


######  Parameters

`device`
    \- The device on which to register the callback
`callbackFunc`
    \- The function to register as a callback
`userData`
    \- A generic pointer to user data. This is passed into the callback function.
`callback`
    \- A handle representing the registered callback instance

###### Returns

cudaSuccesscudaErrorNotSupportedcudaErrorInvalidDevicecudaErrorInvalidValuecudaErrorNotPermittedcudaErrorUnknown

###### Description

Registers `callbackFunc` to receive async notifications.

The `userData` parameter is passed to the callback function at async notification time. Likewise, `callback` is also passed to the callback function to distinguish between multiple registered callbacks.

The callback function being registered should be designed to return quickly (~10ms). Any long running tasks should be queued for execution on an application thread.

Callbacks may not call cudaDeviceRegisterAsyncNotification or cudaDeviceUnregisterAsyncNotification. Doing so will result in cudaErrorNotPermitted. Async notification callbacks execute in an undefined order and may be serialized.

Returns in `*callback` a handle representing the registered callback instance.

###### Returns

cudaSuccess

###### Description

Explicitly destroys and cleans up all resources associated with the current device in the current process. It is the caller's responsibility to ensure that the resources are not accessed or passed in subsequent API calls and doing so will result in undefined behavior. These resources include CUDA types cudaStream_t, cudaEvent_t, cudaArray_t, cudaMipmappedArray_t, cudaPitchedPtr, cudaTextureObject_t, cudaSurfaceObject_t, textureReference, surfaceReference, cudaExternalMemory_t, cudaExternalSemaphore_t and cudaGraphicsResource_t. These resources also include memory allocations by cudaMalloc, cudaMallocHost, cudaMallocManaged and cudaMallocPitch. Any subsequent API call to this device will reinitialize the device.

Note that this function will reset the device immediately. It is the caller's responsibility to ensure that the device is not being accessed by any other host threads from the process when this function is called.

  * cudaDeviceReset() will not destroy memory allocations by cudaMallocAsync() and cudaMallocFromPoolAsync(). These memory allocations need to be destroyed explicitly.

  * If a non-primary CUcontext is current to the thread, cudaDeviceReset() will destroy only the internal CUDA RT state for that CUcontext.


  *

  * Note that this function may also return cudaErrorInitializationError, cudaErrorInsufficientDriver or cudaErrorNoDevice if this call tries to initialize internal CUDA RT state.

  * Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cudaErrorNotPermitted may, but is not guaranteed to, be returned as a diagnostic in such case.


######  Parameters

`cacheConfig`
    \- Requested cache configuration

###### Returns

cudaSuccess

###### Description

On devices where the L1 cache and shared memory use the same hardware resources, this sets through `cacheConfig` the preferred cache configuration for the current device. This is only a preference. The runtime will use the requested configuration if possible, but it is free to choose a different configuration if required to execute the function. Any function preference set via cudaFuncSetCacheConfig ( C API) or cudaFuncSetCacheConfig ( C++ API) will be preferred over this device-wide setting. Setting the device-wide cache configuration to cudaFuncCachePreferNone will cause subsequent kernel launches to prefer to not change the cache configuration unless required to launch the kernel.

This setting does nothing on devices where the size of the L1 cache and shared memory are fixed.

Launching a kernel with a different preference than the most recent preference setting may insert a device-side synchronization point.

The supported cache configurations are:

  * cudaFuncCachePreferNone: no preference for shared memory or L1 (default)

  * cudaFuncCachePreferShared: prefer larger shared memory and smaller L1 cache

  * cudaFuncCachePreferL1: prefer larger L1 cache and smaller shared memory

  * cudaFuncCachePreferEqual: prefer equal size L1 cache and shared memory


  *

  * Note that this function may also return cudaErrorInitializationError, cudaErrorInsufficientDriver or cudaErrorNoDevice if this call tries to initialize internal CUDA RT state.

  * Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cudaErrorNotPermitted may, but is not guaranteed to, be returned as a diagnostic in such case.


######  Parameters

`limit`
    \- Limit to set
`value`
    \- Size of limit

###### Returns

cudaSuccess, cudaErrorUnsupportedLimit, cudaErrorInvalidValue, cudaErrorMemoryAllocation

###### Description

Setting `limit` to `value` is a request by the application to update the current limit maintained by the device. The driver is free to modify the requested value to meet h/w requirements (this could be clamping to minimum or maximum values, rounding up to nearest element size, etc). The application can use cudaDeviceGetLimit() to find out exactly what the limit has been set to.

Setting each cudaLimit has its own specific restrictions, so each is discussed here.

  * cudaLimitStackSize controls the stack size in bytes of each GPU thread.


  * cudaLimitPrintfFifoSize controls the size in bytes of the shared FIFO used by the printf() device system call. Setting cudaLimitPrintfFifoSize must not be performed after launching any kernel that uses the printf() device system call - in such case cudaErrorInvalidValue will be returned.


  * cudaLimitMallocHeapSize controls the size in bytes of the heap used by the malloc() and free() device system calls. Setting cudaLimitMallocHeapSize must not be performed after launching any kernel that uses the malloc() or free() device system calls - in such case cudaErrorInvalidValue will be returned.


  * cudaLimitDevRuntimeSyncDepth controls the maximum nesting depth of a grid at which a thread can safely call cudaDeviceSynchronize(). Setting this limit must be performed before any launch of a kernel that uses the device runtime and calls cudaDeviceSynchronize() above the default sync depth, two levels of grids. Calls to cudaDeviceSynchronize() will fail with error code cudaErrorSyncDepthExceeded if the limitation is violated. This limit can be set smaller than the default or up the maximum launch depth of 24. When setting this limit, keep in mind that additional levels of sync depth require the runtime to reserve large amounts of device memory which can no longer be used for user allocations. If these reservations of device memory fail, cudaDeviceSetLimit will return cudaErrorMemoryAllocation, and the limit can be reset to a lower value. This limit is only applicable to devices of compute capability < 9.0. Attempting to set this limit on devices of other compute capability will results in error cudaErrorUnsupportedLimit being returned.


  * cudaLimitDevRuntimePendingLaunchCount controls the maximum number of outstanding device runtime launches that can be made from the current device. A grid is outstanding from the point of launch up until the grid is known to have been completed. Device runtime launches which violate this limitation fail and return cudaErrorLaunchPendingCountExceeded when cudaGetLastError() is called after launch. If more pending launches than the default (2048 launches) are needed for a module using the device runtime, this limit can be increased. Keep in mind that being able to sustain additional pending launches will require the runtime to reserve larger amounts of device memory upfront which can no longer be used for allocations. If these reservations fail, cudaDeviceSetLimit will return cudaErrorMemoryAllocation, and the limit can be reset to a lower value. This limit is only applicable to devices of compute capability 3.5 and higher. Attempting to set this limit on devices of compute capability less than 3.5 will result in the error cudaErrorUnsupportedLimit being returned.


  * cudaLimitMaxL2FetchGranularity controls the L2 cache fetch granularity. Values can range from 0B to 128B. This is purely a performance hint and it can be ignored or clamped depending on the platform.


  * cudaLimitPersistingL2CacheSize controls size in bytes available for persisting L2 cache. This is purely a performance hint and it can be ignored or clamped depending on the platform.


  *

  * Note that this function may also return cudaErrorInitializationError, cudaErrorInsufficientDriver or cudaErrorNoDevice if this call tries to initialize internal CUDA RT state.

  * Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cudaErrorNotPermitted may, but is not guaranteed to, be returned as a diagnostic in such case.


###### Returns

cudaSuccess, cudaErrorInvalidValuecudaErrorInvalidDevicecudaErrorNotSupported

###### Description

The memory pool must be local to the specified device. Unless a mempool is specified in the cudaMallocAsync call, cudaMallocAsync allocates from the current mempool of the provided stream's device. By default, a device's current memory pool is its default memory pool.

Use cudaMallocFromPoolAsync to specify asynchronous allocations from a device different than the one the stream runs on.

  *

  * Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cudaErrorNotPermitted may, but is not guaranteed to, be returned as a diagnostic in such case.


###### Returns

cudaSuccess, cudaErrorStreamCaptureUnsupported

###### Description

Blocks until the device has completed all preceding requested tasks. cudaDeviceSynchronize() returns an error if one of the preceding tasks has failed. If the cudaDeviceScheduleBlockingSync flag was set for this device, the host thread will block until the device has finished its work.

  * Use of cudaDeviceSynchronize in device code was deprecated in CUDA 11.6 and removed for compute_90+ compilation. For compute capability < 9.0, compile-time opt-in by specifying -D CUDA_FORCE_CDP1_IF_SUPPORTED is required to continue using cudaDeviceSynchronize() in device code for now. Note that this is different from host-side cudaDeviceSynchronize, which is still supported.

  *

  * Note that this function may also return cudaErrorInitializationError, cudaErrorInsufficientDriver or cudaErrorNoDevice if this call tries to initialize internal CUDA RT state.

  * Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cudaErrorNotPermitted may, but is not guaranteed to, be returned as a diagnostic in such case.


######  Parameters

`device`
    \- The device from which to remove `callback`.
`callback`
    \- The callback instance to unregister from receiving async notifications.

###### Returns

cudaSuccesscudaErrorNotSupportedcudaErrorInvalidDevicecudaErrorInvalidValuecudaErrorNotPermittedcudaErrorUnknown

###### Description

Unregisters `callback` so that the corresponding callback function will stop receiving async notifications.

######  Parameters

`device`
    \- Returns the device on which the active host thread executes the device code.

###### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorDeviceUnavailable

###### Description

Returns in `*device` the current device for the calling host thread.

  *

  * Note that this function may also return cudaErrorInitializationError, cudaErrorInsufficientDriver or cudaErrorNoDevice if this call tries to initialize internal CUDA RT state.

  * Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cudaErrorNotPermitted may, but is not guaranteed to, be returned as a diagnostic in such case.


######  Parameters

`count`
    \- Returns the number of devices with compute capability greater or equal to 2.0

###### Returns

cudaSuccess

###### Description

Returns in `*count` the number of devices with compute capability greater or equal to 2.0 that are available for execution.

  *

  * Note that this function may also return cudaErrorInitializationError, cudaErrorInsufficientDriver or cudaErrorNoDevice if this call tries to initialize internal CUDA RT state.

  * Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cudaErrorNotPermitted may, but is not guaranteed to, be returned as a diagnostic in such case.


######  Parameters

`flags`
    \- Pointer to store the device flags

###### Returns

cudaSuccess, cudaErrorInvalidDevice

###### Description

Returns in `flags` the flags for the current device. If there is a current device for the calling thread, the flags for the device are returned. If there is no current device, the flags for the first device are returned, which may be the default flags. Compare to the behavior of cudaSetDeviceFlags.

Typically, the flags returned should match the behavior that will be seen if the calling thread uses a device after this call, without any change to the flags or current device inbetween by this or another thread. Note that if the device is not initialized, it is possible for another thread to change the flags for the current device before it is initialized. Additionally, when using exclusive mode, if this thread has not requested a specific device, it may use a device other than the first device, contrary to the assumption made by this function.

If a context has been created via the driver API and is current to the calling thread, the flags for that context are always returned.

Flags returned by this function may specifically include cudaDeviceMapHost even though it is not accepted by cudaSetDeviceFlags because it is implicit in runtime API flags. The reason for this is that the current context may have been created via the driver API in which case the flag is not implicit and may be unset.

  *

  * Note that this function may also return cudaErrorInitializationError, cudaErrorInsufficientDriver or cudaErrorNoDevice if this call tries to initialize internal CUDA RT state.

  * Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cudaErrorNotPermitted may, but is not guaranteed to, be returned as a diagnostic in such case.


######  Parameters

`prop`
    \- Properties for the specified device
`device`
    \- Device number to get properties for

###### Returns

cudaSuccess, cudaErrorInvalidDevice

###### Description

Returns in `*prop` the properties of device `dev`.

  *

  * Note that this function may also return cudaErrorInitializationError, cudaErrorInsufficientDriver or cudaErrorNoDevice if this call tries to initialize internal CUDA RT state.

  * Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cudaErrorNotPermitted may, but is not guaranteed to, be returned as a diagnostic in such case.


######  Parameters

`device`
    \- Device on which the runtime will initialize itself.
`deviceFlags`
    \- Parameters for device operation.
`flags`
    \- Flags for controlling the device initialization.

###### Returns

cudaSuccess, cudaErrorInvalidDevice

###### Description

This function will initialize the CUDA Runtime structures and primary context on `device` when called, but the context will not be made current to `device`.

When cudaInitDeviceFlagsAreValid is set in `flags`, deviceFlags are applied to the requested device. The values of deviceFlags match those of the flags parameters in cudaSetDeviceFlags. The effect may be verified by cudaGetDeviceFlags.

This function will return an error if the device is in cudaComputeModeExclusiveProcess and is occupied by another process or if the device is in cudaComputeModeProhibited.

  *

  * Note that this function may also return cudaErrorInitializationError, cudaErrorInsufficientDriver or cudaErrorNoDevice if this call tries to initialize internal CUDA RT state.

  * Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cudaErrorNotPermitted may, but is not guaranteed to, be returned as a diagnostic in such case.


######  Parameters

`devPtr`
    \- Device pointer returned by cudaIpcOpenMemHandle

###### Returns

cudaSuccess, cudaErrorMapBufferObjectFailed, cudaErrorNotSupported, cudaErrorInvalidValue

###### Description

Decrements the reference count of the memory returnd by cudaIpcOpenMemHandle by 1. When the reference count reaches 0, this API unmaps the memory. The original allocation in the exporting process as well as imported mappings in other processes will be unaffected.

Any resources used to enable peer access will be freed if this is the last mapping using them.

IPC functionality is restricted to devices with support for unified addressing on Linux and Windows operating systems. IPC functionality on Windows is supported for compatibility purposes but not recommended as it comes with performance cost. Users can test their device for IPC functionality by calling cudaDeviceGetAttribute with cudaDevAttrIpcEventSupport

  * Note that this function may also return cudaErrorInitializationError, cudaErrorInsufficientDriver or cudaErrorNoDevice if this call tries to initialize internal CUDA RT state.

  * Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cudaErrorNotPermitted may, but is not guaranteed to, be returned as a diagnostic in such case.


######  Parameters

`handle`
    \- Pointer to a user allocated cudaIpcEventHandle in which to return the opaque event handle
`event`
    \- Event allocated with cudaEventInterprocess and cudaEventDisableTiming flags.

###### Returns

cudaSuccess, cudaErrorInvalidResourceHandle, cudaErrorMemoryAllocation, cudaErrorMapBufferObjectFailed, cudaErrorNotSupported, cudaErrorInvalidValue

###### Description

Takes as input a previously allocated event. This event must have been created with the cudaEventInterprocess and cudaEventDisableTiming flags set. This opaque handle may be copied into other processes and opened with cudaIpcOpenEventHandle to allow efficient hardware synchronization between GPU work in different processes.

After the event has been been opened in the importing process, cudaEventRecord, cudaEventSynchronize, cudaStreamWaitEvent and cudaEventQuery may be used in either process. Performing operations on the imported event after the exported event has been freed with cudaEventDestroy will result in undefined behavior.

IPC functionality is restricted to devices with support for unified addressing on Linux and Windows operating systems. IPC functionality on Windows is supported for compatibility purposes but not recommended as it comes with performance cost. Users can test their device for IPC functionality by calling cudaDeviceGetAttribute with cudaDevAttrIpcEventSupport

  * Note that this function may also return cudaErrorInitializationError, cudaErrorInsufficientDriver or cudaErrorNoDevice if this call tries to initialize internal CUDA RT state.

  * Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cudaErrorNotPermitted may, but is not guaranteed to, be returned as a diagnostic in such case.


######  Parameters

`handle`
    \- Pointer to user allocated cudaIpcMemHandle to return the handle in.
`devPtr`
    \- Base pointer to previously allocated device memory

###### Returns

cudaSuccess, cudaErrorMemoryAllocation, cudaErrorMapBufferObjectFailed, cudaErrorNotSupported, cudaErrorInvalidValue

###### Description

Takes a pointer to the base of an existing device memory allocation created with cudaMalloc and exports it for use in another process. This is a lightweight operation and may be called multiple times on an allocation without adverse effects.

If a region of memory is freed with cudaFree and a subsequent call to cudaMalloc returns memory with the same device address, cudaIpcGetMemHandle will return a unique handle for the new memory.

IPC functionality is restricted to devices with support for unified addressing on Linux and Windows operating systems. IPC functionality on Windows is supported for compatibility purposes but not recommended as it comes with performance cost. Users can test their device for IPC functionality by calling cudaDeviceGetAttribute with cudaDevAttrIpcEventSupport

  * Note that this function may also return cudaErrorInitializationError, cudaErrorInsufficientDriver or cudaErrorNoDevice if this call tries to initialize internal CUDA RT state.

  * Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cudaErrorNotPermitted may, but is not guaranteed to, be returned as a diagnostic in such case.


######  Parameters

`event`
    \- Returns the imported event
`handle`
    \- Interprocess handle to open

###### Returns

cudaSuccess, cudaErrorMapBufferObjectFailed, cudaErrorNotSupported, cudaErrorInvalidValue, cudaErrorDeviceUninitialized

###### Description

Opens an interprocess event handle exported from another process with cudaIpcGetEventHandle. This function returns a cudaEvent_t that behaves like a locally created event with the cudaEventDisableTiming flag specified. This event must be freed with cudaEventDestroy.

Performing operations on the imported event after the exported event has been freed with cudaEventDestroy will result in undefined behavior.

IPC functionality is restricted to devices with support for unified addressing on Linux and Windows operating systems. IPC functionality on Windows is supported for compatibility purposes but not recommended as it comes with performance cost. Users can test their device for IPC functionality by calling cudaDeviceGetAttribute with cudaDevAttrIpcEventSupport

  * Note that this function may also return cudaErrorInitializationError, cudaErrorInsufficientDriver or cudaErrorNoDevice if this call tries to initialize internal CUDA RT state.

  * Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cudaErrorNotPermitted may, but is not guaranteed to, be returned as a diagnostic in such case.


######  Parameters

`devPtr`
    \- Returned device pointer
`handle`
    \- cudaIpcMemHandle to open
`flags`
    \- Flags for this operation. Must be specified as cudaIpcMemLazyEnablePeerAccess

###### Returns

cudaSuccess, cudaErrorMapBufferObjectFailed, cudaErrorInvalidResourceHandle, cudaErrorDeviceUninitialized, cudaErrorTooManyPeers, cudaErrorNotSupported, cudaErrorInvalidValue

###### Description

Maps memory exported from another process with cudaIpcGetMemHandle into the current device address space. For contexts on different devices cudaIpcOpenMemHandle can attempt to enable peer access between the devices as if the user called cudaDeviceEnablePeerAccess. This behavior is controlled by the cudaIpcMemLazyEnablePeerAccess flag. cudaDeviceCanAccessPeer can determine if a mapping is possible.

cudaIpcOpenMemHandle can open handles to devices that may not be visible in the process calling the API.

Contexts that may open cudaIpcMemHandles are restricted in the following way. cudaIpcMemHandles from each device in a given process may only be opened by one context per device per other process.

If the memory handle has already been opened by the current context, the reference count on the handle is incremented by 1 and the existing device pointer is returned.

Memory returned from cudaIpcOpenMemHandle must be freed with cudaIpcCloseMemHandle.

Calling cudaFree on an exported memory region before calling cudaIpcCloseMemHandle in the importing context will result in undefined behavior.

IPC functionality is restricted to devices with support for unified addressing on Linux and Windows operating systems. IPC functionality on Windows is supported for compatibility purposes but not recommended as it comes with performance cost. Users can test their device for IPC functionality by calling cudaDeviceGetAttribute with cudaDevAttrIpcEventSupport

  * Note that this function may also return cudaErrorInitializationError, cudaErrorInsufficientDriver or cudaErrorNoDevice if this call tries to initialize internal CUDA RT state.

  * Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cudaErrorNotPermitted may, but is not guaranteed to, be returned as a diagnostic in such case.

  * No guarantees are made about the address returned in `*devPtr`. In particular, multiple processes may not receive the same address for the same `handle`.


######  Parameters

`device`
    \- Device on which the active host thread should execute the device code.

###### Returns

cudaSuccess, cudaErrorInvalidDevice, cudaErrorDeviceUnavailable

###### Description

Sets `device` as the current device for the calling host thread. Valid device id's are 0 to (cudaGetDeviceCount() \- 1).

Any device memory subsequently allocated from this host thread using cudaMalloc(), cudaMallocPitch() or cudaMallocArray() will be physically resident on `device`. Any host memory allocated from this host thread using cudaMallocHost() or cudaHostAlloc() or cudaHostRegister() will have its lifetime associated with `device`. Any streams or events created from this host thread will be associated with `device`. Any kernels launched from this host thread using the <<<>>> operator or cudaLaunchKernel() will be executed on `device`.

This call may be made from any host thread, to any device, and at any time. This function will do no synchronization with the previous or new device, and should only take significant time when it initializes the runtime's context state. This call will bind the primary context of the specified device to the calling thread and all the subsequent memory allocations, stream and event creations, and kernel launches will be associated with the primary context. This function will also immediately initialize the runtime state on the primary context, and the context will be current on `device` immediately. This function will return an error if the device is in cudaComputeModeExclusiveProcess and is occupied by another process or if the device is in cudaComputeModeProhibited.

It is not required to call cudaInitDevice before using this function.

  *

  * Note that this function may also return cudaErrorInitializationError, cudaErrorInsufficientDriver or cudaErrorNoDevice if this call tries to initialize internal CUDA RT state.

  * Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cudaErrorNotPermitted may, but is not guaranteed to, be returned as a diagnostic in such case.


######  Parameters

`flags`
    \- Parameters for device operation

###### Returns

cudaSuccess, cudaErrorInvalidValue

###### Description

Records `flags` as the flags for the current device. If the current device has been set and that device has already been initialized, the previous flags are overwritten. If the current device has not been initialized, it is initialized with the provided flags. If no device has been made current to the calling thread, a default device is selected and initialized with the provided flags.

The three LSBs of the `flags` parameter can be used to control how the CPU thread interacts with the OS scheduler when waiting for results from the device.

  * cudaDeviceScheduleAuto: The default value if the `flags` parameter is zero, uses a heuristic based on the number of active CUDA contexts in the process `C` and the number of logical processors in the system `P`. If `C` > `P`, then CUDA will yield to other OS threads when waiting for the device, otherwise CUDA will not yield while waiting for results and actively spin on the processor. Additionally, on Tegra devices, cudaDeviceScheduleAuto uses a heuristic based on the power profile of the platform and may choose cudaDeviceScheduleBlockingSync for low-powered devices.

  * cudaDeviceScheduleSpin: Instruct CUDA to actively spin when waiting for results from the device. This can decrease latency when waiting for the device, but may lower the performance of CPU threads if they are performing work in parallel with the CUDA thread.

  * cudaDeviceScheduleYield: Instruct CUDA to yield its thread when waiting for results from the device. This can increase latency when waiting for the device, but can increase the performance of CPU threads performing work in parallel with the device.

  * cudaDeviceScheduleBlockingSync: Instruct CUDA to block the CPU thread on a synchronization primitive when waiting for the device to finish work.

  * cudaDeviceBlockingSync: Instruct CUDA to block the CPU thread on a synchronization primitive when waiting for the device to finish work.

Deprecated: This flag was deprecated as of CUDA 4.0 and replaced with cudaDeviceScheduleBlockingSync.

  * cudaDeviceMapHost: This flag enables allocating pinned host memory that is accessible to the device. It is implicit for the runtime but may be absent if a context is created using the driver API. If this flag is not set, cudaHostGetDevicePointer() will always return a failure code.

  * cudaDeviceLmemResizeToMax: Instruct CUDA to not reduce local memory after resizing local memory for a kernel. This can prevent thrashing by local memory allocations when launching many kernels with high local memory usage at the cost of potentially increased memory usage.

Deprecated: This flag is deprecated and the behavior enabled by this flag is now the default and cannot be disabled.

  * cudaDeviceSyncMemops: Ensures that synchronous memory operations initiated on this context will always synchronize. See further documentation in the section titled "API Synchronization behavior" to learn more about cases when synchronous memory operations can exhibit asynchronous behavior.


  *

  * Note that this function may also return cudaErrorInitializationError, cudaErrorInsufficientDriver or cudaErrorNoDevice if this call tries to initialize internal CUDA RT state.

  * Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cudaErrorNotPermitted may, but is not guaranteed to, be returned as a diagnostic in such case.


######  Parameters

`device_arr`
    \- List of devices to try
`len`
    \- Number of devices in specified list

###### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevice

###### Description

Sets a list of devices for CUDA execution in priority order using `device_arr`. The parameter `len` specifies the number of elements in the list. CUDA will try devices from the list sequentially until it finds one that works. If this function is not called, or if it is called with a `len` of 0, then CUDA will go back to its default behavior of trying devices sequentially from a default list containing all of the available CUDA devices in the system. If a specified device ID in the list does not exist, this function will return cudaErrorInvalidDevice. If `len` is not 0 and `device_arr` is NULL or if `len` exceeds the number of devices in the system, then cudaErrorInvalidValue is returned.

  *

  * Note that this function may also return cudaErrorInitializationError, cudaErrorInsufficientDriver or cudaErrorNoDevice if this call tries to initialize internal CUDA RT state.

  * Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cudaErrorNotPermitted may, but is not guaranteed to, be returned as a diagnostic in such case.
