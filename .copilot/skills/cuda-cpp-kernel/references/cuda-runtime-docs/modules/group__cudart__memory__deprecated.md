# 6.11. Memory Management [DEPRECATED]

**Source:** group__CUDART__MEMORY__DEPRECATED.html#group__CUDART__MEMORY__DEPRECATED


### Functions

__host__ cudaError_t cudaMemcpyArrayToArray ( cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice )


Copies data between host and device.

######  Parameters

`dst`
    \- Destination memory address
`wOffsetDst`
    \- Destination starting X offset (columns in bytes)
`hOffsetDst`
    \- Destination starting Y offset (rows)
`src`
    \- Source memory address
`wOffsetSrc`
    \- Source starting X offset (columns in bytes)
`hOffsetSrc`
    \- Source starting Y offset (rows)
`count`
    \- Size in bytes to copy
`kind`
    \- Type of transfer

###### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidMemcpyDirection

###### Deprecated

###### Description

Copies `count` bytes from the CUDA array `src` starting at `hOffsetSrc` rows and `wOffsetSrc` bytes from the upper left corner to the CUDA array `dst` starting at `hOffsetDst` rows and `wOffsetDst` bytes from the upper left corner, where `kind` specifies the direction of the copy, and must be one of cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, or cudaMemcpyDefault. Passing cudaMemcpyDefault is recommended, in which case the type of transfer is inferred from the pointer values. However, cudaMemcpyDefault is only allowed on systems that support unified virtual addressing.

  *

  * Note that this function may also return cudaErrorInitializationError, cudaErrorInsufficientDriver or cudaErrorNoDevice if this call tries to initialize internal CUDA RT state.

  * Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cudaErrorNotPermitted may, but is not guaranteed to, be returned as a diagnostic in such case.


######  Parameters

`dst`
    \- Destination memory address
`src`
    \- Source memory address
`wOffset`
    \- Source starting X offset (columns in bytes)
`hOffset`
    \- Source starting Y offset (rows)
`count`
    \- Size in bytes to copy
`kind`
    \- Type of transfer

###### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidMemcpyDirection

###### Deprecated

###### Description

Copies `count` bytes from the CUDA array `src` starting at `hOffset` rows and `wOffset` bytes from the upper left corner to the memory area pointed to by `dst`, where `kind` specifies the direction of the copy, and must be one of cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, or cudaMemcpyDefault. Passing cudaMemcpyDefault is recommended, in which case the type of transfer is inferred from the pointer values. However, cudaMemcpyDefault is only allowed on systems that support unified virtual addressing.

  *

  * This function exhibits synchronous behavior for most use cases.

  * Note that this function may also return cudaErrorInitializationError, cudaErrorInsufficientDriver or cudaErrorNoDevice if this call tries to initialize internal CUDA RT state.

  * Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cudaErrorNotPermitted may, but is not guaranteed to, be returned as a diagnostic in such case.


######  Parameters

`dst`
    \- Destination memory address
`src`
    \- Source memory address
`wOffset`
    \- Source starting X offset (columns in bytes)
`hOffset`
    \- Source starting Y offset (rows)
`count`
    \- Size in bytes to copy
`kind`
    \- Type of transfer
`stream`
    \- Stream identifier

###### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidMemcpyDirection

###### Deprecated

###### Description

Copies `count` bytes from the CUDA array `src` starting at `hOffset` rows and `wOffset` bytes from the upper left corner to the memory area pointed to by `dst`, where `kind` specifies the direction of the copy, and must be one of cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, or cudaMemcpyDefault. Passing cudaMemcpyDefault is recommended, in which case the type of transfer is inferred from the pointer values. However, cudaMemcpyDefault is only allowed on systems that support unified virtual addressing.

cudaMemcpyFromArrayAsync() is asynchronous with respect to the host, so the call may return before the copy is complete. The copy can optionally be associated to a stream by passing a non-zero `stream` argument. If `kind` is cudaMemcpyHostToDevice or cudaMemcpyDeviceToHost and `stream` is non-zero, the copy may overlap with operations in other streams.

  *

  * This function exhibits asynchronous behavior for most use cases.

  * This function uses standard default stream semantics.

  * Note that this function may also return cudaErrorInitializationError, cudaErrorInsufficientDriver or cudaErrorNoDevice if this call tries to initialize internal CUDA RT state.

  * Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cudaErrorNotPermitted may, but is not guaranteed to, be returned as a diagnostic in such case.


######  Parameters

`dst`
    \- Destination memory address
`wOffset`
    \- Destination starting X offset (columns in bytes)
`hOffset`
    \- Destination starting Y offset (rows)
`src`
    \- Source memory address
`count`
    \- Size in bytes to copy
`kind`
    \- Type of transfer

###### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidMemcpyDirection

###### Deprecated

###### Description

Copies `count` bytes from the memory area pointed to by `src` to the CUDA array `dst` starting at `hOffset` rows and `wOffset` bytes from the upper left corner, where `kind` specifies the direction of the copy, and must be one of cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, or cudaMemcpyDefault. Passing cudaMemcpyDefault is recommended, in which case the type of transfer is inferred from the pointer values. However, cudaMemcpyDefault is only allowed on systems that support unified virtual addressing.

  *

  * This function exhibits synchronous behavior for most use cases.

  * Note that this function may also return cudaErrorInitializationError, cudaErrorInsufficientDriver or cudaErrorNoDevice if this call tries to initialize internal CUDA RT state.

  * Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cudaErrorNotPermitted may, but is not guaranteed to, be returned as a diagnostic in such case.


######  Parameters

`dst`
    \- Destination memory address
`wOffset`
    \- Destination starting X offset (columns in bytes)
`hOffset`
    \- Destination starting Y offset (rows)
`src`
    \- Source memory address
`count`
    \- Size in bytes to copy
`kind`
    \- Type of transfer
`stream`
    \- Stream identifier

###### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidMemcpyDirection

###### Deprecated

###### Description

Copies `count` bytes from the memory area pointed to by `src` to the CUDA array `dst` starting at `hOffset` rows and `wOffset` bytes from the upper left corner, where `kind` specifies the direction of the copy, and must be one of cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, or cudaMemcpyDefault. Passing cudaMemcpyDefault is recommended, in which case the type of transfer is inferred from the pointer values. However, cudaMemcpyDefault is only allowed on systems that support unified virtual addressing.

cudaMemcpyToArrayAsync() is asynchronous with respect to the host, so the call may return before the copy is complete. The copy can optionally be associated to a stream by passing a non-zero `stream` argument. If `kind` is cudaMemcpyHostToDevice or cudaMemcpyDeviceToHost and `stream` is non-zero, the copy may overlap with operations in other streams.

  *

  * This function exhibits asynchronous behavior for most use cases.

  * This function uses standard default stream semantics.

  * Note that this function may also return cudaErrorInitializationError, cudaErrorInsufficientDriver or cudaErrorNoDevice if this call tries to initialize internal CUDA RT state.

  * Note that as specified by cudaStreamAddCallback no CUDA function may be called from callback. cudaErrorNotPermitted may, but is not guaranteed to, be returned as a diagnostic in such case.
