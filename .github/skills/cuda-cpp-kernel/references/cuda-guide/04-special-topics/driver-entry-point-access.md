---
url: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/driver-entry-point-access.html
---

# 4.20. Driver Entry Point Access

## 4.20.1. Introduction

The `Driver Entry Point Access APIs` provide a way to retrieve the address of a CUDA driver function. Starting from CUDA 11.3, users can call into available CUDA driver APIs using function pointers obtained from these APIs.

These APIs provide functionality similar to their counterparts, dlsym on POSIX platforms and GetProcAddress on Windows. The provided APIs will let users:

  * Retrieve the address of a driver function using the `CUDA Driver API.`

  * Retrieve the address of a driver function using the `CUDA Runtime API.`

  * Request _per-thread default stream_ version of a CUDA driver function. For more details, see [Retrieve Per-thread Default Stream Versions](#retrieve-per-thread-default-stream-versions).

  * Access new CUDA features on older toolkits but with a newer driver.


## 4.20.2. Driver Function Typedefs

To help retrieve the CUDA Driver API entry points, the CUDA Toolkit provides access to headers containing the function pointer definitions for all CUDA driver APIs. These headers are installed with the CUDA Toolkit and are made available in the toolkit’s `include/` directory. The table below summarizes the header files containing the `typedefs` for each CUDA API header file.

Table 27 Typedefs header files for CUDA driver APIs API header file | API Typedef header file  
---|---  
`cuda.h` | `cudaTypedefs.h`  
`cudaGL.h` | `cudaGLTypedefs.h`  
`cudaProfiler.h` | `cudaProfilerTypedefs.h`  
`cudaVDPAU.h` | `cudaVDPAUTypedefs.h`  
`cudaEGL.h` | `cudaEGLTypedefs.h`  
`cudaD3D9.h` | `cudaD3D9Typedefs.h`  
`cudaD3D10.h` | `cudaD3D10Typedefs.h`  
`cudaD3D11.h` | `cudaD3D11Typedefs.h`  
  
The above headers do not define actual function pointers themselves; they define the typedefs for function pointers. For example, `cudaTypedefs.h` has the below typedefs for the driver API `cuMemAlloc`:
    
    
    typedef CUresult (CUDAAPI *PFN_cuMemAlloc_v3020)(CUdeviceptr_v2 *dptr, size_t bytesize);
    typedef CUresult (CUDAAPI *PFN_cuMemAlloc_v2000)(CUdeviceptr_v1 *dptr, unsigned int bytesize);
    

CUDA driver symbols have a version based naming scheme with a `_v*` extension in its name except for the first version. When the signature or the semantics of a specific CUDA driver API changes, we increment the version number of the corresponding driver symbol. In the case of the `cuMemAlloc` driver API, the first driver symbol name is `cuMemAlloc` and the next symbol name is `cuMemAlloc_v2`. The typedef for the first version which was introduced in CUDA 2.0 (2000) is `PFN_cuMemAlloc_v2000`. The typedef for the next version which was introduced in CUDA 3.2 (3020) is `PFN_cuMemAlloc_v3020`.

The `typedefs` can be used to more easily define a function pointer of the appropriate type in code:
    
    
    PFN_cuMemAlloc_v3020 pfn_cuMemAlloc_v2;
    PFN_cuMemAlloc_v2000 pfn_cuMemAlloc_v1;
    

The above method is preferable if users are interested in a specific version of the API. Additionally, the headers have predefined macros for the latest version of all driver symbols that were available when the installed CUDA toolkit was released; these typedefs do not have a `_v*` suffix. For CUDA 11.3 toolkit, `cuMemAlloc_v2` was the latest version and so we can also define its function pointer as below:
    
    
    PFN_cuMemAlloc pfn_cuMemAlloc;
    

## 4.20.3. Driver Function Retrieval

Using the Driver Entry Point Access APIs and the appropriate typedef, we can get the function pointer to any CUDA driver API.

### 4.20.3.1. Using the Driver API

The driver API requires CUDA version as an argument to get the ABI compatible version for the requested driver symbol. CUDA Driver APIs have a per-function ABI denoted with a `_v*` extension. For example, consider the versions of `cuStreamBeginCapture` and their corresponding `typedefs` from `cudaTypedefs.h`:
    
    
    // cuda.h
    CUresult CUDAAPI cuStreamBeginCapture(CUstream hStream);
    CUresult CUDAAPI cuStreamBeginCapture_v2(CUstream hStream, CUstreamCaptureMode mode);
    
    // cudaTypedefs.h
    typedef CUresult (CUDAAPI *PFN_cuStreamBeginCapture_v10000)(CUstream hStream);
    typedef CUresult (CUDAAPI *PFN_cuStreamBeginCapture_v10010)(CUstream hStream, CUstreamCaptureMode mode);
    

From the above `typedefs` in the code snippet, version suffixes `_v10000` and `_v10010` indicate that the above APIs were introduced in CUDA 10.0 and CUDA 10.1 respectively.
    
    
    #include <cudaTypedefs.h>
    
    // Declare the entry points for cuStreamBeginCapture
    PFN_cuStreamBeginCapture_v10000 pfn_cuStreamBeginCapture_v1;
    PFN_cuStreamBeginCapture_v10010 pfn_cuStreamBeginCapture_v2;
    
    // Get the function pointer to the cuStreamBeginCapture driver symbol
    cuGetProcAddress("cuStreamBeginCapture", &pfn_cuStreamBeginCapture_v1, 10000, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
    // Get the function pointer to the cuStreamBeginCapture_v2 driver symbol
    cuGetProcAddress("cuStreamBeginCapture", &pfn_cuStreamBeginCapture_v2, 10010, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
    

Referring to the code snippet above, to retrieve the address to the `_v1` version of the driver API `cuStreamBeginCapture`, the CUDA version argument should be exactly 10.0 (10000). Similarly, the CUDA version for retrieving the address to the `_v2` version of the API should be 10.1 (10010). Specifying a higher CUDA version for retrieving a specific version of a driver API might not always be portable. For example, using 11030 here would still return the `_v2` symbol, but if a hypothetical `_v3` version is released in CUDA 11.3, the `cuGetProcAddress` API would start returning the newer `_v3` symbol instead when paired with a CUDA 11.3 driver. Since the ABI and function signatures of the `_v2` and `_v3` symbols might differ, calling the `_v3` function using the `_v10010` typedef intended for the `_v2` symbol would exhibit undefined behavior.

To retrieve the latest version of a driver API for a given CUDA Toolkit, we can also specify CUDA_VERSION as the `version` argument and use the unversioned typedef to define the function pointer. Since `_v2` is the latest version of the driver API `cuStreamBeginCapture` in CUDA 11.3, the below code snippet shows a different method to retrieve it.
    
    
    // Assuming we are using CUDA 11.3 Toolkit
    
    #include <cudaTypedefs.h>
    
    // Declare the entry point
    PFN_cuStreamBeginCapture pfn_cuStreamBeginCapture_latest;
    
    // Initialize the entry point. Specifying CUDA_VERSION will give the function pointer to the
    // cuStreamBeginCapture_v2 symbol since it is latest version on CUDA 11.3.
    cuGetProcAddress("cuStreamBeginCapture", &pfn_cuStreamBeginCapture_latest, CUDA_VERSION, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
    

Note that requesting a driver API with an invalid CUDA version will return an error `CUDA_ERROR_NOT_FOUND`. In the above code examples, passing in a version less than 10000 (CUDA 10.0) would be invalid.

### 4.20.3.2. Using the Runtime API

The runtime API `cudaGetDriverEntryPoint` uses the CUDA runtime version to get the ABI compatible version for the requested driver symbol. In the below code snippet, the minimum CUDA runtime version required would be CUDA 11.2 as `cuMemAllocAsync` was introduced then.
    
    
    #include <cudaTypedefs.h>
    
    // Declare the entry point
    PFN_cuMemAllocAsync pfn_cuMemAllocAsync;
    
    // Initialize the entry point. Assuming CUDA runtime version >= 11.2
    cudaGetDriverEntryPoint("cuMemAllocAsync", &pfn_cuMemAllocAsync, cudaEnableDefault, &driverStatus);
    
    // Call the entry point
    if(driverStatus == cudaDriverEntryPointSuccess && pfn_cuMemAllocAsync) {
        pfn_cuMemAllocAsync(...);
    }
    

The runtime API `cudaGetDriverEntryPointByVersion` uses the user provided CUDA version to get the ABI compatible version for the requested driver symbol. This allows more specific control over the requested ABI version.

### 4.20.3.3. Retrieve Per-thread Default Stream Versions

Some CUDA driver APIs can be configured to have _default stream_ or _per-thread default stream_ semantics. Driver APIs having _per-thread default stream_ semantics are suffixed with __ptsz_ or __ptds_ in their name. For example, `cuLaunchKernel` has a _per-thread default stream_ variant named `cuLaunchKernel_ptsz`. With the Driver Entry Point Access APIs, users can request for the _per-thread default stream_ version of the driver API `cuLaunchKernel` instead of the _default stream_ version. Configuring the CUDA driver APIs for _default stream_ or _per-thread default stream_ semantics affects the synchronization behavior. More details can be found [here](https://docs.nvidia.com/cuda/cuda-driver-api/stream-sync-behavior.html#stream-sync-behavior__default-stream).

The _default stream_ or _per-thread default stream_ versions of a driver API can be obtained by one of the following ways:

  * Use the compilation flag `--default-stream per-thread` or define the macro `CUDA_API_PER_THREAD_DEFAULT_STREAM` to get _per-thread default stream_ behavior.

  * Force _default stream_ or _per-thread default stream_ behavior using the flags `CU_GET_PROC_ADDRESS_LEGACY_STREAM/cudaEnableLegacyStream` or `CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM/cudaEnablePerThreadDefaultStream` respectively.


### 4.20.3.4. Access New CUDA features

It is always recommended to install the latest CUDA toolkit to access new CUDA driver features, but if for some reason, a user does not want to update or does not have access to the latest toolkit, the API can be used to access new CUDA features with only an updated CUDA driver. For discussion, let us assume the user is on CUDA 11.3 and wants to use a new driver API `cuFoo` available in the CUDA 12.0 driver. The below code snippet illustrates this use-case:
    
    
    int main()
    {
        // Assuming we have CUDA 12.0 driver installed.
    
        // Manually define the prototype as cudaTypedefs.h in CUDA 11.3 does not have the cuFoo typedef
        typedef CUresult (CUDAAPI *PFN_cuFoo)(...);
        PFN_cuFoo pfn_cuFoo = NULL;
        CUdriverProcAddressQueryResult driverStatus;
    
        // Get the address for cuFoo API using cuGetProcAddress. Specify CUDA version as
        // 12000 since cuFoo was introduced then or get the driver version dynamically
        // using cuDriverGetVersion
        int driverVersion;
        cuDriverGetVersion(&driverVersion);
        CUresult status = cuGetProcAddress("cuFoo", &pfn_cuFoo, driverVersion, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
    
        if (status == CUDA_SUCCESS && pfn_cuFoo) {
            pfn_cuFoo(...);
        }
        else {
            printf("Cannot retrieve the address to cuFoo - driverStatus = %d. Check if the latest driver for CUDA 12.0 is installed.\n", driverStatus);
            assert(0);
        }
    
        // rest of code here
    
    }
    

## 4.20.4. Potential Implications with cuGetProcAddress

Below is a set of concrete and theoretical examples of potential issues with `cuGetProcAddress` and `cudaGetDriverEntryPoint`.

### 4.20.4.1. Implications with cuGetProcAddress vs Implicit Linking

`cuDeviceGetUuid` was introduced in CUDA 9.2. This API has a newer revision (`cuDeviceGetUuid_v2`) introduced in CUDA 11.4. To preserve minor version compatibility, `cuDeviceGetUuid` will not be version bumped to `cuDeviceGetUuid_v2` in cuda.h until CUDA 12.0. This means that calling it by obtaining a function pointer to it via `cuGetProcAddress` might have different behavior. Example using the API directly:
    
    
    #include <cuda.h>
    
    CUuuid uuid;
    CUdevice dev;
    CUresult status;
    
    status = cuDeviceGet(&dev, 0); // Get device 0
    // handle status
    
    status = cuDeviceGetUuid(&uuid, dev) // Get uuid of device 0
    

In this example, assume the user is compiling with CUDA 11.4. Note that this will perform the behavior of `cuDeviceGetUuid`, not _v2 version. Now an example of using `cuGetProcAddress`:
    
    
    #include <cudaTypedefs.h>
    
    CUuuid uuid;
    CUdevice dev;
    CUresult status;
    CUdriverProcAddressQueryResult driverStatus;
    
    status = cuDeviceGet(&dev, 0); // Get device 0
    // handle status
    
    PFN_cuDeviceGetUuid pfn_cuDeviceGetUuid;
    status = cuGetProcAddress("cuDeviceGetUuid", &pfn_cuDeviceGetUuid, CUDA_VERSION, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
    if(CUDA_SUCCESS == status && pfn_cuDeviceGetUuid) {
        // pfn_cuDeviceGetUuid points to ???
    }
    

In this example, assume the user is compiling with CUDA 11.4. This will get the function pointer of `cuDeviceGetUuid_v2`. Calling the function pointer will then invoke the new _v2 function, not the same `cuDeviceGetUuid` as shown in the previous example.

### 4.20.4.2. Compile Time vs Runtime Version Usage in cuGetProcAddress

Let’s take the same issue and make one small tweak. The last example used the compile time constant of CUDA_VERSION to determine which function pointer to obtain. More complications arise if the user queries the driver version dynamically using `cuDriverGetVersion` or `cudaDriverGetVersion` to pass to `cuGetProcAddress`. Example:
    
    
    #include <cudaTypedefs.h>
    
    CUuuid uuid;
    CUdevice dev;
    CUresult status;
    int cudaVersion;
    CUdriverProcAddressQueryResult driverStatus;
    
    status = cuDeviceGet(&dev, 0); // Get device 0
    // handle status
    
    status = cuDriverGetVersion(&cudaVersion);
    // handle status
    
    PFN_cuDeviceGetUuid pfn_cuDeviceGetUuid;
    status = cuGetProcAddress("cuDeviceGetUuid", &pfn_cuDeviceGetUuid, cudaVersion, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
    if(CUDA_SUCCESS == status && pfn_cuDeviceGetUuid) {
        // pfn_cuDeviceGetUuid points to ???
    }
    

In this example, assume the user is compiling with CUDA 11.3. The user would debug, test, and deploy this application with the known behavior of getting `cuDeviceGetUuid` (not the _v2 version). Since CUDA has guaranteed ABI compatibility between minor versions, this same application is expected to run after the driver is upgraded to CUDA 11.4 (without updating the toolkit and runtime) without requiring recompilation. This will have undefined behavior though, because now the typedef for `PFN_cuDeviceGetUuid` will still be of the signature for the original version, but since `cudaVersion` would now be 11040 (CUDA 11.4), `cuGetProcAddress` would return the function pointer to the _v2 version, meaning calling it might have undefined behavior.

Note in this case the original (not the _v2 version) typedef looks like:
    
    
    typedef CUresult (CUDAAPI *PFN_cuDeviceGetUuid_v9020)(CUuuid *uuid, CUdevice_v1 dev);
    

But the _v2 version typedef looks like:
    
    
    typedef CUresult (CUDAAPI *PFN_cuDeviceGetUuid_v11040)(CUuuid *uuid, CUdevice_v1 dev);
    

So in this case, the API/ABI is going to be the same and the runtime API call will likely not cause issues–only the potential for unknown uuid return. In [Implications to API/ABI](#implications-to-api-abi), we discuss a more problematic case of API/ABI compatibility.

### 4.20.4.3. API Version Bumps with Explicit Version Checks

Above, was a specific concrete example. Now for instance let’s use a theoretical example that still has issues with compatibility across driver versions. Example:
    
    
    CUresult cuFoo(int bar); // Introduced in CUDA 11.4
    CUresult cuFoo_v2(int bar); // Introduced in CUDA 11.5
    CUresult cuFoo_v3(int bar, void* jazz); // Introduced in CUDA 11.6
    
    typedef CUresult (CUDAAPI *PFN_cuFoo_v11040)(int bar);
    typedef CUresult (CUDAAPI *PFN_cuFoo_v11050)(int bar);
    typedef CUresult (CUDAAPI *PFN_cuFoo_v11060)(int bar, void* jazz);
    

Notice that the API has been modified twice since original creation in CUDA 11.4 and the latest in CUDA 11.6 also modified the API/ABI interface to the function. The usage in user code compiled against CUDA 11.5 is:
    
    
    #include <cuda.h>
    #include <cudaTypedefs.h>
    
    CUresult status;
    int cudaVersion;
    CUdriverProcAddressQueryResult driverStatus;
    
    status = cuDriverGetVersion(&cudaVersion);
    // handle status
    
    PFN_cuFoo_v11040 pfn_cuFoo_v11040;
    PFN_cuFoo_v11050 pfn_cuFoo_v11050;
    if(cudaVersion < 11050 ) {
        // We know to get the CUDA 11.4 version
        status = cuGetProcAddress("cuFoo", &pfn_cuFoo_v11040, cudaVersion, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
        // Handle status and validating pfn_cuFoo_v11040
    }
    else {
        // Assume >= CUDA 11.5 version we can use the second version
        status = cuGetProcAddress("cuFoo", &pfn_cuFoo_v11050, cudaVersion, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
        // Handle status and validating pfn_cuFoo_v11050
    }
    

In this example, without updates for the new typedef in CUDA 11.6 and recompiling the application with those new typedefs and case handling, the application will get the cuFoo_v3 function pointer returned and any usage of that function would then cause undefined behavior. The point of this example was to illustrate that even explicit version checks for `cuGetProcAddress` may not safely cover the minor version bumps within a CUDA major release.

### 4.20.4.4. Issues with Runtime API Usage

The above examples were focused on the issues with the Driver API usage for obtaining the function pointers to driver APIs. Now we will discuss the potential issues with the Runtime API usage for `cudaApiGetDriverEntryPoint`.

We will start by using the Runtime APIs similar to the above.
    
    
    #include <cuda.h>
    #include <cudaTypedefs.h>
    #include <cuda_runtime.h>
    
    CUresult status;
    cudaError_t error;
    int driverVersion, runtimeVersion;
    CUdriverProcAddressQueryResult driverStatus;
    
    // Ask the runtime for the function
    PFN_cuDeviceGetUuid pfn_cuDeviceGetUuidRuntime;
    error = cudaGetDriverEntryPoint ("cuDeviceGetUuid", &pfn_cuDeviceGetUuidRuntime, cudaEnableDefault, &driverStatus);
    if(cudaSuccess == error && pfn_cuDeviceGetUuidRuntime) {
        // pfn_cuDeviceGetUuid points to ???
    }
    

The function pointer in this example is even more complicated than the driver only examples above because there is no control over which version of the function to obtain; it will always get the API for the current CUDA Runtime version. See the following table for more information:

| Static Runtime Version Linkage  
---|---  
Driver Version Installed | **V11.3** | **V11.4**  
**V11.3** | v1 | v1x  
**V11.4** | v1 | v2  
      
    
    V11.3 => 11.3 CUDA Runtime and Toolkit (includes header files cuda.h and cudaTypedefs.h)
    V11.4 => 11.4 CUDA Runtime and Toolkit (includes header files cuda.h and cudaTypedefs.h)
    v1 => cuDeviceGetUuid
    v2 => cuDeviceGetUuid_v2
    
    x => Implies the typedef function pointer won't match the returned
         function pointer.  In these cases, the typedef at compile time
         using a CUDA 11.4 runtime, would match the _v2 version, but the
         returned function pointer would be the original (non _v2) function.
    

The problem in the table comes in with a newer CUDA 11.4 Runtime and Toolkit and older driver (CUDA 11.3) combination, labeled as v1x in the above. This combination would have the driver returning the pointer to the older function (non _v2), but the typedef used in the application would be for the new function pointer.

### 4.20.4.5. Issues with Runtime API and Dynamic Versioning

More complications arise when we consider different combinations of the CUDA version with which an application is compiled, CUDA runtime version, and CUDA driver version that an application dynamically links against.
    
    
    #include <cuda.h>
    #include <cudaTypedefs.h>
    #include <cuda_runtime.h>
    
    CUresult status;
    cudaError_t error;
    int driverVersion, runtimeVersion;
    CUdriverProcAddressQueryResult driverStatus;
    enum cudaDriverEntryPointQueryResult runtimeStatus;
    
    PFN_cuDeviceGetUuid pfn_cuDeviceGetUuidDriver;
    status = cuGetProcAddress("cuDeviceGetUuid", &pfn_cuDeviceGetUuidDriver, CUDA_VERSION, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
    if(CUDA_SUCCESS == status && pfn_cuDeviceGetUuidDriver) {
        // pfn_cuDeviceGetUuidDriver points to ???
    }
    
    // Ask the runtime for the function
    PFN_cuDeviceGetUuid pfn_cuDeviceGetUuidRuntime;
    error = cudaGetDriverEntryPoint ("cuDeviceGetUuid", &pfn_cuDeviceGetUuidRuntime, cudaEnableDefault, &runtimeStatus);
    if(cudaSuccess == error && pfn_cuDeviceGetUuidRuntime) {
        // pfn_cuDeviceGetUuidRuntime points to ???
    }
    
    // Ask the driver for the function based on the driver version (obtained via runtime)
    error = cudaDriverGetVersion(&driverVersion);
    PFN_cuDeviceGetUuid pfn_cuDeviceGetUuidDriverDriverVer;
    status = cuGetProcAddress ("cuDeviceGetUuid", &pfn_cuDeviceGetUuidDriverDriverVer, driverVersion, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
    if(CUDA_SUCCESS == status && pfn_cuDeviceGetUuidDriverDriverVer) {
        // pfn_cuDeviceGetUuidDriverDriverVer points to ???
    }
    

The following matrix of function pointers is expected:

**Function Pointer** | **Application Compiled/Runtime Dynamic Linked Version/Driver Version**  
---|---  
**(3 = > CUDA 11.3 and 4 => CUDA 11.4)**  
**3/3/3** | **3/3/4** | **3/4/3** | **3/4/4** | **4/3/3** | **4/3/4** | **4/4/3** | **4/4/4**  
`pfn_cuDeviceGetUuidDriver` | t1/v1 | t1/v1 | t1/v1 | t1/v1 | N/A | N/A | **t2/v1** | t2/v2  
`pfn_cuDeviceGetUuidRuntime` | t1/v1 | t1/v1 | t1/v1 | **t1/v2** | N/A | N/A | **t2/v1** | t2/v2  
`pfn_cuDeviceGetUuidDriverDriverVer` | t1/v1 | **t1/v2** | t1/v1 | **t1/v2** | N/A | N/A | **t2/v1** | t2/v2  
      
    
    tX -> Typedef version used at compile time
    vX -> Version returned/used at runtime
    

If the application is compiled against CUDA Version 11.3, it would have the typedef for the original function, but if compiled against CUDA Version 11.4, it would have the typedef for the _v2 function. Because of that, notice the number of cases where the typedef does not match the actual version returned/used.

### 4.20.4.6. Issues with Runtime API allowing CUDA Version

Unless specified otherwise, the CUDA runtime API `cudaGetDriverEntryPointByVersion` will have similar implications as the driver entry point `cuGetProcAddress` since it allows for the user to request a specific CUDA driver version.

### 4.20.4.7. Implications to API/ABI

In the above examples using `cuDeviceGetUuid`, the implications of the mismatched API are minimal, and may not be entirely noticeable to many users as the _v2 was added to support Multi-Instance GPU (MIG) mode. So, on a system without MIG, the user might not even realize they are getting a different API.

More problematic is an API which changes its application signature (and hence ABI) such as `cuCtxCreate`. The _v2 version, introduced in CUDA 3.2 is currently used as the default `cuCtxCreate` when using `cuda.h` but now has a newer version introduced in CUDA 11.4 (`cuCtxCreate_v3`). The API signature has been modified as well, and now takes extra arguments. So, in some of the cases above, where the typedef to the function pointer doesn’t match the returned function pointer, there is a chance for non-obvious ABI incompatibility which would lead to undefined behavior.

For example, assume the following code compiled against a CUDA 11.3 toolkit with a CUDA 11.4 driver installed:
    
    
    PFN_cuCtxCreate cuUnknown;
    CUdriverProcAddressQueryResult driverStatus;
    
    status = cuGetProcAddress("cuCtxCreate", (void**)&cuUnknown, cudaVersion, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
    if(CUDA_SUCCESS == status && cuUnknown) {
        status = cuUnknown(&ctx, 0, dev);
    }
    

Running this code where `cudaVersion` is set to anything >=11040 (indicating CUDA 11.4) could have undefined behavior due to not having adequately supplied all the parameters required for the _v3 version of the `cuCtxCreate_v3` API.

## 4.20.5. Determining cuGetProcAddress Failure Reasons

There are two types of errors with cuGetProcAddress. Those are (1) API/usage errors and (2) inability to find the driver API requested. The first error type will return error codes from the API via the CUresult return value. Things like passing NULL as the `pfn` variable or passing invalid `flags`.

The second error type encodes in the `CUdriverProcAddressQueryResult *symbolStatus` and can be used to help distinguish potential issues with the driver not being able to find the symbol requested. Take the following example:
    
    
    // cuDeviceGetExecAffinitySupport was introduced in release CUDA 11.4
    #include <cuda.h>
    CUdriverProcAddressQueryResult driverStatus;
    cudaVersion = ...;
    status = cuGetProcAddress("cuDeviceGetExecAffinitySupport", &pfn, cudaVersion, 0, &driverStatus);
    if (CUDA_SUCCESS == status) {
        if (CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT == driverStatus) {
            printf("We can use the new feature when you upgrade cudaVersion to 11.4, but CUDA driver is good to go!\n");
            // Indicating cudaVersion was < 11.4 but run against a CUDA driver >= 11.4
        }
        else if (CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND == driverStatus) {
            printf("Please update both CUDA driver and cudaVersion to at least 11.4 to use the new feature!\n");
            // Indicating driver is < 11.4 since string not found, doesn't matter what cudaVersion was
        }
        else if (CU_GET_PROC_ADDRESS_SUCCESS == driverStatus && pfn) {
            printf("You're using cudaVersion and CUDA driver >= 11.4, using new feature!\n");
            pfn();
        }
    }
    

The first case with the return code `CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT` indicates that the `symbol` was found when searching in the CUDA driver but it was added later than the `cudaVersion` supplied. In the example, specifying `cudaVersion` as anything 11030 or less and when running against a CUDA driver >= CUDA 11.4 would give this result of `CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT`. This is because `cuDeviceGetExecAffinitySupport` was added in CUDA 11.4 (11040).

The second case with the return code `CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND` indicates that the `symbol` was not found when searching in the CUDA driver. This can be due to a few reasons such as unsupported CUDA function due to older driver as well as just having a typo. In the latter, similar to the last example if the user had put `symbol` as CUDeviceGetExecAffinitySupport - notice the capital CU to start the string - `cuGetProcAddress` would not be able to find the API because the string doesn’t match. In the former case an example might be the user developing an application against a CUDA driver supporting the new API, and deploying the application against an older CUDA driver. Using the last example, if the developer developed against CUDA 11.4 or later but was deployed against a CUDA 11.3 driver, during their development they may have had a successful `cuGetProcAddress`, but when deploying an application running against a CUDA 11.3 driver the call would no longer work with the `CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND` returned in `driverStatus`.
