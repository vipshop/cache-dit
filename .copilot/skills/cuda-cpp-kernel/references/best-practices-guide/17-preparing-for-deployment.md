# 17. Preparing for Deployment


##  17.1. Testing for CUDA Availability 

When deploying a CUDA application, it is often desirable to ensure that the application will continue to function properly even if the target machine does not have a CUDA-capable GPU and/or a sufficient version of the NVIDIA Driver installed. (Developers targeting a single machine with known configuration may choose to skip this section.)

**Detecting a CUDA-Capable GPU**

When an application will be deployed to target machines of arbitrary/unknown configuration, the application should explicitly test for the existence of a CUDA-capable GPU in order to take appropriate action when no such device is available. The `cudaGetDeviceCount()` function can be used to query for the number of available devices. Like all CUDA Runtime API functions, this function will fail gracefully and return `cudaErrorNoDevice` to the application if there is no CUDA-capable GPU or `cudaErrorInsufficientDriver` if there is not an appropriate version of the NVIDIA Driver installed. If `cudaGetDeviceCount()` reports an error, the application should fall back to an alternative code path.

A system with multiple GPUs may contain GPUs of different hardware versions and capabilities. When using multiple GPUs from the same application, it is recommended to use GPUs of the same type, rather than mixing hardware generations. The `cudaChooseDevice()` function can be used to select the device that most closely matches a desired set of features.

**Detecting Hardware and Software Configuration**

When an application depends on the availability of certain hardware or software capabilities to enable certain functionality, the CUDA API can be queried for details about the configuration of the available device and for the installed software versions.

The `cudaGetDeviceProperties()` function reports various features of the available devices, including the [CUDA Compute Capability](#cuda-compute-capability) of the device (see also the Compute Capabilities section of the CUDA C++ Programming Guide). See [Version Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART____VERSION.html#group__CUDART____VERSION) for details on how to query the available CUDA software API versions.


##  17.2. Error Handling 

All CUDA Runtime API calls return an error code of type `cudaError_t`; the return value will be equal to `cudaSuccess` if no errors have occurred. (The exceptions to this are kernel launches, which return void, and `cudaGetErrorString()`, which returns a character string describing the `cudaError_t` code that was passed into it.) The CUDA Toolkit libraries (`cuBLAS`, `cuFFT`, etc.) likewise return their own sets of error codes.

Since some CUDA API calls and all kernel launches are asynchronous with respect to the host code, errors may be reported to the host asynchronously as well; often this occurs the next time the host and device synchronize with each other, such as during a call to `cudaMemcpy()` or to `cudaDeviceSynchronize()`.

Always check the error return values on all CUDA API functions, even for functions that are not expected to fail, as this will allow the application to detect and recover from errors as soon as possible should they occur. To check for errors occurring during kernel launches using the `<<<...>>>` syntax, which does not return any error code, the return code of `cudaGetLastError()` should be checked immediately after the kernel launch. Applications that do not check for CUDA API errors could at times run to completion without having noticed that the data calculated by the GPU is incomplete, invalid, or uninitialized.

Note

The CUDA Toolkit Samples provide several helper functions for error checking with the various CUDA APIs; these helper functions are located in the `samples/common/inc/helper_cuda.h` file in the CUDA Toolkit.


##  17.3. Building for Maximum Compatibility 

Each generation of CUDA-capable device has an associated _compute capability_ version that indicates the feature set supported by the device (see [CUDA Compute Capability](#cuda-compute-capability)). One or more compute capability versions can be specified to the nvcc compiler while building a file; compiling for the native compute capability for the target GPU(s) of the application is important to ensure that application kernels achieve the best possible performance and are able to use the features that are available on a given generation of GPU.

When an application is built for multiple compute capabilities simultaneously (using several instances of the `-gencode` flag to nvcc), the binaries for the specified compute capabilities are combined into the executable, and the CUDA Driver selects the most appropriate binary at runtime according to the compute capability of the present device. If an appropriate native binary (_cubin_) is not available, but the intermediate _PTX_ code (which targets an abstract virtual instruction set and is used for forward-compatibility) is available, then the kernel will be compiled _Just In Time_ (JIT) (see [Compiler JIT Cache Management Tools](#compiler-jit-cache-management)) from the PTX to the native cubin for the device. If the PTX is also not available, then the kernel launch will fail.

**Windows**
    
    
    nvcc.exe -ccbin "C:\vs2008\VC\bin"
      -Xcompiler "/EHsc /W3 /nologo /O2 /Zi /MT"
      -gencode=arch=compute_30,code=sm_30
      -gencode=arch=compute_35,code=sm_35
      -gencode=arch=compute_50,code=sm_50
      -gencode=arch=compute_60,code=sm_60
      -gencode=arch=compute_70,code=sm_70
      -gencode=arch=compute_75,code=sm_75
      -gencode=arch=compute_75,code=compute_75
      --compile -o "Release\mykernel.cu.obj" "mykernel.cu"
    

**Mac/Linux**
    
    
    /usr/local/cuda/bin/nvcc
      -gencode=arch=compute_30,code=sm_30
      -gencode=arch=compute_35,code=sm_35
      -gencode=arch=compute_50,code=sm_50
      -gencode=arch=compute_60,code=sm_60
      -gencode=arch=compute_70,code=sm_70
      -gencode=arch=compute_75,code=sm_75
      -gencode=arch=compute_75,code=compute_75
      -O2 -o mykernel.o -c mykernel.cu
    

Alternatively, the `nvcc` command-line option `-arch=sm_XX` can be used as a shorthand equivalent to the following more explicit `-gencode=` command-line options described above:
    
    
    -gencode=arch=compute_XX,code=sm_XX
    -gencode=arch=compute_XX,code=compute_XX
    

However, while the `-arch=sm_XX` command-line option does result in inclusion of a PTX back-end target by default (due to the `code=compute_XX` target it implies), it can only specify a single target `cubin` architecture at a time, and it is not possible to use multiple `-arch=` options on the same `nvcc` command line, which is why the examples above use `-gencode=` explicitly.


##  17.4. Distributing the CUDA Runtime and Libraries 

CUDA applications are built against the CUDA Runtime library, which handles device, memory, and kernel management. Unlike the CUDA Driver, the CUDA Runtime guarantees neither forward nor backward binary compatibility across versions. It is therefore best to [redistribute](#redistribution) the CUDA Runtime library with the application when using dynamic linking or else to statically link against the CUDA Runtime. This will ensure that the executable will be able to run even if the user does not have the same CUDA Toolkit installed that the application was built against.

Note

When statically linking to the CUDA Runtime, multiple versions of the runtime can peacably coexist in the same application process simultaneously; for example, if an application uses one version of the CUDA Runtime, and a plugin to that application is statically linked to a different version, that is perfectly acceptable, as long as the installed NVIDIA Driver is sufficient for both.

Statically-linked CUDA Runtime

The easiest option is to statically link against the CUDA Runtime. This is the default if using `nvcc` to link in CUDA 5.5 and later. Static linking makes the executable slightly larger, but it ensures that the correct version of runtime library functions are included in the application binary without requiring separate redistribution of the CUDA Runtime library.

Dynamically-linked CUDA Runtime

If static linking against the CUDA Runtime is impractical for some reason, then a dynamically-linked version of the CUDA Runtime library is also available. (This was the default and only option provided in CUDA versions 5.0 and earlier.)

To use dynamic linking with the CUDA Runtime when using the `nvcc` from CUDA 5.5 or later to link the application, add the `--cudart=shared` flag to the link command line; otherwise the [statically-linked CUDA Runtime library](#statically-linked-cuda-runtime) is used by default.

After the application is dynamically linked against the CUDA Runtime, this version of the runtime library should be [bundled with](#redistribution) the application. It can be copied into the same directory as the application executable or into a subdirectory of that installation path.

Other CUDA Libraries

Although the CUDA Runtime provides the option of static linking, some libraries included in the CUDA Toolkit are available only in dynamically-linked form. As with the [dynamically-linked version of the CUDA Runtime library](#dynamically-linked-cuda-runtime), these libraries should be [bundled with](#redistribution) the application executable when distributing that application.

###  17.4.1. CUDA Toolkit Library Redistribution 

The CUDA Toolkit’s End-User License Agreement (EULA) allows for redistribution of many of the CUDA libraries under certain terms and conditions. This allows applications that depend on these libraries [to redistribute the exact versions](#redistribution-which-files) of the libraries against which they were built and tested, thereby avoiding any trouble for end users who might have a different version of the CUDA Toolkit (or perhaps none at all) installed on their machines. Please refer to the EULA for details.

Note

This does _not_ apply to the NVIDIA Driver; the end user must still download and install an NVIDIA Driver appropriate to their GPU(s) and operating system.

####  17.4.1.1. Which Files to Redistribute 

When redistributing the dynamically-linked versions of one or more CUDA libraries, it is important to identify the exact files that need to be redistributed. The following examples use the cuBLAS library from CUDA Toolkit 5.5 as an illustration:

**Linux**

In a shared library on Linux, there is a string field called the `SONAME` that indicates the binary compatibility level of the library. The `SONAME` of the library against which the application was built must match the filename of the library that is redistributed with the application.

For example, in the standard CUDA Toolkit installation, the files `libcublas.so` and `libcublas.so.5.5` are both symlinks pointing to a specific build of cuBLAS, which is named like `libcublas.so.5.5.x`, where _x_ is the build number (e.g., `libcublas.so.5.5.17`). However, the `SONAME` of this library is given as “`libcublas.so.5.5`”:
    
    
    $ objdump -p /usr/local/cuda/lib64/libcublas.so | grep SONAME
       SONAME               libcublas.so.5.5
    

Because of this, even if `-lcublas` (with no version number specified) is used when linking the application, the `SONAME` found at link time implies that “`libcublas.so.5.5`” is the name of the file that the dynamic loader will look for when loading the application and therefore must be the name of the file (or a symlink to the same) that is redistributed with the application.

The `ldd` tool is useful for identifying the exact filenames of the libraries that the application expects to find at runtime as well as the path, if any, of the copy of that library that the dynamic loader would select when loading the application given the current library search path:
    
    
    $ ldd a.out | grep libcublas
       libcublas.so.5.5 => /usr/local/cuda/lib64/libcublas.so.5.5
    

**Mac**

In a shared library on Mac OS X, there is a field called the `install name` that indicates the expected installation path and filename the library; the CUDA libraries also use this filename to indicate binary compatibility. The value of this field is propagated into an application built against the library and is used to locate the library of the correct version at runtime.

For example, if the install name of the cuBLAS library is given as `@rpath/libcublas.5.5.dylib`, then the library is version 5.5 and the copy of this library redistributed with the application must be named `libcublas.5.5.dylib`, even though only `-lcublas` (with no version number specified) is used at link time. Furthermore, this file should be installed into the `@rpath` of the application; see [Where to Install Redistributed CUDA Libraries](#redistribution-where-to-install).

To view a library’s install name, use the `otool -L` command:
    
    
    $ otool -L a.out
    a.out:
            @rpath/libcublas.5.5.dylib (...)
    

**Windows**

The binary compatibility version of the CUDA libraries on Windows is indicated as part of the filename.

For example, a 64-bit application linked to cuBLAS 5.5 will look for `cublas64_55.dll` at runtime, so this is the file that should be redistributed with that application, even though `cublas.lib` is the file that the application is linked against. For 32-bit applications, the file would be `cublas32_55.dll`.

To verify the exact DLL filename that the application expects to find at runtime, use the `dumpbin` tool from the Visual Studio command prompt:
    
    
    $ dumpbin /IMPORTS a.exe
    Microsoft (R) COFF/PE Dumper Version 10.00.40219.01
    Copyright (C) Microsoft Corporation.  All rights reserved.
    
    
    Dump of file a.exe
    
    File Type: EXECUTABLE IMAGE
    
      Section contains the following imports:
    
        ...
        cublas64_55.dll
        ...
    

####  17.4.1.2. Where to Install Redistributed CUDA Libraries 

Once the correct library files are identified for redistribution, they must be configured for installation into a location where the application will be able to find them.

On Windows, if the CUDA Runtime or other dynamically-linked CUDA Toolkit library is placed in the same directory as the executable, Windows will locate it automatically. On Linux and Mac, the `-rpath` linker option should be used to instruct the executable to search its local path for these libraries before searching the system paths:

**Linux/Mac**
    
    
    nvcc -I $(CUDA_HOME)/include
      -Xlinker "-rpath '$ORIGIN'" --cudart=shared
      -o myprogram myprogram.cu
    

**Windows**
    
    
    nvcc.exe -ccbin "C:\vs2008\VC\bin"
      -Xcompiler "/EHsc /W3 /nologo /O2 /Zi /MT" --cudart=shared
      -o "Release\myprogram.exe" "myprogram.cu"
    

Note

It may be necessary to adjust the value of `-ccbin` to reflect the location of your Visual Studio installation.

To specify an alternate path where the libraries will be distributed, use linker options similar to those below:

**Linux/Mac**
    
    
    nvcc -I $(CUDA_HOME)/include
      -Xlinker "-rpath '$ORIGIN/lib'" --cudart=shared
      -o myprogram myprogram.cu
    

**Windows**
    
    
    nvcc.exe -ccbin "C:\vs2008\VC\bin"
      -Xcompiler "/EHsc /W3 /nologo /O2 /Zi /MT /DELAY" --cudart=shared
      -o "Release\myprogram.exe" "myprogram.cu"
    

For Linux and Mac, the `-rpath` option is used as before. For Windows, the `/DELAY` option is used; this requires that the application call `SetDllDirectory()` before the first call to any CUDA API function in order to specify the directory containing the CUDA DLLs.

Note

For Windows 8, `SetDefaultDLLDirectories()` and `AddDllDirectory()` should be used instead of `SetDllDirectory()`. Please see the MSDN documentation for these routines for more information.
