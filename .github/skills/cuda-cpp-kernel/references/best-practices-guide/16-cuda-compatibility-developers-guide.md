# 16. CUDA Compatibility Developer’s Guide


CUDA Toolkit is released on a monthly release cadence to deliver new features, performance improvements, and critical bug fixes. CUDA compatibility allows users to update the latest CUDA Toolkit software (including the compiler, libraries, and tools) without requiring update to the entire driver stack.


The CUDA software environment consists of three parts:


  * CUDA Toolkit (libraries, CUDA runtime and developer tools) - SDK for developers to build CUDA applications.

  * CUDA driver - User-mode driver component used to run CUDA applications (e.g. libcuda.so on Linux systems).

  * NVIDIA GPU device driver - Kernel-mode driver component for NVIDIA GPUs.


On Linux systems, the CUDA driver and kernel mode components are delivered together in the NVIDIA display driver package. This is shown in Figure 1.


![Components of CUDA](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/_images/CUDA-components.png)

Figure 17 Components of CUDA


The CUDA compiler (nvcc), provides a way to handle CUDA and non-CUDA code (by splitting and steering compilation), along with the CUDA runtime, is part of the CUDA compiler toolchain. The CUDA Runtime API provides developers with high-level C++ interface for simplified management of devices, kernel executions etc., While the CUDA driver API provides ([CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/index.html)) a low-level programming interface for applications to target NVIDIA hardware.


Built on top of these technologies are CUDA libraries, some of which are included in the CUDA Toolkit, while others such as cuDNN may be released independently of the CUDA Toolkit.


##  16.1. CUDA Toolkit Versioning 

Starting with CUDA 11, the toolkit versions are based on an industry-standard semantic versioning scheme: .X.Y.Z, where:

  * .X stands for the major version - APIs have changed and binary compatibility is broken.

  * .Y stands for the minor version - Introduction of new APIs, deprecation of old APIs, and source compatibility might be broken but binary compatibility is maintained.

  * .Z stands for the release/patch version - new updates and patches will increment this.


Each component in the toolkit is recommended to be semantically versioned. From CUDA 11.3 NVRTC is also semantically versioned. We will note some of them later on in the document. The versions of the components in the toolkit are available in this [table](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions).

Compatibility of the CUDA platform is thus intended to address a few scenarios:

  1. NVIDIA driver upgrades to systems with GPUs running in production for enterprises or datacenters can be complex and may need advance planning. Delays in rolling out new NVIDIA drivers could mean that users of such systems may not have access to new features available in CUDA releases. Not requiring driver updates for new CUDA releases can mean that new versions of the software can be made available faster to users.

  2. Many software libraries and applications built on top of CUDA (e.g. math libraries or deep learning frameworks) do not have a direct dependency on the CUDA runtime, compiler or driver. In such cases, users or developers can still benefit from not having to upgrade the entire CUDA Toolkit or driver to use these libraries or frameworks.

  3. Upgrading dependencies is error-prone and time consuming, and in some corner cases, can even change the semantics of a program. Constantly recompiling with the latest CUDA Toolkit means forcing upgrades on the end-customers of an application product. Package managers facilitate this process but unexpected issues can still arise and if a bug is found, it necessitates a repeat of the above upgrade process.


CUDA supports several compatibility choices:

  1. First introduced in CUDA 10, the **CUDA Forward Compatible Upgrade** is designed to allow users to get access to new CUDA features and run applications built with new CUDA releases on systems with older installations of the NVIDIA datacenter driver.

  2. First introduced in CUDA 11.1, **CUDA Enhanced Compatibility** provides two benefits:

     * By leveraging semantic versioning across components in the CUDA Toolkit, an application can be built for one CUDA minor release (for example 11.1) and work across all future minor releases within the major family (i.e. 11.x).

     * The CUDA runtime has relaxed the minimum driver version check and thus no longer requires a driver upgrade when moving to a new minor release.

  3. The CUDA driver ensures backward Binary Compatibility is maintained for compiled CUDA applications. Applications compiled with CUDA toolkit versions as old as 3.2 will run on newer drivers.


##  16.2. Source Compatibility 

We define source compatibility as a set of guarantees provided by the library, where a well-formed application built against a specific version of the library (using the SDK) will continue to build and run without errors when a newer version of the SDK is installed.

Both the CUDA driver and the CUDA runtime are not source compatible across the different SDK releases. APIs can be deprecated and removed. Therefore, an application that compiled successfully on an older version of the toolkit may require changes in order to compile against a newer version of the toolkit.

Developers are notified through deprecation and documentation mechanisms of any current or upcoming changes. This does not mean that application binaries compiled using an older toolkit will not be supported anymore. Application binaries rely on CUDA Driver API interface and even though the CUDA Driver API itself may also have changed across toolkit versions, CUDA guarantees Binary Compatibility of the CUDA Driver API interface.


##  16.3. Binary Compatibility 

We define binary compatibility as a set of guarantees provided by the library, where an application targeting the said library will continue to work when dynamically linked against a different version of the library.

The CUDA Driver API has a versioned C-style ABI, which guarantees that applications that were running against an older driver (for example CUDA 3.2) will still run and function correctly against a modern driver (for example one shipped with CUDA 11.0). This means that even though an application source might need to be changed if it has to be recompiled against a newer CUDA Toolkit in order to use the newer features, replacing the driver components installed in a system with a newer version will always support existing applications and its functions.

The CUDA Driver API thus is binary-compatible (the OS loader can pick up a newer version and the application continues to work) but not source-compatible (rebuilding your application against a newer SDK might require source changes).

![CUDA Toolkit and Minimum Driver Versions](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/_images/CTK-and-min-driver-versions.png)

Figure 18 CUDA Toolkit and Minimum Driver Versions

Before we proceed further on this topic, it’s important for developers to understand the concept of Minimum Driver Version and how that may affect them.

Each version of the CUDA Toolkit (and runtime) requires a minimum version of the NVIDIA driver. Applications compiled against a CUDA Toolkit version will only run on systems with the specified minimum driver version for that toolkit version. Prior to CUDA 11.0, the minimum driver version for a toolkit was the same as the driver shipped with that version of the CUDA Toolkit.

So, when an application is built with CUDA 11.0, it can only run on a system with an R450 or later driver. If such an application is run on a system with the R418 driver installed, CUDA initialization will return an error as can be seen in the example below.

In this example, the deviceQuery sample is compiled with CUDA 11.1 and is run on a system with R418. In this scenario, CUDA initialization returns an error due to the minimum driver requirement.
    
    
    ubuntu@:~/samples/1_Utilities/deviceQuery
    $ make
    /usr/local/cuda-11.1/bin/nvcc -ccbin g++ -I../../common/inc -m64 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86 -o deviceQuery.o -c deviceQuery.cpp
    
    /usr/local/cuda-11.1/bin/nvcc -ccbin g++ -m64 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86 -o deviceQuery deviceQuery.o
    
    $ nvidia-smi
    
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 418.165.02   Driver Version: 418.165.02   CUDA Version: 10.1     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  Tesla T4            On   | 00000000:00:1E.0 Off |                    0 |
    | N/A   42C    P0    28W /  70W |      0MiB / 15079MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    
    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+
    
    
    $ samples/bin/x86_64/linux/release/deviceQuery
    samples/bin/x86_64/linux/release/deviceQuery Starting...
    
     CUDA Device Query (Runtime API) version (CUDART static linking)
    
    cudaGetDeviceCount returned 3
    -> initialization error
    Result = FAIL
    

Refer to the [CUDA Toolkit Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) for details for the minimum driver version and the version of the driver shipped with the toolkit.

###  16.3.1. CUDA Binary (cubin) Compatibility 

A slightly related but important topic is one of application binary compatibility across GPU architectures in CUDA.

CUDA C++ provides a simple path for users familiar with the C++ programming language to easily write programs for execution by the device. Kernels can be written using the CUDA instruction set architecture, called PTX, which is described in the PTX reference manual. It is however usually more effective to use a high-level programming language such as C++. In both cases, kernels must be compiled into binary code by nvcc (called cubins) to execute on the device.

The cubins are architecture-specific. Binary compatibility for cubins is guaranteed from one compute capability minor revision to the next one, but not from one compute capability minor revision to the previous one or across major compute capability revisions. In other words, a cubin object generated for compute capability _X.y_ will only execute on devices of compute capability _X.z_ where _z≥y_.

To execute code on devices of specific compute capability, an application must load binary or PTX code that is compatible with this compute capability. For portability, that is, to be able to execute code on future GPU architectures with higher compute capability (for which no binary code can be generated yet), an application must load PTX code that will be just-in-time compiled by the NVIDIA driver for these future devices.

More information on cubins, PTX and application compatibility can be found in the [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#binary-compatibility).


##  16.4. CUDA Compatibility Across Minor Releases 

By leveraging the semantic versioning, starting with CUDA 11, components in the CUDA Toolkit will remain binary compatible across the minor versions of the toolkit. In order to maintain binary compatibility across minor versions, the CUDA runtime no longer bumps up the minimum driver version required for every minor release - this only happens when a major release is shipped.

One of the main reasons a new toolchain requires a new minimum driver is to handle the JIT compilation of PTX code and the JIT linking of binary code.

In this section, we will review the usage patterns that may require new user workflows when taking advantage of the compatibility features of the CUDA platform.

###  16.4.1. Existing CUDA Applications within Minor Versions of CUDA 
    
    
    $ nvidia-smi
    
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 450.80.02    Driver Version: 450.80.02    CUDA Version: 11.0     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  Tesla T4            On   | 00000000:00:1E.0 Off |                    0 |
    | N/A   39C    P8     9W /  70W |      0MiB / 15109MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
    
    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+
    

When our CUDA 11.1 application (i.e. cudart 11.1 is statically linked) is run on the system, we see that it runs successfully even when the driver reports a 11.0 version - that is, without requiring the driver or other toolkit components to be updated on the system.
    
    
    $ samples/bin/x86_64/linux/release/deviceQuery
    samples/bin/x86_64/linux/release/deviceQuery Starting...
    
     CUDA Device Query (Runtime API) version (CUDART static linking)
    
    Detected 1 CUDA Capable device(s)
    
    Device 0: "Tesla T4"
      CUDA Driver Version / Runtime Version          11.0 / 11.1
      CUDA Capability Major/Minor version number:    7.5
    
      ...<snip>...
    
    deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 11.0, CUDA Runtime Version = 11.1, NumDevs = 1
    Result = PASS
    

By using new CUDA versions, users can benefit from new CUDA programming model APIs, compiler optimizations and math library features.

The following sections discuss some caveats and considerations.

####  16.4.1.1. Handling New CUDA Features and Driver APIs 

A subset of CUDA APIs don’t need a new driver and they can all be used without any driver dependencies. For example, `cuMemMap` APIs or any of APIs introduced prior to CUDA 11.0, such as `cudaDeviceSynchronize`, do not require a driver upgrade. To use other CUDA APIs introduced in a minor release (that require a new driver), one would have to implement fallbacks or fail gracefully. This situation is not different from what is available today where developers use macros to compile out features based on CUDA versions. Users should refer to the CUDA headers and documentation for new CUDA APIs introduced in a release.

When working with a feature exposed in a minor version of the toolkit, the feature might not be available at runtime if the application is running against an older CUDA driver. Users wishing to take advantage of such a feature should query its availability with a dynamic check in the code:
    
    
    static bool hostRegisterFeatureSupported = false;
    static bool hostRegisterIsDeviceAddress = false;
    
    static error_t cuFooFunction(int *ptr)
    {
        int *dptr = null;
        if (hostRegisterFeatureSupported) {
             cudaHostRegister(ptr, size, flags);
             if (hostRegisterIsDeviceAddress) {
                  qptr = ptr;
             }
           else {
              cudaHostGetDevicePointer(&qptr, ptr, 0);
              }
           }
        else {
                // cudaMalloc();
                // cudaMemcpy();
           }
        gemm<<<1,1>>>(dptr);
        cudaDeviceSynchronize();
    }
    
    int main()
    {
        // rest of code here
        cudaDeviceGetAttribute(
               &hostRegisterFeatureSupported,
               cudaDevAttrHostRegisterSupported,
               0);
        cudaDeviceGetAttribute(
               &hostRegisterIsDeviceAddress,
               cudaDevAttrCanUseHostPointerForRegisteredMem,
               0);
        cuFooFunction(/* malloced pointer */);
    }
    

Alternatively the application’s interface might not work at all without a new CUDA driver and then its best to return an error right away:
    
    
    #define MIN_VERSION 11010
    cudaError_t foo()
    {
        int version = 0;
        cudaGetDriverVersion(&version);
        if (version < MIN_VERSION) {
            return CUDA_ERROR_INSUFFICIENT_DRIVER;
        }
        // proceed as normal
    }
    

A new error code is added to indicate that the functionality is missing from the driver you are running against: `cudaErrorCallRequiresNewerDriver`.

####  16.4.1.2. Using PTX 

PTX defines a virtual machine and ISA for general purpose parallel thread execution. PTX programs are translated at load time to the target hardware instruction set via the JIT Compiler which is part of the CUDA driver. As PTX is compiled by the CUDA driver, new toolchains will generate PTX that is not compatible with the older CUDA driver. This is not a problem when PTX is used for future device compatibility (the most common case), but can lead to issues when used for runtime compilation.

For codes continuing to make use of PTX, in order to support compiling on an older driver, your code must be first transformed into device code via the static ptxjitcompiler library or NVRTC with the option of generating code for a specific architecture (e.g. sm_80) rather than a virtual architecture (e.g. compute_80). For this workflow, a new nvptxcompiler_static library is shipped with the CUDA Toolkit.

We can see this usage in the following example:
    
    
    char* compilePTXToNVElf()
    {
        nvPTXCompilerHandle compiler = NULL;
        nvPTXCompileResult status;
    
        size_t elfSize, infoSize, errorSize;
        char *elf, *infoLog, *errorLog;
        int minorVer, majorVer;
    
        const char* compile_options[] = { "--gpu-name=sm_80",
                                          "--device-debug"
        };
    
        nvPTXCompilerGetVersion(&majorVer, &minorVer);
        nvPTXCompilerCreate(&compiler, (size_t)strlen(ptxCode), ptxCode);
        status = nvPTXCompilerCompile(compiler, 2, compile_options);
        if (status != NVPTXCOMPILE_SUCCESS) {
            nvPTXCompilerGetErrorLogSize(compiler, (void*)&errorSize);
    
            if (errorSize != 0) {
                errorLog = (char*)malloc(errorSize+1);
                nvPTXCompilerGetErrorLog(compiler, (void*)errorLog);
                printf("Error log: %s\n", errorLog);
                free(errorLog);
            }
            exit(1);
        }
    
        nvPTXCompilerGetCompiledProgramSize(compiler, &elfSize));
        elf = (char*)malloc(elfSize);
        nvPTXCompilerGetCompiledProgram(compiler, (void*)elf);
        nvPTXCompilerGetInfoLogSize(compiler, (void*)&infoSize);
    
        if (infoSize != 0) {
            infoLog = (char*)malloc(infoSize+1);
            nvPTXCompilerGetInfoLog(compiler, (void*)infoLog);
            printf("Info log: %s\n", infoLog);
            free(infoLog);
        }
    
        nvPTXCompilerDestroy(&compiler);
        return elf;
    }
    

####  16.4.1.3. Dynamic Code Generation 

NVRTC is a runtime compilation library for CUDA C++. It accepts CUDA C++ source code in character string form and creates handles that can be used to obtain the PTX. The PTX string generated by NVRTC can be loaded by cuModuleLoadData and cuModuleLoadDataEx.

Dealing with relocatable objects is not yet supported, therefore the `cuLink`* set of APIs in the CUDA driver will not work with enhanced compatibility. An upgraded driver matching the CUDA runtime version is currently required for those APIs.

As mentioned in the PTX section, the compilation of PTX to device code lives along with the CUDA driver, hence the generated PTX might be newer than what is supported by the driver on the deployment system. When using NVRTC, it is recommended that the resulting PTX code is first transformed to the final device code via the steps outlined by the PTX user workflow. This ensures your code is compatible. Alternatively, NVRTC can generate cubins directly starting with CUDA 11.1. Applications using the new API can load the final device code directly using driver APIs `cuModuleLoadData` and `cuModuleLoadDataEx`.

NVRTC used to support only virtual architectures through the option -arch, since it was only emitting PTX. It will now support actual architectures as well to emit SASS. The interface is augmented to retrieve either the PTX or cubin if an actual architecture is specified.

The example below shows how an existing example can be adapted to use the new features, guarded by the `USE_CUBIN` macro in this case:
    
    
    #include <nvrtc.h>
    #include <cuda.h>
    #include <iostream>
    
    void NVRTC_SAFE_CALL(nvrtcResult result) {
      if (result != NVRTC_SUCCESS) {
        std::cerr << "\nnvrtc error: " << nvrtcGetErrorString(result) << '\n';
        std::exit(1);
      }
    }
    
    void CUDA_SAFE_CALL(CUresult result) {
      if (result != CUDA_SUCCESS) {
        const char *msg;
        cuGetErrorName(result, &msg);
        std::cerr << "\ncuda error: " << msg << '\n';
        std::exit(1);
      }
    }
    
    const char *hello = "                                           \n\
    extern \"C\" __global__ void hello() {                          \n\
      printf(\"hello world\\n\");                                   \n\
    }                                                               \n";
    
    int main()
    {
      nvrtcProgram prog;
      NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog, hello, "hello.cu", 0, NULL, NULL));
    #ifdef USE_CUBIN
      const char *opts[] = {"-arch=sm_70"};
    #else
      const char *opts[] = {"-arch=compute_70"};
    #endif
      nvrtcResult compileResult = nvrtcCompileProgram(prog, 1, opts);
      size_t logSize;
      NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
      char *log = new char[logSize];
      NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));
      std::cout << log << '\n';
      delete[] log;
      if (compileResult != NVRTC_SUCCESS)
        exit(1);
      size_t codeSize;
    #ifdef USE_CUBIN
      NVRTC_SAFE_CALL(nvrtcGetCUBINSize(prog, &codeSize));
      char *code = new char[codeSize];
      NVRTC_SAFE_CALL(nvrtcGetCUBIN(prog, code));
    #else
      NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &codeSize));
      char *code = new char[codeSize];
      NVRTC_SAFE_CALL(nvrtcGetPTX(prog, code));
    #endif
      NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
      CUdevice cuDevice;
      CUcontext context;
      CUmodule module;
      CUfunction kernel;
      CUDA_SAFE_CALL(cuInit(0));
      CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
      CUDA_SAFE_CALL(cuCtxCreate(&context, NULL, 0, cuDevice));
      CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, code, 0, 0, 0));
      CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "hello"));
      CUDA_SAFE_CALL(cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, NULL, NULL, 0));
      CUDA_SAFE_CALL(cuCtxSynchronize());
      CUDA_SAFE_CALL(cuModuleUnload(module));
      CUDA_SAFE_CALL(cuCtxDestroy(context));
      delete[] code;
    }
    

####  16.4.1.4. Recommendations for building a minor-version compatible library 

We recommend that the CUDA runtime be statically linked to minimize dependencies. Verify that your library doesn’t leak dependencies, breakages, namespaces, etc. outside your established ABI contract.

Follow semantic versioning for your library’s soname. Having a semantically versioned ABI means the interfaces need to be maintained and versioned. The library should follow semantic rules and increment the version number when a change is made that affects this ABI contract. Missing dependencies is also a binary compatibility break, hence you should provide fallbacks or guards for functionality that depends on those interfaces. Increment major versions when there are ABI breaking changes such as API deprecation and modifications. New APIs can be added in minor versions.

Conditionally use features to remain compatible against older drivers. If no new features are used (or if they are used conditionally with fallbacks provided) you’ll be able to remain compatible.

Don’t expose ABI structures that can change. A pointer to a structure with a size embedded is a better solution.

When linking with dynamic libraries from the toolkit, the library must be equal to or newer than what is needed by any one of the components involved in the linking of your application. For example, if you link against the CUDA 11.1 dynamic runtime, and use functionality from 11.1, as well as a separate shared library that was linked against the CUDA 11.2 dynamic runtime that requires 11.2 functionality, the final link step must include a CUDA 11.2 or newer dynamic runtime.

####  16.4.1.5. Recommendations for taking advantage of minor version compatibility in your application 

Certain functionality might not be available so you should query where applicable. This is common for building applications that are GPU architecture, platform and compiler agnostic. However we now add “the underlying driver” to that mix.

As with the previous section on library building recommendations, if using the CUDA runtime, we recommend linking to the CUDA runtime statically when building your application. When using the driver APIs directly, we recommend using the new driver entry point access API (`cuGetProcAddress`) documented here: [CUDA Driver API :: CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DRIVER__ENTRY__POINT.html#group__CUDA__DRIVER__ENTRY__POINT).

When using a shared or static library, follow the release notes of said library to determine if the library supports minor version compatibility.
