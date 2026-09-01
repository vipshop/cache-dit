# 18. Deployment Infrastructure Tools


##  18.1. Nvidia-SMI 

The NVIDIA System Management Interface (`nvidia-smi`) is a command line utility that aids in the management and monitoring of NVIDIA GPU devices. This utility allows administrators to query GPU device state and, with the appropriate privileges, permits administrators to modify GPU device state. `nvidia-smi` is targeted at Tesla and certain Quadro GPUs, though limited support is also available on other NVIDIA GPUs. `nvidia-smi` ships with NVIDIA GPU display drivers on Linux, and with 64-bit Windows Server 2008 R2 and Windows 7. `nvidia-smi` can output queried information as XML or as human-readable plain text either to standard output or to a file. See the nvidia-smi documenation for details. Please note that new versions of nvidia-smi are not guaranteed to be backward-compatible with previous versions.

###  18.1.1. Queryable state 

ECC error counts
    

Both correctable single-bit and detectable double-bit errors are reported. Error counts are provided for both the current boot cycle and the lifetime of the GPU.

GPU utilization
    

Current utilization rates are reported for both the compute resources of the GPU and the memory interface.

Active compute process
    

The list of active processes running on the GPU is reported, along with the corresponding process name/ID and allocated GPU memory.

Clocks and performance state
    

Max and current clock rates are reported for several important clock domains, as well as the current GPU performance state (_pstate_).

Temperature and fan speed
    

The current GPU core temperature is reported, along with fan speeds for products with active cooling.

Power management
    

The current board power draw and power limits are reported for products that report these measurements.

Identification
    

Various dynamic and static information is reported, including board serial numbers, PCI device IDs, VBIOS/Inforom version numbers and product names.

###  18.1.2. Modifiable state 

ECC mode
    

Enable and disable ECC reporting.

ECC reset
    

Clear single-bit and double-bit ECC error counts.

Compute mode
    

Indicate whether compute processes can run on the GPU and whether they run exclusively or concurrently with other compute processes.

Persistence mode
    

Indicate whether the NVIDIA driver stays loaded when no applications are connected to the GPU. It is best to enable this option in most circumstances.

GPU reset
    

Reinitialize the GPU hardware and software state via a secondary bus reset.


##  18.2. NVML 

The NVIDIA Management Library (NVML) is a C-based interface that provides direct access to the queries and commands exposed via `nvidia-smi` intended as a platform for building 3rd-party system management applications. The NVML API is shipped with the CUDA Toolkit (since version 8.0) and is also available standalone on the NVIDIA developer website as part of the GPU Deployment Kit through a single header file accompanied by PDF documentation, stub libraries, and sample applications; see <https://developer.nvidia.com/gpu-deployment-kit>. Each new version of NVML is backward-compatible.

An additional set of Perl and Python bindings are provided for the NVML API. These bindings expose the same features as the C-based interface and also provide backwards compatibility. The Perl bindings are provided via CPAN and the Python bindings via PyPI.

All of these products (`nvidia-smi`, NVML, and the NVML language bindings) are updated with each new CUDA release and provide roughly the same functionality.

See <https://developer.nvidia.com/nvidia-management-library-nvml> for additional information.


##  18.3. Cluster Management Tools 

Managing your GPU cluster will help achieve maximum GPU utilization and help you and your users extract the best possible performance. Many of the industry’s most popular cluster management tools support CUDA GPUs via NVML. For a listing of some of these tools, see <https://developer.nvidia.com/cluster-management>.


##  18.4. Compiler JIT Cache Management Tools 

Any PTX device code loaded by an application at runtime is compiled further to binary code by the device driver. This is called _just-in-time compilation_ (_JIT_). Just-in-time compilation increases application load time but allows applications to benefit from latest compiler improvements. It is also the only way for applications to run on devices that did not exist at the time the application was compiled.

When JIT compilation of PTX device code is used, the NVIDIA driver caches the resulting binary code on disk. Some aspects of this behavior such as cache location and maximum cache size can be controlled via the use of environment variables; see Just in Time Compilation of the CUDA C++ Programming Guide.


##  18.5. CUDA_VISIBLE_DEVICES 

It is possible to rearrange the collection of installed CUDA devices that will be visible to and enumerated by a CUDA application prior to the start of that application by way of the `CUDA_VISIBLE_DEVICES` environment variable.

Devices to be made visible to the application should be included as a comma-separated list in terms of the system-wide list of enumerable devices. For example, to use only devices 0 and 2 from the system-wide list of devices, set `CUDA_VISIBLE_DEVICES=0,2` before launching the application. The application will then enumerate these devices as device 0 and device 1, respectively.
