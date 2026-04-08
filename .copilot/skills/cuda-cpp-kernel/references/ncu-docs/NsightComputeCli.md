---
url: https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html
---

[ ![Logo](https://docs.nvidia.com/nsight-compute/_static/nsight-compute.png) ](../index.html)

Nsight Compute

  * [1\. Release Notes](../ReleaseNotes/index.html)
  * [2\. Profiling Guide](../ProfilingGuide/index.html)
  * [3\. Nsight Compute](../NsightCompute/index.html)
  * [4\. Nsight Compute CLI](#)
    * [4.1. Introduction](#introduction)
    * [4.2. Quickstart](#quickstart)
    * [4.3. Usage](#usage)
      * [4.3.1. Modes](#modes)
      * [4.3.2. Multi-Process Support](#multi-process-support)
        * [Platform Support](#platform-support)
        * [MPI Support](#mpi-support)
        * [Mandatory Concurrent Kernels](#mandatory-concurrent-kernels)
      * [4.3.3. Output Pages](#output-pages)
      * [4.3.4. Profile Import](#profile-import)
      * [4.3.5. Filtered Profile Export](#filtered-profile-export)
      * [4.3.6. Metrics and Units](#metrics-and-units)
      * [4.3.7. NVTX Filtering](#nvtx-filtering)
      * [4.3.8. Config File](#config-file)
      * [4.3.9. Kernel Renaming](#kernel-renaming)
      * [4.3.10. CPU Call Stack Filtering](#cpu-call-stack-filtering)
    * [4.4. Command Line Options](#command-line-options)
      * [4.4.1. General](#general)
      * [4.4.2. Launch](#launch)
      * [4.4.3. Attach](#attach)
      * [4.4.4. Profile](#profile)
      * [4.4.5. PM Sampling](#pm-sampling)
      * [4.4.6. Warp Sampling](#warp-sampling)
      * [4.4.7. File](#file)
      * [4.4.8. Console Output](#console-output)
      * [4.4.9. Response File](#response-file)
      * [4.4.10. File Macros](#file-macros)
      * [4.4.11. MPS](#mps)
    * [4.5. Environment Variables](#environment-variables)
    * [4.6. Nvprof Transition Guide](#nvprof-transition-guide)


Developer Interfaces

  * [1\. Customization Guide](../CustomizationGuide/index.html)
  * [2\. Python Report Interface](../PythonReportInterface/index.html)
  * [3\. NvRules API](../NvRulesAPI/index.html)
  * [4\. Occupancy Calculator Python Interface](../OccupancyCalculatorPythonInterface/index.html)


Training

  * [Training](../Training/index.html)


Release Information

  * [Archives](../Archives/index.html)


Copyright and Licenses

  * [Copyright and Licenses](../CopyrightAndLicenses/index.html)


__[NsightCompute](../index.html)

  * [](../index.html) »
  * 4\. Nsight Compute CLI
  *   * v2026.1.0 | [Archive](https://developer.nvidia.com/nsight-compute-history)


* * *

# 4\. Nsight Compute CLI

The User Guide for Nsight Compute CLI.

## 4.1. Introduction

NVIDIA Nsight Compute CLI (ncu) provides a non-interactive way to profile applications from the command line. It can print the results directly on the command line or store them in a report file. It can also be used to simply launch the target application (see [General](index.html#command-line-options-general) for details) and later attach with NVIDIA Nsight Compute or another ncu instance.

## 4.2. Quickstart

  1. **Launch the target application with the command line profiler**

The command line profiler launches the target application, instruments the target API, and collects profile results for the specified kernels. The CLI executable is called ncu. A shortcut with this name is located in the base directory of the NVIDIA Nsight Compute installation. The actual executable is located in the folder `target\windows-desktop-win7-x64` on Windows or `target/linux-desktop-glibc_2_11_3-x64` on Linux. By default, NVIDIA Nsight Compute is installed in `/usr/local/cuda-<cuda-version>/NsightCompute-<version>` on Linux and in `C:\Program Files\NVIDIA Corporation\Nsight Compute <version>` on Windows.

To collect the `basic` set for all kernel launches in the target application, launch:
         
         $ ncu -o profile CuVectorAddMulti.exe
         

The application runs in instrumented mode and for each kernel launch, a profile result is created. The results are written by default to profile.nsight-cuprof. Each output from the compute profiler starts with `==PROF==` The other lines are output from the application itself. For each profiled kernel, the name of the kernel function and the progress of data collection is shown. To collect all requested profile information, it may be required to replay the kernels multiple times. The total number of replay passes per kernel is shown after profiling has completed.
         
         [Vector addition of 1144477 elements]
         ==PROF== Connected to process 5268
         Copy input data from the host memory to the CUDA device
         CUDA kernel launch A with 4471 blocks of 256 threads
         ==PROF== Profiling "vectorAdd_A" - 0: 0%....50%....100% - 46 passes
         CUDA kernel launch B with 4471 blocks of 256 threads
         ==PROF== Profiling "vectorAdd_B" - 1: 0%....50%....100% - 46 passes
         Copy output data from the CUDA device to the host memory
         Done
         ==PROF== Disconnected from process 5268
         ==PROF== Report: profile.ncu-rep
         

  2. **Customizing data collection**

Options are available to specify for which kernels data should be collected. `-c` limits the number of kernel launches collected. `-s` skips the given number of kernels before data collection starts. `-k` allows you to filter the kernels by a regex match of their names. `--kernel-id` allows you to filter kernels by context, stream, name and invocation.

To limit what should be collected for each kernel launch, specify the exact *.section (files) by their identifier using `--section`. Each section file defines a set of metrics to be collected, grouped logically to solve a specific performance question. By default, the sections associated with the `basic` set are collected. Use `--list-sets` to see the list of currently available sets. Use `--list-sections` to see the list of currently available sections. The default search directory and location of pre-defined section files is also called `sections/`. See the [Profiling Guide](../ProfilingGuide/index.html#sets-and-sections) for more details.

Alternatively, you can collect a set of individual metrics using `--metrics`. The available metrics can be queried using `--query-metrics`. For an explanation of the naming conventions and structuring of metrics, see [Metrics Structure](../ProfilingGuide/index.html#metrics-structure).

Most metrics in NVIDIA Nsight Compute are named using a base name and various suffixes, e.g. _sm__throughput.avg.pct_of_peak_sustained_elapsed_. The base name is _sm__throughput and the suffix is avg.pct_of_peak_sustained_elapsed_. This is because most metrics follow the same structure and have the same set of suffixes. You need to pass the base or full name to NVIDIA Nsight Compute when selecting a metric for profiling. Use `--query-metrics-mode suffix --metrics <metrics list>` to see the full names for the chosen metrics.

Some additional metrics do not follow this structured naming. They are documented in the [Metrics Reference](../ProfilingGuide/index.html#metrics-reference).

  3. **Changing command line output**

By default, a temporary file is used to store profiling results, and data is printed to the command line. To permanently store the profiler report, use `-o` to specify the output filename.

Besides storing results in a report file, the command line profiler can print results using different pages. Those pages correspond to the respective pages in the UI’s report. By default, the [Details page](index.html#output-pages) is printed, if no explicit output file is specified. To select a different page or print in addition to storing in an explicit file, use the `--page=<Page>` command. Currently, the following pages are supported: `details, raw, source`.

Use `--csv` to make any output comma separated and easier to process further. See [Console Output](index.html#command-line-options-console-output) for further options, e.g. summary views.

  4. **Open the report in the UI**

The UI executable is called ncu-ui. A shortcut with this name is located in the base directory of the NVIDIA Nsight Compute installation. The actual executable is located in the folder `host\windows-desktop-win7-x64` on Windows or `host/linux-desktop-glibc_2_11_3-x64` on Linux. In the UI window, close the _Start Activity_ dialog and open the report file through _File_ > _Open_ , by dragging the report file into NVIDIA Nsight Compute.

You can also specify the report file as a command line parameter to the executable, i.e. as `ncu-ui <MyReport.ncu-rep>`. Alternatively, when using NVIDIA Nsight Compute CLI on a platform with host support, `--open-in-ui` can be used directly with ncu to open a collected report in the user interface.

The report opens in a new document window. For more information about the report, see the [Profiler Report](../NsightCompute/index.html#profiler-report) for collecting profile information through NVIDIA Nsight Compute.


## 4.3. Usage

### 4.3.1. Modes

Modes change the fundamental behavior of the command line profiler. Depending on which mode is chosen, different [Command Line Options](index.html#command-line-options) become available. For example, [Launch](index.html#command-line-options-launch) is invalid if the _Attach_ mode is selected.

  * **Launch-and-attach:** The target application is launched on the local system with the tool’s injection libraries. Depending on which profiling options are chosen, selected kernels in the application are profiled and the results printed to the console or stored in a report file. The tool exits once the target application finishes or crashes, and once all results are processed.

This is the default, and the only mode that supports profiling of child processes on selected platforms.

  * **Launch:** The target application is launched on the local system with the tool’s injection libraries. As soon as the first intercepted API call is reached (commonly `cuInit()`), all application threads are suspended. The application now expects a tool to attach for profiling. You can attach using NVIDIA Nsight Compute or using the command line profiler’s _Attach_ mode.

  * **Attach:** The tool tries to connect to a target application previously launched using NVIDIA Nsight Compute or using the command line profiler’s _Launch_ mode. The tool can attach to a target on the local system or using a remote connection.


### 4.3.2. Multi-Process Support

The NVIDIA Nsight Compute CLI supports profiling multi-process applications. See [Launch](index.html#command-line-options-launch) for which command line options are available.

#### Platform Support

On x86_64 Windows, NVIDIA Nsight Compute CLI supports profiling 64-bit processes launched from 32-bit applications by default. On x86_64 Linux, launching from 32-bit applications requires you to enable the `support-32bit` option, and the required 32-bit libraries must be installed on your system. On DRIVE OS Linux and DRIVE OS QNX, tracking of 32-bit applications is not supported. Profiling of 32-bit processes is not supported on any platform.

#### MPI Support

NVIDIA Nsight Compute CLI can be used to profile applications launched with the `mpirun` command.

  * To profile all ranks on a node and store all the profiling data in a single report file:
        
        ncu --target-processes all -o <report-name> mpirun [mpi arguments] <app> [app arguments]
        

  * To profile multi-node submissions, ensure that you specify unique report files per rank.
        
        mpirun [mpi arguments] ncu -o report_%q{OMPI_COMM_WORLD_RANK} <app> [app arguments]
        

Alternatively, enable the multiprocess communicator:
        
        mpirun [mpi arguments] ncu --communicator=tcp --communicator-num-peers=<number of MPI ranks> -o report <app> [app arguments]
        

  * To profile a single rank one can use a wrapper script. The following script (called “wrap.sh”) profiles rank 0 only:
        
        #!/bin/bash
        if [[ $OMPI_COMM_WORLD_RANK == 0 ]]; then
           ncu -o report_${OMPI_COMM_WORLD_RANK}  --target-processes all "$@"
        else
           "$@"
        fi
        

and then execute:
        
        mpirun [mpi arguments] ./wrap.sh <app> [app arguments]
        


See below for profiling mandatory concurrent kernels across MPI processes.

#### Mandatory Concurrent Kernels

Multi-process/multi-GPU applications that contain mandatory concurrent kernels (e.g., NCCL and NVSHMEM communication kernels) may be unable to make forward progress when profiled with default settings due to multi-pass data collection and serialization across processes within the same process tree.

**For communication kernels launched from different process trees, use --communicator tcp**

To make the tool synchronize launchs and replays across processes, use `--lockstep-kernel-launch`.

> 
>     mpirun -np 4 ncu --communicator tcp --communicator-num-peers 4 --lockstep-kernel-launch -o report <app>
>     

Additionally, `--lockstep-nvtx-include` and `--lockstep-nvtx-exclude` options can be set to limit synchronization to specific kernel ranges. See [NVTX Filtering](index.html#nvtx-filtering) for additional information.

> 
>     mpirun <args> ncu <args> --lockstep-kernel-launch --lockstep-nvtx-include nccl/ -o report <app>
>     

[\--communicator tcp](index.html#launch) internally uses TCP sockets in a client-server architecture to send and receive message. One process at random is selected as the server process to manage sockets. In a multi-node system, the option `--communicator-tcp-hostname` must be set to manually select one host as the server. Below is an example slurm script that can be used to automate the process

> 
>     #!/bin/bash
>     nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
>     nodes_array=($nodes)
>     head_node=${nodes_array[0]}
>     head_node_addr=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname)
>     
>     srun ncu \
>         --communicator=tcp                           \
>         --communicator-tcp-hostname="$head_node_addr"  \
>         --communicator-tcp-num-peers="$SLURM_NTASKS" \
>         --lockstep-kernel-launch                     \
>         ./application
>     

You can use the [Report Merge Tool](../NsightCompute/index.html#report-merge-tool) and [Clustering Window](../NsightCompute/index.html#clustering-window) to aggregate and/or compare these results.

**For communication kernels launched from the same process tree, use --communicator shmem**

In single-node scenarios where the entire process tree is profiled by the same instance of `ncu`, the [\--communicator shmem](index.html#launch) option can be used. The processes must be able to communicate with each other using shared memory on the target system.

> 
>     ncu --communicator shmem --communicator-shmem-num-peers 2 -f -o report -k regex:nccl.* torchrun --nnodes=1 --nproc_per_node=2 pytorch_app.py
>     

This mode only supports _kernel_ and _range_ replay modes and has limited filter options. If your app launches different types of kernels, select the concurrent ones using a name filter like `-k regex:nccl.*`. A single report is generated for all profile results. By default, the first CUDA context of each group is profiled. The `--devices` option can be used to select the primary GPU device(s) to profile.

Setting `--communicator-shmem-num-peers` to the number of processes participating in the concurrent workload `-k ...` is not required but recommended. If the number of processes is not specified, the communicator will use all available processes per workload. If processes don’t reach the same workload at the same time, and since the number of processes per workload is unknown to the tool, Nsight Compute may try to profile a workload with insufficient participants.

Launcher processes like `mpirun` or `torchrun` are auto-excluded from profiling. You can manually exclude others using the [–target-processes-filter](index.html#launch) option.

### 4.3.3. Output Pages

The command line profiler supports printing results to the console using various pages. Each page has an equivalent in NVIDIA Nsight Compute’s _Profiler Report_. In the command line profiler, they are slightly adapted to fit console output. To select a page, use the `--page` option. By default, the details page is used. Note that if `--page` is not used but `--export` is, no results will be printed to the console.

  * **Details:** This page represents NVIDIA Nsight Compute’s _Details_ page. For every profiled kernel launch, each collected is printed as section as a three-column table, followed by any rule results applied to this section. Rule results not associated with any section are printed after the kernel’s sections.

The first section table column shows the metric name. If the metric was given a label in the section, it is used instead. The second column shows the metric unit, if available. The third column shows the unit value. Both metric unit and value are automatically adjusted to the most fitting order of magnitude. By default, only metrics defined in section headers are shown. This can be changed by passing the `--details-all` option on the command line.

Some metrics will show multiple values, separated by “;”, e.g. memory_l2_transactions_global Kbytes 240; 240; 240; 240; 240. Those are instanced metrics, which have one value per represented instance. An instance can be a streaming multiprocessor, an assembly source line, etc.

  * **Raw:** This page represents NVIDIA Nsight Compute’s _Raw_ page. For every profiled kernel launch, each collected metric is printed as a three-column table. Besides metrics from sections, this includes automatically collected metrics such as device attributes and kernel launch information.

The first column shows the metric name. The second and third columns show the metric unit and value, respectively. Both metric unit and value are automatically adjusted to the most fitting order of magnitude. No unresolved regex:, group:, or breakdown: metrics are included.


### 4.3.4. Profile Import

Using the `--import` option, saved reports can be imported into the command line profiler. When using this flag, most other options are not available, except for certain result filterting options. They are marked as such in the [Profile options](index.html#command-line-options-profile) table.

### 4.3.5. Filtered Profile Export

Using the `--import` and `--export` options together, along with supported filtering options, you can export desired results from one report to another. Most of the filtering [Profile options](index.html#command-line-options-profile) that can be used with `--import` alone are also supported here, except for `--metrics` and `--section`.

### 4.3.6. Metrics and Units

When available and applicable, metrics are shown along with their unit. This is to make it apparent if a metric represents cycles, threads, bytes/s, and so on.

By default, units are scaled automatically so that metric values are shown with a reasonable order of magnitude. Units are scaled using their SI-factors, i.e. byte-based units are scaled using a factor of 1000 and the prefixes K, M, G, etc. Time-based units are also scaled using a factor of 1000, with the prefixes n, u and m. This scaling can be changed using a command line option, see [Console Output](index.html#command-line-options-console-output) options for details.

### 4.3.7. NVTX Filtering

`--nvtx-include <configuration> --nvtx-exclude <configuration>`

These options are used to profile only those kernels which satisfy the conditions mentioned in the configuration. Through these options, you can choose which kernel falls into a specific range or collection of ranges.

You can use both options multiple times, mentioning all the `--nvtx-include` configurations followed by all `--nvtx-exclude` configurations. NVTX filtering requires `--nvtx` option.

NVTX ranges are of two types: NvtxRangeStart/End and NvtxRangePush/Pop. The configuration syntax for both the types are briefly described below. Both range and domain names can contain whitespace. Note that “Domain” and “range” in below example are for illustration purposes only and are not required to mark domain or range names.

  * **Push-Pop Ranges**

Quantifier | Description | Example  
---|---|---  
/ | Delimiter between range names. When only a single range name is given, the delimiter must be appended to indicate that this refers to a push/pop range. | `A_range/` `A_range/B range` `A_range/\*/B range`  
[ | Range is at the bottom of the stack | `[A_range` `[A_range/+/Range Z`  
] | Range is at the top of the stack | `Range Z]` `Range C/\*/Range Z]`  
+ | Only one range between the two other ranges | `B range/+/Range D`  
* | Zero or more range(s) between the two other ranges | `B range/\*/Range Z`  
@ | Specify domain name. If not mentioned, assuming <default domain> | `Domain-A@A_range` `Domain B@A_range/\*/Range Z]`  
  
Include kernels wrapped inside push/pop range ‘A_range’ of ‘<default-domain>’:
        
        ncu --nvtx --nvtx-include "A_range/" CuNvtx.exe
        

Include kernels wrapped inside push/pop range ‘A_range’ of ‘Domain-A’:
        
        ncu --nvtx --nvtx-include "Domain-A@A_range/" CuNvtx.exe
        

Include kernels wrapped inside push/pop range ‘A_range’ of ‘<default domain>’, where ‘A_range’ is at the bottom of the stack:
        
        ncu --nvtx --nvtx-include "[A_range" CuNvtx.exe
        

Include kernels wrapped inside push/pop ranges ‘A_range’ and ‘B range’ of ‘<default domain>’, with zero or many ranges between them:
        
        ncu --nvtx --nvtx-include "A_range/*/B range" CuNvtx.exe
        

Exclude kernels wrapped inside push/pop ranges ‘A_range’ and ‘B range’ of ‘<default domain>’, with zero or many ranges between them:
        
        ncu --nvtx --nvtx-exclude "A_range/*/B range" CuNvtx.exe
        

Include kernels wrapped inside only push/pop range ‘A_range’ of ‘<default domain>’ but not inside ‘B range’ at the top of the stack:
        
        ncu --nvtx --nvtx-include "A_range/" --nvtx-exclude "B range]" CuNvtx.exe
        

  * **Start-End Ranges**

Quantifier | Description | Example  
---|---|---  
, | Delimiter between range names | `A_range,B range` `B range,A_range,Range C`  
@ | Specify domain name. If not mentioned, assuming <default domain> | `Domain-A@A_range` `Domain B@B range,Range Z`  
  
Include kernels wrapped inside start/end range ‘A_range’ of ‘Domain-A’:
        
        ncu --nvtx --nvtx-include "Domain-A@A_range" CuNvtx.exe
        

Include kernels wrapped inside both start/end ranges, ‘A_range’ and ‘B range’ of ‘<default domain>’:
        
        ncu --nvtx --nvtx-include "A_range,B range" CuNvtx.exe
        

Include kernels wrapped inside start/end ranges, ‘A_range’ or ‘B range’ of ‘<default domain>’:
        
        ncu --nvtx --nvtx-include "A_range" --nvtx-include "B range" CuNvtx.exe
        

Include all kernels, except those which are wrapped inside start/end range ‘A_range’ of ‘<default domain>’:
        
        ncu --nvtx --nvtx-exclude "A_range" CuNvtx.exe
        

Include kernels wrapped inside only start/end ‘B range’ and not ‘A_range’ of ‘<default domain>’:
        
        ncu --nvtx --nvtx-include "B range"--nvtx-exclude "A_range" CuNvtx.exe
        

  * **Regular Expression Support**

The configuration syntax for both the types NvtxRangeStart/End and NvtxRangePush/Pop is the same. Additionally, to use regular expressions, follow the following syntax.

    * Provide prefix ‘regex:’ to treat nvtx config as regular expression.
          
          ncu --nvtx --nvtx-include "regex:Domain[A-Z]@Range[0-9]/" CuNvtx.exe
          

The kernels wrapped inside push/pop range with matching regex ‘Range[0-9]’ of domain with matching regex ‘Domain[A-Z]’ are profiled.

    * Provide ‘/’ as a prefix to “[” or “]” only for the range part of the config if “[” or “]” is at the start or at the end of the range part, respectively. This is needed so that NCU can distinguish if “[” or “]” is part of the regex or represents the top/bottom of the stack.
          
          ncu --nvtx --nvtx-include "regex:[0-9]domainA@/[0-9]rangeA,RangeC[0-9/]" CuNvtx.exe
          

The kernels wrapped inside start/end ranges with matching regex ‘[0-9]rangeA’ and ‘RangeC[0-9]’ of domain with matching regex ‘[0-9]domainA’ are profiled.

    * If any quantifier is part of the domain/range name, you need to use ‘\\\’ or ‘\’ as a prefix. For the “$” quantifier, only the ‘\\\’ prefix is valid.

  * **Additional Information**
        
        --nvtx-include "DomainA@RangeA,DomainB@RangeB" //Not a valid config
        

In a single NVTX configuration, multiple ranges with regard to a single domain can be specified. Mentioning ranges from different domains inside a single NVTX config is not supported.
        
        --nvtx-include "A_range\[i\]"
        

Quantifiers `@ , [ ] / * +`, including regular expression quantifiers, can be used in domain/range names using prefix ‘\’. The kernels wrapped inside ‘A_range[i]’ of ‘<default domain>’ in the application are profiled.
        
        --nvtx-include "A_range"  //Start/End configuration
        --nvtx-inlcude "A_range/" //Push/Pop configuration
        --nvtx-inlcude "A_range]" //Push/Pop configuration
        

If the domain/range name contains ‘\’, you need to provide ‘\\\\\\\’ in the config.

Do not use ‘\\\\\\\’ before any quantifier.

Including/Excluding only single range for Push/Pop configuration without specifying stack frame position ‘[’ or ‘]’, use ‘/’ quantifier at the end.
        
        --nvtx-include "A_range/*/B range"
        

The order in which you mention Push/Pop configurations is important. In the above example, ‘A_range’ should be below ‘B range’ in the stack of ranges so that the kernel is profiled.

NVTX filtering honors cudaProfilerStart() and cudaProfilerStop(). There is no support for ranges with no name.


### 4.3.8. Config File

Using the `--config-file on/off` option, parsing parameters from config file can be enabled or disabled.

Using the `--config-file-path <path>` option, default path and name of config file can be overwritten.

By default, config-file with name `config.ncu-cfg` is searched in the current working directory, `$HOME/.config/NVIDIA Corporation` on Linux and `%APPDATA%\NVIDIA Corporation\` on Windows. If a valid config file is found, ncu parses the file and initializes any command line parameters to the values set in the file. If the same command line parameter is also set explicitly during the current invocation, the latter takes precedence.

Parameters can be set under various general modes and ncu command line parameters are used to determine which general-mode needs to be parsed from the config file. See the table below for more details.

Command line parameters | General Mode  
---|---  
ncu –mode launch-and-attach CuVectorAddMulti.exe | Launch-and-attach  
ncu –mode launch CuVectorAddMulti.exe | Launch  
ncu –mode attach | Attach  
ncu –list-sets, ncu –list-sections, ncu –list-rules and ncu –list-metrics | List  
ncu –query-metrics | Query  
ncu -i <MyReport.ncu-rep> | Import  
  
These general modes should be defined in the config file using INI-like syntax as:
    
    
    [<general-mode>]
    <parameter>=<value>
    ;<comments>
    

**Sample usage**
    
    
    [Launch-and-attach]
    -c = 1
    --section = LaunchStats, Occupancy
    [Import]
    --open-in-ui
    -c = 1
    --section = LaunchStats, Occupancy
    

From this configuration, ncu will parse parameters set under `[Launch-and-attach]` block whenever an application is profiled in `launch-and-attach` mode. In the same manner, parameters set under `[Import]` block will be parsed whenever a report is imported. Different modes can be clubbed together if there exists a set of parameters which is common to each mode. Sample shown above can be rewritten after clubbing both modes as:
    
    
    [Launch-and-attach, import]
    -c = 1
    --section = LaunchStats, Occupancy
    [Import]
    --open-in-ui
    

**Additional points**

  * Options like `--open-in-ui` do not expect any value to be set. These options should not be passed any value.

  * Options like `--section` can be passed multiple times in the command line. These options should be written only once under a general-mode with all required values seperated by comma as shown below. Explicitly setting values for these options will not overwrite the config file values. Instead, all values will be composed together and set to the option.
        
        [<general-mode>]
        <parameter>=<value1>,<value2>,...
        


### 4.3.9. Kernel Renaming

In some cases, it gets difficult to distinguish between results using the kernel function or mangled name. However, demangled names can be quite long and complex to understand. To handle such cases, kernel demangled names are auto-simplified to some extent. To see original kernel demangled names, disable kernel renaming using `--rename-kernels off` option. If a simplified kernel demangled name turns out to be not useful, you can rename it through the configuration file. A kernel renaming configuration file should be a YAML file written in the following format:
    
    
    -
     - Original: mergeRanksAndIndicesKernel(unsigned int *, unsigned int *, unsigned int, unsigned int, unsigned int)
     - Renamed: Merge Rank Kernel
    -
     - Original: void mergeSortSharedKernel<(unsigned int)1>(unsigned int *, unsigned int *, unsigned int *, unsigned int *, unsigned int)
     - Renamed: Merge Sort Kernel
    

By default, kernel renaming config file with name `ncu-kernel-renames.yaml` is searched in the similar way as [config file](index.html#config-file) is searched.

To avoid manually writing demangled names in the config file, one can use `--rename-kernels-export on` option to export demangled names from the report to the config file with mappings for renaming them.

Using the `--rename-kernels-path <path>` option, default path and name of the file used while importing renamed names and exporting can be overwritten.

Note that renamed names can later be used to filter kernels in the report using `--kernel-name` and `--kernel-id` options.

Similary, you can use the renamed names to filter the kernels while profiling as well.

**Sample usage**

To profile a kernel with demangled name ‘void SomeComplexKernel(unsigned int, unsigned int, unsigned int)’, you can import the following kernel renaming configuration file:
    
    
    -
     - Original: void SomeComplexKernel(unsigned int, unsigned int, unsigned int)
     - Renamed: MySimpleKernel
    
    
    
    ncu
    --rename-kernels-path /path/to/ncu-kernel-renames.yaml
    --kernel-name-base demangled
    --kernel-name MySimpleKernel
    app.exe
    

If you export the report from the above command, you will see the kernel demangled name renamed to ‘MySimpleKernel’ in the report even if you remove the renaming configuration file. If you want to see the original demangled name in the report, you can always rename it again using the renaming configuration file or simply disable kernel renaming using `--rename-kernels off` option. Note that the renaming can be done on demangled names only. That’s why you need to use `--kernel-name-base demangled` option to use the renamed demangled names for kernel filtering.

You can also use the kernel renaming configuration file while exporting kernels from one report to another using [Filtered Profile Export](index.html#filtered-profile-export) feature. For example, if you rename the kernels in the configuration file as:
    
    
    -
     - Original: KernelA(unsigned int, unsigned int, unsigned int)
     - Renamed: KernelA_Export
    -
     - Original: KernelB(unsigned int, unsigned int, unsigned int)
     - Renamed: KernelB_NoExport
    -
     - Original: KernelC(unsigned int, unsigned int, unsigned int)
     - Renamed: KernelC_Export
    

And then use this configuration file while exporting kernels with the following command:
    
    
    ncu
    --export /path/to/NewReport.ncu-rep
    --import /path/to/OldReport.ncu-rep
    --rename-kernels-path /path/to/ncu-kernel-renames.yaml
    --kernel-name-base demangled
    --kernel-name regex:.*_Export
    

Then, only the kernels with renamed names ‘KernelA_Export’ and ‘KernelC_Export’ will be exported to the new report.

### 4.3.10. CPU Call Stack Filtering

CPU call stack filtering allows you to filter the launches based on their CPU call stacktrace. You can collect two types of CPU call stacks: “Native” and “Python” using the `--call-stack` or `--call-stack-type` option. You can filter the launches based on the collected call stack using the following options:

`--native-include <configuration> --native-exclude <configuration>`

Through these options, you can choose kernel launches via a specific native call stack frame function or set of functions to include or exclude from the profiling.

`--python-include <configuration> --python-exclude <configuration>`

Through these options, you can choose kernel launches via a specific python call stack frame function or set of functions to include or exclude from the profiling.

These options are used to profile only those kernels which satisfy the conditions mentioned in the configurations. You can use each option multiple times, mentioning all the “include” and “exclude” configurations as needed. In order to use these include/exclude options, you must collect the respective call stack type using the `--call-stack` or `--call-stack-type` option. However, if you have already collected the call stack into the report file, you can directly use the include/exclude options to filter the results while importing the report.

Include/Exclude configuration follows the below format:

`<Module>@<File>@<Function>`

where <Module> and <File> are optional. If you do not provide any one of them, your configuration will be searched in all the modules or files respectively. You must provide the <Function> expression otherwise the configuration will be considered invalid.

  * <File>@<Function> usage is similar to the usage of <Domain>@<Range> in [NVTX Filtering](index.html#nvtx-filtering)

> Quantifier | Description | Example  
> ---|---|---  
> / | Delimiter between function names. Function before the delimiter should precede the function after the delimiter in the call stack backtrace. | `FuncA/FuncB` `FuncA/\*/FuncB`  
> [ | Function is at the bottom of the stack | `[FuncA` `[FuncA/+/FuncB`  
> ] | Function is at the top of the stack | `FuncB]` `FuncA/\*/FuncB]`  
> + | Only one function between the two other functions | `FuncA/+/FuncB`  
> * | Zero or more function(s) between the two other functions | `FuncA/\*/FuncB`  
> @ | Specify file name. If not mentioned, assuming <default file> which will match all filenames. You can mention only the filename or a path to the file. | `FileA.py@FuncA` `FileB.cpp@FuncB/\*/FuncD]` `/path/to/FileB.cpp@FuncB]`  
> @@ | Specify module name. If not mentioned, assuming <default module> which will match all modules. Similar to file name, you can mention only the module name or a path to the module. | `ModuleA@@FuncA` `ModuleA@FileA.cpp@FuncA`  
>   
> Include kernels launched via function ‘FuncA’ in file ‘FileA.py’ across all modules:
>         
>         ncu --call-stack-type python --python-include FileA.py@FuncA python CuPythonScript.py
>         
> 
> Include kernels launched via function ‘FuncA’ in file ‘FileA.cpp’ and ‘ModuleA:
>         
>         ncu --call-stack-type native --native-include ModuleA@FileA.cpp@FuncA CuApp.exe
>         
> 
> Exclude kernels launched via function ‘FuncA’ of a native call stack frame but include kernels launched via function ‘FuncB’ of a python call stack frame:
>         
>         ncu --call-stack-type native --call-stack-type python --native-exclude FuncA --python-include FuncB python CuPythonScript.py
>         

  * **Regular Expression Support**

To use regular expressions, follow the following syntax.

    * Provide prefix ‘regex:’ to treat the config as regular expression.

The kernels launched via function with matching regex ‘Func[0-9]’ of file with matching regex ‘File[A-Z]’ are profiled.
          
          ncu --call-stack-type native --native-include "regex:File[A-Z]@Func[0-9]" CuApp.exe
          

The kernels launched via function with matching regex ‘Func[0-9]’ of file with matching regex ‘File[A-Z]’ of module with matching regex ‘Module[A-Z]’ are profiled.
          
          ncu --call-stack-type native --native-include "regex:Module[A-Z]@File[A-Z]@Func[0-9]" CuApp.exe
          

  * **Additional Information**
        
        ncu --call-stack-type native --native-include "FileA.cpp@FuncA,FileB.cpp@FuncB" CuApp.exe
        

This will not work as you might expect. You cannot pass multiple configurations in a single include/exclude option. You must use the option multiple times to pass multiple configurations.
        
        ncu --call-stack-type native --native-include @FuncA --native-include @FileA.cpp@FuncB CuApp.exe
        

Both configurations are invalid as you cannot pass an empty <file> or <module> before the delimiter ‘@’.


Look at [NVTX Push/Pop Range Filtering](index.html#nvtx-filtering) section to understand the function expression format in detail.

## 4.4. Command Line Options

For long command line options, passing a unique initial substring can be sufficient.

### 4.4.1. General

General Command Line Options Option | Description | Default  
---|---|---  
h,help | Show help message |   
v,version | Show version information |   
mode | Select the mode of interaction with the target application

  * **launch-and-attach:** Launch the target application and immediately attach for profiling.
  * **launch:** Launch the target application and suspend in the first intercepted API call, wait for tool to attach.
  * **attach:** Attach to a previously launched application to which no other tool is attached.

| launch-and-attach  
p,port | Base port used for connecting to target applications for `--mode launch/attach` | 49152  
max-connections | Maximum number of ports for connecting to target applications | 64  
config-file | Use config.ncu-cfg config file to set parameters. Searches in the current working directory, in “$HOME/.config/NVIDIA Corporation” on Linux and in “%APPDATA%\NVIDIA Corporation\” on Windows. | on  
config-file-path | Override the default path for config file. |   
  
### 4.4.2. Launch

Launch Command Line Options Option | Description | Default  
---|---|---  
check-exit-code | Check the application exit code and print an error if it is different than 0. If set, `--replay-mode application` will stop after the first pass if the exit code is not 0. | yes  
forward-signals | Forwards SIGINT and SIGTERM signals to the application root process. On Windows, only CTRL_C_EVENT (equivalent to SIGINT) is handled. **Note:** \- On Windows, CTRL_C_EVENT is forwarded to the entire process group, with this option specified NCU will skip the signal. \- On both Windows and Linux, Ctrl+C input from the terminal sends the signal to entire process group and behavior of signal forwarding is unspecified. | false  
injection-path-64 | Override the default path for the injection libraries. The injection libraries are used by the tools to intercept relevant APIs (like CUDA or NVTX). |   
preload-library | Prepend a shared library to be loaded by the application before the injection libraries. This option can be given multiple times and the libraries will be loaded in the order they were specified. |   
call-stack | Enable CPU Call Stack collection. | false  
call-stack-type | Set the call stack type(s) that should be collected. More than one type may be specified. Implies –call-stack. Available modes are:

  * **native:** Collect a regular CPU call stack.
  * **python:** Collect a Python CPU call stack.

Note that Python call stack collection requires CPython version 3.9 or later. | native **Examples** `--call-stack-type native --call-stack-type python`  
communicator | Enable Multiprocess Communicator.

  * **none:** Disables the communicator.
  * **tcp:** TCP-based inter-process tree communication for multiple ncu instances.
  * **shmem:** Shared memory-based intra-process tree communication for one ncu instance.

| none  
communicator-tcp-num-peers | Set number of expected communicator peers (number of NCU instances). | 2  
communicator-tcp-hostname | Set hostname/IP address for the TCP communicator server. | 127.0.0.1  
communicator-tcp-port | Set port for the TCP communicator server. | 49217  
lockstep-kernel-launch | Launch kernels in lockstep across all communicator instances. | false  
lockstep-nvtx-include | Adds include statement to the NVTX filter, for kernels to be synchronized. See [NVTX expressions](index.html#nvtx-filtering). |   
lockstep-nvtx-exclude | Adds exclude statement to the NVTX filter, for kernels to be synchronized. See [NVTX expressions](index.html#nvtx-filtering). |   
nvtx | Enable NVTX support for tools. | false  
nvtx-push-pop-scope | Specify scope of push-pop ranges.

  * **thread:** NVTX Push/Pop ranges are scoped per thread.
  * **process:** NVTX push/pop ranges are scoped per process.

| thread  
target-processes | Select the processes you want to profile. Available modes are:

  * **application-only** Profile only the root application process.
  * **all** Profile the application and all its child processes.

| all  
target-processes-filter | Set the comma separated expressions to filter which processes are profiled.

  * `<process name>` Set the exact process name to include for profiling.
  * `regex:<expression>` Set the regex to filter matching process name profiling. On shells that recognize regular expression symbols as special characters (e.g. Linux bash), the expression needs to be escaped with quotes, e.g. `--target-processes-filter regex:".*Process"`. When using `regex:`, the expression must not include any commas.
  * `exclude:<process name>` Set the exact process name to exclude for profiling.
  * `exclude-tree:<process name>` Set the exact process name to exclude for profiling and further process tracking. None of its child processes will be profiled, even if they match a positive filter. This option is not available on Windows.

The executable name part of the process will be considered in the match. Processing of filters stops at the first match. If any positive filter is specified, no process that is not matching a positive filter is profiled. | **Examples** `--target-processes-filter MatrixMul` Filter all processes having executable name exactly as “MatrixMul”. `--target-processes-filter regex:Matrix`Filter all processes that include the string “Matrix” in their executable name, e.g. “MatrixMul” and “MatrixAdd”. `--target-processes-filter MatrixMul,MatrixAdd`Filter all processes having executable name exactly as “MatrixMul” or “MatrixAdd”. `--target-processes-filter exclude:MatrixMul.exe` Exclude only “MatrixMul.exe”. `--target-processes-filter exclude-tree:ChildLauncher,ParentProcess` Exclude “ChildLauncher” and all its sub-processes. Include (only) “ParentProcess”, but not if it’s a child of “ChildLauncher”.  
support-32bit | Support profiling processes launched from 32-bit applications. This option is only available on x86_64 Linux. On Windows, tracking 32-bit applications is enabled by default. | no  
null-stdin | Launch the application with ‘/dev/null’ as its standard input. This avoids applications reading from standard input being stopped by `SIGTTIN` signals and hanging when running as backgrounded processes. | false  
  
### 4.4.3. Attach

Attach Command Line Options Option | Description | Default  
---|---|---  
hostname | Set the hostname or IP address for connecting to the machine on which the target application is running. When attaching to a local target application, use 127.0.0.1. | 127.0.0.1  
  
### 4.4.4. Profile

Profile Command Line Options Option | Description | Default/Examples  
---|---|---  
devices | List the GPU devices to enable profiling on, separated by comma. [1](#fn1) | All devices **Examples** `--devices 0,2`  
filter-mode | Set the filtering mode for kernel launches. Available modes:

  * **global:** Apply provided launch filters on kernel launches collectively.
  * **per-gpu:** Apply provided launch filters on kernel launches separately on each device. Effective launch filters for this mode are `--launch-count` and `--launch-skip`
  * **per-launch-config:** Apply kernel filters and launch filters on kernel launches separately for each GPU launch parameter i.e. Grid Size, Block Size and Shared Memory.

| global  
kernel-id | Set the identifier to use for matching kernels. If the kernel does not match the identifier, it will be ignored for profiling. The identifier must be of the following format: _context-id:stream-id:[name-operator:]kernel-name:invocation-nr_

  * **context-id** is the CUDA context ID or regular expression of context id, NVTX name.
  * **stream-id** is the CUDA stream ID or regular expression of stream id, NVTX name.
  * **name-operator** is an optional operator to _kernel-name_. Currently, _regex_ is the only supported operator.
  * **kernel-name** is the expression to match the kernel name. By default, this is a full, literal match to what is specified by `--kernel-name-base`. When specifying the optional _regex_ name operator, this is a partial regular expression match to what is specified by `--kernel-name-base`.
  * **invocation-nr** is the N’th invocation of matching kernel filter i.e. ctx id, stream id, kernel name, grid dimensions, block dimensions and shared memory bytes are all considered for invocation count. If ctx id or stream id is not provided then respective id is not considered for invocation count. If Multiple invocations can also be specified using regular expressions. Multiple invocations can also be specified using regular expressions.

If the context/stream ID is a positive number, it will be strictly matched against the CUDA context/stream ID. Otherwise it will be treated as a regular expression and matched against the context/stream name specified using the NVTX library. [1](#fn1) | **Examples** `--kernel-id ::foo:2` For kernel “foo”, match the second invocation. `--kernel-id :::".*5|3"` For all kernels, match the third invocation, and all for which the invocation number ends in “5”. `--kernel-id ::regex:^.*foo$:` Match all kernels ending in “foo”. `--kernel-id ::regex:^(?!foo):` Match all kernels except those starting with “foo”. Note that depending on your OS and shell, ` you might need to quote the expression, e.g. using single quotes in Linux _bash_ : `--kernel-id ::regex:'^(?!foo)':` `--kernel-id 1|5:2::7` Match all seventh kernel invocations of kernels lauched from context 1 + stream 2, and context 5 + stream 2.  
k,kernel-name | Set the expression to use when matching kernel names.

  * `<kernel name>` Set the kernel name for an exact match.
  * `regex:<expression>` Set the regex to use for matching the kernel name. On shells that recognize regular expression symbols as special characters (e.g. Linux bash), the expression needs to be escaped with quotes, e.g. `--kernel-name regex:".*Foo"`.

If the kernel name or the provided expression do not match, it will be ignored for profiling. [1](#fn1) | **Examples** `-k foo` Match all kernels named exactly “foo”. `-k regex:foo` Match all kernels that include the string “foo”, e.g. “foo” and “fooBar”. `-k regex:"foo|bar"` Match all kernels including the strings “foo” or “bar”, e.g. “foo”, “foobar”, “_bar2”.  
kernel-name-base | Set the basis for `--kernel-name`, and `--kernel-id` kernel-name. [1](#fn1) Options are:

  * **function:** Function name without parameters, templates etc. e.g. `dmatrixmul`
  * **demangled:** Demangled function name, including parameters, templates, etc. e.g. `dmatrixmul(float*,int,int)`. Use [Kernel Renaming](index.html#kernel-renaming) to rename the demangled name.
  * **mangled:** Mangled function name. e.g. `_Z10dmatrixmulPfiiS_iiS_`

| function  
rename-kernels | Perform simplification on the kernel demangled names. Rename demangled names using a config file. See [Kernel Renaming](index.html#kernel-renaming) for more details. | on  
rename-kernels-export | Export demangled names from the report to a new file and specify mappings for renaming them. Use `--rename-kernels-path` option to specify the export file path. | off  
rename-kernels-path | Override the default path of the configuration file which should be used while importing renamed kernels or exporting demangled names. Only valid while using `--rename-kernels` or `--rename-kernels-export`. |   
c,launch-count | Limit the number of profiled kernel launches. The count is only incremented for launches that match the kernel filters.[1](#fn1) |   
s,launch-skip | Set the number of kernel launches to skip before starting to profile kernels. The number takes into account only launches that match the kernel filters. [1](#fn1) | 0  
launch-skip-before-match | Set the number of kernel launches to skip before starting to profile. The count is incremented for all launches, regardless of the kernel filters. [1](#fn1) | 0  
range-filter | Filter to profile specified instance(s) of matching NVTX ranges or start/stop ranges created through cu(da)ProfilerStart/Stop APIs. Specify in format _[yes/no/on/off]:[start/stop range instance(s)]:[NVTX range instance(s)]_

  * [yes/no/on/off] : default is ‘no/off’. If set to ‘yes/on’ then NVTX range numbering starts from 1 inside every start/stop range.
  * provide numbers in regex form e.g, [2-4] or 2|3|4 to profile 2nd, 3rd and 4th instance of the matching range.
  * NVTX range numbers will be counted for matching range provided using –nvtx-include.

| **Examples** `--range-filter :2:3 --nvtx-include A/` Match 2nd start/stop range and also 3rd NVTX push/pop range A in the app. `--range-filter yes:2:3 --nvtx-include A/` Match 3rd NVTX push/pop range A from 2nd start/stop range.  
kill | Terminate the target application when the requested –launch-count was profiled. Allowed values:

  * on/off
  * yes/no

| no  
replay-mode | Mechanism used for replaying a kernel launch multiple times to collect all requested profiling data:

  * **kernel:** Replay individual kernel launches “transparently” during the execution of the application. See [Kernel Replay](../ProfilingGuide/index.html#kernel-replay) for more details.
  * **application:** Relaunch the entire application multiple times. Requires deterministic program execution. See [Application Replay](../ProfilingGuide/index.html#application-replay) for more details.
  * **range:** Replay ranges of CUDA API calls and kernel launches “transparently” during the execution of the application. Ranges must be defined using `cu(da)ProfilerStart/Stop` API pairs or [NVTX expressions](index.html#nvtx-filtering). See [Range Replay](../ProfilingGuide/index.html#range-replay) for more details.
  * **app-range:** Profile ranges without API capture by relaunching the entire application multiple times. Requires deterministic program execution. Ranges must be defined using `cu(da)ProfilerStart/Stop` API pairs or [NVTX expressions](index.html#nvtx-filtering). See [Application Range Replay](../ProfilingGuide/index.html#application-range-replay) for more details.

| kernel  
app-replay-buffer | Application replay buffer location.

  * **file:** Replay pass data is buffered in a temporary file. The report is created after profiling completed. This mode is more scalable, as the amount of required memory does not scale with the number of profiled kernels.
  * **memory:** Replay pass data is buffered in memory, and the report is created while profiling. This mode can result in better performance if the filesystem is slow, but the amount of required memory scales with the number of profiled kernels.

| file  
app-replay-match | Application replay kernel matching strategy. For all options, kernels are matched on a per-process and per-device (GPU) basis. Below options are used to configure the applied strategy in more detail.

  * **name:** Kernels are matched in the following order: 1. (mangled) name, 2. order of execution
  * **grid:** Kernels are matched in the following order: 1. (mangled) name, 2. CUDA grid/block size, 3. order of execution
  * **all:** Kernels are matched in the following order: 1. (mangled) name, 2. CUDA grid/block size, 3. CUDA context ID, 4. CUDA stream ID, 5. order of execution

| grid  
app-replay-mode | Application replay kernel matching mode:

  * **strict:** Requires all filtered kernels to match across all replay passes in the exact order.
  * **balanced:** Requires all filtered kernels to match across all replay passes, without strict ordering.
  * **relaxed:** Produces results only for filtered kernels that could be matched across replay passes. Unmatched kernels are dropped.

| balanced  
range-replay-options | Range replay options, separated by comma. Below options are supported:

  * **enable-greedy-sync** Insert ctx sync for applicable deferred APIs during capture.
  * **disable-host-save** Disable saving device-mapped host allocations for every kernel but capture it only once before the first kernel launch.
  * **disable-host-restore** Disable restoring device-written host allocations.
  * **disable-dependent-kernel-detection** Disable dependent kernel detection before capturing device-mapped host allocations for each kernel.

| none  
graph-profiling | CUDA graph profiling mode:

  * **node** Profile individual kernel nodes as regular CUDA kernels.
  * **graph** Profile entire graphs as one workload (but disable profiling of individual graph kernel nodes). See the [Profiling Guide](../ProfilingGuide/index.html#graph-profiling) for more information on this mode.

| node  
list-sets | List all section sets found in the searched section folders and exit. For each set, the associated sections are shown, as well as the estimated number of metrics collected as part of this set. This number can be used as an estimate of the relative profiling overhead per kernel launch of this set. |   
set | Identifier of section set to collect. If not specified, the `basic` set is collected. The full set of sections can be collected with `--set full`. | If no `--set` option is given, the `basic` set is collected. If not specified and `--section` or `--metrics` are used, no sets are collected. Use `--list-sets` to see which set is the default.  
list-sections | List all sections found in the searched section folders and exit. |   
section | Add a section identifier to collect in one of the following ways:

  * `<section identifier>` Set the section identifier for an exact match.
  * `regex:<expression>` Regular expression allows matching full section identifier. For example, `.*Stats`, matches all sections ending with ‘Stats’. On shells that recognize regular expression symbols as special characters (e.g. Linux bash), the expression needs to be escaped with quotes, e.g. `--section "regex:.*Stats"`.

This option is ignored when used with `--import` and `--page raw` or `--page source`. [1](#fn1) | If no `--section` options are given, the sections associated with the `basic` set are collected. If no sets are found, all sections are collected.  
section-folder | Add a non-recursive search path for `.section` and `.py` rule files. Section files in this folder will be made available to the `--section` option. Individual files from directories including an `.ncu-ignore` file are ignored. | If no `--section-folder` options are given, the `sections` folder is added by default.  
section-folder-recursive | Add a recursive search path for `.section` and `.py` rule files. Section files in this folder and all folders below will be made available to the `--section` option. Individual files from directories including an `.ncu-ignore` file are ignored. | If no `--section-folder` options are given, the `sections` folder is added by default.  
list-rules | List all rules found in the searched section folders and exit. |   
apply-rules | Apply active and applicable rules to each profiling result. Use `--rule` to limit which rules to apply. Allowed values:

  * on/off
  * yes/no

| yes  
rule | Add a rule identifier to apply. Implies `--apply-rules yes`. | If no `--rule` options are given, all applicable rules in the `sections` folder are applied.  
import-sass | Import ELF cubins into the report. Cubins include SASS, PTX and meta information. Allowed values:

  * on/off
  * yes/no

Disabling this is not recommended, as it removes most source-related metrics and analysis. | yes  
import-source | If available from -lineinfo, correlated CUDA source files are permanently imported into the report. Allowed values:

  * on/off
  * yes/no

Use `--source-folders` option to provide missing source files. | no  
source-folders | Add comma separated recursive search paths for missing CUDA source files to import into the report. |   
list-metrics | List all metrics collected from active sections. If the list of active sections is restricted using the `--section` option, only metrics from those sections will be listed. |   
query-metrics | Query available metrics for the devices on system. Use `--devices` and `--chips` to filter which devices to query. Note that by default, listed metric names need to be appended a valid suffix in order for them to become valid metrics. See `--query-metrics-mode` for how to get the list of valid suffixes, or check the [Profiling Guide](../ProfilingGuide/index.html#metrics-structure). |   
query-metrics-mode | Set the mode for querying metrics. Implies `--query-metrics`. Available modes:

  * **base:** Only the base names of the metrics.
  * **suffix:** Suffix names for the base metrics. This gives the list of all metrics derived from the base metrics. Use `--metrics` to specify the base metrics to query.
  * **all:** Full names for all metrics. This gives the list of all base metrics and their suffix metrics.

| base  
query-metrics-collection | Set which metric collection kind to query. Implies `--query-metrics`. Available collections:

  * **device:** Query CUDA device attributes.
  * **groups:** Query metric groups available for profiling.
  * **launch:** Query launch attributes.
  * **numa:** Query NUMA topology metrics.
  * **nvlink:** Query NVLink topology metrics.
  * **occupancy:** Query occupancy calculation metrics.
  * **pmsampling:** Query metrics available for [PM sampling](../ProfilingGuide/index.html#pm-sampling).
  * **profiling:** Query metrics available for profiling.
  * **source:** Query source metrics available for profiling.
  * **stats:** Query metrics generated by the profiler to inform about profiling statistics.
  * **warpsampling:** Query metrics available for [Warp sampling](../ProfilingGuide/index.html#warp-sampling).

| profiling  
metrics | Specify all metrics to be profiled, separated by comma. If no `--section` options are given, only the temporary section containing all metrics listed using this option is collected. If `--section` options are given in addition to `--metrics`, all metrics from those sections and from `--metrics` are collected. Names passed to this option support the following prefixes:

  * `regex:<expression>` expands to all metrics that partially match the expression. Enclose the regular expression in ^…$ to force a full match.
  * `group:<name>` lists all metrics of the metric group with that name. See section files for valid group names.
  * `breakdown:<metric>` expands to the input metrics of the high-level throughput metric.
  * `pmsampling:<metric>` collects the metric using PM sampling. Only single-pass metrics that don’t require SASS-patching (_sass_) are supported. Using this prefix adds a timeline element to the report’s details page.

Combining multiple prefixes is not supported. If a metric requires a suffix to be valid, and neither `regex:` nor `group:` are used, this option automatically expands the name to all available first-level sub-metrics. When importing a report, `:group` and `:breakdown` are not supported. When using `regex:`, the expression must not include any commas. [1](#fn1) |   
metric-distribution-groups | Set the number of groups to distribute metrics to. | 1  
disable-extra-suffixes | Disable the collection of extra suffixes (avg, min, max, sum) for all metrics. Only collect what is explicitly specified. |   
list-chips | List all supported chips that can be used with `--chips`. |   
chips | Specify the chips for querying metrics, separated by comma. | **Examples** `--chips tu102,gh100`  
profile-from-start | Set if application should be profiled from its start. Allowed values:

  * on/off
  * yes/no

| yes  
disable-profiler-start-stop | Disable profiler start/stop. When enabled, `cu(da)ProfilerStart/Stop` API calls are ignored. |   
quiet | Suppress all profiling output. |   
verbose | Make profiler output more verbose. |   
cache-control | Control the behavior of the GPU caches during profiling. Allowed values:

  * **all:** All GPU caches are flushed before each kernel replay iteration during profiling. While metric values in the execution environment of the application might be slightly different without invalidating the caches, this mode offers the most reproducible metric results across the replay passes and also across multiple runs of the target application.
  * **none:** No GPU caches are flushed during profiling. This can improve performance and better replicates the application behavior if only a single kernel replay pass is necessary for metric collection. However, some metric results will vary depending on prior GPU work, and between replay iterations. This can lead to inconsistent and out-of-bounds metric values.

| all  
clock-control | Control the behavior of the GPU clocks during profiling. Allowed values:

  * **base:** GPC and memory clocks are locked to their respective base (rated tdp) frequency during profiling.
  * **boost:** GPC and memory clocks are locked to their respective turbo boost frequency during profiling. If setting boost clock is not supported by GPU or driver, falls back to locking clocks to ‘base’ if possible.
  * **force-boost:** GPC and memory clocks are locked to their respective turbo boost frequency during profiling. Supports Ampere+ on NVIDIA kernel mode driver 560 and Turing+ on driver 580 or newer.
  * **none:** No GPC or memory frequencies are changed during profiling.
  * **reset:** Reset GPC and memory clocks for all or the selected devices and exit. Use if a previous, killed execution of ncu left the GPU clocks in a locked state.

This has no impact on thermal throttling. Note that actual clocks might still vary, depending on the level of driver support for this feature. As an alternative, use `nvidia-smi` to lock the clocks externally and set this option to `none`. | boost  
pipeline-boost-state | Control the Tensor Core boost state. Setting stable Tensor Core boosting is recommended for application performance profiling ensuring predictive run to run performance. Allowed values:

  * **stable:** Set the Tensor Core boost state to stable.
  * **dynamic:** Set the Tensor Core boost state to dynamic.

| stable  
nvtx-include | Adds an include statement to the [NVTX filter](index.html#nvtx-filtering), which allows selecting kernels to profile based on NVTX ranges. [1](#fn1) |   
nvtx-exclude | Adds an exclude statement to the [NVTX filter](index.html#nvtx-filtering), which allows selecting kernels to profile based on NVTX ranges. [1](#fn1) |   
native-include | Adds an include statement to the [Native call stack filtering](index.html#cpu-stack-filtering), which allows selecting kernels to profile based on native CPU stack frames. [1](#fn1) |   
native-exclude | Adds an exclude statement to the [Native call stack filtering](index.html#cpu-stack-filtering), which allows selecting kernels to profile based on native CPU stack frames. [1](#fn1) |   
python-include | Adds an include statement to the [Python call stack filtering](index.html#cpu-stack-filtering), which allows selecting kernels to profile based on python CPU stack frames. [1](#fn1) |   
python-exclude | Adds an exclude statement to the [Python call stack filtering](index.html#cpu-stack-filtering), which allows selecting kernels to profile based on python CPU stack frames. [1](#fn1) |   
  
1([1](#id7),[2](#id8),[3](#id9),[4](#id10),[5](#id11),[6](#id12),[7](#id13),[8](#id14),[9](#id15),[10](#id16),[11](#id17),[12](#id18),[13](#id19),[14](#id20),[15](#id21))
    

This filtering option is available when using `--import`.

### 4.4.5. PM Sampling

These options apply to [PM sampling](../ProfilingGuide/index.html#pm-sampling). See [here](index.html#warp-sampling) for options used in warp state sampling.

PM Sampling Command Line Options Option | Description | Default  
---|---|---  
pm-sampling-interval | Set the PM sampling interval in cycles or ns (depending on the architecture), or determine dynamically when 0. | 0 (auto)  
pm-sampling-buffer-size - | Set the size of the device-sided allocation for PM sampling in bytes, or determine dynamically when 0. | 0 (auto)  
pm-sampling-max-passes - | Set the maximum number of passes used for PM sampling, or determine dynamically when 0. | 0 (auto)  
warp-samples-per-interval | Set the number of warp samples per PM sampling interval, or determine dynamically when 0. | 0 (auto)  
disable-pm-warp-sampling | Disables warp state sampling during PM sampling. | By default, warp state sampling is enabled with the PM sampling.  
  
### 4.4.6. Warp Sampling

These options apply to [Warp State Sampling](../ProfilingGuide/index.html#statistical-sampler). See [here](index.html#pm-sampling) for options used in PM sampling.

Warp Sampling Command Line Options Option | Description | Default  
---|---|---  
warp-sampling-interval | Set the sampling period in the range of [0..31]. The actual frequency is 2 ^ (5 + value) cycles. If set to ‘auto’, the profiler tries to automatically determine a high sampling frequency without skipping samples or overflowing the output buffer. | auto  
warp-sampling-max-passes | Set maximum number of passes used for sampling (see the [Profiling Guide](../ProfilingGuide/index.html#overhead) for more details on profiling overhead). | 5  
warp-sampling-buffer-size | Set the size of the device-sided allocation for samples in bytes. | 32*1024*1024  
  
### 4.4.7. File

File Command Line Options Option | Description | Default  
---|---|---  
log-file | Send all tool output to the specified file, or one of the standard channels. The file will be overwritten. If the file doesn’t exist, a new one will be created.”stdout” as the whole file name indicates standard output channel (stdout). “stderr” as the whole file name indicates standard error channel (stderr).” | If `--log-file` is not set , profile results will be printed on the console.  
o,export | Set the output file for writing the profile report. If not set, a temporary file will be used which is removed afterwards. Use with `--import` option to save filtered results. See [Filtered Profile Export](index.html#filtered-profile-export) for more details. The specified name supports macro expansion. See [File Macros](index.html#command-line-options-file-macros) for more details. | If `--export` is set and no `--page` option is given, no profile results will be printed on the console.  
f,force-overwrite | Force overwriting all output files. | By default, the profiler won’t overwrite existing output files and show an error instead.  
i,import | Set the input file for reading the profile results. |   
open-in-ui | Open report in UI instead of showing result on terminal. (Only available on host platforms) |   
section-folder-restore | Restores stock files to the default section folder or the folder specified by an accompanying _–section-folder_ option. If the operation will overwrite modified files then the _–force-overwrite_ option is required. |   
  
### 4.4.8. Console Output

Console Output Command Line Options Option | Description | Default  
---|---|---  
csv | Use comma-separated values as console output. Implies –print-units base by default. |   
page | Select the report page to print console output for. Available pages are:

  * **details** Show results grouped as sections, include rule results. Some metrics that are collected by default (e.g. device attributes) are omitted if not specified explicitly in any section or using `--metrics`.
  * **raw** Show all collected metrics by kernel launch.
  * **source** Show source. See `--print-source` to select the source view.
  * **session** Show launch settings, session info, process info and device attributes.

| **details**. If no `--page` option is given and `--export` is set, no results are printed to the console output.  
print-source | Select the source view:

  * **sass** Show SASS (assembly) instructions for each kernel launch.
  * **ptx** Show PTX source of every cubin whose at least one kernel is profiled.
  * **cuda** Show entire CUDA-C source file which has kernel code as per kernel launch. CLI shows CUDA source only if file exists on the host machine.
  * **cuda,sass** Show SASS CUDA-C source correlation for each kernel launch. CLI shows CUDA source only if file exists on the host machine.

Metric correlation with source is available in **sass** , and **cuda,sass** source view. Metrics specified with `--metrics` and specified section file with `--section` are correlated. Consider restricting the number of selected metrics such that values fit into a single output row. | sass  
resolve-source-file | Resolve CUDA source file in the `--page source` output. Provide comma separated files full path. |   
print-details | Select which part of a section should be shown in the details page output:

  * **header** Show all metrics from header of the section.
  * **body** Show all metrics from body of the section.
  * **all** Show all metrics from the section.

Replaces deprecated option `--details-all`. | header  
print-metric-name | Select one of the option to show it in the Metric Name column:

  * **label** Show metric label.
  * **name** Show metric name.
  * **label-name** Show both metric label and metric name.

| label  
print-units | Select the mode for scaling of metric units. Available modes are:

  * **auto** Show all metrics automatically scaled to the most fitting order of magnitude.
  * **base** Show all metrics in their base unit.

| auto  
print-metric-attribution | Show the attribution level for metrics of Green Context results. | false  
print-fp | Show all numeric metrics in the console output as floating point numbers. | false  
print-kernel-base | Set the basis for kernel name output. See `--kernel-regex-base` for options. | demangled  
print-metric-instances | Set output mode for metrics with instance values:

  * **none** Only show GPU aggregate value.
  * **values** Show GPU aggregate followed by all instance values.
  * **details** Show aggregate value, followed by correlation IDs and instance values

| none  
print-nvtx-rename | Select how NVTX should be used for renaming:

  * **none** Don’t use NVTX for renaming.
  * **kernel** Rename kernels with the most recent enclosing NVTX push/pop range.

| none  
print-rule-details | Print additional details for rule results, such as tables and metrics that are key performance indicators. Currently has no effect in CSV mode. | false  
print-summary | Select the summary output mode. Available modes are:

  * **none** No summary.
  * **per-gpu** Summary for each GPU.
  * **per-kernel** Summary for each kernel type.
  * **per-nvtx** Summary for each NVTX context.

| none  
  
### 4.4.9. Response File

Response files can be specified by adding `@FileName` to the command line. The file name must immediately follow the `@` character. The content of each response file is inserted in place of the corresponding response file option.

### 4.4.10. File Macros

The file name specified with option `-o` or `--export` supports the following macro expansions. Occurrences of these macros in the report file name are replaced by the corresponding character sequence. If not specified otherwise, the macros cannot be used as part of the file path.

Macro Expansions Macro | Description  
---|---  
%h | Expands to the host name of the machine on which the command line profiler is running.  
%q{ENV_NAME} | Expands to the content of the variable with the given name `ENV_NAME` from the environment of the command line profiler.  
%p | Expands to the process ID of the command line profiler.  
%i | Expands to the lowest unused positive integer number that guarantees the resulting file name is not yet used. This macro can only be used once in the output file name.  
%% | Expands to a single `%` character in the output file name. This macro can be used in the file path and the file name.  
  
### 4.4.11. MPS

MPS Command Line Options Option | Description | Default  
---|---|---  
mps | Select MPS behavior. Available modes are:

>   * **none** No MPS support
>   * **client** Launch MPS client process
>   * **primary-client** Launch primary MPS client process
>   * **control** Launch MPS ncu control
> 


| none  
mps-num-clients | Number of MPS client processes. |   
mps-timeout | Timeout (seconds) to discover MPS client processes. |   
Behavior for Combinations of `mps-timeout` and `mps-num-clients` mps-timeout | mps-num-clients | Behavior  
---|---|---  
Specified | Not Specified | Control Process waits for the specified timeout and if after the timeout no clients are discovered, it exits.  
Not Specified | Specified | Control process waits until the specified number of clients are discovered.  
Specified | Specified | Control Process waits up to the specified timeout and if number of clients discovered are not at least the number of clients specified, it exits.  
Not Specified | Not Specified | **Error** : At least one of `--mps-timeout` or `--mps-num-clients` must be specified.  
  
## 4.5. Environment Variables

The following environment variables can be set before launching NVIDIA Nsight Compute CLI, or the UI, respectively.

Environment Variables Name | Description | Default/Values  
---|---|---  
NV_COMPUTE_PROFILER_DISABLE_STOCK_FILE_DEPLOYMENT | Disable file deployment to the versioned `Sections` directory, using section and rule files from the stock directory within the installation instead. By default, the versioned directory from the user’s documents folder is used to ensure that any potential user updates are taken into account. Only supported in the NVIDIA Nsight Compute CLI. | Default: unset Set to “1” to disable deployment.  
NV_COMPUTE_PROFILER_LOCAL_CONNECTION_OVERRIDE | Override the default local connection mechanism between frontend and profiled target processes. The default mechanism is platform-dependent. This should only be used if there are connection problems between frontend and target processes in a local launch. | Default: unset (use default mechanism) Set to “uds” to use Unix Domain Socket connections (available on Posix platforms, only). Set to “tcp” to use TCP (available on all platforms). Set to “named-pipes” to use Windows Named Pipes (available on Windows, only).  
NV_COMPUTE_PROFILER_DISABLE_SW_PRE_PASS | Disable the instruction-level software (SW) metric pre-pass. When collecting SW-patched metrics, such as `inst_executed`, the pre-pass is used to determine which functions are executed as part of the kernel and should be patched. This requires a separate replay pass, and if only instruction-level SW metrics are to be collected, prevents single-pass data collection. Disabling the pre-pass can improve performance if memory save-and-restore is undesirable and application replay is not possible. | Default: unset (use pre-pass when applicable) Set to “1” to disable pre-pass.  
NV_COMPUTE_PROFILER_DISABLE_CONCURRENT_PROFILING | Disable concurrent profiling. If set, only one NCU process will be allowed to profile a kernel in the system at any given time. The default behavior is one NCU process per device or MIG instance. | Default: unset Set to “1” to disable concurrent profiling.  
NV_COMPUTE_PROFILER_CUDA_LOG_LINES | Number of CUDA error log lines to be included in NVLOG output on unrecoverable profiling errors. Logs are reset after each profiling pass. | Default: 5  
  
## 4.6. Nvprof Transition Guide

Refer to the [older documentation](../Archives/index.html).

Notices

Notices

ALL NVIDIA DESIGN SPECIFICATIONS, REFERENCE BOARDS, FILES, DRAWINGS, DIAGNOSTICS, LISTS, AND OTHER DOCUMENTS (TOGETHER AND SEPARATELY, “MATERIALS”) ARE BEING PROVIDED “AS IS.” NVIDIA MAKES NO WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO THE MATERIALS, AND EXPRESSLY DISCLAIMS ALL IMPLIED WARRANTIES OF NONINFRINGEMENT, MERCHANTABILITY, AND FITNESS FOR A PARTICULAR PURPOSE.

Information furnished is believed to be accurate and reliable. However, NVIDIA Corporation assumes no responsibility for the consequences of use of such information or for any infringement of patents or other rights of third parties that may result from its use. No license is granted by implication of otherwise under any patent rights of NVIDIA Corporation. Specifications mentioned in this publication are subject to change without notice. This publication supersedes and replaces all other information previously supplied. NVIDIA Corporation products are not authorized as critical components in life support devices or systems without express written approval of NVIDIA Corporation.

Trademarks

NVIDIA and the NVIDIA logo are trademarks or registered trademarks of NVIDIA Corporation in the U.S. and other countries. Other company and product names may be trademarks of the respective companies with which they are associated.

* * *
