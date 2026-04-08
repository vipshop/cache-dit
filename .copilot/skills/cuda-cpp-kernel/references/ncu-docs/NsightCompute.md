---
url: https://docs.nvidia.com/nsight-compute/NsightCompute/index.html
---

[ ![Logo](https://docs.nvidia.com/nsight-compute/_static/nsight-compute.png) ](../index.html)

Nsight Compute

  * [1\. Release Notes](../ReleaseNotes/index.html)
  * [2\. Profiling Guide](../ProfilingGuide/index.html)
  * [3\. Nsight Compute](#)
    * [3.1. Introduction](#introduction)
      * [3.1.1. Overview](#overview)
    * [3.2. Quickstart](#quickstart)
      * [3.2.1. Interactive Profile Activity](#interactive-profile-activity)
      * [3.2.2. Non-Interactive Profile Activity](#non-interactive-profile-activity)
      * [3.2.3. System Trace Activity](#system-trace-activity)
      * [3.2.4. Navigate the Report](#navigate-the-report)
    * [3.3. Start Activity Dialog](#start-activity-dialog)
      * [3.3.1. Remote Connections](#remote-connections)
      * [3.3.2. Interactive Profile Activity](#connection-activity-interactive)
      * [3.3.3. Profile Activity](#profile-activity)
      * [3.3.4. Reset](#reset)
    * [3.4. Main Menu and Toolbar](#main-menu-and-toolbar)
      * [3.4.1. Main Menu](#id2)
      * [3.4.2. Main Toolbar](#main-toolbar)
      * [3.4.3. Status Banners](#status-banners)
    * [3.5. Tool Windows](#tool-windows)
      * [3.5.1. API Statistics](#api-statistics)
      * [3.5.2. API Stream](#api-stream)
      * [3.5.3. Baselines](#baselines)
      * [3.5.4. Metric Details](#metric-details)
      * [3.5.5. Launch Details](#launch-details)
        * [Header](#header)
        * [Body](#body)
      * [3.5.6. Function Stats](#function-stats)
        * [Accessing Function Stats](#accessing-function-stats)
        * [Basic Workflow](#basic-workflow)
        * [Understanding the Function Stats Table](#understanding-the-function-stats-table)
        * [Data Source](#data-source)
        * [Metric Naming](#metric-naming)
      * [3.5.7. NVTX](#nvtx)
      * [3.5.8. CPU Call Stack](#cpu-call-stack)
      * [3.5.9. Resources](#resources)
        * [Memory Allocations](#memory-allocations)
        * [Graphviz DOT and SVG exports](#graphviz-dot-and-svg-exports)
      * [3.5.10. CUDA Graph Viewer](#cuda-graph-viewer)
      * [3.5.11. Search](#search)
      * [3.5.12. Metric Selection](#metric-selection)
    * [3.6. Profiler Report](#profiler-report)
      * [3.6.1. Header](#profiler-report-header)
      * [3.6.2. Report Pages](#report-pages)
        * [Summary Page](#summary-page)
          * [Properties](#properties)
        * [Details Page](#details-page)
          * [Overview](#id5)
          * [Memory](#memory)
          * [Occupancy](#occupancy)
          * [Range Replay](#range-replay)
          * [Rooflines](#rooflines)
          * [Source](#source)
          * [Timelines](#timelines)
        * [Source Page](#source-page)
          * [Navigation](#navigation)
          * [Metrics](#metrics)
            * [Metrics Correlation](#metrics-correlation)
            * [Pre-Defined Source Metrics](#pre-defined-source-metrics)
            * [Register Dependencies](#register-dependencies)
          * [Profiles](#profiles)
          * [Additional Tables](#additional-tables)
            * [Instructions & Dependencies Table](#instructions-dependencies-table)
            * [Inline Functions Table](#inline-functions-table)
            * [Source Markers Table](#source-markers-table)
            * [Statistics Table](#statistics-table)
          * [Limitations](#limitations)
        * [Context Page](#context-page)
        * [Comments Page](#comments-page)
        * [Raw Page](#raw-page)
        * [Session Page](#session-page)
      * [3.6.3. Metrics and Units](#metrics-and-units)
      * [3.6.4. Filtered Profiler Report](#filtered-profiler-report)
    * [3.7. Baselines](#id8)
    * [3.8. Standalone Source Viewer](#standalone-source-viewer)
    * [3.9. Source Comparison](#source-comparison)
    * [3.10. Report Merge Tool](#report-merge-tool)
      * [3.10.1. File Selection Tab](#file-selection-tab)
      * [3.10.2. Metric Filters Tab](#metric-filters-tab)
      * [3.10.3. Result Filters Tab](#result-filters-tab)
        * [Result Merge Operations](#result-merge-operations)
        * [Merged Report](#merged-report)
    * [3.11. Clustering Window](#clustering-window)
      * [3.11.1. How to Use Clustering](#how-to-use-clustering)
      * [3.11.2. Clustering Tree](#clustering-tree)
      * [3.11.3. Similarity Matrix](#similarity-matrix)
    * [3.12. Occupancy Calculator](#occupancy-calculator)
      * [3.12.1. Tables](#tables)
      * [3.12.2. Utilization](#utilization)
      * [3.12.3. Graphs](#graphs)
      * [3.12.4. GPU Data](#gpu-data)
    * [3.13. Green Contexts support](#green-contexts-support)
    * [3.14. Acceleration Structure Viewer](#acceleration-structure-viewer)
      * [3.14.1. Navigation](#as-viewer-nav)
      * [3.14.2. Filtering and Highlighting](#filtering-and-highlighting)
      * [3.14.3. Rendering Options](#rendering-options)
      * [3.14.4. Exporting](#exporting)
    * [3.15. Options](#options)
      * [3.15.1. Profile](#profile)
        * [Report Sections](#report-sections)
        * [Report Rules](#report-rules)
        * [Report UI](#report-ui)
        * [Report Baselines](#report-baselines)
        * [Report Metrics](#report-metrics)
        * [Report Summary Page](#report-summary-page)
        * [Report Details Page](#report-details-page)
        * [Report Source Page](#report-source-page)
        * [API Stream View](#api-stream-view)
      * [3.15.2. Fonts and Colors](#fonts-and-colors)
        * [Fonts](#fonts)
        * [Colors](#colors)
      * [3.15.3. Environment](#environment)
        * [Visual Experience](#visual-experience)
        * [Windowing](#windowing)
        * [Documents Folder](#documents-folder)
        * [Startup Behavior](#startup-behavior)
        * [Updates](#updates)
      * [3.15.4. Connection](#connection)
        * [Target Connection Properties](#target-connection-properties)
        * [Host Connection Properties](#host-connection-properties)
        * [SSH ProxyJump Connection Properties](#ssh-proxyjump-connection-properties)
      * [3.15.5. Timeline](#timeline)
        * [Basic Settings](#basic-settings)
      * [3.15.6. Source Lookup](#source-lookup)
      * [3.15.7. Send Feedback…](#send-feedback)
    * [3.16. Projects](#projects)
      * [3.16.1. Project Dialogs](#project-dialogs)
      * [3.16.2. Project Explorer](#project-explorer)
    * [3.17. Visual Studio Integration Guide](#visual-studio-integration-guide)
      * [3.17.1. Visual Studio Integration Overview](#visual-studio-integration-overview)
  * [4\. Nsight Compute CLI](../NsightComputeCli/index.html)


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
  * 3\. Nsight Compute
  *   * v2026.1.0 | [Archive](https://developer.nvidia.com/nsight-compute-history)


* * *

# 3\. Nsight Compute

The User Guide for Nsight Compute.

## 3.1. Introduction

NVIDIA Nsight Compute (UI) User Manual: Detailed documentation on views, controls, and workflows.

### 3.1.1. Overview

This document is a user guide to the next-generation NVIDIA Nsight Compute profiling tools. NVIDIA Nsight Compute is an interactive kernel profiler for CUDA applications. It provides detailed performance metrics and API debugging via a user interface and command line tool. In addition, its baseline feature allows users to compare results within the tool. NVIDIA Nsight Compute provides a customizable and data-driven user interface and metric collection and can be extended with analysis scripts for post-processing results.

**Important Features**

  * Interactive kernel profiler and API debugger

  * Graphical profile report

  * Result comparison across one or multiple reports within the tool

  * Fast Data Collection

  * UI and Command Line interface

  * Fully customizable reports and analysis rules


## 3.2. Quickstart

The following sections provide brief step-by-step guides of how to setup and run NVIDIA Nsight Compute to collect profile information. All directories are relative to the base directory of NVIDIA Nsight Compute, unless specified otherwise.

The UI executable is called ncu-ui. A shortcut with this name is located in the base directory of the NVIDIA Nsight Compute installation. The actual executable is located in the folder `host\windows-desktop-win7-x64` on Windows or `host/linux-desktop-glibc_2_11_3-x64` on Linux. By default, when installing from a Linux `.run` file, NVIDIA Nsight Compute is located in `/usr/local/cuda-<cuda-version>/nsight-compute-<version>`. When installing from a `.deb` or `.rpm` package, it is located in `/opt/nvidia/nsight-compute/<version>` to be consistent with [Nsight Systems](https://developer.nvidia.com/nsight-systems). In Windows, the default path is `C:\Program Files\NVIDIA Corporation\Nsight Compute <version>`.

After starting NVIDIA Nsight Compute, by default the _Welcome Page_ is opened. The _Start_ section allows the user to start a new activity, open an existing report, create a new project or load an existing project. The _Continue_ section provides links to recently opened reports and projects. The _Explore_ section provides information about what is new in the latest release, as well as links to additional training. See [Environment](index.html#options-environment) on how to change the start-up action.

![../_images/welcome-page.png](https://docs.nvidia.com/nsight-compute/_images/welcome-page.png)

Welcome Page

### 3.2.1. Interactive Profile Activity

  1. **Launch the target application from NVIDIA Nsight Compute**

When starting NVIDIA Nsight Compute, the _Welcome Page_ will appear. Click on _Quick Launch_ to open the _Connection_ dialog. If the _Connection_ dialog doesn’t appear, you can open it using the _Connect_ button from the main toolbar, as long as you are not currently connected. Select your target platform on the left-hand side and your connection target (machine) from the _Connection_ drop down. If you have your local target platform selected, `localhost` will become available as a connection. Use the + button to add a new connection target. Then, continue by filling in the details in the _Launch_ tab. In the _Activity_ panel, select the Interactive Profile activity to initiate a session that allows controlling the execution of the target application and selecting the kernels of interest interactively. Press _Launch_ to start the session.

![../_images/quick-start-interactive-profiling-connect.png](https://docs.nvidia.com/nsight-compute/_images/quick-start-interactive-profiling-connect.png)
  2. **Launch the target application with tools instrumentation from the command line**

The ncu can act as a simple wrapper that forces the target application to load the necessary libraries for tools instrumentation. The parameter `--mode=launch` specifies that the target application should be launched and suspended before the first instrumented API call. That way the application waits until we connect with the UI.
         
         $ ncu --mode=launch CuVectorAddDrv.exe
         

  3. **Launch NVIDIA Nsight Compute and connect to target application**

![../_images/quick-start-interactive-profiling-attach.png](https://docs.nvidia.com/nsight-compute/_images/quick-start-interactive-profiling-attach.png)

Select the target machine at the top of the dialog to connect and update the list of attachable applications. By default, _localhost_ is pre-selected if the target matches your current local platform. Select the _Attach_ tab and the target application of interest and press _Attach_. Once connected, the layout of NVIDIA Nsight Compute changes into stepping mode that allows you to control the execution of any calls into the instrumented API. When connected, the _API Stream_ window indicates that the target application waits before the very first API call.

![../_images/quick-start-interactive-profiling-connected.png](https://docs.nvidia.com/nsight-compute/_images/quick-start-interactive-profiling-connected.png)
  4. **Control application execution**

Use the _API Stream_ window to step the calls into the instrumented API. The dropdown at the top allows switching between different CPU threads of the application. _Step In_ (F11), _Step Over_ (F10), and _Step Out_ (Shift + F11) are available from the _Debug_ menu or the corresponding toolbar buttons. While stepping, function return values and function parameters are captured.

![../_images/quick-start-interactive-profiling-api-stream.png](https://docs.nvidia.com/nsight-compute/_images/quick-start-interactive-profiling-api-stream.png)

Use _Resume_ (F5) and _Pause_ to allow the program to run freely. Freeze control is available to define the behavior of threads currently not in focus, i.e. selected in the thread drop down. By default, the _API Stream_ stops on any API call that returns an error code. This can be toggled in the _Debug_ menu by _Break On API Error_.

  5. **Isolate a kernel launch**

To quickly isolate a kernel launch for profiling, use the _Run to Next Kernel_ button in the toolbar of the _API Stream_ window to jump to the next kernel launch. The execution will stop before the kernel launch is executed.

![../_images/quick-start-interactive-profiling-next-launch.png](https://docs.nvidia.com/nsight-compute/_images/quick-start-interactive-profiling-next-launch.png)
  6. **Profile a kernel launch**

Once the execution of the target application is suspended at a kernel launch, additional actions become available in the UI. These actions are either available from the menu or from the toolbar. Please note that the actions are disabled, if the API stream is not at a qualifying state (not at a kernel launch or launching on an unsupported GPU). To profile, press _Profile Kernel_ and wait until the result is shown in the [Profiler Report](index.html#profiler-report). Profiling progress is reported in the lower right corner status bar.

Instead of manually selecting _Profile_ , it is also possible to enable _Auto Profile_ from the _Profile_ menu. If enabled, each kernel matching the current kernel filter (if any) will be profiled using the current section configuration. This is especially useful if an application is to be profiled unattended, or the number of kernel launches to be profiled is very large. Sections can be enabled or disabled using the [Metric Selection](index.html#tool-window-sections-info) tool window.

_Profile Series_ allows to configure the collection of a set of profile results at once. Each result in the set is profiled with varying parameters. Series are useful to investigate the behavior of a kernel across a large set of parameters without the need to recompile and rerun the application many times.


For a detailed description of the options available in this activity, see [Interactive Profile Activity](index.html#connection-activity-interactive).

### 3.2.2. Non-Interactive Profile Activity

  1. **Launch the target application from NVIDIA Nsight Compute**

When starting NVIDIA Nsight Compute, the _Welcome Page_ will appear. Click on _Start Activity_ to open the _Start Activity_ dialog. If the _Start Activity_ dialog doesn’t appear, you can open it using the _Start Activity_ button from the main toolbar, as long as you are not currently connected. Select your target platform on the left-hand side and your localhost from the _Connection_ drop down. Then, fill in the launch details. In the _Activity_ panel, select the _Profile_ activity to initiate a session that pre-configures the profile session and launches the command line profiler to collect the data. Provide the _Output File_ name to enable starting the session with the _Launch_ button.

![../_images/quick-start-profiling-connect.png](https://docs.nvidia.com/nsight-compute/_images/quick-start-profiling-connect.png)
  2. **Additional Launch Options**

For more details on these options, see [Command Line Options](../NsightComputeCli/index.html#command-line-options). The options are grouped into tabs: The _Filter_ tab exposes the options to specify which kernels should be profiled. Options include the kernel regex filter, the number of launches to skip, and the total number of launches to profile. The _Sections_ tab allows you to select which sections should be collected for each kernel launch. Hover over a section to see its description as a tool-tip. To change the sections that are enabled by default, use the [Metric Selection](index.html#tool-window-sections-info) tool window. The _Sampling_ tab allows you to configure sampling options for each kernel launch. The _Other_ tab includes the option to collect NVTX information or custom metrics via the `--metrics` option.

![../_images/quick-start-profiling-options-sections.png](https://docs.nvidia.com/nsight-compute/_images/quick-start-profiling-options-sections.png)


For a detailed description of the options available in this activity, see [Profile Activity](index.html#connection-activity-non-interactive).

### 3.2.3. System Trace Activity

  1. **Launch the target application from NVIDIA Nsight Compute**

When starting NVIDIA Nsight Compute, the _Welcome Page_ will appear. Click on _Start Activity_ to open the _Start Activity_ dialog. If the _Start Activity_ dialog doesn’t appear, you can open it using the _Start Activity_ button from the main toolbar, as long as you are not currently connected. Select your local target platform on the left-hand side and your localhost from the _Connection_ drop down. Then, fill in the launch details. In the _Activity_ panel, select the _System Trace_ activity to initiate a session with pre-configured settings. Press _Launch_ to start the session.

![../_images/quick-start-system-trace-connect.png](https://docs.nvidia.com/nsight-compute/_images/quick-start-system-trace-connect.png)
  2. **Additional Launch Options**

For more details on these options, see [System-Wide Profiling Options](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#linux-system-wide-profiling-options).

![../_images/quick-start-system-trace-options.png](https://docs.nvidia.com/nsight-compute/_images/quick-start-system-trace-options.png)
  3. Once the session is completed, the [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) report is opened in a new document. By default, the timeline view is shown. It provides detailed information of the activity of the CPU and GPUs and helps understanding the overall behavior and performance of application. Once a CUDA kernel is identified to be on the critical path and not meeting the performance expectations, right click on the kernel launch on timeline and select _Profile Kernel_ from the context menu. A new [Start Activity](index.html#connection-dialog) opens up that is already preconfigured to profile the selected kernel launch. Proceed with optimizing the selected kernel using [Non-Interactive Profile Activity](index.html#quick-start-non-interactive)

![../_images/quick-start-system-trace-timeline.png](https://docs.nvidia.com/nsight-compute/_images/quick-start-system-trace-timeline.png)


### 3.2.4. Navigate the Report

  1. **Navigate the report**

The profile report comes up by default on the _Summary_ page. It shows an overview table to summarize all results in the report. It also shows rule information for the selected row.

> ![../_images/quick-start-report-summary.png](https://docs.nvidia.com/nsight-compute/_images/quick-start-report-summary.png)

You can switch between different [Report Pages](index.html#profiler-report-pages) using the tab bar on the top-left of the report. You can also use _Ctrl + Shift + N_ and _Ctrl + Shift + P_ shortcut keys or corresponding toolbar button to navigate next and previous pages, respectively. A report can contain any number of results. The _Current_ dropdown allows switching between the different results in a report.

![../_images/quick-start-report.png](https://docs.nvidia.com/nsight-compute/_images/quick-start-report.png)
  2. **Diffing multiple results**

On the _Details_ page, use the _Compare - Add Baseline_ button for the current result to become the baseline all other results from this report and any other report opened in the same instance of NVIDIA Nsight Compute get compared to. When a baseline is set, every element on the Details page shows two values: The current value of the result in focus and the corresponding value of the baseline or the percentage of change from the corresponding baseline value.

![../_images/quick-start-baseline.png](https://docs.nvidia.com/nsight-compute/_images/quick-start-baseline.png)

Use the _Clear Baselines_ entry from the same group button, the Profile menu or the corresponding toolbar button to remove all baselines. For more information see [Baselines](index.html#baselines).

  3. **Following rules**

On the _Details_ page, many sections provide rules with valuable information on detected problems and optimization suggestions. Rules can be user-defined too. For more information, see the [Customization Guide](../CustomizationGuide/index.html#rule-system).

![../_images/quick-start-rule.png](https://docs.nvidia.com/nsight-compute/_images/quick-start-rule.png)


## 3.3. Start Activity Dialog

Use the _Start Activity Dialog_ to launch and attach to applications on your local and remote platforms. Start by selecting the _Target Platform_ for profiling. By default (and if supported) your local platform will be selected. Select the platform on which you would like to start the target application or connect to a running process.

![../_images/connection-dialog.png](https://docs.nvidia.com/nsight-compute/_images/connection-dialog.png)

When using a remote platform, you will be asked to select or create a _Connection_ in the top drop down. To create a new connection, select _+_ and enter your connection details. When using the local platform, _localhost_ will be selected as the default and no further connection settings are required. You can still create or select a remote connection, if profiling will be on a remote system of the same platform.

Depending on your target platform, select either _Launch_ or _Remote Launch_ to launch an application for profiling on the target. Note that _Remote Launch_ will only be available if supported on the target platform.

Fill in the following launch details for the application:

  * **Application Executable:** Specifies the root application to launch. Note that this may not be the final application that you wish to profile. It can be a script or launcher that creates other processes.

  * **Working Directory:** The directory in which the application will be launched.

  * **Command Line Arguments:** Specify the arguments to pass to the application executable.

  * **Environment:** The environment variables to set for the launched application.


Select _Attach_ to attach the profiler to an application already running on the target platform. This application must have been started using another NVIDIA Nsight Compute CLI instance. The list will show all application processes running on the target system which can be attached. Select the refresh button to re-create this list.

Finally, select the _Activity_ to be run on the target for the launched or attached application. Note that not all activities are necessarily compatible with all targets and connection options. Currently, the following activities exist:

  * [Interactive Profile Activity](index.html#connection-activity-interactive)

  * [Profile Activity](index.html#connection-activity-non-interactive)

  * [System Trace Activity](index.html#quick-start-system-trace)

  * [Occupancy Calculator](index.html#occupancy-calculator)


### 3.3.1. Remote Connections

Remote devices that support SSH can also be configured as a target in the _Start Activity Dialog_. To configure a remote device, ensure an SSH-capable _Target Platform_ is selected, then press the _+_ button. The following configuration dialog will be presented.

![../_images/add-remote-connection.png](https://docs.nvidia.com/nsight-compute/_images/add-remote-connection.png)

NVIDIA Nsight Compute supports both password and private key authentication methods. In this dialog, select the authentication method and enter the following information:

  * **Password**

    * **IP/Host Name:** The IP address or host name of the target device.

    * **User Name:** The user name to be used for the SSH connection.

    * **Password:** The user password to be used for the SSH connection.

    * **Port:** The port to be used for the SSH connection. (The default value is 22)

    * **Deployment Directory:** The directory to use on the target device to deploy supporting files. The specified user must have write permissions to this location.

    * **Connection Name:** The name of the remote connection that will show up in the _Start Activity Dialog_. If not set, it will default to <User>@<Host>:<Port>.

  * **Private Key**

![../_images/add-remote-connection-private-key.png](https://docs.nvidia.com/nsight-compute/_images/add-remote-connection-private-key.png)
    * **IP/Host Name:** The IP address or host name of the target device.

    * **User Name:** The user name to be used for the SSH connection.

    * **SSH Private Key:** The private key that is used to authenticate to SSH server.

    * **SSH Key Passphrase:** The passphrase for your private key.

    * **Port:** The port to be used for the SSH connection. (The default value is 22)

    * **Deployment Directory:** The directory to use on the target device to deploy supporting files. The specified user must have write permissions to this location.

    * **Connection Name:** The name of the remote connection that will show up in the _Start Activity Dialog_. If not set, it will default to <User>@<Host>:<Port>.


In addition to keyfiles specified by path and plain password authentication, NVIDIA Nsight Compute supports keyboard-interactive authentication, standard keyfile path searching and SSH agents.

You can also specify placeholders in the _Deployment Directory_ field. These placeholders will be replaced with the actual values when the connection is used. The following placeholders are supported.

Deployment Directory Placeholders Placeholder | Description  
---|---  
%C | User cache directory. Resolves to $XDG_CACHE_HOME, if empty resolves to $HOME/.cache  
%E | User config directory. Resolves to $XDG_CONFIG_HOME, if empty resolves to $HOME/.config  
%h | User home directory. Resolves to $HOME, if empty resolves to /root  
%H | Host name. Resolves to $HOSTNAME, if empty resolves to host  
%T | Temporary directory. Resolves to $TMPDIR, if empty resolves to /tmp  
%u | User name. Resolves to $USER, if empty resolves to root  
%S | State directory. Resolves to $XDG_STATE_HOME, if empty resolves to $HOME/.local/state  
  
For example, if you specify the _Deployment Directory_ as _%T/nsight-compute_ , the actual directory used will be _/tmp/nsight-compute_ on the remote device.

When all information is entered, click the _Add_ button to make use of this new connection.

When a remote connection is selected in the _Start Activity Dialog_ , the _Application Executable_ file browser will browse the remote file system using the configured SSH connection, allowing the user to select the target application on the remote device.

When an activity is launched on a remote device, the following steps are taken:

  1. The command line profiler and supporting files are copied into the _Deployment Directory_ on the remote device. (Only files that do not exist or are out of date are copied.)

  2. Communication channels are opened to prepare for the traffic between the UI and the _Application Executable_.

     * For _Interactive Profile_ activities, a _SOCKS proxy_ is started on the host machine.

     * For _Non-Interactive Profile_ activities, a remote forwarding channel is opened on the target machine to tunnel profiling information back to the host.

  3. The _Application Executable_ is executed on the remote device.

     * For _Interactive Profile_ activities, a connection is established to the remote application and the profiling session begins.

     * For _Non-Interactive Profile_ activities, the remote application is executed under the command line profiler and the specified report file is generated.

  4. For non-interactive profiling activities, the generated report file is copied back to the host, and opened.


The progress of each of these steps is presented in the _Progress Log_.

![../_images/progress-log.png](https://docs.nvidia.com/nsight-compute/_images/progress-log.png)

Progress Log

Note that once either activity type has been launched remotely, the tools necessary for further profiling sessions can be found in the _Deployment Directory_ on the remote device.

On Linux and Mac host platforms, NVIDIA Nsight Compute supports SSH remote profiling on target machines which are not directly addressable from the machine the UI is running on through the `ProxyJump` and `ProxyCommand` SSH options.

These options can be used to specify intermediate hosts to connect to or actual commands to run to obtain a socket connected to the SSH server on the target host and can be added to your SSH configuration file.

Note that for both options, NVIDIA Nsight Compute runs external commands and does not implement any mechanism to authenticate to the intermediate hosts using the credentials entered in the [Start Activity Dialog](index.html#connection-dialog). These credentials will only be used to authenticate to the final target in the chain of machines.

When using the `ProxyJump` option NVIDIA Nsight Compute uses the _OpenSSH client_ to establish the connection to the intermediate hosts. This means that in order to use `ProxyJump` or `ProxyCommand`, a version of OpenSSH supporting these options must be installed on the host machine.

A common way to authenticate to the intermediate hosts in this case is to use a _SSH agent_ and have it hold the private keys used for authentication.

Since the _OpenSSH SSH client_ is used, you can also use the _SSH askpass_ mechanism to handle these authentications in an interactive manner.

It might happen on slow networks that connections used for remote profiling through SSH time out. If this is the case, the `ConnectTimeout` option can be used to set the desired timeout value.

A known limitation of the remote profiling through SSH is that problems may arise if NVIDIA Nsight Compute tries to do remote profiling through _SSH_ by connecting to the same machine it is running on. In this case, the workaround is to do local profiling through `localhost`.

For more information about available options for the _OpenSSH client_ and the ecosystem of tools it can be used with for authentication refer to the official [manual pages](https://www.openssh.com/manual.html).

### 3.3.2. Interactive Profile Activity

The _Interactive Profile_ activity allows you to initiate a session that controls the execution of the target application, similar to a debugger. You can step API calls and workloads (CUDA kernels), pause and resume, and interactively select the kernels of interest and which metrics to collect.

This activity does currently not support profiling or attaching to child processes.

  * **Enable CPU Call Stack**

Collect the CPU-sided Call Stack at the location of each profiled kernel launch.

  * **CPU Call Stack Types**

If “Enable CPU Call Stack” is set to “Yes”, the type(s) of call stack may be selected here.

  * **Enable NVTX Support**

Collect NVTX information provided by the application or its libraries. Required to support stepping to specific NVTX contexts.

  * **Disable Profiling Start/Stop**

Ignore calls to `cu(da)ProfilerStart` or `cu(da)ProfilerStop` made by the application.

  * **Enable Profiling From Start**

Enables profiling from the application start. Disabling this is useful if the application calls `cu(da)ProfilerStart` and kernels before the first call to this API should not be profiled. Note that disabling this does not prevent you from manually profiling kernels.

  * **Cache Control**

Control the behavior of the GPU caches during profiling. Allowed values: For _Flush All_ , all GPU caches are flushed before each kernel replay iteration during profiling. While metric values in the execution environment of the application might be slightly different without invalidating the caches, this mode offers the most reproducible metric results across the replay passes and also across multiple runs of the target application.

For _Flush None_ , no GPU caches are flushed during profiling. This can improve performance and better replicates the application behavior if only a single kernel replay pass is necessary for metric collection. However, some metric results will vary depending on prior GPU work, and between replay iterations. This can lead to inconsistent and out-of-bounds metric values.

  * **Clock Control**

Control the behavior of the GPU clocks during profiling. Allowed values: For _Base_ , GPC and memory clocks are locked to their respective base frequency during profiling. This has no impact on thermal throttling. For _Boost_ , GPC and memory clocks are locked to their respective turbo boost frequency during profiling. If setting boost clock is not supported by the GPU or driver, falls back to locking clocks to ‘base’ if possible. For _Force-Boost_ , GPC and memory clocks are locked to their respective turbo boost frequency during profiling. Supports Ampere+ on NVIDIA kernel mode driver 560 and Turing+ on driver 580 or newer. For _None_ , no GPC or memory frequencies are changed during profiling.

  * **Import Source**

Enables permanently importing available source files into the report. Missing source files are searched in [Source Lookup](index.html#options-source-lookup) folders. Source information must be embedded in the executable, e.g. via the `-lineinfo` compiler option. Imported files are used in the _CUDA-C_ view on the [Source Page](index.html#profiler-report-source-page).


  * **Graph Profiling**

Set if CUDA graphs should be stepped and profiled as individual _Nodes_ or as complete _Graphs_. See the [Profiling Guide](../ProfilingGuide/index.html#graph-profiling) for more information on this mode.


  * **Pipeline Boost State**

Control the Tensor Core boost state. It is recommended to set the stable Tensor Core boosting for application profiling, ensuring predictive run-to-run performance. By default, the boost state is set to `Auto`, which attempts to set the stable boost state on supported platforms and drivers.


### 3.3.3. Profile Activity

The _Profile_ activity provides a traditional, pre-configurable profiler. After configuring which kernels to profile, which metrics to collect, etc, the application is run under the profiler without interactive control. The activity completes once the application terminates. For applications that normally do not terminate on their own, e.g. interactive user interfaces, you can cancel the activity once all expected kernels are profiled.

This activity does not support attaching to processes previously launched via NVIDIA Nsight Compute. These processes will be shown grayed out in the _Attach_ tab.

  * **Output File**

Path to report file where the collected profile should be stored. If not present, the report extension `.ncu-rep` is added automatically. The placeholder `%i` is supported for the filename component. It is replaced by a sequentially increasing number to create a unique filename. This maps to the `--export` command line option.

  * **Force Overwrite**

If set, existing report file are overwritten. This maps to the `--force-overwrite` command line option.

  * **Target Processes**

Select the processes you want to profile. In mode _Application Only_ , only the root application process is profiled. In mode _all_ , the root application process and all its child processes are profiled. This maps to the `--target-processes` command line option.

  * **Replay Mode**

Select the method for replaying kernel launches multiple times. In mode _Kernel_ , individual kernel launches are replayed transparently during the single execution of the target application. In mode _Application_ , the entire target application is relaunched multiple times. In each iteration, additional data for the target kernel launches is collected. Application replay requires the program execution to be deterministic. This maps to the `--replay-mode` command line option. See the [Profiling Guide](../ProfilingGuide/index.html#kernel-replay) for more details on the replay modes.


  * **Graph Profiling**

Set if CUDA graphs should be profiled as individual _Nodes_ or as complete _Graphs_.


  * **Additional Options**

All remaining options map to their command line profiler equivalents. See the [Command Line Options](../NsightComputeCli/index.html#command-line-options) for details.


### 3.3.4. Reset

Entries in the connection dialog are saved as part of the current [project](index.html#projects). When working in a custom project, simply close the project to reset the dialog.

When not working in a custom project, entries are stored as part of the _default project_. You can delete all information from the default project by closing NVIDIA Nsight Compute and then [deleting the project file from disk](index.html#projects).

## 3.4. Main Menu and Toolbar

Information on the main menu and toolbar.

![../_images/main-menu.png](https://docs.nvidia.com/nsight-compute/_images/main-menu.png)

### 3.4.1. Main Menu

  * File

    * **New Project** Create new profiling [Projects](index.html#projects) with the [New Project Dialog](index.html#projects-dialog).

    * **Open Project** Open an existing profiling project.

    * **Recent Projects** Open an existing profiling project from the list of recently used projects.

    * **Save Project** Save the current profiling project.

    * **Save Project As** Save the current profiling project with a new filename.

    * **Close Project** Close the current profiling project.

    * **New File** Create a new file.

    * **Open File** Open an existing file.

    * **Open Remote File**

Download an existing file from a remote host and open it locally. The opened file will only exist in memory and will not be written to the local machine’s disk unless the user explicitly saves it. For more information concerning the selection of a remote host to download the file from, see the section about [Remote Connections](index.html#remote-connections).

Only a subset of file types that are supported locally can be opened from a remote target. The following table lists file types that can be opened remotely.

Remote File Type Support Extensions | Description | Supported  
---|---|---  
ncu-rep | Nsight Compute Profiler Report | Yes  
ncu-repz | Nsight Compute Profiler Report (zstd compressed) | Yes  
ncu-occ | Occupancy Calculator File | Yes  
ncu-bvh | OptiX AS Viewer File | Yes (except on macOS)  
section | Section Description | No  
cubin | Cubin File | No  
cuh,h,hpp | Header File | No  
c,cpp,cu | Source File | No  
txt | Text file | No  
nsight-cuprof-report | Nsight Compute Profiler Report (legacy) | Yes  
    * **Save** Save the current file

    * **Save As** Save a copy of the current file with a different name or type or in a different location.

    * **Save All Files** Save all open files.

    * **Close** Close the current file.

    * **Close All Files** Close all open files.

    * **Recent Files** Open an existing file from the list of recently used files.

    * **Exit** Exit Nsight Compute.

  * Connection

    * **Start Activity** Open the [Start Activity Dialog](index.html#connection-dialog) to launch or attach to a target application. Disabled when already connected.

    * **Disconnect** Disconnect from the current target application, allows the application to continue normally and potentially re-attach.

    * **Terminate** Disconnect from and terminate the current target application immediately.

  * Debug

    * **Pause** Pause the target application at the next intercepted API call or launch.

    * **Resume** Resume the target application.

    * **Step In** Step into the current API call or launch to the next nested call, if any, or the subsequent API call, otherwise.

    * **Step Over** Step over the current API call or launch and suspend at the next, non-nested API call or launch.

    * **Step Out** Step out of the current nested API call or launch to the next, non-parent API call or launch one level above.

    * **Freeze API**

When disabled, all CPU threads are enabled and continue to run during stepping or resume, and all threads stop as soon as at least one thread arrives at the next API call or launch. This also means that during stepping or resume the currently selected thread might change as the old selected thread makes no forward progress and the API Stream automatically switches to the thread with a new API call or launch. When enabled, only the currently selected CPU thread is enabled. All other threads are disabled and blocked.

Stepping now completes if the current thread arrives at the next API call or launch. The selected thread never changes. However, if the selected thread does not call any further API calls or waits at a barrier for another thread to make progress, stepping may not complete and hang indefinitely. In this case, pause, select another thread, and continue stepping until the original thread is unblocked. In this mode, only the selected thread will ever make forward progress.

    * **Break On API Error** When enabled, during resume or stepping, execution is suspended as soon as an API call returns an error code. Note that some non-zero return codes are expected and not necessarily errors. In addition, certain errors may be ignored if they don’t happen on a nested API call, as the top-level call may still handle them.

    * **Run to Next Kernel** See [API Stream](index.html#tool-window-api-stream) tool window.

    * **Run to Next API Call** See [API Stream](index.html#tool-window-api-stream) tool window.

    * **Run to Next Range Start** See [API Stream](index.html#tool-window-api-stream) tool window.

    * **Run to Next Range End** See [API Stream](index.html#tool-window-api-stream) tool window.

    * **API Statistics** Opens the [API Statistics](index.html#tool-window-api-statistics) tool window

    * **API Stream** Opens the [API Stream](index.html#tool-window-api-stream) tool window

    * **Resources** Opens the [Resources](index.html#tool-window-resources) tool window

    * **NVTX** Opens the [NVTX](index.html#tool-window-nvtx) tool window

  * Profile

    * **Profile Kernel** When suspended at a kernel launch, select the profile using the current configuration.

    * **Profile Series** When suspended at a kernel launch, open the Profile Series configuration dialog to setup and collect a series of profile results.

    * **Auto Profile** Enable or disable auto profiling. If enabled, each kernel matching the current kernel filter (if any) will be profiled using the current section configuration.

    * **Baselines** Opens the [Baselines](index.html#tool-window-baselines) tool window.

    * **Clear Baselines** Clear all current baselines.

    * **Import Source** Permanently import resolved source files into the report. Existing content may be overwritten.

    * **Section/Rules Info** Opens the [Metric Selection](index.html#tool-window-sections-info) tool window.

    * **Function Stats** Opens the [Function Stats](index.html#tool-window-function-stats) tool window.

  * Tools

    * **Project Explorer** Opens the [Project Explorer](index.html#projects-explorer) tool window.

    * **Output Messages** Opens the Output Messages tool window.

    * **Options** Opens the [Options](index.html#options) dialog.

    * **Search** Opens the [Search](index.html#tool-window-search) tool window.

  * Window

    * **Save Window Layout** Allows you to specify a name for the current layout. The layouts are saved to a Layouts folder in the documents directory as named “.nvlayout” files.

    * **Apply Window Layout** Once you have saved a layout, you can restore them by using the “Apply Window Layout” menu entry. Simply select the entry from sub-menu you want to apply.

    * **Manage Window Layout** Allows you to delete or rename old layouts.

    * **Restore Default Layout** Restore views to their original size and position.

    * **Show Welcome Page** Opens the [Welcome Page](index.html#quick-start__fig-welcome-page).

  * Help

    * **Documentation** Opens the latest documentation for NVIDIA Nsight Compute online.

    * **Documentation (local)** Opens the local HTML documentation for NVIDIA Nsight Compute that has shipped with the tool.

    * **Check For Updates** Checks online if a newer version of NVIDIA Nsight Compute is available for download.

    * **Reset Application Data** Reset all NVIDIA Nsight Compute configuration data saved on disk, including option settings, default paths, recent project references etc. This will not delete saved reports.

    * **Send Feedback** Opens a dialog that allows you to send bug reports and suggestions for features. Optionally, the feedback includes basic system information, screenshots, or additional files (such as profile reports).

    * **About** Opens the About dialog with information about the version of NVIDIA Nsight Compute.


### 3.4.2. Main Toolbar

The main toolbar shows commonly used operations from the main menu. See [Main Menu](index.html#main-menu) for their description.

### 3.4.3. Status Banners

Status banners are used to display important messages, such as profiler errors. The message can be dismissed by clicking the ‘X’ button. The number of banners shown at the same time is limited and old messages can get dismissed automatically if new ones appear. Use the _Output Messages_ window to see the complete message history.

![../_images/status-banner.png](https://docs.nvidia.com/nsight-compute/_images/status-banner.png)

## 3.5. Tool Windows

### 3.5.1. API Statistics

The _API Statistics_ window is available when NVIDIA Nsight Compute is connected to a target application. It opens by default as soon as the connection is established. It can be re-opened using _Debug > API Statistics_ from the main menu.

![../_images/tool-window-api-statistics.png](https://docs.nvidia.com/nsight-compute/_images/tool-window-api-statistics.png)

Whenever the target application is suspended, it shows a summary of tracked API calls with some statistical information, such as the number of calls, their total, average, minimum and maximum duration. Note that this view cannot be used as a replacement for [Nsight Systems](https://developer.nvidia.com/nsight-systems) when trying to optimize CPU performance of your application.

The _Reset_ button deletes all statistics collected to the current point and starts a new collection. Use the _Export to CSV_ button to export the current statistics to a CSV file.

### 3.5.2. API Stream

The _API Stream_ window is available when NVIDIA Nsight Compute is connected to a target application. It opens by default as soon as the connection is established. It can be re-opened using _Debug > API Stream_ from the main menu.

![../_images/tool-window-api-stream.png](https://docs.nvidia.com/nsight-compute/_images/tool-window-api-stream.png)

Whenever the target application is suspended, the window shows the history of API calls and traced kernel launches. The currently suspended API call or kernel launch (activity) is marked with a yellow arrow. If the suspension is at a subcall, the parent call is marked with a green arrow. The API call or kernel is suspended before being executed.

For each activity, further information is shown such as the kernel name or the function parameters (_Func Parameters_) and return value (_Func Return_). Note that the function return value will only become available once you step out or over the API call.

Use the _Current Thread_ dropdown to switch between the active threads. The dropdown shows the thread ID followed by the current API name. One of several options can be chosen in the trigger dropdown, which are executed by the adjacent _> >_ button. _Run to Next Kernel_ resumes execution until the next kernel launch is found in any enabled thread. _Run to Next API Call_ resumes execution until the next API call matching _Next Trigger_ is found in any enabled thread. _Run to Next Range Start_ resumes execution until the next start of an active profiler range is found. Profiler ranges are defined by using the `cu(da)ProfilerStart/Stop` API calls. _Run to Next Range Stop_ resumes execution until the next stop of an active profiler range is found. The _API Level_ dropdown changes which API levels are shown in the stream. The _Export to CSV_ button exports the currently visible stream to a CSV file.

### 3.5.3. Baselines

The _Baselines_ tool window can be opened by clicking the _Baselines_ entry in the _Profile_ menu. It provides a centralized place from which to manage configured baselines. (Refer to [Baselines](index.html#baselines), for information on how to create baselines from profile results.)

![../_images/tool-window-baselines.png](https://docs.nvidia.com/nsight-compute/_images/tool-window-baselines.png)

The baseline visibility can be controlled by clicking on the check box in a table row. When the check box is checked, the baseline will be visible in the summary header as well as all graphs in all sections. When unchecked the baseline will be hidden and will not contribute to metric difference calculations.

The baseline color can be changed by double-clicking on the color swatch in the table row. The color dialog which is opened provides the ability to choose an arbitrary color as well as offers a palette of predefined colors associated with the stock baseline color rotation.

The baseline name can be changed by double-clicking on the _Name_ column in the table row. The name must not be empty and must be less than the _Maximum Baseline Name Length_ as specified in the options dialog.

The z-order of a selected baseline can be changed by clicking the _Move Baseline Up_ and _Move Baseline Down_ buttons in the tool bar. When a baseline is moved up or down its new position will be reflected in the report header as well as in each graph. Currently, only one baseline may be moved at a time.

The selected baselines may be removed by clicking on the _Clear Selected Baselines_ button in the tool bar. All baselines can be removed at once by clicking on the _Clear All Baselines_ button, from either the global tool bar or the tool window tool bar.

The configured baselines can be saved to a file by clicking on the _Save Baselines_ button in the tool bar. By default baseline files use the `.ncu-bln` extension. Baseline files can be opened locally and/or shared with other users.

Baseline information can be loaded by clicking on the _Load Baselines_ button in the tool bar. When a baseline file is loaded, currently configured baselines will be replaced. A dialog will be presented to the user to confirm this operation when necessary.

Differences between the current result and the baselines can be visualized with graphical bars for metrics in Details page section headers. Use the _Difference Bars_ drop down to select the visualization mode. Bars are extending from left to right and have a fixed maximum.

### 3.5.4. Metric Details

The _Metric Details_ tool window can be opened using the _Metric Details_ entry in the _Profile_ menu or the respective tool bar button. When a report and the tool window are open, a metric can be selected in the report to display additional information and its value for the current result in the tool window. It also contains a search bar to look up metrics in the focused report’s current result.

![../_images/tool-window-metric-details.png](https://docs.nvidia.com/nsight-compute/_images/tool-window-metric-details.png)

Report metrics can be selected on the [Details Page](index.html#profiler-report-details-page) or on the [Raw Page](index.html#profiler-report-raw-page). The window will show basic information (name, unit and raw value of the metric) as well as additional information, such as its extended description. Note that all shown values are for the respective metric in the current result, even if a different result in e.g. the Raw Page is selected.

For results collected for [Green Contexts](../ProfilingGuide/index.html#cuda-green-contexts) applications, the _Metric Details_ tool window also shows an additional row with the _Attribution_ level of the selected performance metric (e.g., Context or Green Context, if applicable).

The search bar can be used to open metrics in the focused report. It shows available matches as you type. The entered string must match from the start of the metric name.

By default, selecting or searching for a new metric updates the current _Default Tab_. You can click the pin button located in the upper-left corner of the tab control to create a copy of the default tab, unless the same metric is already pinned. This makes it possible to save multiple tabs and quickly switch between them to compare values.

Some metrics contain [Instance Values](../ProfilingGuide/index.html#metrics-structure). When available, they are listed in the tool window. Instance values can have a _Correlation ID_ that allows correlating the individual value with its associated entity, e.g. a function address or instruction name.

For metrics collected with [PM sampling](../ProfilingGuide/index.html#pm-sampling), the correlation ID is the GPU timestamp in nanoseconds. It is shown relative to the first timestamp for this metric. Note that the instance values for PM sampling metrics are currently not [context-switched](../ProfilingGuide/index.html#ctx-switch-trace) , even if this is enabled in the corresponding timeline UI.

### 3.5.5. Launch Details

The _Launch Details_ tool window can be opened using the _Launch Details_ entry in the _Profile_ menu or the respective tool bar button. When a result containing multiple sub-launches is selected and this tool window is open, it will display information about each sub-launch contained in the result.

[![../_images/tool-window-launch-details.png](https://docs.nvidia.com/nsight-compute/_images/tool-window-launch-details.png)](../_images/tool-window-launch-details.png)

This tool window is split into two sections:

  * a header displaying information applying to the result as a whole

  * a body displaying information specific to the viewed sub-launch


#### Header

On the left side of its header, this tool window displays the selected result’s name and the number of sub-launches it is comprised of.

The right side contains a combo box that allows selection of the sub-launch the body should represent. Each element of the combo box contains an index for the sub-launch as well as the name of the function that it launched if available.

#### Body

The body of this tool window displays a table with sub-launch-specific metrics. This table has four columns:

  * _Metric Name_ : the name of the metric

  * _Metric Unit_ : the unit for metric values

  * _Instance Value_ : the value of this metric for the selected sub-launch

  * _Aggregate Value_ : the aggregate value for this metric over all sub-launches in the selected result


### 3.5.6. Function Stats

The _Function Stats_ tool window presents a table of all functions in your profile result with top stalls for each function and all stall reasons. This window is connected with the PM Sampling timeline and automatically updates to show data for the selected time range in the timeline.

#### Accessing Function Stats

The Function Stats tool window can be opened using:
    

  * The _Function Stats_ entry in the _Profile_ menu. See [Main Menu](index.html#main-menu) for details.

  * The main tool bar button


#### Basic Workflow

  1. Profile with PM Warp Sampling enabled (default: on).

  2. Open the tool window using the **Function Stats** tool bar button.

  3. Sort by a column (default is sorted by **All Samples**).

  4. Click a function to navigate to it on the Source page (if available).
    

**Note:** The source view is not updated with the warp sampling data collected with PM sampling.


![../_images/tool-window-function-stats.png](https://docs.nvidia.com/nsight-compute/_images/tool-window-function-stats.png)

The Function Stats tool window

#### Understanding the Function Stats Table

Each row corresponds to a **function**. The columns include:

**Samples (All)** : Reports the samples collected for this function as % of total samples collected. This also includes all the stall reasons as a stacked bar chart.

**Samples (Not Issued)** : Reports the samples collected for this function as % of total samples collected where warps were not issued. This also includes all the stall reasons as a stacked bar chart.

**Top Stall #1** : Reports the stall reason with the highest incidence for the function.

**Top Stall #2** : Reports the stall reason with the second highest incidence for the function.

**Top Stall #3** : Reports the stall reason with the third highest incidence for the function.

Note

By default, the table is **sorted by All Samples in decreasing order**.

See the _Warp Stall Reasons_ tables in the [Metrics Reference](../ProfilingGuide/index.html#warp-stall-reasons) for a description of the individual warp scheduler states.

#### Data Source

  * Collection is performed by PM Sampling with warp-state sampling enabled (**default**).

  * The shipped `PmSampling.section` file comes pre-configured with PM Sampling and Warp State Sampling metrics required to enable this window.

  * At least one **PM Sampling** metric is required with the attribute `PmWarpSampling` declared (`Visible` or `Hidden`) to enable PM Warp Sampling data collection for this window. Use:

    * `PmWarpSampling: Visible` — show the metric with the timeline UI.

    * `PmWarpSampling: Hidden` — hide the metric in the timeline UI but still collected.

  * The feature is **enabled by default** on all GPUs GA10X and newer. Disable globally with `--disable-pm-warp-sampling`.

  * Only **one** PM Sampling pass with warp-state sampling is supported per profile session.

  * **Note:** `warpsampling:` metrics alone do **not** populate the per-function view; at least one **PM Sampling** metric with the attribute `PmWarpSampling` must be present.


#### Metric Naming

Warp Stall Sampling metrics are the **same underlying source counter metrics** ; they are requested by prefixing the counter name with `warpsampling:` to indicate collection via the PM Sampling path (**PM Warp Sampling**) and to render them on a timeline. Example: \- `warpsampling:smsp__pcsamp_warps_issue_stalled_barrier`

In addition, a new group of **Warp Sampling** metrics that can be collected in this mode is available: \- group:smsp__pmwarpsamp_warp_stall_reasons

### 3.5.7. NVTX

The _NVTX_ window is available when NVIDIA Nsight Compute is connected to a target application. If closed, it can be re-opened using _Debug > NVTX_ from the main menu. Whenever the target application is suspended, the window shows the state of all active NVTX domains and ranges in the currently selected thread. Note that [NVTX](https://devblogs.nvidia.com/cuda-pro-tip-generate-custom-application-profile-timelines-nvtx/) information is only tracked if the launching command line profiler instance was started with `--nvtx` or NVTX was enabled in the NVIDIA Nsight Compute launch dialog.

![../_images/tool-window-nvtx.png](https://docs.nvidia.com/nsight-compute/_images/tool-window-nvtx.png)

Use the _Current Thread_ dropdown in the [API Stream](index.html#tool-window-api-stream) window to change the currently selected thread. NVIDIA Nsight Compute supports NVTX named resources, such as threads, CUDA devices, CUDA contexts, etc. If a resource is named using NVTX, the appropriate UI elements will be updated.

![../_images/tool-window-nvtx-resources.png](https://docs.nvidia.com/nsight-compute/_images/tool-window-nvtx-resources.png)

### 3.5.8. CPU Call Stack

The _CPU Call Stack_ window is available when NVIDIA Nsight Compute is connected to a target application. If closed, it can be re-opened using _Debug > CPU Call Stack_ from the main menu. Whenever the target application is suspended, the window shows all enabled CPU call stacks for the currently selected thread.

![../_images/tool-window-callstack.png](https://docs.nvidia.com/nsight-compute/_images/tool-window-callstack.png)

Use the _Call Stack Type_ dropdown menu to switch between stack types in case multiple stack types were enabled (e.g., _Native_ , _Python_). Note that Python call stack collection requires CPython version 3.9 or later.

### 3.5.9. Resources

The _Resources_ window is available when NVIDIA Nsight Compute is connected to a target application. It shows information about the currently known resources, such as CUDA devices, CUDA streams or kernels. The window is updated every time the target application is suspended. If closed, it can be re-opened using _Debug > Resources_ from the main menu.

![../_images/tool-window-resources.png](https://docs.nvidia.com/nsight-compute/_images/tool-window-resources.png)

Using the dropdown on the top, different views can be selected, where each view is specific to one kind of resource (context, stream, kernel, …). The _Filter_ edit allows you to create filter expressions using the column headers of the currently selected resource.

The resource table shows all information for each resource instance. Each instance has a unique ID, the _API Call ID_ when this resource was created, its handle, associated handles, and further parameters. When a resource is destroyed, it is removed from its table.

#### Memory Allocations

When using the asynchronous malloc/free APIs, the resource view for _Memory Allocation_ will also include the memory objects created in this manner. These memory objects have a non-zero memory pool handle. The _Mode_ column will indicate which code path was taken during the allocation of the corresponding object. The modes are:

  * REUSE_STREAM_SUBPOOL: The memory object was allocated in memory that was previously freed. The memory was backed by the memory pool set as current for the stream on which the allocation was made.

  * USE_EXISTING_POOL_MEMORY: The memory object was allocated in memory that was previously freed. The memory is backed by the default memory pool of the stream on which the allocation was made.

  * REUSE_EVENT_DEPENDENCIES: The memory object was allocated in memory that was previously freed in another stream of the same context. A stream ordering dependency of the allocating stream on the free action existed. Cuda events and null stream interactions can create the required stream ordered dependencies.

  * REUSE_OPPORTUNISTIC: The memory object was allocated in memory that was previously freed in another stream of the same context. However, no dependency between the free and allocation existed. This mode requires that the free be already committed at the time the allocation is requested. Changes in execution behavior might result in different modes for multiple runs of the application.

  * REUSE_INTERNAL_DEPENDENCIES: The memory object was allocated in memory that was previously freed in another stream of the same context. New internal stream dependencies may have been added in order to establish the stream ordering required to reuse a piece of memory previously released.

  * REQUEST_NEW_ALLOCATION: New memory had to be allocated for this memory object as no viable reusable pool memory was found. The allocation performance is comparable to using the non-asynchronous malloc/free APIs.


#### Graphviz DOT and SVG exports

Some of the shown _Resources_ can also be exported to _GraphViz DOT_ or SVG* files using the `Export to GraphViz` or `Export to SVG` buttons.

When exporting _OptiX traversable handles_ , the traversable graph node types will be encoded using shapes and colors as described in the following table.

OptiX Traversable Graph Node Types Node Type | Shape | Color  
---|---|---  
IAS | Hexagon | #8DD3C7  
Triangle GAS | Box | #FFFFB3  
AABB GAS | Box | #FCCDE5  
Curve GAS | Box | #CCEBC5  
Sphere GAS | Box | #BEBADA  
Static Transform | Diamond | #FB8072  
SRT Transform | Diamond | #FDB462  
Matrix Motion Transform | Diamond | #80B1D3  
Error | Paralellogram | #D9D9D9  
  
### 3.5.10. CUDA Graph Viewer

The _CUDA Graph Viewer_ provides real-time visualization and inspection of CUDA Graphs during interactive profiling sessions with NVIDIA Nsight Compute. This window becomes available when the target application creates a CUDA Graph while connected to the profiler.

By default, the viewer opens automatically upon graph creation. To disable this behavior, navigate to _Tools > Options > Profile > CUDA Graph Viewer > Auto-Open Graph Viewer_ and set it to _No_. If the window is closed, you can reopen it by clicking the _CUDA Graph Viewer_ button in _Resources > Graphs: Graphs_ or in any other _Graphs:_ resource from the main menu. The viewer displays the current state of all active CUDA Graphs whenever the target application is suspended.

![../_images/tool-window-cuda-graph-viewer.png](https://docs.nvidia.com/nsight-compute/_images/tool-window-cuda-graph-viewer.png)

Use the dropdown menu to select which CUDA Graph to view when multiple graphs are active. The window provides two tabs for examining CUDA Graphs:

  * **Source Graph** : Displays the original graph structure as defined in the application source code before instantiation. This view reflects the logical organization of nodes and dependencies as specified by the developer.

  * **Instantiated Graph** : Shows the graph as it exists after instantiation and optimization by the CUDA runtime. This view may differ from the source graph due to runtime optimizations, node merging, or dependency resolution.


Note that the graph layouts are optimized for interactive visualization and may differ from those generated by CUDA API functions such as `cudaGraphDebugDotPrint` or exported as [DOT or SVG](index.html#graphviz-dot-and-svg-exports) from the _Resources_ window. The viewer uses hierarchical layout algorithms to improve readability and highlight execution flow.

You can toggle the visibility of node and edge labels using the _Show/Hide Labels_ button. Hover over nodes, edges, or the graph canvas to view detailed tooltips.

**Execution Tracking**

The viewer provides dynamic execution tracking during both graph construction and execution:

  * When a source graph is being built, the _Source Graph_ tab automatically shows the graph in its current state. As nodes are added through CUDA API calls, the viewer updates in real-time to highlight newly created nodes and their connections.

  * When an instantiated graph is launched, the _Instantiated Graph_ tab automatically displays the graph execution state. As you step through the application, the viewer highlights nodes currently being executed and visually distinguishes completed nodes from pending ones. Execution flow through conditional graph and child graph nodes is clearly indicated.


### 3.5.11. Search

The _Search_ tool window can be opened using the _Search_ entry in the _Tools_ menu, or from the tool bar’s search bar. It can be used to search for terms throughout Nsight Compute.

Your queries can also be searched in the online developer forums, providing you with additional information and help.

![../_images/tool-window-search.png](https://docs.nvidia.com/nsight-compute/_images/tool-window-search.png)

Enter your query into the tool menu’s or window’s search bar and press _Enter_ or use the _Start search_ button. Available sources are checked for matches in the background and results are displayed in the tool window’s table. While a search is running, you can press the _Cancel search_ button to stop it. Previous searches are available from a history dropdown that is shown when clicking the tool window’s search bar with the mouse. Select a history entry to replace the current search bar content with it.

![../_images/tool-window-search-results.png](https://docs.nvidia.com/nsight-compute/_images/tool-window-search-results.png)

Each result is associated with the original source. While some results are plain text, others are links that can be clicked to jump to the location and see them in context. Below is a (non-comprehensive) list of supported sources. The list of available sources may be extended in the future.

  * Local **Documentation**

  * **Knowledgebase** entries

  * **Metrics** in the focused report

  * Nsight Compute UI **Options**

  * Loaded **Sections**

  * Online **Developer Forums** entries


### 3.5.12. Metric Selection

The _Metric Selection_ window can be opened from the main menu using _Profile > Metric Selection_. It tracks all metric sets, sections and rules currently loaded in NVIDIA Nsight Compute, independent from a specific connection or report. The directory to load those files from can be configured in the [Profile](index.html#options-profile) options dialog. It is used to inspect available sets, sections and rules, as well as to configure which should be collected, and which rules should be applied. You can also specify a comma separated list of individual metrics, that should be collected. The window has two views, which can be selected using the dropdown in its header.

The **Metric Sets** view shows all available metric sets. Each set is associated with a number of metrics sections. You can choose a set appropriate to the level of detail for which you want to collect performance metrics. Sets which collect more detailed information normally incur higher runtime overhead during profiling.

![../_images/tool-window-section-sets.png](https://docs.nvidia.com/nsight-compute/_images/tool-window-section-sets.png)

When enabling a set in this view, the associated metric sections are enabled in the _Metric Sections/Rules_ view. When disabling a set in this view, the associated sections in the _Metric Sections/Rules_ view are disabled. If no set is enabled, or if sections are manually enabled/disabled in the _Metric Sections/Rules_ view, the <_custom_ > entry is marked active to represent that no section set is currently enabled. Note that the _basic_ set is enabled by default.

Whenever a kernel is profiled manually, or when auto-profiling is enabled, only sections enabled in the **Metric Sections/Rules** view and individual metrics specified in input box are collected. Similarly, whenever rules are applied, only rules enabled in this view are active.

![../_images/tool-window-sections.png](https://docs.nvidia.com/nsight-compute/_images/tool-window-sections.png)

The enabled states of sections and rules are persisted across NVIDIA Nsight Compute launches. The _Reload_ button reloads all sections and rules from disk again. If a new section or rule is found, it will be enabled if possible. If any errors occur while loading a rule, they will be listed in an extra entry with a warning icon and a description of the error.

Use the _Enable All_ and _Disable All_ checkboxes to enable or disable all sections and rules at once. The Filter text box can be used to filter what is currently shown in the view. It does not alter activation of any entry.

The table shows sections and rules with their activation status, their relationship and further parameters, such as associated metrics or the original file on disk. Rules associated with a section are shown as children of their section entry. Rules independent of any section are shown under an additional _Independent Rules_ entry.

Clicking an entry in the table’s _Filename_ column opens this file as a document. It can be edited and saved (ctrl + s) directly in NVIDIA Nsight Compute. After editing the file, _Reload_ must be selected to apply those changes. Document also supports text search (ctrl + f), zoom in (ctrl + mouse scroll down), zoom out (ctrl + mouse scroll up) functionalities.

When a section or rule file is modified, the entry in the _State_ column will show _User Modified_ to reflect that it has been modified from its default state. When a _User Modified_ row is selected, the _Restore_ button will be enabled. Clicking the Restore button will restore the entry to its default state and automatically _Reload_ the sections and rules.

Similarly, when a stock section or rule file is removed from the configured _Sections Directory_ (specified in the [Profile](index.html#options-profile) options dialog), the _State_ column will show _User Deleted_. _User Deleted_ files can also be restored using the _Restore_ button.

Section and rule files that are created by the user (and not shipped with NVIDIA Nsight Compute) will show up as _User Created_ in the _state column_.

See the [Sections and Rules](../ProfilingGuide/index.html#sections-and-rules) for the list of default sections for NVIDIA Nsight Compute.

## 3.6. Profiler Report

The profiler report contains all the information collected during profiling for each kernel launch. In the user interface, it consists of a header with general information, as well as controls to switch between report pages or individual collected launches.

### 3.6.1. Header

![../_images/profiler-report-header.png](https://docs.nvidia.com/nsight-compute/_images/profiler-report-header.png)

The top of the report shows a table with information about the selected profile result (as _Current_) and potentially additional [baselines](index.html#tool-window-baselines). For many values in this table, tooltips provide additional information or data, e.g., the tooltip of the column _Attributes_ provides additional information about the context type and resources used for the launch.

The _Result_ dropdown can be used to switch between all collected kernel launches. The information displayed in each page commonly represents the selected launch instance. On some pages (e.g. _Raw_), information for all launches is shown and the selected instance is highlighted. You can type in this dropdown to quickly filter and find a kernel launch.

The _Apply Filters_ button opens the filter dialog. You can use more than one filter to narrow down your results. On the filter dialog, enter your filter parameters and press OK button. The _Launch_ dropdown, [Summary Page](index.html#profiler-report-summary-page) table, and [Raw Page](index.html#profiler-report-raw-page) table will be filtered accordingly. Select the arrow dropdown to access the _Clear Filters_ button, which removes all filters.

![../_images/profiler-report-header-filter-dialog.png](https://docs.nvidia.com/nsight-compute/_images/profiler-report-header-filter-dialog.png)

Filter Dialog

Underneath the current and baseline results are the tabs for switching between the report pages. The pages themselves are explained in detail in the [next section](index.html#profiler-report-pages).

Each group button to the right of the page tabs opens a context menu that features related actions. Some actions may be enabled only when the related report page is selected.

  * Compare

    * **Add Baseline** promotes the current result in focus to become the baseline of all other results from this report and any other report opened in the same instance of NVIDIA Nsight Compute.

    * **Clear Baselines** removes all currently active baselines. You may also use the [Baselines](index.html#tool-window-baselines) tool window to manage baseline for comparison.

    * **Source Comparison** navigates to the [Source Comparison](index.html#source-comparison) document in case at least two profile results are available for comparison.

  * Tools

    * **Occupancy Calculator** opens the [Occupancy Calculator](index.html#occupancy-calculator) in a new document.

    * **Metric Details Windows** opens the [Metric Details](index.html#tool-window-metric-details) tool window. When the window is open and a metric is selected elsewhere in the report, it shows detailed information about it.

    * **Launch Details Windows** opens the [Launch Details](index.html#tool-window-launch-details) tool window. When the window is open and a result containing multiple sub-launches is selected, it displays information about each sub-launch in the result.

  * View

    * **Show/Hide Rules Output** toggles the visibility of rule results.

    * **Show/Hide Section Descriptions** toggles the visibility of section descriptions on the Details and Session pages.

    * **Show/Hide Green Context Markers** toggles the visibility of markers for _attributable_ Green Context metrics.`

    * **Expand Sections** expands all sections to show their body contents, not only header and rule output. Note that sections may have multiple bodies and the visible one can be chosen using the dropdown in the section header.

    * **Collapse Sections** collapses all sections to show only their header and rule output.

  * Export

    * **Copy as Image** \- Copies the contents of the page to the clipboard as an image.

    * **Save as Image** \- Saves the contents of the page to a file as an image.

    * **Save as PDF** \- Saves the contents of the page to a file as a PDF.

    * **Export to CSV** \- Exports the contents of the page to CSV format.

  * More (three bars icon)

    * **Apply Rules** applies all rules available for this report. If rules had been applied previously, those results will be replaced. By default, rules are applied immediately once the kernel launch has been profiled. This can be changed in the options under _Tools > Options > Profile > Report UI > Apply Applicable Rules Automatically_.

    * **Reset to Default** resets the page to a default state by removing any persisted settings.


### 3.6.2. Report Pages

Use the _Page_ dropdown in the header to switch between the report pages.

By default, when opening a report with a single profile result, the [Details Page](index.html#profiler-report-details-page) is shown. When opening a report with multiple results, the [Summary Page](index.html#profiler-report-summary-page) is selected instead. You can change the default report page in the [Profile](index.html#options-profile) options.

#### Summary Page

The _Summary_ page shows a table of all collected results in the report, as well as a list of the most important rule outputs (_Prioritized Rules_) which are ordered by the estimated speedup that could potential be obtained by following their guidance. _Prioritized Rules_ are shown by default and can be toggled with the [R] button on the upper right of the page.

![../_images/profiler-report-pages-summary.png](https://docs.nvidia.com/nsight-compute/_images/profiler-report-pages-summary.png)

Summary page with Summary Table and Prioritized Rules.

The _Summary Table_ gives you a quick comparison overview across all profiled workloads. It contains a number of important, pre-selected metrics which can be customized as explained below. Its columns can be sorted by clicking the column header. You can transpose the table with the _Transpose_ button. Aggregate of all results per each counter metric is shown in the table header along with the column name. You can change the aggregated values by selecting the desired results for multiple metrics simultaneously. When selecting any entry by single-click, a list of its _Prioritized Rules_ will be shown below the table. Double-click any entry to make the result the currently active one and switch to the [Details Page](index.html#profiler-report-details-page) page to inspect its performance data. By default, kernel demangled names are simplified, renamed and shown in an optimized manner. This behavior can be changed with [Rename Demangled Names](index.html#options-profile) option. If an auto-simplified name is not useful, you can rename it through a configuration file. You can also persist the updated names directly in the report by double-clicking on the name, renaming and saving the report. Use [Rename Kernels Config Path](index.html#options-profile) option to specify the configuration file which should be used while importing renamed kernels or exporting demangled names with mappings to rename them. To export names to a new file, click _Export_ button and use _Rename Kernels Config_ option. See [Kernel Renaming](../NsightComputeCli/index.html#kernel-renaming) for more details on configuration file usage.

![../_images/profiler-report-pages-summary-table.png](https://docs.nvidia.com/nsight-compute/_images/profiler-report-pages-summary-table.png)

You can configure the list of metrics included in this table in the [Profile](index.html#options-profile) options dialog. If a metric has multiple instance values, the number of instances is shown after its standard value. A metric with ten instance values could for example look like this: `35.48 {10}`. In the [Profile](index.html#options-profile) options dialog, you can select that all instance values should be shown individually. You can also inspect the instances values of a metric result in the [Metric Details](index.html#tool-window-metric-details) tool window.

In addition to metrics, you can also configure the table to include any of the following properties:

##### Properties

> Properties `property__api_call_id` | ID of the API call associated with this profile result.  
> ---|---  
> `property__block_size` | Block Size. If the result contains multiple launches, this will contain the maximum value for each dimension of the block.  
> `property__creation_time` | Local collection time.  
> `property__demangled_name` | Kernel demangled name, potentially renamed.  
> `property__device_name` | GPU device name.  
> `property__estimated_speedup` | Maximal relative speedup achievable for this profile result as estimated by the guided analysis rules.  
> `property__function_name` | Kernel function name.  
> `property__grid_dimensions` | Grid Dimensions. If the result contains multiple launches, this will contain the maximum value for each dimension of the grid.  
> `property__grid_offset` | Grid Offset.  
> `property__grid_size` | Grid Size. If the result contains multiple launches, this will contain the maximum value for each dimension of the grid.  
> `property__issues_detected` | Number of issues detected by guided analysis rules for this profile result.  
> `property__kernel_id` | Kernel ID.  
> `property__mangled_name` | Kernel mangled name.  
> `property__original_demangled_name` | Original kernel demangled name without any renaming.  
> `property__process_name` | Process name.  
> `property__range_name` | Range name.  
> `property__result_type` | Result Type. This property shows workload type and execution model of the profile result.  
> `property__runtime_improvement` | Runtime improvement corresponding to the estimated speedup.  
> `property__series_id` | ID of the profile series.  
> `property__series_parameters` | Profile series parameters.  
> `property__thread_id` | CPU thread ID.  
  
For [Range Replay](../ProfilingGuide/index.html#range-replay) reports, a smaller set of columns is shown by default, as not all apply to such results.

For the currently selected metric result the _Prioritized Rules_ show the most impactful rule results with respect to the estimated potential speedup. Clicking on any of the rule names on the left allows you to easily navigate to the containing section on the details page. With the downward-facing arrow on the right a table with the relevant _key performance indicators_ can be toggled. This table contains the metrics which should be tracked when optimizing performance according to the rule guidance.

![../_images/profiler-report-pages-summary-rules.png](https://docs.nvidia.com/nsight-compute/_images/profiler-report-pages-summary-rules.png)

_Prioritized Rules_ with key performance indicators table.

#### Details Page

##### Overview

The _Details_ page is the main page for all metric data collected during a kernel launch. The page is split into individual sections. Each section consists of a header table and an optional body that can be expanded. You can expand or collapse the body of each section by clicking on its respective header. The sections are completely user defined and can be changed easily by updating their respective files. For more information on customizing sections, see the [Customization Guide](../CustomizationGuide/index.html#abstract). For a list of sections shipped with NVIDIA Nsight Compute, see [Sections and Rules](../ProfilingGuide/index.html#sections-and-rules).

By default, once a new profile result is collected, all applicable rules are applied. Any rule results will be shown as _Recommendations_ on this page. Most rule results will contain an optimization advice along with an estimate of the improvement that could be achieved when successfully implementing this advice. Other rule results will be purely informative or have a warning icon to indicate a problem that occurred during execution (e.g., an optional metric that could not be collected). Results with error icons typically indicate an error while applying the rule.

Estimates of potential improvement are shown below the rule result’s name and exist in two types. _Global estimates_ (“Est. Speedup”) are an approximation of the decrease in workload runtime, whereas _local estimates_ (“Est. Local Speedup”) are an approximation of the increase in efficiency of the hardware utilization of the particular performance problem the rule addresses.

![../_images/profiler-report-pages-section-with-rule.png](https://docs.nvidia.com/nsight-compute/_images/profiler-report-pages-section-with-rule.png)

Rule results often point out performance problems and guide through the analysis process.

If a rule result references another report section, it will appear as a link in the recommendation. Select the link to scroll to the respective section. If the section was not collected in the same profile result, enable it in the [Metric Selection](index.html#tool-window-sections-info) tool window.

You can add or edit comments in each section of the _Details_ view by clicking on the comment button (speech bubble). The comment icon will be highlighted in sections that contain a comment. Comments are persisted in the report and are summarized in the [Comments Page](index.html#profiler-report-comments-page).

![../_images/profiler-report-pages-details-comments.png](https://docs.nvidia.com/nsight-compute/_images/profiler-report-pages-details-comments.png)

Use the Comments button to annotate sections.

Besides their header, sections typically have one or more _bodies_ with additional charts or tables. Click the triangle _Expander_ icon in the top-left corner of each section to show or hide those. If a section has multiple bodies, a dropdown in their top-right corner allows you to switch between them.

![../_images/profiler-report-pages-section-bodies.png](https://docs.nvidia.com/nsight-compute/_images/profiler-report-pages-section-bodies.png)

Sections with multiple bodies have a dropdown to switch between them.

##### Memory

If enabled, the _Memory Workload Analysis_ section contains a Memory chart that visualizes data transfers, cache hit rates, instructions and memory requests. More information on how to use and read this chart can be found in the [Profiling Guide](../ProfilingGuide/index.html#memory-chart).

##### Occupancy

You can open the [Occupancy Calculator](index.html#occupancy-calculator) by clicking on the calculator button in the report header or in the header of the _Occupancy Section_.

##### Range Replay

Note that for [Range Replay](../ProfilingGuide/index.html#range-replay) results some UI elements, analysis rules, metrics or section body items such as charts or tables might not be available, as they only apply to kernel launch-based results. The filters can be checked in the corresponding section files.

##### Rooflines

If enabled, the _GPU Speed Of Light Roofline Chart_ section contains a Roofline chart that is particularly helpful for visualizing kernel performance at a glance. (To enable roofline charts in the report, ensure that the section is enabled when profiling.) More information on how to use and read this chart can be found in [Roofline Charts](../ProfilingGuide/index.html#roofline). NVIDIA Nsight Compute ships with several different definitions for roofline charts, including hierarchical rooflines. These additional rooflines are defined in different section files. While not part of the _full_ section set, a new section set called _roofline_ was added to collect and show all rooflines in one report. The idea of hierarchical rooflines is that they define multiple ceilings that represent the limiters of a hardware hierarchy. For example, a hierarchical roofline focusing on the memory hierarchy could have ceilings for the throughputs of the L1 cache, L2 cache and device memory. If the achieved performance of a kernel is limited by one of the ceilings of a hierarchical roofline, it can indicate that the corresponding unit of the hierarchy is a potential bottleneck.

![../_images/profiler-report-pages-section-rooflines.png](https://docs.nvidia.com/nsight-compute/_images/profiler-report-pages-section-rooflines.png)

Sample roofline chart.

The roofline chart can be zoomed and panned for more effective data analysis, using the controls in the table below. When the [Metric Details](index.html#tool-window-metric-details) tool window is open, you can click on an achieved value in the chart to see its metric formula in the tool window.

Roofline Chart Zoom and Pan Controls Zoom In | Zoom Out | Zoom Reset | Pan  
---|---|---|---  
  
  * Click the Zoom In button in the top right corner of the chart.
  * Click the left mouse button and drag to create a rectangle that bounds the area of interest.
  * Press the plus (+) key.
  * Use Ctrl + MouseWheel (Windows and Linux only)

| 

  * Click the Zoom Out button in the top right corner of the chart.
  * Click the right mouse button.
  * Press the minus (-) key.
  * Use Ctrl + MouseWheel (Windows and Linux only)

| 

  * Click the Zoom Reset button in the top right corner of the chart.
  * Press the Escape (Esc) key.

| 

  * Use Ctrl (Command on Mac) + LeftMouseButton to grab the chart, then move the mouse.
  * Use the cursor keys.

  
  
##### Source

Sections such as _Source Counters_ can contain source hot spot tables. These tables indicate the N highest or lowest values of one or more metrics in your kernel source code. Select the location links to navigate directly to this location in the [Source Page](index.html#profiler-report-source-page). Hover the mouse over a value to see which metrics contribute to it.

![../_images/profiler-report-pages-details-source-table.png](https://docs.nvidia.com/nsight-compute/_images/profiler-report-pages-details-source-table.png)

Hot spot tables point out performance problems in your source.

##### Timelines

When collecting metrics with [PM sampling](../ProfilingGuide/index.html#pm-sampling), they can be viewed in a _timeline_. The timeline shows metrics selected in the respective section file or on the command line with their labels/names and their values over time.

![../_images/profiler-report-pages-details-timeline.png](https://docs.nvidia.com/nsight-compute/_images/profiler-report-pages-details-timeline.png)

Different metrics may be collected in different passes (replays) of the workload, as only a limited number of them can be sampled in the same pass. Context switch trace is used to filter the collected data to only include samples from the profiled contexts and to align it in the timeline.

You can hover the mouse over a metric row label to see further information on the metrics in the row. Hovering over a sample on the timeline shows the metric values at that timestamp within the current row. With the [Metric Details](index.html#tool-window-metric-details) tool window open, click to select a value on the timeline and show the metric and all its raw timestamps (absolute and relative) correlated values in the tool window.

You can also use the [Metric Details](index.html#tool-window-metric-details) tool window to inspect profiler metrics generated during PM sampling. These provide information about the used sampling intervals, buffer sizes, dropped samples and other properties for each collection pass. A detailed list can be found in the [metrics reference](../ProfilingGuide/index.html#metrics-reference).

The timeline has a context menu for further actions regarding copying, zooming or adjusting the viewed data:

>   * Metric rows on the timeline can be scaled (y-axis) based on a theoretical HW peak, or based on the maximum profiled value in the workload. Select the _Scale to Peak_ /_Scale to Max Value_ options in the context menu to switch between the two modes. Note that HW peak info may not be available for all rows.
> 
>   * When zoomed out, the bar height may be reduced to represent that multiple samples are aggregated into one bar. The _Show/Hide Max Bars_ options in the context menu can be used to enable/disable showing the maximum value for each time range across all samples in that range, even when zoomed out.
> 
>   * In addition, the _Enable/Disable Context Switch Filter_ option can be used to enable or disable the filtering of the timeline data with [context switch](../ProfilingGuide/index.html#pm-sampling) information, if it is available. When the context switch filter is enabled (the default), samples from each pass group are only shown for the active contexts. When the context switch filter is disabled, the raw collected sampling data is shown along with a separate row for each pass group’s context switch trace.
> 
>   * When the context menu option is not available, the report does not include context switch trace data. In this case, the option _Enable/Disable Workload Alignment_ is shown instead if workload execution trace is available. When enabled, it aligns all passes based on their first workload execution timestamp.
> 
> 


The timeline row Workload Execution shows each kernel’s start and end timestamp. When the context switch filter is enabled, kernel execution is only shown for one of the passes for the active contexts. When the context switch filter is disabled, kernel execution is shown for all the passes.

#### Source Page

The _Source_ page correlates assembly (SASS) with high-level code such as CUDA-C, Python or PTX. In addition, it displays instruction-correlated metrics to help pinpoint performance problems in your code.

![../_images/profiler-report-pages-source.png](https://docs.nvidia.com/nsight-compute/_images/profiler-report-pages-source.png)

Source Correlation

The page can be switched between different _Views_ to focus on a specific source layer or see two layers side-by-side. This includes SASS, PTX and Source (CUDA-C, Fortran, Python, …), as well as their combinations. Which options are available depends on the source information embedded into the executable.

The high-level Source (CUDA-C) view is available if the application was built with the `-lineinfo` or `--generate-line-info` nvcc flag to correlate SASS and source. When using separate linking at the ELF level, there is no PTX available in the ELF that would correspond to the final SASS. As such, NVIDIA Nsight Compute does not show any PTX even though it would be available statically in the executable and could be shown with `cuobjdump -all -lptx`. However, this is a pre-linked version of the PTX and cannot be reliably used for correlation.

##### Navigation

The _View_ dropdown can be used to select different code (correlation) options: SASS, PTX and Source (CUDA-C, Fortran, Python, …).

In side-by-side views, when selecting a line in the left-hand- or right-hand-side, any correlated lines in the opposite view are highlighted. However, when the [Show Single File For Multi-File Sources](index.html#options-profile) option is set to _Yes_ , the target file or source object must already be selected in the respective view for those correlated lines to be shown.

The locations of correlated lines are displayed on the minimap. Clicking on the minimap will scroll to the corresponding source line. The source correlation navigation controls allow you to move to the previous or next block of correlated lines.

![../_images/profiler-report-pages-correlation-navigation.png](https://docs.nvidia.com/nsight-compute/_images/profiler-report-pages-correlation-navigation.png)

Correlation Navigation

In side-by-side views, the view which has blue border is called _focused view_. Focused view can be changed by clicking anywhere in the other view. Control groups which are bordered with blue color will work for a focused view only.

The _Source_ dropdown allows you to switch between the files or functions that provide the content in the view. When a different source entry is selected, the view scrolls to the start of this file or function.

If a view contains multiple source files or functions, [+] and [-] buttons are shown. These can be used to expand or collapse the view, thereby showing or hiding the file or function content except for its header. These will work for both the views if that group is in the linked state otherwise in an unlinked state it will work for a focused view. If collapsed, all [metrics](index.html#profiler-report-source-page-metrics) are shown aggregated to provide a quick overview.

![../_images/profiler-report-pages-source-collapse.png](https://docs.nvidia.com/nsight-compute/_images/profiler-report-pages-source-collapse.png)

Collapsed Source View

The SASS instruction disassembly includes a symbol label column. Symbol labels are displayed alongside instruction addresses, matching the output style of the `nvdisasm` tool. These labels are interactive hyperlinks and clicking a symbol label will immediately navigate to the corresponding address line within the view. Labels are shown for branch targets and function calls, making it easy to follow control flow and symbol references directly from the disassembly. This feature allows you to quickly explore code flow and symbol references.

![../_images/profiler-report-pages-symbol-labels.png](https://docs.nvidia.com/nsight-compute/_images/profiler-report-pages-symbol-labels.png)

Symbol Labels

You can use the search functionality to find information within your data efficiently. This feature allows users to search within a specific column by selecting it through a click anywhere in the column. If no column is selected, the search defaults to the _Source_ column. The functionality supports advanced search expressions, such as regular expressions (regex) and value comparisons (e.g., “>= 5”). The search control can be opened using the search button on the common control toolbar or by pressing the Ctrl+F keys. It can be closed using the cross button or by pressing the Esc key.

![../_images/profiler-report-pages-source-search.png](https://docs.nvidia.com/nsight-compute/_images/profiler-report-pages-source-search.png)

Search Control

The SASS view is filtered to only show functions that were executed in the launch. You can toggle the [Show Only Executed Functions](index.html#options-profile) option to change this, but performance of this page may be negatively affected for large binaries. It is possible that some SASS instructions are shown as _N/A_. Those instructions are not currently exposed publicly.

In side-by-side views, the _Navigate By_ dropdowns are linked with each other by default, thereby changing column names from one dropdown will change it in the other view only if column is available. These dropdowns can be unlinked with the link-unlink button provided just before it.

![../_images/profiler-report-pages-source-navigate-by.png](https://docs.nvidia.com/nsight-compute/_images/profiler-report-pages-source-navigate-by.png)

Linked state of “Navigate By” Dropdowns

Only filenames are shown in the view, together with a _File Not Found_ error, if the source files cannot be found in their original location. This can occur, for example, if the report was moved to a different system. Select a filename and click the _Resolve_ button above to specify where this source can be found on the local filesystem. However, the view always shows the source files if the [import source](index.html#connection-activity-interactive) option was selected during profiling, and the files were available at that time. If a file is found in its original or any source lookup location, but its attributes don’t match, a _File Mismatch_ error is shown. See the [Source Lookup](index.html#options-source-lookup) options for changing file lookup behavior.

![../_images/profiler-report-pages-source-resolve.png](https://docs.nvidia.com/nsight-compute/_images/profiler-report-pages-source-resolve.png)

Resolve Source

If the report was collected using remote profiling, and automatic resolution of remote files is enabled in the [Profile](index.html#options-profile) options, NVIDIA Nsight Compute will attempt to load the source from the remote target. If the connection credentials are not yet available in the current NVIDIA Nsight Compute instance, they are prompted in a dialog. Loading from a remote target is currently only available for Linux x86_64 targets and Linux and Windows hosts.

##### Metrics

###### Metrics Correlation

The page is most useful when inspecting performance information and metrics correlated with your code. Metrics are shown in columns, which can be enabled or disabled using the _Column Chooser_ accessible using the column header right click menu.

![../_images/profiler-report-pages-source-column-chooser.png](https://docs.nvidia.com/nsight-compute/_images/profiler-report-pages-source-column-chooser.png)

Column Chooser

To not move out of view when scrolling horizontally, columns can be fixed. By default, the _Source_ column is fixed to the left, enabling easy inspection of all metrics correlated to a source line. To change fixing of columns, right click the column header and select _Freeze_ or _Unfreeze_ , respectively.

![../_images/profiler-report-pages-fix-column.png](https://docs.nvidia.com/nsight-compute/_images/profiler-report-pages-fix-column.png)

Column Freezing/Unfreezing

The heatmap on the right-hand side of each view can be used to quickly identify locations with high metric values of the currently selected metric in the dropdown. The heatmap uses a black-body radiation color scale where black denotes the lowest mapped value and white the highest, respectively. The current scale is shown when clicking and holding the heatmap with the right mouse button.

![../_images/profiler-report-pages-source-heatmap.png](https://docs.nvidia.com/nsight-compute/_images/profiler-report-pages-source-heatmap.png)

Heatmap Color Scale

By default, applicable metrics are shown as percentage values relative to their sum across the launch. A bar is filling from left to right to indicate the value at a specific source location relative to this metric’s maximum within the launch. The [%] and [+-] buttons can be used to switch the display from relative to absolute and from abbreviated absolute to full-precision absolute, respectively. For relative values and bars, the [circle/pie] button can be used to switch the display between relative to global (launch) and relative to local (function/file) scope. This button is disabled when the view is collapsed, as percentages are always relative to the global launch scope in this case.

![../_images/profiler-report-pages-source-rel-abs.png](https://docs.nvidia.com/nsight-compute/_images/profiler-report-pages-source-rel-abs.png)

Relative and Absolute Metric Values.

###### Pre-Defined Source Metrics

  * **Metric Pipelines**

The [metric pipelines](../ProfilingGuide/index.html#pipelines) this instruction (can) execute in. Some instructions can execute in one of multiple pipelines, and the decision is dynamic at runtime. Some instructions execute in both of two pipelines, and the association is static. Other instructions execute in only a single pipeline.

  * **Scoreboard Dependencies**

> See [Instructions & Scoreboards Table](../NsightCompute/index.html#profiler-report-source-page-additional-tables) for information on the scoreboard dependencies.

  * **Live Registers**

Number of registers that need to be kept valid by the compiler. A high value indicates that many registers are required at this code location, potentially increasing the register pressure and the maximum number of register required by the kernel.

The total number of registers reported as `launch__registers_per_thread` may be significantly higher than the maximum live registers. The compiler may need to allocate specific registers that can creates holes in the allocation, thereby affecting `launch__registers_per_thread`, even if the maximum live registers is smaller. This may happen due to ABI restrictions, or restrictions enforced by particular hardware instructions. The compiler may not have a complete picture of which registers may be used in either callee or caller and has to obey ABI conventions, thereby allocating different registers even if some register could have theoretically been re-used.

  * **Attributed Stalls**

Number of warp stall samples of type long/short scoreboard not issued, attributed to this scoreboard producer location.

Use this to identify source locations that cause the most stalls. The stalls are attributed to the producer location, which is the location of the line that produced the scoreboard. This may not be the same as the location of the instruction that is stalled, as it may be waiting on a scoreboard produced by another line.

  * **Instruction Mix/Category**

Instruction Mix (high-level source) or Instruction Category (SASS) show the breakdown of, or the instruction category, respectively, for each line.

  * **Warp Stall Sampling (All Samples)**[1](#fn1)

The number of samples from the [Statistical Sampler](../ProfilingGuide/index.html#statistical-sampler) at this program location.

  * **Warp Stall Sampling (Not-issued Samples)**[2](#fn2)

The number of samples from the [Statistical Sampler](../ProfilingGuide/index.html#statistical-sampler) at this program location on cycles the warp scheduler issued no instructions. Note that _(Not Issued)_ samples may be taken on a different profiling pass than _(All)_ samples mentioned above, so their values do not strictly correlate.

This metric is only available on devices with compute capability 7.0 or higher.

  * **Instructions Executed**

Number of times the source (instruction) was executed per individual warp, independent of the number of participating threads within each warp.

  * **Thread Instructions Executed**

Number of times the source (instruction) was executed by any thread, regardless of predicate presence or evaluation.

  * **Predicated-On Thread Instructions Executed**

Number of times the source (instruction) was executed by any active, predicated-on thread. For instructions that are executed unconditionally (i.e. without predicate), this is the number of active threads in the warp, multiplied with the respective _Instructions Executed_ value.

  * **Avg. Threads Executed**

Average number of thread-level executed instructions per warp, regardless of their predicate.

  * **Avg. Predicated-On Threads Executed**

Average number of predicated-on thread-level executed instructions per warp.

  * **Divergent Branches**

Number of divergent branch targets, including fallthrough. Incremented only when there are two or more active threads with divergent targets. Divergent branches can lead to warp stalls due to resolving the branch or instruction cache misses.

  * **Information on Memory Operations**

**Label** | **Name** | **Description**  
---|---|---  
Address Space | memory_type | The accessed address space (global/local/shared).  
Access Operation | memory_access_type | The type of memory access (e.g. load or store).  
Access Size | memory_access_size_type | The size of the memory access, in bits.  
L1 Tag Requests Global | memory_l1_tag_requests_global | Number of L1 tag requests generated by global memory instructions.  
L1 Conflicts Shared N-Way | derived__memory_l1_conflicts_shared_nway | Average N-way conflict in L1 per shared memory instruction. A 1-way access has no conflicts and resolves in a single pass. Note: This is a derived metric which can not be collected directly.  
L1 Wavefronts Shared Excessive | derived__memory_l1_wavefronts_shared_excessive | Excessive number of wavefronts in L1 from shared memory instructions, because not all not predicated-off threads performed the operation. Note: This is a derived metric which can not be collected directly.  
L1 Wavefronts Shared | memory_l1_wavefronts_shared | Number of wavefronts in L1 from shared memory instructions.  
L1 Wavefronts Shared Ideal | memory_l1_wavefronts_shared_ideal | Ideal number of wavefronts in L1 from shared memory instructions, assuming each not predicated-off thread performed the operation.  
L2 Theoretical Sectors Global Excessive | derived__memory_l2_theoretical_sectors_global_excessive | Excessive theoretical number of sectors requested in L2 from global memory instructions, because not all not predicated-off threads performed the operation. Note: This is a derived metric which can not be collected directly.  
L2 Theoretical Sectors Global | memory_l2_theoretical_sectors_global | Theoretical number of sectors requested in L2 from global memory instructions.  
L2 Theoretical Sectors Global Ideal | memory_l2_theoretical_sectors_global_ideal | Ideal number of sectors requested in L2 from global memory instructions, assuming each not predicated-off thread performed the operation.  
L2 Theoretical Sectors Local | memory_l2_theoretical_sectors_local | Theoretical number of sectors requested in L2 from local memory instructions.  
  
All _L1/L2 Sectors/Wavefronts/Requests_ metrics give the number of achieved (actually required), ideal, and excessive (achieved - ideal) sectors/wavefronts/requests. _Ideal_ metrics indicate the number that would needed, given each not predicated-off thread performed the operation of given width. _Excessive_ metrics indicate the required surplus over the ideal case. Reducing divergence between threads can reduce the excess amount and result in less work for the respective HW units.


Several of the above metrics on memory operations were renamed in version 2021.2 as follows:

**Old name** | **New name**  
---|---  
memory_l2_sectors_global | memory_l2_theoretical_sectors_global  
memory_l2_sectors_global_ideal | memory_l2_theoretical_sectors_global_ideal  
memory_l2_sectors_local | memory_l2_theoretical_sectors_local  
memory_l1_sectors_global | memory_l1_tag_requests_global  
memory_l1_sectors_shared | memory_l1_wavefronts_shared  
memory_l1_sectors_shared_ideal | memory_l1_wavefronts_shared_ideal  
  
  * **L2 Explicit Evict Policy Metrics**

Starting with the NVIDIA Ampere architecture the eviction policy of the L2 cache can be tuned to match the kernel’s access pattern. The eviction policy can be either set implicitly for a memory window (for more details see [CUaccessProperty](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaAccessPolicyWindow.html)) or set explicitly per executed memory instruction. If set explicitly, the desired eviction behavior for the cases of an L2 cache hit or miss are passed as input to the instruction. For more details refer to CUDA’s [Cache Eviction Priority Hints](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-eviction-priority-hints).

**Label** | **Name** | **Description**  
---|---|---  
L2 Explicit Evict Policies | smsp__inst_executed_memdesc_explicit_evict_type | Comma separated list of configured explicit eviction policies. As the policies can be set dynamically at runtime, this list includes all policies that were part of any executed instruction.  
L2 Explicit Hit Policy Evict First | smsp__inst_executed_memdesc_explicit_hitprop_evict_first | Number of times a memory instruction was executed by any warp which had the `evict_first` policy set in case the access leads to a cache hit in L2. Data cached with this policy will be first in the eviction priority order and will likely be evicted when cache eviction is required. This policy is suitable for streaming data.  
L2 Explicit Hit Policy Evict Last | smsp__inst_executed_memdesc_explicit_hitprop_evict_last | Number of times a memory instruction was executed by any warp which had the `evict_last` policy set in case the access leads to a cache hit in L2. Data cached with this policy will be last in the eviction priority order and will likely be evicted only after other data with `evict_normal` or `evict_first` eviction policy is already evicted. This policy is suitable for data that should remain persistent in cache.  
L2 Explicit Hit Policy Evict Normal | smsp__inst_executed_memdesc_explicit_hitprop_evict_normal | Number of times a memory instruction was executed by any warp which had the `evict_normal` (default) policy set in case the access leads to a cache hit in L2.  
L2 Explicit Hit Policy Evict Normal Demote | smsp__inst_executed_memdesc_explicit_hitprop_evict_normal_demote | Number of times a memory instruction was executed by any warp which had the `evict_normal_demote` policy set in case the access leads to a cache hit in L2.  
L2 Explicit Miss Policy Evict First | smsp__inst_executed_memdesc_explicit_missprop_evict_first | Number of times a memory instruction was executed by any warp which had the `evict_first` policy set in case the access leads to a cache miss in L2. Data cached with this policy will be first in the eviction priority order and will likely be evicted cache eviction is required. This policy is suitable for streaming data.  
L2 Explicit Miss Policy Evict Normal | smsp__inst_executed_memdesc_explicit_missprop_evict_normal | Number of times a memory instruction was executed by any warp which had the `evict_normal` (default) policy set in case the access leads to a cache miss in L2.  
  * **Individual Warp Stall Sampling Metrics**

All _stall_*_ metrics show the information combined in _Warp Stall Sampling_ individually. See [Statistical Sampler](../ProfilingGuide/index.html#statistical-sampler) for their descriptions.

  * See the [Customization Guide](../CustomizationGuide/index.html#abstract) on how to add additional metrics for this view and the [Metrics Reference](../ProfilingGuide/index.html#metrics-reference) for further information on available metrics.


###### Register Dependencies

Dependencies between registers are displayed in the SASS view. When a register is read, all the potential addresses where it could have been written are found. The links between these lines are drawn in the view. All dependencies for registers, predicates, uniform registers and uniform predicates are shown in their respective columns.

[![../_images/profiler-report-pages-source-register-dependencies.png](https://docs.nvidia.com/nsight-compute/_images/profiler-report-pages-source-register-dependencies.png)](../_images/profiler-report-pages-source-register-dependencies.png)

Register Dependencies

The picture above shows some dependencies for a simple CUDA kernel. On the first row, which is line 9 of the SASS code, we can see _writes_ on registers R2 and R3, represented by _filled triangles pointing to the left_. These registers are then read on lines 17, 20 and 23, and this is represented by _regular triangles pointing to the right_. There are also some lines where both types of triangles are on the same line, which means that a read and a write occured for the same register.

The lines are colored with a gradient that helps visualizing how long the dependencies are.

Dependencies across source files and functions are not tracked.

The Register Dependencies Tracking feature is enabled by default, but can be disabled completely in _Tools > Options > Profile > Report Source Page > Enable Register Dependencies_.

[1](#id6)
    

This metric was previously called Sampling Data (All).

[2](#id7)
    

This metric was previously called Sampling Data (Not Issued).

##### Profiles

The icon next to the _View_ dropdown can be used to manage _Source View Profiles_.

![../_images/profiler-report-pages-source-profiles-button.png](https://docs.nvidia.com/nsight-compute/_images/profiler-report-pages-source-profiles-button.png)

This button opens a dialog that shows you the list of saved source view profiles. Such profiles can be created using the `+` button in the dialog. Profiles let you store the column properties of all views in the report to a file. Such properties include column visibility, freeze state, width, order and the selected navigation metric. Double-click on a saved profile to apply it to any opened report. This updates the column properties mentioned above from the selected profile in all views.

![../_images/profiler-report-pages-source-profiles.png](https://docs.nvidia.com/nsight-compute/_images/profiler-report-pages-source-profiles.png)

Profiles are useful for configuring views to your preferences, or for a certain use case. Start by choosing metric columns from the _Column Chooser_. Next, configure other properties like freezing column, changing width or order and setting a heatmap metric in the _Navigation_ dropdown before creating the profile. Once a profile is created, you can always use this profile on any opened report to hide all non-required columns or to restore your configured properties. Simply select the profile from the source view profiles dialog and apply it. You can also make the profile default by clicking on the _star_ button to automatically apply it while opening a report.

Note that the column properties are stored separately for each _View_ in the profile and when applied, only those views will be updated which are present in the selected profile. You will not see the metric columns that are not available in your report even if those were configured to be visible in the source profile you have applied.

##### Additional Tables

###### Instructions & Dependencies Table

If **Scoreboard Dependencies** is selected, this table shows the following information for the selected line:

  * **Self Instruction Mix** : The breakdown of all instruction categories. Use this to understand the composition of the selection, and which instruction categories have the most (scoreboard) warp stalls.

  * **Input Scoreboard Dependencies** : The scoreboard-driven data dependencies of the selection. Data dependencies are predecessor operations that need to complete before code in the selection can proceed. Use this to understand on which instructions the selection is waiting (stalled) on.

The _Attributed Stalls_ column shows the number of warp stalls from the selection attributed to this dependency. As a result, dependencies causing more stalls in the selection are sorted higher in the table. This data is also available in the source views to locate lines causing the most stalls.

  * **Output Scoreboard Dependencies** : The consumers of scoreboards produced by the selection. A scoreboard producer is an operation producing a scoreboard a consumer is waiting on. Use this to understand who is waiting (stalled) on the selection. Note that these lines may wait for other instructions than the current selection, too.


Relative percentage values in these tables are calculated based on the total value of the respective data in the entire view, not just in the selection.

The _Dependencies_ tables show in their _Scoreboards_ columns which of the GPU’s six scoreboards are responsible for the dependency. Use the _Scoreboard Dependencies_ column in the SASS view to see individual scoreboard dependencies between assembly instructions. Lines at the same offset in this column are for the same scoreboard.

The _Scoreboard Stalls_ column uses different colors to distinguish between long and short scoreboard warp stalls.

![../_images/profiler-report-pages-instruction-mix-table.png](https://docs.nvidia.com/nsight-compute/_images/profiler-report-pages-instruction-mix-table.png)

Scoreboard Dependencies Table

If **Register Dependencies** is selected, this table shows the following information for the selected line:

  * **Self Instruction Mix** : The breakdown of all instruction categories. Use this to understand the composition of the selection.

  * **Input Register Dependencies** : The general purpose register-driven data dependencies of the selection. Data dependencies are predecessor operations that need to complete before code in the selection can proceed. Use this to understand which instructions produce/write registers the selection is waiting to consume/read.

The _Attributed Live Registers_ column shows the number of live general-purpose registers attributed to instructions of this category. This is the maximum number of live registers across all consumers of dependencies from this producer. This data is also available in the source views to locate lines causing the most live registers.

The _Output Registers_ column shows the number of register dependencies produced by instructions of this category. This is the sum of all general-purpose registers written by this producer. This data is also available in the source views to locate lines producing the most dependencies.

  * **Output Register Dependencies** : This table lists all consumers/readers of general-purpose registers produced/written by the selected line. A register producer is an operation writing a register a consumer is waiting on for reading. Use this to understand which instructions are waiting on the selection to produce/write registers. Note that these lines may wait for other instructions than the current selection, too.


![../_images/profiler-report-pages-register-dependencies-table.png](https://docs.nvidia.com/nsight-compute/_images/profiler-report-pages-register-dependencies-table.png)

Register Dependencies Table

For both scoreboard and register views, the dependencies are grouped by location and instruction category. If the location does not match the selected line, it is shown as a clickable link. Clicking on the link will navigate to the corresponding line. If the selection is in a high-level (e.g., CUDA, Python) view, the locations are mapped to the high-level language if possible. If not possible due to missing correlation information, or if the selection is in the SASS view, the locations are shown in the low-level language.

###### Inline Functions Table

This table shows all call sites where the selected function from the _Source_ view is inlined.

In the _Source_ view, correlated metric values of every inline function source line are an aggregation of metric values of related SASS lines from all call sites. The table provides such SASS lines info for each call site individually to help in identifying which call site contributes how much to the overall metric values.

![../_images/profiler-report-pages-inline-functions-table.png](https://docs.nvidia.com/nsight-compute/_images/profiler-report-pages-inline-functions-table.png)

Inline Functions Table

###### Source Markers Table

The code in the different _Views_ can also contain warnings, errors or just notifications that are displayed as _Source Markers_ in the left header, as shown below. These can be generated from multiple sources, but as of now only NvRules are supported.

![../_images/profiler-report-pages-source-markers.png](https://docs.nvidia.com/nsight-compute/_images/profiler-report-pages-source-markers.png)

Source Markers

This table shows all lines containing _Source Markers_ in the open View. The user can click on the line number to navigate to the corresponding line in the view.

![../_images/profiler-report-pages-source-markers-table.png](https://docs.nvidia.com/nsight-compute/_images/profiler-report-pages-source-markers-table.png)

Source Markers Table

###### Statistics Table

This table allows the user to select multiple lines in the source code above and calculate a variety of statistics. They can be used to check if anything seems unusual, like an unexpected gap between minimum and maximum.

Cells will be empty if the metric has either not been collected or its statistical value is not meaningful in the context of the calculation. For example, a memory access or any other non-numeric metric can’t have a minimum or maximum since there is no defined order.

![../_images/profiler-report-pages-source-statistics.png](https://docs.nvidia.com/nsight-compute/_images/profiler-report-pages-source-statistics.png)

Statistics Table

##### Limitations

Graph Profiling

When profiling complete CUDA graphs, instruction-level source metrics are not available.

#### Context Page

The _CPU Call Stack_ section of this report page shows the CPU call stack(s) for the executing CPU thread at the time the kernel was launched. For this information to show up in the profiler report, the option to collect CPU call stacks had to be enabled in the [Connection Dialog](index.html#connection-dialog) or using the corresponding NVIDIA Nsight Compute CLI command line parameter.

![../_images/profiler-report-pages-callstack.png](https://docs.nvidia.com/nsight-compute/_images/profiler-report-pages-callstack.png)

NVIDIA Nsight Compute supports to collect _native_ CPU call stacks as well as call stacks for _Python_ applications. Either or both types can be selected in the _Activity_ menu of the [Connection Dialog](index.html#connection-dialog) (via the “CPU Call Stack Types” option), or using the NVIDIA Nsight Compute [CLI command line parameter](../NsightComputeCli/index.html#command-line-options-launch) –call-stack-type. In case both types are enabled, a dropdown menu will appear to select the desired call stack type.

![../_images/profiler-report-pages-callstack-python.png](https://docs.nvidia.com/nsight-compute/_images/profiler-report-pages-callstack-python.png)

Note that Python call stack collection requires CPython version 3.9 or later.

The _NVTX State_ section of this report page shows the NVTX context when the kernel was launched. All thread-specific information is with respect to the thread of the kernel’s launch API call. Note that NVTX information is only collected if the profiler is started with NVTX support enabled, either in the [Connection Dialog](index.html#connection-dialog) or using the NVIDIA Nsight Compute CLI command line parameter.

![../_images/profiler-report-pages-nvtx.png](https://docs.nvidia.com/nsight-compute/_images/profiler-report-pages-nvtx.png)

This page has been renamed from “Call Stack / NVTX Page”.

#### Comments Page

The _Comments_ page aggregates all section comments in a single view and allows the user to edit those comments on any launch instance or section, as well as on the overall report. Comments are persisted with the report. If a section comment is added, the comment icon of the respective section in the [Details Page](index.html#profiler-report-details-page) will be highlighted.

#### Raw Page

The _Raw_ page shows a list of all collected metrics with their units per profiled kernel launch. It can be exported, for example, to CSV format for further analysis. The page features a filter edit to quickly find specific metrics. You can transpose the table of kernels and metrics by using the _Transpose_ button.

If a metric has multiple instance values, the number of instances is shown after the standard value. This metric for example has ten instance values: `35.48 {10}`. You can select in the [Profile](index.html#options-profile) options dialog that all instance values should be shown individually or inspect the metric result in the [Metric Details](index.html#tool-window-metric-details) tool window.

#### Session Page

This _Session_ page contains basic information about the report and the machine, as well as device attributes of all devices for which launches were profiled. When switching between launch instances, the respective device attributes are highlighted.

### 3.6.3. Metrics and Units

Numeric metric values are shown in various places in the report, including the header and tables and charts on most pages. NVIDIA Nsight Compute supports various ways to display those metrics and their values.

When available and applicable to the UI component, metrics are shown along with their unit. This is to make it apparent if a metric represents cycles, threads, bytes/s, and so on. The unit will normally be shown in rectangular brackets, e.g. `Metric Name [bytes] 128`.

By default, units are scaled automatically so that metric values are shown with a reasonable order of magnitude. Units are scaled using their SI-factors, i.e. byte-based units are scaled using a factor of 1000 and the prefixes K, M, G, etc. Time-based units are also scaled using a factor of 1000, with the prefixes n, u and m. This scaling can be disabled in the [Profile](index.html#options-profile) options.

Metrics which could not be collected are shown as `n/a` and assigned a warning icon. If the metric floating point value is out of the regular range (i.e. `nan` (Not a number) or `inf` (infinite)), they are also assigned a warning icon. The exception are metrics for which these values are expected and which are allow-listed internally.

### 3.6.4. Filtered Profiler Report

A filtered profiler report is a subset of the full profiler report. You can save filtered or selected results from the full profiler report to a new file. To save the filtered results, you can use the File > Save Filtered Results As menu option, which will save the [Filtered Results](../NsightCompute/index.html#profiler-report-header-filter-dialog) to a new file. Alternatively, you can select one or more results on the [Summary Page](../NsightCompute/index.html#summary-page) or [Raw Page](../NsightCompute/index.html#raw-page) and save them to a new file using the File > Save Selected Results As menu option or the right-click Save Result(s) menu option.

## 3.7. Baselines

NVIDIA Nsight Compute supports diffing collected results across one or multiple reports using Baselines. Each result in any report can be promoted to a baseline. This causes metric values from all results in all reports to show the difference to the baseline. If multiple baselines are selected simultaneously, metric values are compared to the average across all current baselines. Baselines are not stored with a report and are only available as long as the same NVIDIA Nsight Compute instance is open, unless they are saved to a `ncu-bln` file from the [Baselines tool window](index.html#tool-window-baselines).

![../_images/baselines.png](https://docs.nvidia.com/nsight-compute/_images/baselines.png)

Profiler report with one baseline

Select _Add Baseline_ to promote the current result in focus to become a baseline. If a baseline is set, most metrics on the [Details Page](index.html#profiler-report-details-page), [Raw Page](index.html#profiler-report-raw-page) and [Summary Page](index.html#profiler-report-summary-page) show two values: the current value of the result in focus, and the corresponding value of the baseline or the percentage of change from the corresponding baseline value. (Note that an infinite percentage gain, _inf%_ , may be displayed when the baseline value for the metric is zero, while the focus value is not.)

If multiple baselines are selected, each metric will show the following notation:
    
    
    <focus value> (<difference to baselines average [%]>, z=<standard score>@<number of values>)
    

The standard score is the difference between the current value and the average across all baselines, normalized by the standard deviation. If the number of metric values contributing to the standard score equals the number of results (current and all baselines), the @<number of values> notation is omitted.

![../_images/baselines-multiple.png](https://docs.nvidia.com/nsight-compute/_images/baselines-multiple.png)

Profiler report with multiple baselines

Baseline added for the current result in focus is always shown on the top. However, the actual order of added baselines is shown in the [Baselines tool window](index.html#tool-window-baselines).

![../_images/baselines-tool-window.png](https://docs.nvidia.com/nsight-compute/_images/baselines-tool-window.png)

Baselines tool window with mutliple baselines

Double-clicking on a baseline name allows the user to edit the displayed name. Edits are committed by pressing `Enter/Return` or upon loss of focus, and abandoned by pressing `Esc`. Hovering over the baseline color icon allows the user to remove this specific baseline from the list.

Use the _Clear Baselines_ entry from the dropdown button, the [Profile](index.html#options-profile) menu, or the corresponding toolbar button to remove all baselines.

Baseline changes can also be made in the [Baselines tool window](index.html#tool-window-baselines).

## 3.8. Standalone Source Viewer

NVIDIA Nsight Compute includes a standalone source viewer for _cubin_ files. This view is identical to the [Source Page](index.html#profiler-report-source-page), except that it won’t include any performance metrics.

Cubin files can be opened from the _File_ > _Open_ main menu command. The SM Selection dialog will be shown before opening the standalone source view. If available, the SM version present in the file name is pre-selected. For example, if your file name is `mergeSort.sm_80.cubin` then SM 8.0 will be pre-selected in the dialog. Choose the appropriate SM version from the drop down menu if it’s not included in the file name.

![../_images/sm-selection-dialog.png](https://docs.nvidia.com/nsight-compute/_images/sm-selection-dialog.png)

SM Selection Dialog

Click Ok button to open [Standalone Source Viewer](index.html#cubin-viewer).

![../_images/cubin-viewer.png](https://docs.nvidia.com/nsight-compute/_images/cubin-viewer.png)

Standalone Source Viewer

## 3.9. Source Comparison

Source comparison provides a way to see the source files of two profile results side by side. It enables to quickly identify source differences and understand changes in metric values.

To compare two results side by side add one result as a baseline, navigate to the other result, and then click the _Source Comparison_ button located in the report header.

For example, if you want to compare kernel XYZ from report R1 with kernel XYZ from report R2, first open report R1, add the profile result for kernel XYZ as baseline, open report R2, choose kernel XYZ, and then click the Source Comparison button.

Source comparison will be shown only with first added baseline result.

![../_images/source-comparison-from-header.png](https://docs.nvidia.com/nsight-compute/_images/source-comparison-from-header.png)

Source Comparison Button

![../_images/source-comparison-document.png](https://docs.nvidia.com/nsight-compute/_images/source-comparison-document.png)

Source Comparison

Currently only high-level Source (CUDA-C) view and SASS view are supported for comparison. The source difference heatmap is located on the very left side. Each of its partitions represents a side of the source diff. Clicking on the heatmap will scroll to the respective source line.

Navigation to the previous or next difference is supported using the navigation buttons or the keyboard shortcuts _Ctrl + 1_ and _Ctrl + 2_.

![../_images/source-comparison-navigation-buttons.png](https://docs.nvidia.com/nsight-compute/_images/source-comparison-navigation-buttons.png)

Source Comparison Navigation Buttons

On the SASS view, the _Diff By_ drop down menu allows you to choose the diff basis based on either _Opcode_ or _Full Instruction_. For the latter, all instruction modifiers and arguments are considered for the comparison in addition to the opcode.

![../_images/source-comparison-diff-by-menu.png](https://docs.nvidia.com/nsight-compute/_images/source-comparison-diff-by-menu.png)

Source Comparison Diff By Menu

## 3.10. Report Merge Tool

NVIDIA Nsight Compute provides a _Report Merge Tool_ utility that enables users to combine multiple Nsight Compute reports with the `.ncu-rep` extension into a single file. It is particularly useful for multi-GPU systems and scenarios when comparing and analyzing several reports individually becomes impractical. The report created with ReportMergeTool is fully compatible with Nsight Compute, and can be used for visualization and analysis in the UI and on the command line.

There are 2 options for using the MergeTool.

  1. **Launching from the UI menu**

Select the Merge Tool item from Tools in the [Main Menu](index.html#main-menu).

  2. **Launching from the Command Line**

Call the binary in `$NCU_INSTALL_PATH/extras/ReportUtils/ReportMergeTool`.

It accepts a directory as an input, and it will merge all the reports under that directory. Other features follow exactly the same logic as the UI.


### 3.10.1. File Selection Tab

This tab handles the selection of reports and allows to set the output name and/or path for the resulting merged report. There are 2 ways of selecting report files to merge.

  1. **Selecting from system files**

This option shows a hierarchical view of `.ncu-rep` files in the system, and the containing folders. Individual reports can be selected together with entire folders. The _Report Directory_ input changes the root directory of the hierarchical view.

![../_images/mergetool-files.png](https://docs.nvidia.com/nsight-compute/_images/mergetool-files.png)

Selection from System Files

  2. **Selecting from currently open reports**

When some reports are already open in Nsight Compute, MergeTool will suggest to merge them first. To change select from the system files, set _Select from_ combo box item to _System Files_ , or use the _Report Directory_ selector.


### 3.10.2. Metric Filters Tab

This tab guides the selection and merging process of individual metrics. Selecting the _Merge Operation_ defines which metrics are merged and kept across all report results. _Aggregation Operation_ then tells how these metrics are merged, and how the resulting value is calculated from multiple values. For any other filtering operation metric names can be provided in _Results_ text box, or _Metric Sets/Sections_ can be selected.

![../_images/mergetool-metrics.png](https://docs.nvidia.com/nsight-compute/_images/mergetool-metrics.png)

Metric Filter Tab

Metric Merge Operations

Metric Merge Operations dictate how the metric set for matching results is maintained in the merged report.

  * **Union** keeps all the metrics that appear at least once across the matching results in all reports. There can only be one metric set per result, metrics that don’t exist in some reports will appear empty in those cases.

  * **Intersection** keeps only the metrics that exist in all matching results across the reports. This option guarantees that no metrics will have missing values in the output. Note that if reports contain no matching results, they cannot be merged, which may result in some metrics being empty in the final output.


Aggregation Operations

Whenever two or more report results are merged, the metric values are aggregated. Aggregation operations tell the MergeTool how the metrics are combined together. As a result, the merged report will contain one of these four options as a metric value.

  * **Average** of the combined metric values

  * **Sum**

  * **Minimum**

  * **Maximum**


Metric Selection: Metric List, Sets, Sections

Apart from Merge Operations, the metric list for the output report can be manipulated using filtering. The simplest way to set the metric list is to enter the metrics as a comma-separated list in the _Metrics_ text box.

![../_images/mergetool-metrics-sections.png](https://docs.nvidia.com/nsight-compute/_images/mergetool-metrics-sections.png)

Metric Sets/Sections

Additionally, metric sets and sections can be used, by loading a view similar to [Metric Selection](index.html#metric-selection). Selecting the sets and sections modifies the appearance of the reports by changing the sections that are displayed on the _Details_ page. The metrics of the reports are not affected. To remove unwanted sections, select the _Remove Unselected Sections_ checkbox.

### 3.10.3. Result Filters Tab

This tab guides the selection and merging process of individual results. It allows to select the [Merge Operation](index.html#mergetool-results-merge-op) to define which results are merged and kept.

For any other filtering operations result names can be provided in the _Results_ text box as a comma-separated list. If this list is set, only the provided results will appear in the output.

![../_images/mergetool-results.png](https://docs.nvidia.com/nsight-compute/_images/mergetool-results.png)

Result Filter Tab

#### Result Merge Operations

Result merge operations, similarly to the metric merge operations, tell the MergeTool which results to keep and which results to discard. Additionally, they can manage the merge process by matching the results from the merged reports. The matched results are the ones that will be aggregated together into a single result. The four available options provide control over which results are merged together and how to handle unmatched results.

  * **Union** merges only matching results from every report, keeps unmatched results as is. When multiple iterations of the same kernel appear in a report, they are kept separate and are only merged with the corresponding iteration of this kernel in other reports.

_For example:_ Merging Report1 with results name {kernel_A, kernel_B} and Report2 with {kernel_A, kernel_C} results in {kernel_Amerged , kernel_B, kernel_C}. i.e.,

\\(\\{A, B\\} + \\{A, C\\} = \\{A^{merged}, B, C\\}\\)

_Note:_ if multiple results share the same name, only matching ones are merged. i.e.,

\\(\\{A_{1}, A_{2}, A_{3}\\} + \\{A_{1}, A_{2}\\} = \\{A_{1}^{merged}, A_{2}^{merged}, A_{3}\\}\\)

  * **Intersection** merges matching results from every report in the same way as _Union_ does, but discards all the unmatched results.

\\(\\{A, B\\} + \\{A, C\\} = \\{A^{merged}\\}\\)

  * **Collapse** merges all results sharing the same name. When multiple iterations of the same kernel appear in a report, merges them together.

\\(\\{A_{1}, A_{2}, A_{3}\\} + \\{A_{1}, A_{2}\\} = \\{A^{merged}\\}\\)

  * **Concat** keeps all the results in their original form and saves them in a single file, no metric aggregation is done.

\\(\\{A, B\\} + \\{A, C\\} = \\{A_{1}, B, A_{2}, C\\}\\)


#### Merged Report

To be able to track how a merged report was created, each merged report contains a comment with the list of reports that were merged, the merge operation, the aggregation operation and the sections that were selected.

![../_images/mergetool-comment.png](https://docs.nvidia.com/nsight-compute/_images/mergetool-comment.png)

Comment with merge details in the merged report

## 3.11. Clustering Window

The **Clustering Window** helps you analyze and compare multiple profiling reports by grouping similar reports together. This makes it easier to identify performance patterns and find relationships between different profiling sessions.

![../_images/clustering-window.png](https://docs.nvidia.com/nsight-compute/_images/clustering-window.png)

Clustering Window

### 3.11.1. How to Use Clustering

  1. **Open Clustering Window** : From the main menu, select _Tools_ > _Clustering Window_. The window can be docked or undocked like other tool windows.

  2. **Select Reports** : Click _Select Reports_ and choose multiple `.ncu-rep` files from the file dialog.

  3. **Configure Parameters** (Optional):

     * _Minimum Cluster Size_ : How many reports are needed to form a cluster (default: 3)

     * _Maximum Cluster Size_ : Maximum reports per cluster (0 = no limit)

  4. **Run Analysis** : Click _Cluster Reports_ to analyze the selected reports and generate clusters.


### 3.11.2. Clustering Tree

![../_images/clustering-tree.png](https://docs.nvidia.com/nsight-compute/_images/clustering-tree.png)

Clustering Tree

The clustering tree shows how reports are grouped together hierarchically:

  * **Reports and Clusters** : Individual reports are shown as leaf nodes in the tree with unique IDs, and clusters are shown as nodes with a **+** sign.

  * **Selecting Clusters** : Click the **+** nodes to select a cluster for merging. Selected clusters will be shown in green, and the corresponding entries in the similarity matrix below will be highlighted with a dotted border.

  * **Highlighting** : When you hover over any cluster or report in the tree, the corresponding entries are highlighted in the tree and the similarity matrix below.


The tree structure helps you understand:

  * Which reports are most similar to each other (grouped in the same cluster)

  * The hierarchical relationships between different groups of reports

  * How the clustering algorithm has organized your data


### 3.11.3. Similarity Matrix

![../_images/clustering-matrix.png](https://docs.nvidia.com/nsight-compute/_images/clustering-matrix.png)

Similarity Matrix

The similarity matrix displays a grid showing how similar each pair of reports is between each other. This matrix is used to generate the clustering tree - reports with higher similarity scores are grouped closer together in the tree structure.

  * **Rows and Columns** : Each row displays a report name along with its ID, while columns show report IDs that are used in the Clustering Tree.

  * **Similarity Scores** : Each cell contains a similarity value calculated based on the metric values from the reports.

  * **Color Coding** :

    * Darker colors indicate higher similarity between reports

    * Lighter colors indicate lower similarity

    * The highlighted entries in the similarity matrix are the ones that are being hovered over in the clustering tree.

  * **Similarity Score Values** : Hover over any cell to see the exact similarity score between those two reports.


Control Buttons

  * _Select Reports_ opens a file dialog to choose the `.ncu-rep` files you want to analyze.

  * _Run Clustering_ starts the clustering analysis using the selected reports and current parameters.

  * _Reset to Default_ restores clustering parameters to their default values.

  * _Clear Clusters_ clears the current cluster selection.

  * _Suggest Clusters_ automatically recommends optimal clustering parameters based on your data characteristics. The suggestion will cover most of the reports, but outliers may be excluded.

  * _Merge Clusters_ creates a merged report from the selected clusters. The merged reports can be processed using the [Report Merge Tool](index.html#report-merge-tool.html) for further analysis and customization.


## 3.12. Occupancy Calculator

NVIDIA Nsight Compute provides an _Occupancy Calculator_ that allows you to compute the multiprocessor occupancy of a GPU for a given CUDA kernel. It replaces the previously provided CUDA Occupancy Calculator spreadsheet.

The Occupancy Calculator can be opened directly from a profile report or as a new activity. The occupancy calculator data can be saved to a file using _File > Save_. By default, the file uses the `.ncu-occ` extension. The occupancy calculator file can be opened using _File > Open File_

  1. **Launching from the Connection Dialog**

Select the Occupancy Calculator activity from the connection dialog. You can optionally specify an occupancy calculator data file, which is used to initialize the calculator with the data from the saved file. Click the _Launch_ button to open the Occupancy Calculator.

![../_images/occupancy-calculator-activity.png](https://docs.nvidia.com/nsight-compute/_images/occupancy-calculator-activity.png)
  2. **Launching from the Profiler Report**

The Occupancy Calculator can be opened from the _Profiler Report_ using the calculator button located in the report header or in the header of the _Occupancy_ section on the _Detail Page_.

![../_images/occupancy-calculator-from-header.png](https://docs.nvidia.com/nsight-compute/_images/occupancy-calculator-from-header.png)

Details page header

![../_images/occupancy-calculator-from-section.png](https://docs.nvidia.com/nsight-compute/_images/occupancy-calculator-from-section.png)

Occupancy section header


The user interface consists of an input section as well as tables and graphs that display information about GPU occupancy. To use the calculator, change the input values in the input section, click the _Apply_ button and examine the tables and graphs.

### 3.12.1. Tables

The tables show the occupancy, as well as the number of active threads, warps, and thread blocks per multiprocessor, and the maximum number of active blocks on the GPU.

![../_images/occupancy-calculator-tables.png](https://docs.nvidia.com/nsight-compute/_images/occupancy-calculator-tables.png)

Tables

### 3.12.2. Utilization

These stacked bar graphs show the portion of resources that is allocated to the blocks, the unallocated portion due to some other resource being a limiter, and lastly the unused portion that has the size less than the minimum amount required by a block.

![../_images/occupancy-calculator-utilization.png](https://docs.nvidia.com/nsight-compute/_images/occupancy-calculator-utilization.png)

Utilization

### 3.12.3. Graphs

The graphs show the occupancy for your chosen block size as a blue circle, and for all other possible block sizes as a line graph. The “Show As” option allows you to switch between the occupancy in percent and absolute number of warps on Y-axis. When you hover over the graphs, the nearest data point will be highlighted and its X and Y-axis values will appear in a tooltip.

![../_images/occupancy-calculator-graphs.png](https://docs.nvidia.com/nsight-compute/_images/occupancy-calculator-graphs.png)

Graphs

### 3.12.4. GPU Data

The _GPU Data_ shows the properties of all supported devices.

![../_images/occupancy-calculator-gpu-data.png](https://docs.nvidia.com/nsight-compute/_images/occupancy-calculator-gpu-data.png)

GPU Data

## 3.13. Green Contexts support

NVIDIA Nsight Compute provides a number of features to ease the analysis of applications using [CUDA Green Contexts](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS), see the [Profiling Guide](../ProfilingGuide/index.html#special-configurations-green-contexts) for an overview about profiling such applications.

To identify Green Context results quickly, the icon of the _Attributes_ button in the [Profiler Report Header](index.html#profiler-report-header) turns into a _leaf_ for Green Context results. Clicking on this button navigates to the _Launch Statistics_ section of the result (if available), which contains [additional metrics](../ProfilingGuide/index.html#id28) related to the Green Context (e.g., its ID and the number of SMs used). Some of this information is also available from the tooltip when hovering over the _Attributes_ button.

To distinguish between Green-Contexts _attributable_ and _non-attributable_ metrics when navigating the [Details page](index.html#details-page), _Green Context markers_ may be used. These can be toggled via the _View_ menu of the [Report Header](index.html#profiler-report-header) and adds a _leaf_ icon behind _attributable_ metrics in all section headers. When hovering over attributable metrics, the tooltip also shows a hint about the scaling of such metrics. Additionally, the [Metrics Details](index.html#metric-details) tool window shows an _Attribution_ row for metrics of Green Context results.

![../_images/green-contexts-details-page.png](https://docs.nvidia.com/nsight-compute/_images/green-contexts-details-page.png)

_Details_ page of a Green Context result showing _Attributes_ tooltip, _Metrics Details_ tool window and Green Context markers.

If a report contains Green Context results, the [Session page](index.html#session-page) includes an additional _Green Context Resources_ section. It contains a table with information about the resources used by each Green Context, such as the TPCs and workqueue resources assigned to it. The _Result IDs_ column allows you to navigate to each profile result that uses the corresponding Green Context.

![../_images/green-contexts-session-page.png](https://docs.nvidia.com/nsight-compute/_images/green-contexts-session-page.png)

When profiling Green Context applications interactively, the [Resources](index.html#resources) tool window also allows to track the state of all Green Contexts, their resources, as well as their associated streams.

![../_images/green-contexts-resource-tool-window-with-tpc-mask.png](https://docs.nvidia.com/nsight-compute/_images/green-contexts-resource-tool-window-with-tpc-mask.png)

## 3.14. Acceleration Structure Viewer

The _Acceleration Structure Viewer_ allows inspection of acceleration structures built using the OptiX API. In modern ray tracing APIs like OptiX, _acceleration structures_ are data structures describing the rendered scene’s geometries that will be intersected when performing ray tracing operations. More information concerning acceleration structures can be found in the [OptiX programming guide](https://raytracing-docs.nvidia.com/optix7/guide/index.html#acceleration_structures#acceleration-structures).

It is the responsibility of the user to set these up and pass them to the OptiX API which translates them to internal data structures that perform well on modern GPUs. The description created by the user can be very error-prone and it is sometimes hard to understand why the rendered result is not as expected. The _Acceleration Structure Viewer_ is a component allowing OptiX users to inspect the acceleration structures they build before launching a ray tracing pipeline.

The _Acceleration Structure Viewer_ is opened through a button in the [Resources](index.html#tool-window-resources) window. The button will only be available when the currently viewed resource is _OptiX: TraversableHandles_. It opens the currently selected handle.

![../_images/as-viewer-open-button.png](https://docs.nvidia.com/nsight-compute/_images/as-viewer-open-button.png)

The viewer is multi-paned: it shows a hierarchical view of the acceleration structure on the left, a graphical view of the acceleration structure in the middle, and controls and options on the right. In the hierarchical tree view on the left of the viewer the _instance acceleration structures (IAS)_ , _geometry acceleration structures (GAS)_ , child instances and child geometries are shown. In addition to this, some general properties for each of them is shown such as their primitive count, surface area and size on the device.

![../_images/as-viewer.png](https://docs.nvidia.com/nsight-compute/_images/as-viewer.png)

In the hierarchical view on the left of the _Acceleration Structure Viewer_ , the following information is displayed where applicable.

Acceleration Structure Hierarchical Columns Column | Description  
---|---  
Name | An identifier for each row in the hierarchy. Click on the check box next to the name to show or hide the selected geometry or hierarchy. Double-click on this entry to jump to the item in the rendering view.  
# Prims | The number of primitives that make up this acceleration structure.  
Surface Area | A calculation of the total surface area for the AABB that bounds the particular entry.  
Size | The size of the output buffer on the device holding this _acceleration structure_.  
  
Performance analysis tools are accessible in the bottom left corner on the main view. These tools help identify potential performance problems that are outlined in the [RTX Ray Tracing Best Practices Guide](https://developer.nvidia.com/blog/best-practices-using-nvidia-rtx-ray-tracing). These analysis tools aim to give a broad picture of acceleration structures that may exhibit sub-optimal performance. To find the most optimal solution, profiling and experimentation is recommended but these tools may paint a better picture as to why one structure performs poorly compared to another.

Acceleration Structure Analysis Tools Action | Description  
---|---  
Instance Overlaps | Identifies instance AABBs that overlap with other instances in 3D. Consider merging GASes when instance world-space AABBs overlap significantly to potentially increase performance.  
Instance Heatmap | This allows you to set the threshold used by the AABB heatmap rendered in the visualizer.  
  
### 3.14.1. Navigation

The _Acceleration Structure Viewer_ supports multiple navigation modes. The navigation mode can be changed using the combo box in the camera controls pane, to the right of the rendering pane. The keyboard and mouse bindings for each mode are as follows:

Acceleration Structure Key Bindings Binding | Fly Camera | Dolly Camera | Orbit Camera  
---|---|---|---  
**WASD/Arrow Keys** | Move forward, backward, left, right | Move forward, backward, left, right | Track (Move up, down, left, right)  
**E/Q** | Move up/down | Move up/down | n/a  
**Z/C** | Increase/decrease field of view | Increase/decrease field of view | Increase/decrease field of view  
**Shift/Ctrl** | Move faster/slower | Move faster/slower | Move faster/slower  
**Mousewheel** | Zoom in/out | Zoom in/out | Zoom in/out  
**LMB + Drag** | Rotate in place | Rotate left/right, move forward/backward | Rotate around the geometry  
**RMB + Drag** | Zoom in/out | Rotate in place | Zoom in/out  
**MMB + Drag** | Track (Move up, down, left, right) | Track (Move up, down, left, right) | Track (Move up, down, left, right)  
**Alt** | Temporarily switch to Orbit Camera | Temporarily switch to Orbit Camera | n/a  
**F/Double Click** | Focus on the selected geometry | Focus on the selected geometry | Focus on the selected geometry  
  
Based on the coordinate system of the input geometry, you may need to change the **Up Direction** setting to Z-Axis or the **Coordinates** setting to RHS. To reset the camera to its original location, click **Reset Camera**.

There are also a selection of Camera Controls for fast and precise navigation. To save a position, use the bookmarks controls. Each node within the acceleration structure hierarchy can also be double-clicked to quickly navigate to that location.

![../_images/as-viewer-cam.png](https://docs.nvidia.com/nsight-compute/_images/as-viewer-cam.png)

### 3.14.2. Filtering and Highlighting

The acceleration structure view supports acceleration structure filtering as well as highlighting of data matching particular characteristics. The checkboxes next to each geometry allow users to toggle the rendering of each traversable.

Geometry instances can also be selected by clicking on them in the main graphical view. Additionally, right clicking in the main graphical view gives options to hide or show all geometry, hide the selected geometry, or hide all but the selected geometry.

![../_images/as-viewer-display-filter.png](https://docs.nvidia.com/nsight-compute/_images/as-viewer-display-filter.png)

Beyond filtering, the view also supports highlight-based identification of geometry specified with particular flags. Checking each highlight option will identify those resources matching that flag, colorizing for easy identification. Clicking an entry in this section will dim all geometry that does **not** meet the filter criteria allowing items that match the filter to standout. Selecting multiple filters requires the passing geometry to meet all selected filters (e.g., AND logic). Additionally, the heading text will be updated to reflect the number of items that meet this filter criteria.

![../_images/as-viewer-property-filter.png](https://docs.nvidia.com/nsight-compute/_images/as-viewer-property-filter.png)

### 3.14.3. Rendering Options

Under the highlight controls, additional rendering options are available. These include methods to control the geometry colors and the ability to toggle the drawing of wireframes for meshes and AABBs.

![../_images/as-viewer-rendering-options.png](https://docs.nvidia.com/nsight-compute/_images/as-viewer-rendering-options.png)

### 3.14.4. Exporting

The data displayed in the acceleration structure viewer document can be saved to file. Exporting an _Acceleration Structure Viewer_ document allows for persisting the data you have collected beyond the immediate analysis session. This capability is particularly valuable for comparing different revisions of your geometry or sharing with others. Bookmarks are persisted as well.

## 3.15. Options

NVIDIA Nsight Compute options can be accessed via the main menu under _Tools_ > _Options_. All options are persisted on disk and available the next time NVIDIA Nsight Compute is launched. When an option is changed from its default setting, its label will become bold. You can use the _Restore Defaults_ button to restore all options to their default values.

![../_images/options-profile.png](https://docs.nvidia.com/nsight-compute/_images/options-profile.png)

Profile options

### 3.15.1. Profile

#### Report Sections

Configure the directories and import settings for section files and rules.

NVIDIA Nsight Compute Report Section Options Option Name | Description | Values  
---|---|---  
Sections Directory | Directory from which to import section files and rules. Relative paths are with respect to the NVIDIA Nsight Compute installation directory. |   
Include Sub- Directories | Recursively include section files and rules from sub-directories. | Yes (Default)/No  
  
#### Report Rules

Configure how rules are applied and reloaded during profiling sessions.

NVIDIA Nsight Compute Report Rules Options Option Name | Description | Values  
---|---|---  
Apply Applicable Rules Automatically | Automatically apply active and applicable rules. | Yes (Default)/No  
Reload Rules Before Applying | Force a rule reload before applying the rule to ensure changes in the rule script are recognized. | Yes/No (Default)  
  
#### Report UI

Customize the user interface behavior for report viewing and navigation.

NVIDIA Nsight Compute Report UI Options Option Name | Description | Values  
---|---|---  
Default Report Page | The report page to show when a report is generated or opened. _Auto_ lets the tool decide the best page to show when opening a report. | 

  * Summary (Default)
  * Details
  * Source
  * Context
  * Comments
  * Raw
  * Session
  * Auto

  
Function Name Mode | Determines how function/kernel names are shown. | 

  * Auto (Default)
  * Demangled
  * Function
  * Mangled

  
NVTX Rename Mode | Determines how NVTX information is used for renaming. Range replay results are always renamed when possible. | 

  * None
  * Kernel
  * Resources (Default)
  * All

  
Show Metrics Aggregation | Show aggregate of all results per each counter metric in the table header and aggregated value of randomly selected metrics in the bottom-right label. | Yes (Default)/No  
Show Y-axis labels | Display Y-axis labels on the timeline to improve the interpretation of minimum and maximum metric value. | Yes (Default)/No  
Rename Demangled Names | Perform auto-simplification on kernel demangled names or import renamed names from a configuration file. | Yes (Default)/No  
Rename Kernels Config Path | Use a configuration file to rename multiple demangled names and export demangled names from the report. | ncu-kernel-renames.yaml  
  
#### Report Baselines

Configure baseline display settings for result comparison.

NVIDIA Nsight Compute Report Baselines Options Option Name | Description | Values  
---|---|---  
Maximum Baseline Name Length | The maximum length of baseline names. | 1..N (Default: 40)  
Number of Full Baselines to Display | Number of baselines to display in the report header with all details in addition to the current result or the baseline added for the current result. | 0..N (Default: 2)  
  
#### Report Metrics

Configure how metrics are displayed and processed in reports.

NVIDIA Nsight Compute Report Metrics Options Option Name | Description | Values  
---|---|---  
Auto-Convert Metric Units | Auto-adjust displayed metric units and values (e.g. Bytes to KBytes). | Yes (Default)/No  
Show Instanced Metric Values | Show the individual values of instanced metrics in tables. | Yes/No (Default)  
Show Metrics As Floating Point | Show all numeric metrics as floating-point numbers. | Yes/No (Default)  
Show Knowledge Base Information | Show information from the knowledge base in (metric) tooltips to explain terminology. Note: Nsight Compute needs to be restarted for this option to take effect. | Yes (Default)/No  
  
#### Report Summary Page

Configure settings for the Summary page in reports.

NVIDIA Nsight Compute Report Summary Page Options Option Name | Description | Values  
---|---|---  
Metrics/ Properties | List of metrics and properties to show on the summary page. Comma-separated list of metric entries. Each entry has the format {Label:MetricName}. |   
  
#### Report Details Page

Configure settings for the Details page in reports.

NVIDIA Nsight Compute Report Summary Page Options Option Name | Description | Values  
---|---|---  
Default Section Body Visibility | Controls the default visibility of section bodies when opening a report. | Default (Default)/All  
  
#### Report Source Page

Configure settings for the Source page view.

NVIDIA Nsight Compute Source Page Options Option Name | Description | Values  
---|---|---  
Delay Load ‘Source’ Page | Delays loading the content of the report page until the page becomes visible. Avoids processing costs and memory overhead until the report page is opened. | Yes/No (Default)  
Show Single File For Multi-File Sources | Shows a single file in each Source page view, even for multi-file sources. | Yes/No (Default)  
Show Only Executed Functions | Shows only executed functions in the source page views. Disabling this can impact performance. | Yes (Default)/No  
Default Metric Value Mode | Default setting for the mode in which the metric values are displayed. | Relative (Default)/Absolute  
Default Metric Precision Mode | Default setting for the precision in which absolute metric values are displayed. | Abbreviated (Default)/Full  
Auto-Resolve Remote Source Files | Automatically try to resolve remote source files on the source page (e.g. via SSH) if the connection is still registered. | Yes/No (Default)  
Enable Register Dependencies | Track dependencies between SASS registers/predicates and display them in the SASS view. | Yes (Default)/No  
SASS Analysis Size Threshold (KB) | Enable SASS flow graph analysis for functions below this threshold. SASS analysis is required for Live Register and Register Dependency information. Set to -1 to enable analysis for all functions. | -1..N (Default: 1024)  
Enable ELF Verification | Enable ELF (cubin) verification to run every time before SASS analysis. This should only be enabled when working with applications compiled before CUDA 11.0 or when encountering source page issues. | Yes/No (Default)  
  
#### API Stream View

Configure settings for the API Stream View.

NVIDIA Nsight Compute API Stream View Options Option Name | Description | Values  
---|---|---  
API Call History | Number of recent API calls shown in API Stream View. | 1..N (Default: 100)  
  
### 3.15.2. Fonts and Colors

#### Fonts

Configure the fonts used throughout the application interface. General fonts affect all UI elements except code, while code fonts are specifically used for source code display and similar content.

NVIDIA Nsight Compute Font Options Option Name | Description | Values  
---|---|---  
General Font | General font used for all non-code UI elements. | Select an installed font via “Change…” button. (Default: Roboto@9pt)  
Code Font | Font used for source code and code-like UI elements. | Select an installed font via “Change…” button. (Default: Cascadia Mono@9pt)  
  
#### Colors

Customize the visual appearance of the application through theme selection and color scale preferences. These settings affect the overall look and feel of the interface and how data is represented visually.

NVIDIA Nsight Compute Color Options Option Name | Description | Values  
---|---|---  
Color Theme | The currently selected application color theme. | 

  * Dark (Default)
  * Light

  
Default Color Scale | Default color map used to represent a qualitative scale of values. | 

  * Flammenmeer
  * Viridis
  * Inferno
  * Magma
  * Plasma
  * Jet (Default)

  
  
### 3.15.3. Environment

#### Visual Experience

Configure how NVIDIA Nsight Compute handles display and windowing behaviors, particularly for multi-monitor setups.

NVIDIA Nsight Compute Visual Experience Options Option Name | Description | Values  
---|---|---  
Use Enhanced Windowing Experience | Disable Mixed DPI Scaling if unwanted artifacts are detected when using monitors with different DPIs. Requires app restart. | Yes (Default)/No  
  
#### Windowing

Control window behavior and management settings.

NVIDIA Nsight Compute Windowing Options Option Name | Description | Values  
---|---|---  
Floating Windows Always on Top | Configure floating tool windows to always stay on top of the main window. Requires app restart. | Yes/No (Default)  
  
#### Documents Folder

Configure where NVIDIA Nsight Compute stores project files and documentation.

NVIDIA Nsight Compute Documents Folder Options Option Name | Description | Values  
---|---|---  
Documents Folder | The folder where projects and documents will be saved. |   
  
#### Startup Behavior

Define how NVIDIA Nsight Compute behaves when launched.

NVIDIA Nsight Compute Startup Behavior Options Option Name | Description | Values  
---|---|---  
At Startup | What to do when the host is launched. | 

  * Show Welcome Page (Default)
  * Show Quick Launch Dialog
  * Load Last Project
  * Show Empty Environment

  
  
#### Updates

Configure version update notification settings.

NVIDIA Nsight Compute Updates Options Option Name | Description | Values  
---|---|---  
Show version update notifications | Show notifications when a new version of this product is available. | Yes (Default)/No  
  
### 3.15.4. Connection

Connection properties are grouped into _Target Connection Options_ and _Host Connection Properties_.

#### Target Connection Properties

The _Target Connection Properties_ determine how the host connects to the target application during an _Interactive Profile Activity_. This connection is used to transfer profile information to the host during the profile session.

NVIDIA Nsight Compute Target Connection Properties Option Name | Description | Values  
---|---|---  
Base Port | Base port used to establish a connection from the host to the target application during an _Interactive Profile_ activity (both local and remote). | 1-65535 (Default: 49152)  
Maximum Ports | Maximum number of ports to try (starting from _Base Port_) when attempting to connect to the target application. | 2-65534 (Default: 64)  
  
#### Host Connection Properties

The _Host Connection Properties_ determine how the command line profiler will connect to the host application during a _Profile Activity_. This connection is used to transfer profile information to the host during the profile session.

NVIDIA Nsight Compute Host Connection Options Option Name | Description | Values  
---|---|---  
Base Port | Base port used to establish a connection from the command line profiler to the host application during a _Profile_ activity (both local and remote). | 1-65535 (Default: 50152)  
Maximum Ports | Maximum number of ports to try (starting from _Base Port_) when attempting to connect to the host application. | 1-100 (Default: 10)  
  
#### SSH ProxyJump Connection Properties

The _SSH ProxyJump Connection Properties_ configure how NVIDIA Nsight Compute handles connections through SSH jump hosts. This is useful when profiling applications on target machines that are not directly accessible and require connecting through intermediate SSH servers. The ProxyJump feature allows for secure, multi-hop SSH connections while maintaining the profiling functionality.

NVIDIA Nsight Compute SSH ProxyJump Connection Options Option Name | Description | Values  
---|---|---  
Process Scan Timeout (seconds) | Maximum time to try searching for attachable process when using SSH ProxyJump connection. | 1-65535 (Default: 10)  
  
### 3.15.5. Timeline

#### Basic Settings

The _Timeline Basic Settings_ control the visualization and behavior of the Timeline view in NVIDIA Nsight Compute.

NVIDIA Nsight Compute Timeline Options Option Name | Description | Values  
---|---|---  
Show correlation arrows | Show arrows on correlated items on the Timeline | Yes (Default)/No  
  
### 3.15.6. Source Lookup

The Source Lookup options control how NVIDIA Nsight Compute locates and validates CUDA source files when displaying source code in the Source page. These settings are particularly important for debugging and profiling sessions where source files may have moved from their original compilation locations or when working with different development environments. The options help ensure accurate source code resolution and provide flexibility in handling file location mismatches.

NVIDIA Nsight Compute Source Lookup Options Option Name | Description | Values  
---|---|---  
Program Source Locations | Set program source search paths. These paths are used to resolve CUDA-C source files on the Source page if the respective file cannot be found in its original location. Files which cannot be found are marked with a _File Not Found_ error. See the _Ignore File Properties_ option for files that are found but don’t match. |   
Ignore File Properties | Ignore file properties (e.g. timestamp, size) for source resolution. If this is disabled, all file properties like modification timestamp and file size are checked against the information stored by the compiler in the application during compilation. If a file with the same name exists on a source lookup path, but not all properties match, it won’t be used for resolution (and a _File Mismatch_ error will be shown). | Yes/No (Default)  
  
### 3.15.7. Send Feedback…

NVIDIA Nsight Compute Send Feedback Options Option Name | Description | Values  
---|---|---  
Collect Usage and Platform Data | Choose whether or not you wish to allow NVIDIA Nsight Compute to collect usage and platform data. | Yes (Default)/No  
  
## 3.16. Projects

NVIDIA Nsight Compute uses _Project Files_ to group and organize profiling reports. At any given time, only one project can be open in NVIDIA Nsight Compute. Collected reports are automatically assigned to the current project. Reports stored on disk can be assigned to a project at any time. In addition to profiling reports, related files such as notes or source code can be associated with the project for future reference.

Note that only references to reports or other files are saved in the project file. Those references can become invalid, for example when associated files are deleted, removed or not available on the current system, in case the project file was moved itself.

NVIDIA Nsight Compute uses the `ncu-proj` file extension for project files.

When no custom project is current, a _default project_ is used to store e.g. the current [Start Activity Dialog](index.html#connection-dialog) entries. To remove all information from the default project, you must close NVIDIA Nsight Compute and then delete the file from disk.

  * On Windows, the file is located at `<USER>\AppData\Local\NVIDIA Corporation\NVIDIA Nsight Compute\`

  * On Linux, the file is located at `<USER>/.local/share/NVIDIA Corporation/NVIDIA Nsight Compute/`

  * On macOS, the file is located at `<USER>/Library/Application Support/NVIDIA Corporation/NVIDIA Nsight Compute/`


### 3.16.1. Project Dialogs

**New Project**

Creates a new project. The project must be given a name, which will also be used for the project file. You can select the location where the project file should be saved on disk. Select whether a new directory with the project name should be created in that location.

### 3.16.2. Project Explorer

The _Project Explorer_ window allows you to inspect and manage the current project. It shows the project name as well as all _Items_ (profile reports and other files) associated with it. Right-click on any entry to see further actions, such as adding, removing or grouping items. Type in the _Search project_ toolbar at the top to filter the currently shown entries.

![../_images/projects-explorer.png](https://docs.nvidia.com/nsight-compute/_images/projects-explorer.png)

Project Explorer

## 3.17. Visual Studio Integration Guide

This guide provides information on using NVIDIA Nsight Compute within Microsoft Visual Studio, using the [NVIDIA Nsight Integration](https://developer.nvidia.com/nsight-tools-visual-studio-integration) Visual Studio extension, allowing for a seamless development workflow.

### 3.17.1. Visual Studio Integration Overview

NVIDIA Nsight Integration is a Visual Studio extension that allows you to access the power of NVIDIA Nsight Compute from within Visual Studio.

When NVIDIA Nsight Compute is installed along with NVIDIA Nsight Integration, NVIDIA Nsight Compute activities will appear under the NVIDIA ‘Nsight’ menu in the Visual Studio menu bar. These activities launch NVIDIA Nsight Compute with the current project settings and executable.

For more information about using NVIDIA Nsight Compute from within Visual Studio, please visit

  * [NVIDIA Nsight Integration Overview](https://developer.nvidia.com/nsight-tools-visual-studio-integration)

  * [NVIDIA Nsight Integration User Guide](https://docs.nvidia.com/nsight-vs-integration/index.html)


Notices

Notices

ALL NVIDIA DESIGN SPECIFICATIONS, REFERENCE BOARDS, FILES, DRAWINGS, DIAGNOSTICS, LISTS, AND OTHER DOCUMENTS (TOGETHER AND SEPARATELY, “MATERIALS”) ARE BEING PROVIDED “AS IS.” NVIDIA MAKES NO WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO THE MATERIALS, AND EXPRESSLY DISCLAIMS ALL IMPLIED WARRANTIES OF NONINFRINGEMENT, MERCHANTABILITY, AND FITNESS FOR A PARTICULAR PURPOSE.

Information furnished is believed to be accurate and reliable. However, NVIDIA Corporation assumes no responsibility for the consequences of use of such information or for any infringement of patents or other rights of third parties that may result from its use. No license is granted by implication of otherwise under any patent rights of NVIDIA Corporation. Specifications mentioned in this publication are subject to change without notice. This publication supersedes and replaces all other information previously supplied. NVIDIA Corporation products are not authorized as critical components in life support devices or systems without express written approval of NVIDIA Corporation.

Trademarks

NVIDIA and the NVIDIA logo are trademarks or registered trademarks of NVIDIA Corporation in the U.S. and other countries. Other company and product names may be trademarks of the respective companies with which they are associated.

* * *
