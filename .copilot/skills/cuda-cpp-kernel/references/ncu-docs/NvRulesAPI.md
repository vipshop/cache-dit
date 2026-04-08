---
url: https://docs.nvidia.com/nsight-compute/NvRulesAPI/index.html
---

[ ![Logo](https://docs.nvidia.com/nsight-compute/_static/nsight-compute.png) ](../index.html)

Nsight Compute

  * [1\. Release Notes](../ReleaseNotes/index.html)
  * [2\. Profiling Guide](../ProfilingGuide/index.html)
  * [3\. Nsight Compute](../NsightCompute/index.html)
  * [4\. Nsight Compute CLI](../NsightComputeCli/index.html)


Developer Interfaces

  * [1\. Customization Guide](../CustomizationGuide/index.html)
  * [2\. Python Report Interface](../PythonReportInterface/index.html)
  * [3\. NvRules API](#)
  * [4\. Occupancy Calculator Python Interface](../OccupancyCalculatorPythonInterface/index.html)


Training

  * [Training](../Training/index.html)


Release Information

  * [Archives](../Archives/index.html)


Copyright and Licenses

  * [Copyright and Licenses](../CopyrightAndLicenses/index.html)


__[NsightCompute](../index.html)

  * [](../index.html) »
  * 3\. NvRules API
  *   * v2026.1.0 | [Archive](https://developer.nvidia.com/nsight-compute-history)


* * *

# 3\. NvRules API

_class _NvRules.IAction
    

Bases: [`object`](https://docs.python.org/3/library/functions.html#object "\(in Python v3.14\)")

The [`IAction`](#NvRules.IAction "NvRules.IAction") represents a profile result such as a CUDA kernel in a single range or a range itself in range-based profiling, for which zero or more metrics were collected.

NameBase_DEMANGLED
    

Name base for demangled names.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

NameBase_FUNCTION
    

Name base for function signature names.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

NameBase_MANGLED
    

Name base for mangled names.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

WorkloadType_CMDLIST
    

Workload type for CBL command lists.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

WorkloadType_GRAPH
    

Workload type for CUDA graphs.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

WorkloadType_KERNEL
    

Workload type for CUDA kernels or CUDA graph kernel nodes.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

WorkloadType_RANGE
    

Workload type for result ranges.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

__getitem__(_key_)
    

Get an [`IMetric`](#NvRules.IMetric "NvRules.IMetric") object contained in this [`IAction`](#NvRules.IAction "NvRules.IAction") by its name.

Parameters
    

**key** ([`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)")) – The name of the [`IMetric`](#NvRules.IMetric "NvRules.IMetric") object to retrieve.

Returns
    

An [`IMetric`](#NvRules.IMetric "NvRules.IMetric") object.

Return type
    

[`IMetric`](#NvRules.IMetric "NvRules.IMetric")

Raises
    

  * [**TypeError**](https://docs.python.org/3/library/exceptions.html#TypeError "\(in Python v3.14\)") – If `key` is not of type [`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)").

  * [**KeyError**](https://docs.python.org/3/library/exceptions.html#KeyError "\(in Python v3.14\)") – If `key` is not the name of any [`IMetric`](#NvRules.IMetric "NvRules.IMetric") object.


__iter__()
    

Get an [iterator](https://docs.python.org/3/glossary.html#term-iterator "\(in Python v3.14\)") over the metric names of this [`IAction`](#NvRules.IAction "NvRules.IAction").

Returns
    

An [iterator](https://docs.python.org/3/glossary.html#term-iterator "\(in Python v3.14\)") over the metric names.

Return type
    

[iterator](https://docs.python.org/3/glossary.html#term-iterator "\(in Python v3.14\)") of [`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)")

__len__()
    

Get the number of [`IMetric`](#NvRules.IMetric "NvRules.IMetric") objects of this [`IAction`](#NvRules.IAction "NvRules.IAction").

Returns
    

The number of [`IMetric`](#NvRules.IMetric "NvRules.IMetric") objects.

Return type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

__str__()
    

Get a human-readable representation of this [`IAction`](#NvRules.IAction "NvRules.IAction").

Returns
    

The name of the kernel the [`IAction`](#NvRules.IAction "NvRules.IAction") object represents.

Return type
    

[`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)")

metric_by_name(_metric_name_)
    

Get a single [`IMetric`](#NvRules.IMetric "NvRules.IMetric") by name.

Parameters
    

**metric_name** ([`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)")) – The name of the [`IMetric`](#NvRules.IMetric "NvRules.IMetric") to retrieve.

Returns
    

The [`IMetric`](#NvRules.IMetric "NvRules.IMetric") object or [`None`](https://docs.python.org/3/library/constants.html#None "\(in Python v3.14\)") if no such metric exists.

Return type
    

[`IMetric`](#NvRules.IMetric "NvRules.IMetric") | [`None`](https://docs.python.org/3/library/constants.html#None "\(in Python v3.14\)")

metric_names()
    

Get the names of all metrics of this [`IAction`](#NvRules.IAction "NvRules.IAction").

Returns
    

The names of all metrics.

Return type
    

[`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.14\)") of [`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)")

name(_* args_)
    

Get the name of the result the [`IAction`](#NvRules.IAction "NvRules.IAction") object represents.

Parameters
    

**name_base** ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)"), optional) – The desired name base. Defaults to [`NameBase_FUNCTION`](#NvRules.IAction.NameBase_FUNCTION "NvRules.IAction.NameBase_FUNCTION").

Returns
    

The name of the result (potentially in a specific name base).

Return type
    

[`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)")

nvtx_state()
    

Get the NVTX state associated with this action.

Returns
    

The associated [`INvtxState`](#NvRules.INvtxState "NvRules.INvtxState") or [`None`](https://docs.python.org/3/library/constants.html#None "\(in Python v3.14\)") if no state is associated.

Return type
    

[`INvtxState`](#NvRules.INvtxState "NvRules.INvtxState") | [`None`](https://docs.python.org/3/library/constants.html#None "\(in Python v3.14\)")

ptx_by_pc(_address_)
    

Get the PTX associated with an address.

Parameters
    

**address** ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) – The address to get PTX for.

Returns
    

The PTX associated with the given address. If no PTX is available, the empty string will be returned.

Return type
    

[`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)")

sass_by_pc(_address_)
    

Get the SASS associated with an address.

Parameters
    

**address** ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) – The address to get SASS for.

Returns
    

The SASS associated with the given address. If no SASS is available, the empty string will be returned.

Return type
    

[`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)")

source_files()
    

Get the source files associated with this action along with their content.

If content is not available for a file (e.g. because it hadn’t been imported into the report), the file name will map to an empty string.

Returns
    

A dictionary mapping source files to their content.

Return type
    

[`dict`](https://docs.python.org/3/library/stdtypes.html#dict "\(in Python v3.14\)") of [`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)") to [`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)")

source_info(_address_)
    

Get the source info for a function address within this action.

Addresses are commonly obtained as correlation IDs of source-correlated metrics.

Parameters
    

**address** ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) – The address to get source info for.

Returns
    

The `ISourceInfo` associated to the given address. If no source info is available, [`None`](https://docs.python.org/3/library/constants.html#None "\(in Python v3.14\)") is returned.

Return type
    

`ISourceInfo` | [`None`](https://docs.python.org/3/library/constants.html#None "\(in Python v3.14\)")

workload_type()
    

Get the workload type of the action.

Returns
    

The workload type.

Return type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

_class _NvRules.IContext
    

Bases: `NvRules.IBaseContext`

The [`IContext`](#NvRules.IContext "NvRules.IContext") class is the top-level object representing an open report.

It can be created by calling the `load_report` function.

__getitem__(_key_)
    

Get one or more [`IRange`](#NvRules.IRange "NvRules.IRange") objects by index or by slice.

Parameters
    

**key** ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)") | [`slice`](https://docs.python.org/3/library/functions.html#slice "\(in Python v3.14\)")) – The index or slice to retrieve.

Returns: [`IRange`](#NvRules.IRange "NvRules.IRange") | [`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.14\)") of [`IRange`](#NvRules.IRange "NvRules.IRange"): An [`IRange`](#NvRules.IRange "NvRules.IRange") object or a [`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.14\)") of [`IRange`](#NvRules.IRange "NvRules.IRange") objects.

Raises
    

  * [**TypeError**](https://docs.python.org/3/library/exceptions.html#TypeError "\(in Python v3.14\)") – If `key` is not of type [`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)") or [`slice`](https://docs.python.org/3/library/functions.html#slice "\(in Python v3.14\)").

  * [**IndexError**](https://docs.python.org/3/library/exceptions.html#IndexError "\(in Python v3.14\)") – If `key` is out of range for the [`IContext`](#NvRules.IContext "NvRules.IContext").


__iter__()
    

Get an [iterator](https://docs.python.org/3/glossary.html#term-iterator "\(in Python v3.14\)") over the [`IRange`](#NvRules.IRange "NvRules.IRange") objects of this [`IContext`](#NvRules.IContext "NvRules.IContext").

Returns
    

An [iterator](https://docs.python.org/3/glossary.html#term-iterator "\(in Python v3.14\)") over the [`IRange`](#NvRules.IRange "NvRules.IRange") objects.

Return type
    

[iterator](https://docs.python.org/3/glossary.html#term-iterator "\(in Python v3.14\)") of [`IRange`](#NvRules.IRange "NvRules.IRange")

__len__()
    

Get the number of [`IRange`](#NvRules.IRange "NvRules.IRange") objects in this [`IContext`](#NvRules.IContext "NvRules.IContext").

Returns
    

The number of [`IRange`](#NvRules.IRange "NvRules.IRange") objects.

Return type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

controller()
    

Get the controller object.

Returns
    

The controller object.

Return type
    

[`IController`](#NvRules.IController "NvRules.IController")

frontend()
    

Get the frontend object.

Returns
    

The frontend object.

Return type
    

[`IFrontend`](#NvRules.IFrontend "NvRules.IFrontend")

num_ranges()
    

Get the number of [`IRange`](#NvRules.IRange "NvRules.IRange") objects in this [`IContext`](#NvRules.IContext "NvRules.IContext").

Returns
    

The number of [`IRange`](#NvRules.IRange "NvRules.IRange") objects.

Return type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

range_by_idx(_idx_)
    

Get an [`IRange`](#NvRules.IRange "NvRules.IRange") object by index.

Parameters
    

**key** ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) – The index to retrieve.

Returns
    

An [`IRange`](#NvRules.IRange "NvRules.IRange") object or [`None`](https://docs.python.org/3/library/constants.html#None "\(in Python v3.14\)") if the index is out of range.

Return type
    

[`IRange`](#NvRules.IRange "NvRules.IRange") | [`None`](https://docs.python.org/3/library/constants.html#None "\(in Python v3.14\)")

_class _NvRules.IController
    

Bases: [`object`](https://docs.python.org/3/library/functions.html#object "\(in Python v3.14\)")

Controller interface.

The controller can be used to interact with the tool runtime, e.g., to signal the tools to propose a follow-up rule.

get_message_vault()
    

Get an [`IMessageVault`](#NvRules.IMessageVault "NvRules.IMessageVault") object that can be used for message passing between rules.

Returns
    

[`IMessageVault`](#NvRules.IMessageVault "NvRules.IMessageVault")

propose_rule(_rule_)
    

Propose the specified rule in the current context.

Parameters
    

**rule** ([`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)")) – The proposed rule identifier.

_class _NvRules.IEvaluator
    

Bases: `NvRules.IBaseContext`

Evaluator interface.

The evaluator is used during rule setup to pass information about rule dependencies to the tool. For most cases, its Python wrapper functions [`require_metrics`](#NvRules.require_metrics "NvRules.require_metrics") and [`require_rules`](#NvRules.require_rules "NvRules.require_rules") should be used instead for convenience.

require_metric(_metric_)
    

Define that the specified metric must have been collected in order for the calling rule to be applied.

Parameters
    

**metric** ([`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)")) – The metric name to require.

require_rule(_rule_)
    

Define that the specified rule must be available and ready to be applied in order for the calling rule to be applied itself.

Parameters
    

**rule** ([`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)")) – The rule identifier to require.

_class _NvRules.IFrontend
    

Bases: [`object`](https://docs.python.org/3/library/functions.html#object "\(in Python v3.14\)")

The frontend is responsible for relaying messages and results to the caller via user interfaces, logs or output files.

Deprecated Attributes and Their Replacement:
    

MarkerKind_SASS - [`MarkerKind.SASS`](#NvRules.MarkerKind.SASS "NvRules.MarkerKind.SASS")

MarkerKind_SOURCE - [`MarkerKind.SOURCE`](#NvRules.MarkerKind.SOURCE "NvRules.MarkerKind.SOURCE")

MsgType_MSG_NONE - [`MsgType.NONE`](#NvRules.MsgType.NONE "NvRules.MsgType.NONE")

MsgType_MSG_OK - [`MsgType.OK`](#NvRules.MsgType.OK "NvRules.MsgType.OK")

MsgType_MSG_OPTIMIZATION - [`MsgType.OPTIMIZATION`](#NvRules.MsgType.OPTIMIZATION "NvRules.MsgType.OPTIMIZATION")

MsgType_MSG_WARNING - [`MsgType.WARNING`](#NvRules.MsgType.WARNING "NvRules.MsgType.WARNING")

MsgType_MSG_ERROR - [`MsgType.ERROR`](#NvRules.MsgType.ERROR "NvRules.MsgType.ERROR")

Severity_SEVERITY_DEFAULT
    

The default severity.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

Severity_SEVERITY_LOW
    

Severity if low.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

Severity_SEVERITY_HIGH
    

Severity if high.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

SpeedupType_LOCAL
    

The proportional increase in efficiency of the hardware usage when viewing the performance problem in isolation.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

SpeedupType_GLOBAL
    

The proportional reduction in runtime of the entire workload.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

focus_metric(_message_id_ , _metric_name_ , _metric_value_ , _severity_ , _info_)
    

Create a rule focus metric message.

Issues a focus metric message to the frontend, e.g. to indicate a key metric that triggered the rule output.

Parameters
    

  * **message_id** ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) – The prior message to associate with.

  * **metric_name** ([`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)")) – Name of the focus metric.

  * **metric_value** ([`float`](https://docs.python.org/3/library/functions.html#float "\(in Python v3.14\)")) – Value of the focus metric.

  * **severity** ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) – Indicates the impact orseverity on the result.

  * **info** ([`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)")) – Descriptive string for further information, e.g., the calculation leading to this metric being focused.


Returns
    

[`True`](https://docs.python.org/3/library/constants.html#True "\(in Python v3.14\)") if the focus metric could be set successfully,
    

[`False`](https://docs.python.org/3/library/constants.html#False "\(in Python v3.14\)") otherwise.

Return type
    

[`bool`](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)")

generate_table(_message_id_ , _header_ , _data_ , _config =None_)
    

Generate a table in the frontend.

This function attaches a table to the pre-existing rule message given by message_id. It can be called multiple times to attach multiple tables to the same message.

The table can be customized by passing a dict to the config parameter. The table can contain a title, description, a column to sort by, and a global styling. Additionally, the table can have per-column configurations, which can specify a header tooltip, relative column width, and styling for the header and data cells.

This is an example of a valid config dict:
    
    
    config = {
        "title": "My table title",
        "description": "Short description of the table",
        "sort_by": {
            "column": "MyColumn",  # specify column by name or index
            "order": "ascending",  # in ascending order (default)
        },
        "per_column_configs": {  # overwrites global styling for individual columns
            "MyColumn": {  # can use column name or index
                "tooltip": "Explanation of MyColumn",  # tooltip for the header
                "relative_width": 0.5,  # relative width of the column
                "style": {
                    "header": {"bold": True},
                    "data": {"italic": True},
                }
            },
        },
    }
    

Parameters
    

  * **message_id** ([_int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) – The message to which to attach the table to.

  * **header** ([_list_](https://docs.python.org/3/library/stdtypes.html#list "\(in Python v3.14\)") _[_[_str_](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)") _]_) – The column labels of the table.

  * **data** ([_list_](https://docs.python.org/3/library/stdtypes.html#list "\(in Python v3.14\)") _[_[_list_](https://docs.python.org/3/library/stdtypes.html#list "\(in Python v3.14\)") _[_[_int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)") _|_[_float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.14\)") _|_[_str_](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)") _|__Any_ _]__]_) – 

The table data in row-major format. Any refers to any type that implements __str__. Each column must only have elements of the same type, and all columns must have the same length.

class:str values may contain substrings with the following special link formats
    
    * @url:<hypertext>:<external link>@ - To add a external link for a hypertext.

    * @sass:<address>:<hypertext>@ - To add a link to the hypertext to open the SASS address line on the Source page.

    * @source:<file name>:<line number>:<hypertext>@ - To add a link to the hypertext to open the source file at the specified line number on the Source page.

    * @section:<section identifier>:<hypertext>@ - To add a link to the hypertext to jump to the respective section.

  * **config** ([_dict_](https://docs.python.org/3/library/stdtypes.html#dict "\(in Python v3.14\)") _[_[_str_](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)") _,__Any_ _]__|__None_) – Configuration options for the table. Defaults to None.


Raises
    

  * [**TypeError**](https://docs.python.org/3/library/exceptions.html#TypeError "\(in Python v3.14\)") – If types of elements within a column are mixed.

  * [**ValueError**](https://docs.python.org/3/library/exceptions.html#ValueError "\(in Python v3.14\)") – If columns have different lengths, or len(header) does not match the number of columns in data.


load_chart_from_file(_filename_)
    

Load a ProfilerSection google protocol buffer chart from a file.

Parameters
    

**filename** ([`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)")) – The file name.

message(_* args_)
    

Issues a message to the frontend.

Parameters
    

  * **type** ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)"), optional) – The message type.

  * **str** ([`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)")) – 

The message content.

The message may contain substrings with the following special link formats:
    
    * @url:<hypertext>:<external link>@ - To add a external link for a hypertext.

    * @sass:<address>:<text>@ - To add a SASS address to the cell.

    * @source:<file name>:<line number>:<hypertext>@ - To add a link to the hypertext to open the source file at the specified line number on the Source page.

    * @section:<section identifier>:<hypertext>@ - To add a link to the hypertext to jump to the respective section.

  * **name** ([`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)"), optional) – The name of the message.


Returns
    

A message ID that is unique in this rule invocation.

Return type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

receive_dict_from_parent(_parent_id_)
    

Receive a dictionary from a parent rule.

Receive a dictionary of type dict[str,float] sent using [`IFrontend.send_dict_to_children`](#NvRules.IFrontend.send_dict_to_children "NvRules.IFrontend.send_dict_to_children"). If the parent id does not represent a pre-specified parent rule of this rule, or in case the parent rule has not been executed, an empty dict will be returned.

Parameters
    

**parent_id** ([`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)")) – ID of the pre-specified parent rule.

Returns
    

The received dictionary.
    

An empty dict if the parent rule has not been executed.

Return type
    

[`dict`](https://docs.python.org/3/library/stdtypes.html#dict "\(in Python v3.14\)")

send_dict_to_children(_dict_)
    

Send a dictionary to all child rules.

Sends a Python dictionary of type dict[str,float] to all rules that specify this rule as a parent rule. Child rules can retrieve the message using [`IFrontend.receive_dict_from_parent`](#NvRules.IFrontend.receive_dict_from_parent "NvRules.IFrontend.receive_dict_from_parent"). In case this function is called repeatedly, the dict is updated accordingly, thereby adding new key-value pairs, and overwriting values of pre-existing keys.

Parameters
    

**dict** ([`dict`](https://docs.python.org/3/library/stdtypes.html#dict "\(in Python v3.14\)")) – The dictionary to send.

source_marker(_* args_)
    

Create a rule source marker.

Parameters
    

  * **str** ([`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)")) – The source marker message.

  * **address_or_line** ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) – The source location in the appropriate kind of source.

  * **kind** ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) – The the kind of source.

  * **file_name** ([`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)"), optional) – The file name.

  * **type** ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)"), optional) – The urgency associated with this marker. By default [`MsgType.NONE`](#NvRules.MsgType.NONE "NvRules.MsgType.NONE").


speedup(_message_id_ , _type_ , _estimated_speedup_)
    

Rule estimated speedup message.

Issues an estimated speedup associated with a message to the frontend.

Parameters
    

  * **message_id** ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) – ID of the existing message.

  * **type** ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) – The SpeedupType. If GLOBAL, it indicates what proportional decrease in workload runtime could potentially be achieved, when following the guidelines of the rule. If LOCAL, it indicates what increase in the efficiency of the hardware usage within the context of the performance problem could be achieved.


Returns
    

[`True`](https://docs.python.org/3/library/constants.html#True "\(in Python v3.14\)") if the speedup could be set successfully,
    

[`False`](https://docs.python.org/3/library/constants.html#False "\(in Python v3.14\)") otherwise.

Return type
    

[`bool`](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)")

_class _NvRules.IMessageVault
    

Bases: [`object`](https://docs.python.org/3/library/functions.html#object "\(in Python v3.14\)")

Passes messages between rules.

Get(_ruleId_)
    

Retrieve the message associated with a rule from the vault.

In case the rule is unknown, an empty message is returned.

Parameters
    

**ruleId** ([`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)")) – The identifier of the rule.

Returns
    

The message associated with the rule.

Return type
    

[`dict`](https://docs.python.org/3/library/stdtypes.html#dict "\(in Python v3.14\)")

Put(_ruleId_ , _message_)
    

Commit a message associated with a rule to the vault.

In case multiple messages associated with the same rule are committed, the messages are merged, in such a way that new key-value pairs are added, and values of pre-existing keys are updated.

Parameters
    

  * **ruleId** ([`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)")) – The identifier of the rule.

  * **message** ([`dict`](https://docs.python.org/3/library/stdtypes.html#dict "\(in Python v3.14\)")) – The message to commit.


_class _NvRules.IMetric
    

Bases: [`object`](https://docs.python.org/3/library/functions.html#object "\(in Python v3.14\)")

Represents a single, named metric. An [`IMetric`](#NvRules.IMetric "NvRules.IMetric") can carry one value or multiple ones if it is an instanced metric.

MetricType_OTHER
    

Metric type for metrics that do not fit in any other category.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

MetricType_COUNTER
    

Metric type for counter metrics.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

MetricType_RATIO
    

Metric type for ratio metrics.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

MetricType_THROUGHPUT
    

Metric type for throughput metrics.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

MetricSubtype_NONE
    

Metric subtype for metrics that do not have a subtype.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

MetricSubtype_PEAK_SUSTAINED
    

Metric subtype for peak sustained metrics.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

MetricSubtype_PEAK_SUSTAINED_ACTIVE
    

Metric subtype for peak sustained active metrics.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

MetricSubtype_PEAK_SUSTAINED_ACTIVE_PER_SECOND
    

Metric subtype for peak sustained active per-second metrics.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

MetricSubtype_PEAK_SUSTAINED_ELAPSED
    

Metric subtype for peak sustained elapsed metrics.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

MetricSubtype_PEAK_SUSTAINED_ELAPSED_PER_SECOND
    

Metric subtype for peak sustained elapsed per-second metrics.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

MetricSubtype_PER_CYCLE_ACTIVE
    

Metric subtype for per-cycle active metrics.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

MetricSubtype_PER_CYCLE_ELAPSED
    

Metric subtype for per-cycle elapsed metrics.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

MetricSubtype_PER_SECOND
    

Metric subtype for per-second metrics.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

MetricSubtype_PCT_OF_PEAK_SUSTAINED_ACTIVE
    

Metric subtype for percentage of peak sustained active metrics.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

MetricSubtype_PCT_OF_PEAK_SUSTAINED_ELAPSED
    

Metric subtype for percentage of peak sustained elapsed metrics.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

MetricSubtype_MAX_RATE
    

Metric subtype for max rate metrics.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

MetricSubtype_PCT
    

Metric subtype for percentage metrics.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

MetricSubtype_RATIO
    

Metric subtype for ratio metrics.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

RollupOperation_NONE
    

No rollup operation.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

RollupOperation_AVG
    

Average rollup operation.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

RollupOperation_MAX
    

Maximum rollup operation.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

RollupOperation_MIN
    

Minimum rollup operation.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

RollupOperation_SUM
    

Sum rollup operation.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

ValueKind_UNKNOWN
    

Unknown value kind.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

ValueKind_ANY
    

Undefined value kind.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

ValueKind_STRING
    

String value kind.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

ValueKind_FLOAT
    

Float value kind.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

ValueKind_DOUBLE
    

Double value kind.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

ValueKind_UINT32
    

Unsigned 32-bit integer value kind.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

ValueKind_UINT64
    

Unsigned 64-bit integer value kind.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

__str__()
    

Get a human-readable representation of this [`IMetric`](#NvRules.IMetric "NvRules.IMetric").

Returns
    

The name of the [`IMetric`](#NvRules.IMetric "NvRules.IMetric").

Return type
    

[`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)")

as_double(_* args_)
    

Get the metric value or metric instance value as a [`float`](https://docs.python.org/3/library/functions.html#float "\(in Python v3.14\)").

Parameters
    

**instance** ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)"), optional) – If provided, the index of the instance value to retrieve instead a metric value.

Returns
    

The metric value or metric instance value requested. If the value cannot be casted to a [`float`](https://docs.python.org/3/library/functions.html#float "\(in Python v3.14\)"), this function will return `0.`.

Return type
    

[`float`](https://docs.python.org/3/library/functions.html#float "\(in Python v3.14\)")

as_string(_* args_)
    

Get the metric value or metric instance value as a [`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)").

Parameters
    

**instance** ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)"), optional) – If provided, the index of the instance value to retrieve instead a metric value.

Returns
    

The metric value or metric instance value requested. If the value cannot be casted to a [`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)"), this function will return [`None`](https://docs.python.org/3/library/constants.html#None "\(in Python v3.14\)").

Return type
    

[`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)") | [`None`](https://docs.python.org/3/library/constants.html#None "\(in Python v3.14\)")

as_uint64(_* args_)
    

Get the metric value or metric instance value as an [`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)").

Parameters
    

**instance** ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)"), optional) – If provided, the index of the instance value to retrieve instead a metric value.

Returns
    

The metric value or metric instance value requested. If the value cannot be casted to a [`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)"), this function will return `0`.

Return type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

correlation_ids()
    

Get a metric object for this metric’s instance value’s correlation IDs.

Returns a new [`IMetric`](#NvRules.IMetric "NvRules.IMetric") representing the correlation IDs for the metric’s instance values. Use [`IMetric.has_correlation_ids`](#NvRules.IMetric.has_correlation_ids "NvRules.IMetric.has_correlation_ids") to check if this metric has correlation IDs for its instance values. Correlation IDs are used to associate instance values with the instance their value represents. In the returned new metric object, the correlation IDs are that object’s instance values.

If the metric does not have any correlation IDs, this function will return [`None`](https://docs.python.org/3/library/constants.html#None "\(in Python v3.14\)").

Returns
    

The new [`IMetric`](#NvRules.IMetric "NvRules.IMetric") object representing the correlation IDs for this metric’s instance values or [`None`](https://docs.python.org/3/library/constants.html#None "\(in Python v3.14\)") if the metric has no correlation IDs.

Return type
    

[`IMetric`](#NvRules.IMetric "NvRules.IMetric") | [`None`](https://docs.python.org/3/library/constants.html#None "\(in Python v3.14\)")

description()
    

Get the metric description.

Returns
    

The description of the metric.

Return type
    

[`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)")

has_correlation_ids()
    

Check if the metric has correlation IDs.

Returns
    

[`True`](https://docs.python.org/3/library/constants.html#True "\(in Python v3.14\)") if the metric has correlation IDs matching its instance values, [`False`](https://docs.python.org/3/library/constants.html#False "\(in Python v3.14\)") otherwise.

Return type
    

[`bool`](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)")

has_value(_* args_)
    

Check if the metric or metric instance has a value.

Parameters
    

**instance** ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)"), optional) – If provided, the index of the instance metric to check.

Returns
    

[`True`](https://docs.python.org/3/library/constants.html#True "\(in Python v3.14\)") if the metric or metric instance has a value, `False`` otherwise.

Return type
    

[`bool`](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)")

kind(_* args_)
    

Get the metric or metric instance value kind.

Parameters
    

**instance** ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)"), optional) – If provided, the index of the instance metric to get the value kind for.

Returns
    

The metric or metric instance value kind.

Return type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

metric_subtype()
    

Get the metric subtype.

Returns
    

The metric subtype.

Return type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

metric_type()
    

Get the metric type.

Returns
    

The metric type.

Return type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

name()
    

Get the metric name.

Returns
    

The metric name.

Return type
    

[`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)")

num_instances()
    

Get the number of instance values for this metric.

Not all metrics have instance values. If a metric has instance values, it may also have [`IMetric.correlation_ids`](#NvRules.IMetric.correlation_ids "NvRules.IMetric.correlation_ids") matching these instance values.

Returns
    

The number of instances for this metric.

Return type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

rollup_operation()
    

Get the type of rollup operation for this metric.

Returns
    

The rollup operation type.

Return type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

unit()
    

Get the metric unit.

Returns
    

The metric unit.

Return type
    

[`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)")

value(_idx =None_)
    

Get the value of this [`IMetric`](#NvRules.IMetric "NvRules.IMetric").

This is a convenience function that will wrap the logic of invoking the correct `IMetric.as_*` method based on the value kind of this [`IMetric`](#NvRules.IMetric "NvRules.IMetric").

Parameters
    

**idx** ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)"), optional) – The index of the value to get.

Returns
    

The value of this [`IMetric`](#NvRules.IMetric "NvRules.IMetric") as [`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)"), [`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)") or [`float`](https://docs.python.org/3/library/functions.html#float "\(in Python v3.14\)"). If no value is available, this will return [`None`](https://docs.python.org/3/library/constants.html#None "\(in Python v3.14\)").

Return type
    

[`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)") | [`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)") | [`float`](https://docs.python.org/3/library/functions.html#float "\(in Python v3.14\)") | [`None`](https://docs.python.org/3/library/constants.html#None "\(in Python v3.14\)")

_class _NvRules.IMutableMetric
    

Bases: [`NvRules.IMetric`](#NvRules.IMetric "NvRules.IMetric")

Represents a single, named mutable metric. An [`IMutableMetric`](#NvRules.IMutableMetric "NvRules.IMutableMetric") can carry one value or multiple ones if it is an instanced metric. In comparison to [`IMetric`](#NvRules.IMetric "NvRules.IMetric"), [`IMutableMetric`](#NvRules.IMutableMetric "NvRules.IMutableMetric") can be modified by assigning it a new value and/or instance values. The metric kind is determined by the assigned value(s).

mutable_correlation_ids()
    

Get a mutable metric object for this metric’s instance value’s correlation IDs.

Returns a new [`IMutableMetric`](#NvRules.IMutableMetric "NvRules.IMutableMetric") representing the correlation IDs for the metric’s instance values. Correlation IDs are used to associate instance values with the instance their value represents. In the returned new metric object, the correlation IDs are that object’s instance values.

Returns
    

The new [`IMutableMetric`](#NvRules.IMutableMetric "NvRules.IMutableMetric") object representing the correlation IDs for this metric’s instance values.

Return type
    

[`IMutableMetric`](#NvRules.IMutableMetric "NvRules.IMutableMetric")

set_double(_* args_)
    

Assign the metric an instance [`float`](https://docs.python.org/3/library/functions.html#float "\(in Python v3.14\)") value.

Parameters
    

  * **instance** ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)"), optional) – The instance to assign the value to.

  * **value_kind** ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) – Acceptable kinds are FLOAT, DOUBLE or ANY (for which the implementation chooses the kind internally)

  * **value** ([`float`](https://docs.python.org/3/library/functions.html#float "\(in Python v3.14\)")) – The value to assign.


Returns
    

[`True`](https://docs.python.org/3/library/constants.html#True "\(in Python v3.14\)") if the instance value was set, [`False`](https://docs.python.org/3/library/constants.html#False "\(in Python v3.14\)") otherwise.

Return type
    

[`bool`](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)")

set_string(_* args_)
    

Assign the metric an instance [`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)") value.

Parameters
    

  * **instance** ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)"), optional) – The instance to assign the value to.

  * **value_kind** ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) – Acceptable kinds are STRING or ANY (for which the implementation chooses the kind internally)

  * **value** ([`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)")) – The value to assign.


Returns
    

[`True`](https://docs.python.org/3/library/constants.html#True "\(in Python v3.14\)") if the instance value was set, [`False`](https://docs.python.org/3/library/constants.html#False "\(in Python v3.14\)") otherwise.

Return type
    

[`bool`](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)")

set_uint64(_* args_)
    

Assign the metric a [`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)") value.

Parameters
    

  * **instance** ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)"), optional) – The instance to assign the value to.

  * **value_kind** ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) – Acceptable kinds are UINT32, UINT64 or ANY (for which the implementation chooses the kind internally)

  * **value** ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) – The value to assign.


Returns
    

[`True`](https://docs.python.org/3/library/constants.html#True "\(in Python v3.14\)") if the value was set, [`False`](https://docs.python.org/3/library/constants.html#False "\(in Python v3.14\)") otherwise.

Return type
    

[`bool`](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)")

_class _NvRules.INvtxDomainInfo
    

Bases: [`object`](https://docs.python.org/3/library/functions.html#object "\(in Python v3.14\)")

Represents a single NVTX domain of the NVTX state, including all ranges associated with this domain.

__str__()
    

Get a human-readable representation of this [`INvtxDomainInfo`](#NvRules.INvtxDomainInfo "NvRules.INvtxDomainInfo").

Returns
    

The name of the [`INvtxDomainInfo`](#NvRules.INvtxDomainInfo "NvRules.INvtxDomainInfo").

Return type
    

[`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)")

name()
    

Get a human-readable representation of this [`INvtxDomainInfo`](#NvRules.INvtxDomainInfo "NvRules.INvtxDomainInfo").

Returns
    

The name of the [`INvtxDomainInfo`](#NvRules.INvtxDomainInfo "NvRules.INvtxDomainInfo").

Return type
    

[`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)")

push_pop_range(_idx_)
    

Get a push/pop range object by index.

The index is identical to the range’s order on the call stack.

Returns
    

The requested [`INvtxRange`](#NvRules.INvtxRange "NvRules.INvtxRange") or [`None`](https://docs.python.org/3/library/constants.html#None "\(in Python v3.14\)") if the index is out of range.

Return type
    

[`INvtxRange`](#NvRules.INvtxRange "NvRules.INvtxRange") | [`None`](https://docs.python.org/3/library/constants.html#None "\(in Python v3.14\)")

push_pop_ranges()
    

Get a sorted list of push/pop range names.

Get the sorted list of stacked push/pop range names in this domain, associated with the current [`INvtxState`](#NvRules.INvtxState "NvRules.INvtxState").

Returns
    

The sorted names of all push/pop ranges.

Return type
    

[`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.14\)") of [`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)")

start_end_range(_idx_)
    

Get a start/end range object by index.

Returns
    

The requested [`INvtxRange`](#NvRules.INvtxRange "NvRules.INvtxRange") or [`None`](https://docs.python.org/3/library/constants.html#None "\(in Python v3.14\)") if the index is out of range.

Return type
    

[`INvtxRange`](#NvRules.INvtxRange "NvRules.INvtxRange") | [`None`](https://docs.python.org/3/library/constants.html#None "\(in Python v3.14\)")

start_end_ranges()
    

Get a sorted list of start/end range names.

Get the sorted list of start/end range names in this domain, associated with the current [`INvtxState`](#NvRules.INvtxState "NvRules.INvtxState").

Returns
    

The sorted names of all start/end ranges.

Return type
    

[`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.14\)") of [`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)")

_class _NvRules.INvtxRange
    

Bases: [`object`](https://docs.python.org/3/library/functions.html#object "\(in Python v3.14\)")

Represents a single NVTX Push/Pop or Start/End range.

PayloadType_PAYLOAD_UNKNOWN
    

Payload type for ranges of unknown type.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

PayloadType_PAYLOAD_INT32
    

Payload type for ranges of INT32 type.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

PayloadType_PAYLOAD_INT64
    

Payload type for ranges of INT64 type.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

PayloadType_PAYLOAD_UINT32
    

Payload type for ranges of UINT32 type.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

PayloadType_PAYLOAD_UINT64
    

Payload type for ranges of UINT64 type.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

PayloadType_PAYLOAD_FLOAT
    

Payload type for ranges of float type.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

PayloadType_PAYLOAD_DOUBLE
    

Payload type for ranges of double type .

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

PayloadType_PAYLOAD_JSON
    

Payload type for ranges of JSON type.

Type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

category()
    

Get the category attribute value.

Returns
    

The category attribute value. If [`INvtxRange.has_attributes`](#NvRules.INvtxRange.has_attributes "NvRules.INvtxRange.has_attributes") returns [`False`](https://docs.python.org/3/library/constants.html#False "\(in Python v3.14\)"), this will return `0`.

Return type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

color()
    

Get the color attribute value.

Returns
    

The color attribute value. If [`INvtxRange.has_attributes`](#NvRules.INvtxRange.has_attributes "NvRules.INvtxRange.has_attributes") returns [`False`](https://docs.python.org/3/library/constants.html#False "\(in Python v3.14\)"), this will return `0`.

Return type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

has_attributes()
    

Check if range has event attributes.

Returns
    

[`True`](https://docs.python.org/3/library/constants.html#True "\(in Python v3.14\)") if the range has event attributes, [`False`](https://docs.python.org/3/library/constants.html#False "\(in Python v3.14\)") otherwise.

Return type
    

[`bool`](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)")

message()
    

Get the message attribute value.

Returns
    

The message attribute value. If [`INvtxRange.has_attributes`](#NvRules.INvtxRange.has_attributes "NvRules.INvtxRange.has_attributes") returns [`False`](https://docs.python.org/3/library/constants.html#False "\(in Python v3.14\)"), this will return the empty string.

Return type
    

[`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)")

name()
    

Get the range’s name.

Returns
    

The range’s name.

Return type
    

[`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)")

payload_as_double()
    

Get the payload attribute value as a [`float`](https://docs.python.org/3/library/functions.html#float "\(in Python v3.14\)").

Returns
    

The payload attribute’s value as a [`float`](https://docs.python.org/3/library/functions.html#float "\(in Python v3.14\)"). If the value cannot be casted to a [`float`](https://docs.python.org/3/library/functions.html#float "\(in Python v3.14\)"), this function will return `0.`.

Return type
    

[`float`](https://docs.python.org/3/library/functions.html#float "\(in Python v3.14\)")

payload_as_string()
    

Get the payload attribute value as a [`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)").

Returns
    

The payload attribute’s value as a [`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)"). If the value cannot be casted to a [`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)"), this function will return the empty string.

Return type
    

[`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)")

payload_as_uint64()
    

Get the payload attribute value as an [`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)").

Returns
    

The payload attribute’s value as a [`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)"). If the value cannot be casted to a [`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)"), this function will return `0`.

Return type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

payload_type()
    

Get the payload type as an [`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)").

Returns
    

The payload type.

Return type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

_class _NvRules.INvtxState
    

Bases: [`object`](https://docs.python.org/3/library/functions.html#object "\(in Python v3.14\)")

Represents the NVTX (Nvidia Tools Extensions) state associated with a single [`IAction`](#NvRules.IAction "NvRules.IAction").

__getitem__(_key_)
    

Get an [`INvtxDomainInfo`](#NvRules.INvtxDomainInfo "NvRules.INvtxDomainInfo") object by ID.

Parameters
    

**key** ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) – The ID of the [`INvtxDomainInfo`](#NvRules.INvtxDomainInfo "NvRules.INvtxDomainInfo") object.

Returns
    

An [`INvtxDomainInfo`](#NvRules.INvtxDomainInfo "NvRules.INvtxDomainInfo") object.

Return type
    

[`INvtxDomainInfo`](#NvRules.INvtxDomainInfo "NvRules.INvtxDomainInfo")

Raises
    

  * [**TypeError**](https://docs.python.org/3/library/exceptions.html#TypeError "\(in Python v3.14\)") – If `key` is not of type [`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)").

  * [**KeyError**](https://docs.python.org/3/library/exceptions.html#KeyError "\(in Python v3.14\)") – If `key` is not a valid ID.


__iter__()
    

Get an [iterator](https://docs.python.org/3/glossary.html#term-iterator "\(in Python v3.14\)") over the [`INvtxDomainInfo`](#NvRules.INvtxDomainInfo "NvRules.INvtxDomainInfo") objects of this [`INvtxState`](#NvRules.INvtxState "NvRules.INvtxState").

Returns
    

An [iterator](https://docs.python.org/3/glossary.html#term-iterator "\(in Python v3.14\)") over the [`INvtxDomainInfo`](#NvRules.INvtxDomainInfo "NvRules.INvtxDomainInfo") objects.

Return type
    

[iterator](https://docs.python.org/3/glossary.html#term-iterator "\(in Python v3.14\)")

__len__()
    

Get the number of [`INvtxDomainInfo`](#NvRules.INvtxDomainInfo "NvRules.INvtxDomainInfo") objects of this [`INvtxState`](#NvRules.INvtxState "NvRules.INvtxState").

Returns
    

The number of [`INvtxDomainInfo`](#NvRules.INvtxDomainInfo "NvRules.INvtxDomainInfo") objects.

Return type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

domain_by_id(_id_)
    

Get a [`INvtxDomainInfo`](#NvRules.INvtxDomainInfo "NvRules.INvtxDomainInfo") object by ID.

Use [`INvtxState.domains`](#NvRules.INvtxState.domains "NvRules.INvtxState.domains") to retrieve the list of valid domain IDs.

Parameters
    

**id** ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) – The ID of the request domain.

Returns
    

The requested [`INvtxDomainInfo`](#NvRules.INvtxDomainInfo "NvRules.INvtxDomainInfo") object.

Return type
    

[`INvtxDomainInfo`](#NvRules.INvtxDomainInfo "NvRules.INvtxDomainInfo") | [`None`](https://docs.python.org/3/library/constants.html#None "\(in Python v3.14\)")

domains()
    

Get the list of domain IDs in this state.

Returns
    

The tuple of valid domain IDs.

Return type
    

[`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.14\)") of [`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

_class _NvRules.IRange
    

Bases: [`object`](https://docs.python.org/3/library/functions.html#object "\(in Python v3.14\)")

Represents a serial, ordered stream of execution, such as a CUDA stream. It holds one or more actions that were logically executing in this range.

__getitem__(_key_)
    

Get one or more [`IAction`](#NvRules.IAction "NvRules.IAction") objects by index or by [`slice`](https://docs.python.org/3/library/functions.html#slice "\(in Python v3.14\)").

Parameters
    

**key** ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)") | [`slice`](https://docs.python.org/3/library/functions.html#slice "\(in Python v3.14\)")) – The index or [`slice`](https://docs.python.org/3/library/functions.html#slice "\(in Python v3.14\)") to retrieve.

Returns
    

An [`IAction`](#NvRules.IAction "NvRules.IAction") object or a [`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.14\)") of [`IAction`](#NvRules.IAction "NvRules.IAction") objects.

Return type
    

[`IAction`](#NvRules.IAction "NvRules.IAction") | [`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.14\)") of [`IAction`](#NvRules.IAction "NvRules.IAction")

Raises
    

  * [**TypeError**](https://docs.python.org/3/library/exceptions.html#TypeError "\(in Python v3.14\)") – If `key` is not of type [`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)") or [`slice`](https://docs.python.org/3/library/functions.html#slice "\(in Python v3.14\)").

  * [**IndexError**](https://docs.python.org/3/library/exceptions.html#IndexError "\(in Python v3.14\)") – If `key` is out of range for the [`IRange`](#NvRules.IRange "NvRules.IRange").


__iter__()
    

Get an [iterator](https://docs.python.org/3/glossary.html#term-iterator "\(in Python v3.14\)") over the [`IAction`](#NvRules.IAction "NvRules.IAction") objects of this class:IRange.

Returns
    

An [iterator](https://docs.python.org/3/glossary.html#term-iterator "\(in Python v3.14\)") over the [`IAction`](#NvRules.IAction "NvRules.IAction") objects.

Return type
    

[iterator](https://docs.python.org/3/glossary.html#term-iterator "\(in Python v3.14\)") of [`IAction`](#NvRules.IAction "NvRules.IAction")

__len__()
    

Get the number of [`IAction`](#NvRules.IAction "NvRules.IAction") objects in this [`IRange`](#NvRules.IRange "NvRules.IRange").

Returns
    

The number of class:IAction objects.

Return type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

action_by_idx(_idx_)
    

Get an [`IAction`](#NvRules.IAction "NvRules.IAction") objects by index.

Parameters
    

**key** ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) – The index to retrieve.

Returns
    

An [`IAction`](#NvRules.IAction "NvRules.IAction") object or [`None`](https://docs.python.org/3/library/constants.html#None "\(in Python v3.14\)") if the index is out of range.

Return type
    

[`IAction`](#NvRules.IAction "NvRules.IAction") | [`None`](https://docs.python.org/3/library/constants.html#None "\(in Python v3.14\)")

actions_by_nvtx(_includes_ , _excludes_)
    

Get a set of indices to IAction objects by their NVTX state. The state is defined using a series of _includes_ and _excludes_.

Parameters
    

  * **includes** ([iterable](https://docs.python.org/3/glossary.html#term-iterable "\(in Python v3.14\)") of [`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)")) – The NVTX states the result should be part of.

  * **excludes** ([iterable](https://docs.python.org/3/glossary.html#term-iterable "\(in Python v3.14\)") of [`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)")) – The NVTX states the result should not be part of.


Returns
    

A [`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.14\)") of indices to [`IAction`](#NvRules.IAction "NvRules.IAction") matching the desired NVTX state.

Return type
    

[`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.14\)") of [`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

num_actions()
    

Get the number of [`IAction`](#NvRules.IAction "NvRules.IAction") objects in this [`IRange`](#NvRules.IRange "NvRules.IRange").

Returns
    

The number of class:IAction objects.

Return type
    

[`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")

_class _NvRules.MarkerKind
    

Bases: [`enum.IntEnum`](https://docs.python.org/3/library/enum.html#enum.IntEnum "\(in Python v3.14\)")

Enum representing the kind of a source marker.

SASS
    

The marker will be associated with a SASS instruction.

SOURCE
    

The marker will be associated with a Source line.

NONE
    

No specific kind of marker.

_class _NvRules.MsgType
    

Bases: [`enum.IntEnum`](https://docs.python.org/3/library/enum.html#enum.IntEnum "\(in Python v3.14\)")

Enum representing the type of the message.

NONE
    

No specific type for this message.

OK
    

The message is informative.

OPTIMIZATION
    

The message represents a suggestion for performance optimization.

WARNING
    

The message represents a warning or fixable issue.

ERROR
    

The message represents an error, potentially in executing the rule.

NvRules.get_context(_h_)
    

Return the [`IContext`](#NvRules.IContext "NvRules.IContext") object from the context handle.

Parameters
    

**h** ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) – The context handle.

Returns
    

The evaluator object.

Return type
    

[`IContext`](#NvRules.IContext "NvRules.IContext")

NvRules.get_evaluator(_h_)
    

Return the [`IEvaluator`](#NvRules.IEvaluator "NvRules.IEvaluator") object from the context handle.

Parameters
    

**h** ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) – The context handle.

Returns
    

The evaluator object.

Return type
    

[`IEvaluator`](#NvRules.IEvaluator "NvRules.IEvaluator")

NvRules.get_version(_h_)
    

Get version number of this interface.

Parameters
    

**h** ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) – The context handle.

Returns
    

Returns the version number of this interface as a string of the form <year>.<major>.<minor>. It matches the Nsight Compute version this interface originates from.

Return type
    

[`str`](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)")

NvRules.require_metrics(_handle_ , _metrics_)
    

Convenience wrapper for [`NvRules.IEvaluator.require_metric`](#NvRules.IEvaluator.require_metric "NvRules.IEvaluator.require_metric").

Parameters
    

  * **handle** ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) – The context handle, obtained from [`get_context`](#NvRules.get_context "NvRules.get_context").

  * **metrics** ([`list`](https://docs.python.org/3/library/stdtypes.html#list "\(in Python v3.14\)")) – List of metric names.


NvRules.require_rules(_handle_ , _rules_)
    

Convenience wrapper for [`NvRules.IEvaluator.require_rule`](#NvRules.IEvaluator.require_rule "NvRules.IEvaluator.require_rule").

Parameters
    

  * **handle** ([`int`](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")) – The context handle, obtained from [`get_context`](#NvRules.get_context "NvRules.get_context").

  * **rules** ([`list`](https://docs.python.org/3/library/stdtypes.html#list "\(in Python v3.14\)")) – List of rule identifiers.


Notices

Notices

ALL NVIDIA DESIGN SPECIFICATIONS, REFERENCE BOARDS, FILES, DRAWINGS, DIAGNOSTICS, LISTS, AND OTHER DOCUMENTS (TOGETHER AND SEPARATELY, “MATERIALS”) ARE BEING PROVIDED “AS IS.” NVIDIA MAKES NO WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO THE MATERIALS, AND EXPRESSLY DISCLAIMS ALL IMPLIED WARRANTIES OF NONINFRINGEMENT, MERCHANTABILITY, AND FITNESS FOR A PARTICULAR PURPOSE.

Information furnished is believed to be accurate and reliable. However, NVIDIA Corporation assumes no responsibility for the consequences of use of such information or for any infringement of patents or other rights of third parties that may result from its use. No license is granted by implication of otherwise under any patent rights of NVIDIA Corporation. Specifications mentioned in this publication are subject to change without notice. This publication supersedes and replaces all other information previously supplied. NVIDIA Corporation products are not authorized as critical components in life support devices or systems without express written approval of NVIDIA Corporation.

Trademarks

NVIDIA and the NVIDIA logo are trademarks or registered trademarks of NVIDIA Corporation in the U.S. and other countries. Other company and product names may be trademarks of the respective companies with which they are associated.

* * *
