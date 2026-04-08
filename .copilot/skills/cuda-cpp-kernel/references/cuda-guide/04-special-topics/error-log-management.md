---
url: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/error-log-management.html
---

# 4.8. Error Log Management

The _Error Log Management_ mechanism allows for CUDA API errors to be reported to developers in a plain-English format that describes the cause of the issue.

## 4.8.1. Background

Traditionally, the only indication of a failed CUDA API call is the return of a non-zero code. As of CUDA Toolkit 12.9, the CUDA Runtime defines over 100 different return codes for error conditions, but many of them are generic and give the developer no assistance with debugging the cause.

## 4.8.2. Activation

Set the _CUDA_LOG_FILE_ environment variable. Acceptable values are _stdout_ , _stderr_ , or a valid path on the system to write a file. The log buffer can be dumped via API even if _CUDA_LOG_FILE_ was not set before program execution. NOTE: An error-free execution may not print any logs.

## 4.8.3. Output

Logs are output in the following format:
    
    
    [Time][TID][Source][Severity][API Entry Point] Message
    

The following line is an actual error message that is generated if the developer tries to dump the Error Log Management logs to an unallocated buffer:
    
    
    [22:21:32.099][25642][CUDA][E][cuLogsDumpToMemory] buffer cannot be NULL
    

Where before, all the developer would have gotten is _CUDA_ERROR_INVALID_VALUE_ in the return code and possibly “invalid argument” if _cuGetErrorString_ is called.

## 4.8.4. API Description

The CUDA Driver provides APIs in two categories for interacting with the Error Log Management feature.

This feature allows developers to register callback functions to be used whenever an error log is generated, where the callback signature is:
    
    
    void callbackFunc(void *data, CUlogLevel logLevel, char *message, size_t length)
    

Callbacks are registered with this API:
    
    
    CUresult cuLogsRegisterCallback(CUlogsCallback callbackFunc, void *userData, CUlogsCallbackHandle *callback_out)
    

Where _userData_ is passed to the callback function without modifications. _callback_out_ should be stored by the caller for use in _cuLogsUnregisterCallback_.
    
    
    CUresult cuLogsUnregisterCallback(CUlogsCallbackHandle callback)
    

The other set of API functions are for managing the output of logs. An important concept is the log iterator, which points to the current end of the buffer:
    
    
    CUresult cuLogsCurrent(CUlogIterator *iterator_out, unsigned int flags)
    

The iterator position can be kept by the calling software in situations where a dump of the entire log buffer is not desired. Currently, the flags parameter must be 0, with additional options reserved for future CUDA releases.

At any time, the error log buffer can be dumped to either a file or memory with these functions:
    
    
    CUresult cuLogsDumpToFile(CUlogIterator *iterator, const char *pathToFile, unsigned int flags)
    CUresult cuLogsDumpToMemory(CUlogIterator *iterator, char *buffer, size_t *size, unsigned int flags)
    

If _iterator_ is NULL, the entire buffer will be dumped, up to the maximum of 100 entries. If _iterator_ is not NULL, logs will be dumped starting from that entry and the value of _iterator_ will be updated to the current end of the logs, as if _cuLogsCurrent_ had been called. If there have been more than 100 log entries into the buffer, a note will be added at the start of the dump noting this.

The flags parameter must be 0, with additional options reserved for future CUDA releases.

The _cuLogsDumpToMemory_ function has additional considerations:

  1. The buffer itself will be null-terminated, but each individual log entry will only be separated by a newline (n) character.

  2. The maximum size of the buffer is 25600 bytes.

  3. If the value provided in _size_ is not sufficient to store all desired logs, a note will be added as the first entry and the oldest entries that do not fit will not be dumped.

  4. After returning, _size_ will contain the actual number of bytes written to the provided buffer.


## 4.8.5. Limitations and Known Issues

  1. The log buffer is limited to 100 entries. After this limit is reached, the oldest entries will be replaced and log dumps will contain a line noting the rollover.

  2. Not all CUDA APIs are covered yet. This is an ongoing project to provide better usage error reporting for all APIs.

  3. The Error Log Management log location (if given) will not be tested for validity until/unless a log is generated.

  4. The Error Log Management APIs are currently only available via the CUDA Driver. Equivalent APIs will be added to the CUDA Runtime in a future release.

  5. The log messages are not localized to any language and all provided logs are in US English.
