# Adapted from: https://github.com/vllm-project/vllm/tree/main/vllm/platforms
import torch
from abc import ABC


class BasePlatform(ABC):
    device_type: str
    device_control_env_var: str
    dispatch_key: str
    dist_backend: str
    full_dist_backend: str

    @staticmethod
    def empty_cache(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def ipc_collect(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_device_name():
        raise NotImplementedError

    @staticmethod
    def device_ctx(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def default_device(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def synchronize(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def device_count(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def is_accelerator_available(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def current_device(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def reset_peak_memory_stats(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def max_memory_allocated(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_device_properties(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def set_device(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_device_capability(*args, **kwargs):
        raise NotImplementedError


class CpuPlatform(BasePlatform):
    device_type: str = "cpu"
    dispatch_key: str = "CPU"
    device_control_env_var = "CPU_VISIBLE_MEMORY_NODES"
    dist_backend: str = "gloo"
    full_dist_backend: str = "cpu:gloo"

    @staticmethod
    def default_device():
        return torch.device("cpu")

    @staticmethod
    def get_device_name():
        return "CPU"

    @staticmethod
    def is_accelerator_available():
        return False


class CudaPlatform(BasePlatform):
    device_type: str = "cuda"
    device_control_env_var: str = "CUDA_VISIBLE_DEVICES"
    dispatch_key: str = "CUDA"
    dist_backend: str = "nccl"
    full_dist_backend: str = "cuda:nccl"

    @staticmethod
    def empty_cache():
        torch.cuda.empty_cache()

    @staticmethod
    def ipc_collect():
        torch.cuda.ipc_collect()

    @staticmethod
    def get_device_name():
        return torch.cuda.get_device_name()

    @staticmethod
    def device_ctx(device):
        return torch.cuda.device(device)

    @staticmethod
    def default_device():
        return torch.device("cuda")

    @staticmethod
    def synchronize(device=None):
        torch.cuda.synchronize(device)

    @staticmethod
    def device_count():
        return torch.cuda.device_count()

    @staticmethod
    def is_accelerator_available():
        return torch.cuda.is_available()

    @staticmethod
    def current_device():
        return torch.cuda.current_device()

    @staticmethod
    def reset_peak_memory_stats(device=None):
        return torch.cuda.reset_peak_memory_stats(device)

    @staticmethod
    def max_memory_allocated(device=None):
        return torch.cuda.max_memory_allocated(device)

    @staticmethod
    def get_device_properties(device=None):
        return torch.cuda.get_device_properties(device)

    @staticmethod
    def set_device(device):
        return torch.cuda.set_device(device)

    @staticmethod
    def get_device_capability(device=None):
        return torch.cuda.get_device_capability(device)


class NPUPlatform(BasePlatform):
    device_type: str = "npu"
    device_control_env_var: str = "ASCEND_RT_VISIBLE_DEVICES"
    dispatch_key: str = "PrivateUse1"
    dist_backend: str = "hccl"
    full_dist_backend: str = "npu:hccl"

    @staticmethod
    def empty_cache():
        torch.npu.empty_cache()

    @staticmethod
    def ipc_collect():
        """
        torch.npu.ipc_collect() is not implemented yet.
        """
        pass

    @staticmethod
    def get_device_name():
        return torch.npu.get_device_name()

    @staticmethod
    def device_ctx(device):
        return torch.npu.device(device)

    @staticmethod
    def default_device():
        return torch.device("npu")

    @staticmethod
    def synchronize(device=None):
        torch.npu.synchronize(device)

    @staticmethod
    def device_count():
        return torch.npu.device_count()

    @staticmethod
    def is_accelerator_available():
        return torch.npu.is_available()

    @staticmethod
    def current_device():
        return torch.npu.current_device()

    @staticmethod
    def reset_peak_memory_stats(device=None):
        return torch.npu.reset_peak_memory_stats(device)

    @staticmethod
    def max_memory_allocated(device=None):
        return torch.npu.max_memory_allocated(device)

    @staticmethod
    def get_device_properties(device=None):
        return torch.npu.get_device_properties(device)

    @staticmethod
    def set_device(device):
        return torch.npu.set_device(device)
