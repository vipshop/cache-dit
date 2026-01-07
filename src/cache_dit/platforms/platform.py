import torch


class CpuPlatform:
    device_name: str = "cpu"
    device_type: str = "cpu"
    dist_backend: str = "gloo"


class CudaPlatform:
    device_name: str = "cuda"
    device_type: str = "cuda"
    device_control_env_var: str = "CUDA_VISIBLE_DEVICES"
    dispatch_key: str = "CUDA"
    dist_backend: str = "nccl"

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


class NPUPlatform:
    device_name: str = "npu"
    device_type: str = "npu"
    device_control_env_var: str = "ASCEND_RT_VISIBLE_DEVICES"
    dispatch_key: str = "PrivateUse1"
    dist_backend: str = "hccl"

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
