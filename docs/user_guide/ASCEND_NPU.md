# Ascend NPU Support 

# Installation

This document describes how to install NPU environment manually.

## Requirements

- OS: Linux
- Python: >= 3.10, < 3.12
- A hardware with Ascend NPU. It's usually the Atlas 800 A2 series.
- Software:

    | Software      | Supported version                | Note                                      |
    |---------------|----------------------------------|-------------------------------------------|
    | Ascend HDK    | Refer to [here](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/releasenote/releasenote_0000.html) | Required for CANN |
    | CANN          | == 8.3.RC2                       | Required for vllm-ascend and torch-npu    |
    | torch-npu     | == 2.8.0             | Required for cache-dit|
    | torch         | == 2.8.0                          | Required for torch-npu and cache-dit        |
    | NNAL          | == 8.3.RC2                       | Required for libatb.so, enables advanced tensor operations |

There are two installation methods:
- **Using pip**: first prepare env manually or via CANN image, then install `cache-dit` using pip.
- **Using docker**: use the `vllm-ascend` pre-built docker image directly. (**Recommended**, no need for installing torch and torch_npu manually)

## Configure Ascend CANN environment

Before installation, you need to make sure firmware/driver and CANN are installed correctly, refer to [Ascend Environment Setup Guide](https://ascend.github.io/docs/sources/ascend/quick_install.html) for more details.

### Configure hardware environment

To verify that the Ascend NPU firmware and driver were correctly installed, run:

```bash
npu-smi info
```

Refer to [Ascend Environment Setup Guide](https://ascend.github.io/docs/sources/ascend/quick_install.html) for more details.

### Configure software environment
The easiest way to prepare your software environment is using CANN image directly:
CANN image can be found in Ascend official community website: [here](https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884)

```{note}
The CANN prebuilt image includes NNAL (Ascend Neural Network Acceleration Library) which provides libatb.so for advanced tensor operations. No additional installation is required when using the prebuilt image.
```

```{code-block} bash
# Update DEVICE according to your device (/dev/davinci[0-7])
export DEVICE=/dev/davinci7
# Update the vllm-ascend image
export IMAGE=quay.io/ascend/cann:|cann_image_tag|
docker run --rm \
    --name vllm-ascend-env \
    --shm-size=1g \
    --device $DEVICE \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    -it $IMAGE bash
```

### Pytorch and Ascend Pytorch Adapter Setup
#### torch
- aarch64
```{code-block} bash
pip3 install torch==2.8.0
```
- x86
```{code-block} bash
pip3 install torch==2.8.0+cpu  --index-url https://download.pytorch.org/whl/cpu
```
If install failed by using pip command, you can get `torch-2.8.0*.whl` file by [Link](https://download.pytorch.org/whl/torch/) and install manually.

#### torch_npu
- Strongly recommend install torch_npu by acquire `torch_npu-2.8.0*.whl` file by [Link](https://gitcode.com/Ascend/pytorch/releases) and install manually.

For more detail about Ascend Pytorch Adapter installation, please refer [https://gitcode.com/Ascend/pytorch](https://gitcode.com/Ascend/pytorch)

## Set up using Docker

`vllm-ascend` offers Docker images for deployment. You can just pull the **prebuilt image** from the image repository [ascend/vllm-ascend](https://quay.io/repository/ascend/vllm-ascend?tab=tags) and run it with bash.

## Ascend relevant Environment variables
```{code-block} bash
# Make sure CANN_path is set to your CANN installation directory, e.g., export CANN_path=/usr/local/Ascend
source $CANN_path/ascend-toolkit/set_env.sh
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

Once it is done, you can start to set up `cache-dit`.
