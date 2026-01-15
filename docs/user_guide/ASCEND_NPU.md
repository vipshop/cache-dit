# Ascend NPU Support 

🔥We are excited to announce that Cache-DiT now provides **native** support for **Ascend NPU**. Theoretically, **nearly all** models supported by Cache-DiT can run on Ascend NPU with most of Cache-DiT’s optimization technologies, including:

- **Hybrid Cache Acceleration** ([**DBCache**](https://cache-dit.readthedocs.io/en/latest/user_guide/CACHE_API/#dbcache-dual-block-cache), DBPrune, [**TaylorSeer**](https://cache-dit.readthedocs.io/en/latest/user_guide/CACHE_API/#hybrid-taylorseer-calibrator), [**SCM**](https://cache-dit.readthedocs.io/en/latest/user_guide/CACHE_API/#scm-steps-computation-masking) and more)
- **Context Parallelism** (w/ Extended Diffusers' CP APIs, [**UAA**](https://cache-dit.readthedocs.io/en/latest/user_guide/CONTEXT_PARALLEL/#uaa-ulysses-anything-attention), Async Ulysses, ...)
- **Tensor Parallelism** (w/ PyTorch native DTensor and Tensor Parallelism APIs)
- **Text Encoder Parallelism** (w/ PyTorch native DTensor and Tensor Parallelism APIs)
- **Auto Encoder (VAE) Parallelism** (w/ Data or Tile Parallelism, avoid OOM)
- **ControlNet Parallelism** (w/ Context Parallelism for ControlNet module)
- Built-in **HTTP serving** deployment support with simple REST APIs

Please refer to **[Ascend NPU Supported Matrix](../supported_matrix/ASCEND_NPU.md)** for more details.

## Features Support 

|Device|Hybrid Cache|Context Parallel|Tensor Parallel|Text Encoder Parallel|Auto Encoder(VAE) Parallel|
|:---|:---:|:---:|:---:|:---:|:---:|
|Atlas 800T A2|✅|✅|✅|✅|✅|
|Atlas 800I A2|✅|✅|✅|✅|✅|

## Environment Requirements

There are two installation methods:  

- **Using pip**: first prepare env manually or via CANN image, then install `cache-dit` using pip.  
- **Using docker**: use the [Ascend NPU community: vllm-ascend](https://quay.io/repository/ascend/vllm-ascend?tab=tags) pre-built docker image as the base image for **cache-dit** directly. (**Recommended**, no need for installing torch and torch_npu manually)

## Install NPU SDKs Manually

This section describes how to install NPU environment manually.

### Requirements

OS: Linux; Python: >= 3.10, < 3.12; A hardware with Ascend NPU. It's usually the Atlas 800 A2 series; Softwares:

| Software      | Supported version                | Note                                      |
|---------------|----------------------------------|-------------------------------------------|
| Ascend HDK    | Refer to [here](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/releasenote/releasenote_0000.html) | Required for CANN |
| CANN          | == 8.3.RC2                       | Required for cache-dit and torch-npu    |
| torch-npu     | == 2.8.0             | Required for cache-dit|
| torch         | == 2.8.0                          | Required for torch-npu and cache-dit        |
| NNAL          | == 8.3.RC2                       | Required for libatb.so, enables advanced tensor operations |


### Configure CANN environment. 

Before installation, you need to make sure firmware/driver and CANN are installed correctly, refer to [Ascend Environment Setup Guide](https://ascend.github.io/docs/sources/ascend/quick_install.html) for more details. To verify that the Ascend NPU firmware and driver were correctly installed, run:

```bash
npu-smi info
```

Please refer to [Ascend Environment Setup Guide](https://ascend.github.io/docs/sources/ascend/quick_install.html) for more details.

### Configure software environment.      

The easiest way to prepare your software environment is using CANN image directly. We recommend using the [Ascend NPU community: vllm-ascend](https://quay.io/repository/ascend/vllm-ascend?tab=tags) pre-built docker image as the base image of Ascend NPU for **cache-dit**. CANN image can be found in Ascend official community website: [here](https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884). The CANN prebuilt image includes NNAL (Ascend Neural Network Acceleration Library) which provides libatb.so for advanced tensor operations. No additional installation is required when using the prebuilt image.

```bash
# Update DEVICE according to your device (/dev/davinci[0-7])
export DEVICE=/dev/davinci7
# Update the pre-built image
export IMAGE=quay.io/ascend/cann:|cann_image_tag|
docker run --rm \
    --name cache-dit-ascend \
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

### Install PyTorch 

If install failed by using pip command, you can get `torch-2.8.0*.whl` file by [Link](https://download.pytorch.org/whl/torch/) and install manually.

```bash
# torch: aarch64
pip3 install torch==2.8.0
# torch: x86
pip3 install torch==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu
```

### Install torch_npu

Strongly recommend install torch_npu by acquire `torch_npu-2.8.0*.whl` file by [Link](https://gitcode.com/Ascend/pytorch/releases) and install manually. For more detail about Ascend Pytorch Adapter installation, please refer [https://gitcode.com/Ascend/pytorch](https://gitcode.com/Ascend/pytorch)

### Install Extra Dependences

```bash
pip install --no-deps torchvision==0.16.0 
pip install einops sentencepiece 
```

## Use prebuilt Docker Image

We recommend using the prebuilt image from the [Ascend NPU community: vllm-ascend](https://quay.io/repository/ascend/vllm-ascend?tab=tags) as the base image of Ascend NPU for **cache-dit**. You can just pull the **prebuilt image** from the image [repository](https://quay.io/repository/ascend/vllm-ascend?tab=tags) and run it with bash. For example:

```bash
# Download pre-built image for Ascend NPU
docker pull quay.io/ascend/vllm-ascend:v0.12.0rc1

# Use the pre-built image for cache-dit
docker run \
    --name cache-dit-ascend \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    --net=host \
    --shm-size=80g \
    --privileged=true \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /data:/data \
    -itd quay.io/ascend/vllm-ascend:v0.12.0rc1 bash
```

## Ascend Environment variables
```bash
# Make sure CANN_path is set to your CANN installation path
# e.g., export CANN_path=/usr/local/Ascend
source $CANN_path/ascend-toolkit/set_env.sh
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# Set NPU devices by ASCEND_RT_VISIBLE_DEVICES env
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

Once it is done, you can start to set up `cache-dit`.

## Install Cache-DiT Library

You can install the stable release of `cache-dit` from PyPI:

```bash
pip3 install -U cache-dit
```
Or you can install the latest develop version from GitHub:

```bash
pip3 install git+https://github.com/vipshop/cache-dit.git
```
Please also install the latest main branch of diffusers for context parallelism:  
```bash
pip3 install git+https://github.com/huggingface/diffusers.git # or >= 0.36.0
```

## Exmaples and Benchmark

After the environment configuration is complete, users can refer to the **[Quick Examples](../EXAMPLES.md)**, **[Ascend NPU Benchmark](../benchmark/ASCEND_NPU.md)** and **[Ascend NPU Supported Matrix](../supported_matrix/ASCEND_NPU.md)** for more details.

```bash
pip3 install torch==2.9.1 transformers accelerate torchao bitsandbytes torchvision 
pip3 install opencv-python-headless einops imageio-ffmpeg ftfy 
pip3 install git+https://github.com/huggingface/diffusers.git # latest or >= 0.36.0
pip3 install git+https://github.com/vipshop/cache-dit.git # latest

git clone https://github.com/vipshop/cache-dit.git && cd cache-dit/examples
```

### Single NPU Inference

The easiest way to enable hybrid cache acceleration for DiTs with cache-dit is to start with single NPU inference. For examples:  

```bash
# use default model path, e.g, "black-forest-labs/FLUX.1-dev"
python3 generate.py flux 
python3 generate.py qwen_image
python3 generate.py flux --cache
python3 generate.py qwen_image --cache
```

### Multi-NPUs Inference 

cache-dit is designed to work seamlessly with CPU or Sequential Offloading, 🔥Context Parallelism, 🔥Tensor Parallelism. For examples:

```bash
torchrun --nproc_per_node=4 generate.py flux --parallel ulysses 
torchrun --nproc_per_node=4 generate.py qwen_image --parallel ulysses
torchrun --nproc_per_node=4 generate.py flux --parallel ulysses --cache
torchrun --nproc_per_node=4 generate.py qwen_image --parallel ulysses --cache
```
