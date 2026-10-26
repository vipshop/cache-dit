# Huawei Ascend NPU Supported Matrix  

<div id="supported"></div>

Currently, **cache-dit** library supports almost **Any** Diffusion Transformers (with **Transformer Blocks** that match the specific Input and Output **patterns**). Please check [🎉Examples](https://github.com/vipshop/cache-dit/blob/main/examples) for more details. Here are just some of the tested models listed.

Theoretically, almost all models supported by Cache-DiT can run on Ascend NPU. Here, only some of the models we have tested are listed. We will continue testing more models for Ascend NPU, so stay tuned for updates!

## Transformer Optimization  

|📚Models|Hybrid Cache|Context Parallel|Tensor Parallel|
|:---|:---:|:---:|:---:|
|FLUX.2-Klein-4B|✅|✅|✅|
|FLUX.2-Klein-base-4B|✅|✅|✅|
|FLUX.2-Klein-9B|✅|✅|✅|
|FLUX.2-Klein-base-9B|✅|✅|✅|
|FLUX.2-dev|✅|✅|✅|
|FLUX.1-dev|✅|✅|✅|
|FLUX.1-Fill-dev|✅|✅|✅|
|FLUX.1-Kontext-dev|✅|✅|✅|
|Z-Image-Turbo*|✅|✅|✅|
|Qwen-Image|✅|✅|✅|
|Qwen-Image-Layered|✅|✅|✅|
|Qwen-Image-2512|✅|✅|✅|
|Qwen-Image-Edit|✅|✅|✅|
|Qwen-Image-Edit-2509|✅|✅|✅|
|Qwen-Image-Edit-2511|✅|✅|✅|
|Qwen-Image-Lightning|✅|✅|✅|
|Qwen-Image-Edit-Lightning|✅|✅|✅|
|Qwen-Image-Edit-2509-Lightning|✅|✅|✅|
|Qwen-Image-Edit-2511-Lightning|✅|✅|✅|
|Wan-2.2-T2V|✅|✅|✅|
|Wan-2.2-I2V|✅|✅|✅|
|Wan-2.1-T2V|✅|✅|✅|
|Wan-2.1-I2V|✅|✅|✅|
|LongCat-Image|✅|✅|✅|
|LongCat-Image-Edit|✅|✅|✅|
|Ovis-Image|✅|✅|✅|


## Text Encoder & VAE Optimization

|📚Models|Text Encoder Parallel|AutoEncoder(VAE) Parallel|
|:---|:---:|:---:|  
|FLUX.2-Klein-4B|✅|✅|
|FLUX.2-Klein-base-4B|✅|✅|
|FLUX.2-Klein-9B|✅|✅|
|FLUX.2-Klein-base-9B|✅|✅|
|FLUX.2-dev|✅|✅|
|FLUX.1-dev|✅|✅|
|FLUX.1-Fill-dev|✅|✅|
|FLUX.1-Kontext-dev|✅|✅|
|Z-Image-Turbo*|✅|✅|
|Qwen-Image|✅|✅|✅|
|Qwen-Image-Layered|✅|✅|✅|
|Qwen-Image-2512|✅|✅|✅|
|Qwen-Image-Edit|✅|✅|
|Qwen-Image-Edit-2509|✅|✅|
|Qwen-Image-Edit-2511|✅|✅|✅|
|Qwen-Image-Lightning|✅|✅|
|Qwen-Image-Edit-Lightning|✅|✅|
|Qwen-Image-Edit-2509-Lightning|✅|✅|
|Qwen-Image-Edit-2511-Lightning|✅|✅|
|Wan-2.2-T2V|✅|✅|
|Wan-2.2-I2V|✅|✅|
|Wan-2.1-T2V|✅|✅|
|Wan-2.1-I2V|✅|✅|
|LongCat-Image|✅|✅|
|LongCat-Image-Edit|✅|✅|
|Ovis-Image|✅|✅|

Z-Image-Turbo*: Since diffusers does not support this model by NPU, you need to merge this PR into your local diffusers repo: https://github.com/huggingface/diffusers/pull/12979
