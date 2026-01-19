# Ascend NPU Benchmark

## Atlas 800I A2

### FLUX.1-dev

<p align="center">
    Ulysses: Standard Ulysses Attention, <b>Async Ulysses</b>: Ulysses Attenton with Async QKV Projection, <b>NPU Attn Backend</b>: NPU Attention Backend, <b>UAA: Ulysses Anything Attenton</b>
</p>

|800I A2x1| 800I A2x2 w/ Ulysses|w/ Async Ulysses| w/ Async Ulysses + NPU Attn Backend|
|:---:|:---:|:---:|:---:|  
|FLUX.1, 16.13s|**🎉11.45s**|10.47s|**🎉10.34s**|
|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/npu_sample/flux.1024x1024.C0_Q0_NONE.png" width=180px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/npu_sample/flux.1024x1024.C0_Q0_NONE_Ulysses2.png" width=180px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/npu_sample/flux.1024x1024.C0_Q0_NONE_Ulysses2_ulysses_async.png" width=180px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/npu_sample/flux.1024x1024.C0_Q0_NONE_Ulysses2_ulysses_async_native_npu.png" width=180px>


### Qwen-Image-Edit

|800I A2x2 w/ UAA + TEP| 800I A2x4 w/ UAA + TEP|w/ UAA + TEP + Async Ulysses| w/ UAA + TEP + Async Ulysses + NPU Attn Backend|
|:---:|:---:|:---:|:---:|
|Qwen-Image-Edit, 134.76s|**🎉67.44s**|64.82s|**🎉64.43s**|
|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/npu_sample/qwen_image_edit.1024x1024.C0_Q0_NONE_Ulysses2_TEP_ulysses_anything.png" width=180px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/npu_sample/qwen_image_edit.1024x1024.C0_Q0_NONE_Ulysses4_TEP_ulysses_anything.png" width=180px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/npu_sample/qwen_image_edit.1024x1024.C0_Q0_NONE_Ulysses4_TEP_ulysses_anything_ulysses_async.png" width=180px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/npu_sample/qwen_image_edit.1024x1024.C0_Q0_NONE_Ulysses4_TEP_ulysses_anything_ulysses_async_native_npu.png" width=180px>|


### Z-Image-Turbo w/ NPU Attn Backend

|800I A2x1| 800I A2x2 w/ Ulysses|w/ Async Ulysses|  
|:---:|:---:|:---:|  
|Z-Image-Turbo, 3.39s|**🎉2.49s**|**🎉2.38s**|
|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/npu_sample/zimage.1024x1024.C0_Q0_NONE_native_npu.png" width=180px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/npu_sample/zimage.1024x1024.C0_Q0_NONE_Ulysses2_native_npu.png" width=180px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/npu_sample/zimage.1024x1024.C0_Q0_NONE_Ulysses2_ulysses_async_native_npu.png" width=180px>
