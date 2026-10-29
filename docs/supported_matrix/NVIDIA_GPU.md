# Supported Matrix  

<div id="supported"></div>

Currently, **cache-dit** library supports almost **Any** Diffusion Transformers (with **Transformer Blocks** that match the specific Input and Output **patterns**). Please check [🎉Examples](https://github.com/vipshop/cache-dit/blob/main/examples) for more details. Here are just some of the tested models listed.

## Transformers Optimization
One Model Series may contain many pipelines. cache-dit applies optimizations at the Transformer level; thus,any pipelines that include the supported transformer are already supported by cache-dit. ✅: supported now; ✖️: not supported now; **[🤖Q](https://github.com/nunchaku-tech/nunchaku)**: **[nunchaku](https://github.com/nunchaku-tech/nunchaku)** w/ SVDQ W4A4; 

|📚Models: `🤗70+`|Hybrid Cache|Context Parallel|Tensor Parallel|
|:---|:---:|:---:|:---:|
|SANA-WM-I2V|✅|✖️|✅|
|FLUX.2-Klein-9b-kv|✖️|✖️|✅|
|FLUX.2-Klein-4B|✅|✅|✅|
|FLUX.2-Klein-base-4B|✅|✅|✅|
|FLUX.2-Klein-9B|✅|✅|✅|
|FLUX.2-Klein-base-9B|✅|✅|✅|
|Helios-Base|✅|✅|✅|  
|Helios-Mid|✅|✅|✅|  
|Helios-Distilled|✅|✅|✅|  
|FireRed-Image-Edit-1.0|✅|✅|✅|  
|FireRed-Image-Edit-1.1|✅|✅|✅|  
|GLM-Image-T2I|✅|✖️|✅|
|GLM-Image-I2I|✅|✖️|✅|
|Z-Image|✅|✅|✅|
|Ming-Image-0.1-Design|✅|✅|✅|
|Ming-Image-0.1-Design-Layer|✅|✅|✅|
|LTX-2-I2V|✅|✅|✅|
|LTX-2-T2V|✅|✅|✅|
|LTX-2.5-DFR (8 steps)|✅|✅|✅|
|LTX-2.5-T2V|✅|✅|✅|
|LTX-2.5-I2V|✅|✅|✅|
|Qwen-Image-2512|✅|✅|✅|
|Qwen-Image-2.1-Turbo (8 steps)|✅|✅|✅|
|Z-Image-Turbo `🤖Q`|✅|✅|✖️|
|Qwen-Image-Layered|✅|✅|✅|
|Qwen-Image-Edit-2511-Lightning|✅|✅|✅|
|Qwen-Image-Edit-2511|✅|✅|✅|
|LongCat-Image|✅|✅|✅|
|LongCat-Image-Edit|✅|✅|✅|
|LongCat-AudioDiT-1B|✅|✅|✅|
|Z-Image-Turbo|✅|✅|✅|
|Z-Image-Turbo-Fun-ControlNet-2.0|✅|✅|✅|
|Z-Image-Turbo-Fun-ControlNet-2.1|✅|✅|✅|
|Ovis-Image|✅|✅|✅|
|HiDream-O1-Image|✅|✅|✅|
|Joy-Image-Edit|✅|✅|✅|
|Bria-Fibo|✅|✅|✅|
|Bria-Fibo-Edit|✅|✅|✅|
|ERNIE-Image|✅|✅|✅|
|ERNIE-Image-Turbo|✅|✅|✅|
|Krea-2-Turbo|✅|✅|✅|
|FLUX.2-dev|✅|✅|✅|
|FLUX.1-dev|✅|✅|✅|
|FLUX.1-Fill-dev|✅|✅|✅|
|FLUX.1-Kontext-dev|✅|✅|✅|
|Qwen-Image|✅|✅|✅|
|Qwen-Image-2.1|✅|✅|✅|
|Qwen-Image-2.1-Turbo|✅|✅|✅|
|Qwen-Image-Edit|✅|✅|✅|
|Qwen-Image-Edit-2509|✅|✅|✅|
|Qwen-Image-ControlNet|✅|✅|✅|
|Qwen-Image-ControlNet-Inpainting|✅|✅|✅|
|Qwen-Image-Lightning|✅|✅|✅|
|Qwen-Image-Edit-Lightning|✅|✅|✅|
|Qwen-Image-Edit-2509-Lightning|✅|✅|✅|
|Any-Flow|✅|✅|✅|
|Wan-Animate-2|✅|✅|✅|
|Wan-Animate-2-Distilled|✅|✅|✅|
|Wan-2.2-T2V|✅|✅|✅|
|Wan-2.2-I2V|✅|✅|✅|
|Wan-2.2-VACE-Fun|✅|✅|✅|
|Wan-2.1-T2V|✅|✅|✅|
|Wan-2.1-I2V|✅|✅|✅|
|Wan-2.1-FLF2V|✅|✅|✅|
|Wan-2.1-VACE|✅|✅|✅|
|HunyuanImage-2.1|✅|✅|✅|
|HunyuanVideo-1.5|✅|✖️|✖️|
|HunyuanVideo|✅|✅|✅|
|FLUX.1-dev `🤖Q`|✅|✅|✖️|
|FLUX.1-Fill-dev `🤖Q`|✅|✅|✖️|
|FLUX.1-Kontext-dev `🤖Q`|✅|✅|✖️|
|Qwen-Image `🤖Q`|✅|✅|✖️|
|Qwen-Image-Edit `🤖Q`|✅|✅|✖️|
|Qwen-Image-Edit-2509 `🤖Q`|✅|✅|✖️|
|Qwen-Image-Lightning `🤖Q`|✅|✅|✖️|
|Qwen-Image-Edit-Lightning `🤖Q`|✅|✅|✖️|
|Qwen-Image-Edit-2509-Lightning `🤖Q`|✅|✅|✖️|
|SkyReels-V2-T2V|✅|✅|✅|
|Motif-Video-T2V|✅|✅|✅|
|Motif-Video-I2V|✅|✅|✅|
|LongCat-Video|✅|✖️|✖️|
|ChronoEdit-14B|✅|✅|✅|
|Kandinsky-5.0-T2V-Lite|✅|✅️|✅️|
|Kandinsky-6.0-T2VA|✅|✅|✅|
|NucleusMoE-Image|✅|✖️|✅|
|PRX-512-t2i-sft|✅|✖️|✖️|
|PRXPixel-t2i|✅|✅|✅|
|Anima-Base-t2i|✅|✅|✅|
|LTX-Video-v0.9.8|✅|✅|✅|
|LTX-Video-v0.9.7|✅|✅|✅|
|CogVideoX|✅|✅|✅|
|CogVideoX-1.5|✅|✅|✅|
|CogView-4|✅|✅|✅|
|CogView-3-Plus|✅|✅|✅|
|Chroma1-HD|✅|✅|✅|
|PixArt-Sigma-XL-2-1024-MS|✅|✅|✅|
|PixArt-XL-2-1024-MS|✅|✅|✅|
|VisualCloze-512|✅|✅|✅|
|ConsisID-preview|✅|✅|✅|
|mochi-1-preview|✅|✖️|✅|
|Lumina-Image-2.0|✅|✖️|✅|
|HiDream-I1-Full|✅|✖️|✖️|
|HunyuanDiT|✅|✖️|✅|
|Sana-1600M-1024px|✅|✖️|✖️|
|DiT-XL-2-256|✅|✅|✖️|
|Allegro-T2V|✅|✖️|✖️|
|OmniGen-2|✅|✖️|✖️|
|stable-diffusion-3.5-large|✅|✖️|✅|
|Amused-512|✅|✖️|✖️|
|AuraFlow|✅|✖️|✖️|

## Text Encoder & VAE Optimization

|📚Models: `🤗70+`|Text Encoder Parallel|AutoEncoder(VAE) Parallel|
|:---|:---:|:---:|
|FLUX.2-Klein-9b-kv|✅|✅|
|FLUX.2-Klein-4B|✅|✅|
|FLUX.2-Klein-base-4B|✅|✅|
|FLUX.2-Klein-9B|✅|✅|
|FLUX.2-Klein-base-9B|✅|✅|
|FLUX.2-dev|✅|✅|
|Helios-Base|✅|✅| 
|Helios-Mid|✅|✅|  
|Helios-Distilled|✅|✅| 
|FireRed-Image-Edit-1.0|✅|✅|  
|FireRed-Image-Edit-1.1|✅|✅|  
|GLM-Image-T2I|✖️|✅|
|GLM-Image-I2I|✖️|✅|
|Z-Image|✅|✅|
|Ming-Image-0.1-Design|✅|✅|
|Ming-Image-0.1-Design-Layer|✅|✅|
|LTX-2-I2V|✅|✅|
|LTX-2-T2V|✅|✅|
|LTX-2.5-DFR (8 steps)|✅|✅|
|LTX-2.5-T2V|✅|✅|
|LTX-2.5-I2V|✅|✅|
|Qwen-Image-2512|✅|✅|
|Z-Image-Turbo `🤖Q`|✅|✅|
|Qwen-Image-Layered|✅|✅|
|Qwen-Image-Edit-2511-Lightning|✅|✅|
|Qwen-Image-Edit-2511|✅|✅|
|LongCat-Image|✅|✅|
|LongCat-Image-Edit|✅|✅|
|Z-Image-Turbo|✅|✅|
|Z-Image-Turbo-Fun-ControlNet-2.0|✅|✅|
|Z-Image-Turbo-Fun-ControlNet-2.1|✅|✅|
|Ovis-Image|✅|✅|
|Joy-Image-Edit|✅|✅|
|Bria-Fibo|✅|✅|
|Bria-Fibo-Edit|✅|✅|
|ERNIE-Image|✅|✅|
|ERNIE-Image-Turbo|✅|✅|
|Krea-2-Turbo|✅|✅|
|FLUX.1-dev|✅|✅|
|FLUX.1-Fill-dev|✅|✅|
|FLUX.1-Kontext-dev|✅|✅|
|Qwen-Image|✅|✅|
|Qwen-Image-Edit|✅|✅|
|Qwen-Image-Edit-2509|✅|✅|
|Qwen-Image-ControlNet|✅|✅|
|Qwen-Image-ControlNet-Inpainting|✅|✅|
|Qwen-Image-Lightning|✅|✅|
|Qwen-Image-Edit-Lightning|✅|✅|
|Qwen-Image-Edit-2509-Lightning|✅|✅|
|Any-Flow|✅|✅|
|Wan-Animate-2|✅|✅|
|Wan-Animate-2-Distilled|✅|✅|
|Wan-2.2-T2V|✅|✅|
|Wan-2.2-I2V|✅|✅|
|Wan-2.2-VACE-Fun|✅|✅|
|Wan-2.1-T2V|✅|✅|
|Wan-2.1-I2V|✅|✅|
|Wan-2.1-FLF2V|✅|✅|
|Wan-2.1-VACE|✅|✅|
|HunyuanImage-2.1|✅|✖️|
|HunyuanVideo-1.5|✅|✖️|
|HunyuanVideo|✅|✅|
|FLUX.1-dev `🤖Q`|✅|✅|
|FLUX.1-Fill-dev `🤖Q`|✅|✅|
|FLUX.1-Kontext-dev `🤖Q`|✅|✅|
|Qwen-Image `🤖Q`|✅|✅|
|Qwen-Image-Edit `🤖Q`|✅|✅|
|Qwen-Image-Edit-2509 `🤖Q`|✅|✅|
|Qwen-Image-Lightning `🤖Q`|✅|✅|
|Qwen-Image-Edit-Lightning `🤖Q`|✅|✅|
|Qwen-Image-Edit-2509-Lightning `🤖Q`|✅|✅|
|SkyReels-V2-T2V|✅|✅|
|Motif-Video-T2V|✅|✅|
|Motif-Video-I2V|✅|✅|
|ChronoEdit-14B|✅|✅|
|Kandinsky-5.0-T2V-Lite|✅|✅|
|Kandinsky-6.0-T2VA|✅|✅|
|SANA-WM-I2V|✅|✅|
|NucleusMoE-Image|✅|✅|
|PRX-512-t2i-sft|✅|✖️|
|PRXPixel-t2i|✅|✖️|
|Anima-Base-t2i|✅|✅|
|LTX-Video-v0.9.8|✅|✖️|
|LTX-Video-v0.9.7|✅|✖️|
|CogVideoX|✅|✖️|
|CogVideoX-1.5|✅|✖️|
|CogView-4|✅|✅|
|CogView-3-Plus|✅|✅|
|Chroma1-HD|✅|✅|
|PixArt-Sigma-XL-2-1024-MS|✅|✅|
|PixArt-XL-2-1024-MS|✅|✅|
|VisualCloze-512|✅|✅|
|ConsisID-preview|✅|✖️|
|mochi-1-preview|✅|✖️|
|Lumina-Image-2.0|✅|✅|
|HiDream-I1-Full|✅|✅|
|HunyuanDiT|✅|✅|
|Sana-1600M-1024px|✅|✖️|
|DiT-XL-2-256|✅|✅|
|Allegro-T2V|✅|✖️|
|OmniGen-2|✅|✅|
|stable-diffusion-3.5-large|✖️|✅|
|Amused-512|✅|✖️|
|AuraFlow|✅|✅|

## ControlNet Optimization

|Models|ControlNet Parallel|
|:---|:---:|
|Z-Image-Turbo-Fun-ControlNet-2.0|✅|
|Z-Image-Turbo-Fun-ControlNet-2.1|✅|
|Qwen-Image-ControlNet|TODO|
|Qwen-Image-ControlNet-Inpainting|TODO|
