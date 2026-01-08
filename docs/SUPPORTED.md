# Supported DiTs  

<div id="supported"></div>

Currently, **cache-dit** library supports almost **Any** Diffusion Transformers (with **Transformer Blocks** that match the specific Input and Output **patterns**). Please check [🎉Examples](https://github.com/vipshop/cache-dit/blob/main/examples) for more details. Here are just some of the tested models listed.

```python
>>> import cache_dit
>>> cache_dit.supported_pipelines()
(32, ['Flux*', 'Mochi*', 'CogVideoX*', 'Wan*', 'HunyuanVideo*', 'QwenImage*', 'LTX*', 'Allegro*',
'CogView3Plus*', 'CogView4*', 'Cosmos*', 'EasyAnimate*', 'SkyReelsV2*', 'StableDiffusion3*',
'ConsisID*', 'DiT*', 'Amused*', 'Bria*', 'Lumina*', 'OmniGen*', 'PixArt*', 'Sana*', 'StableAudio*',
'VisualCloze*', 'AuraFlow*', 'Chroma*', 'ShapE*', 'HiDream*', 'HunyuanDiT*', 'HunyuanDiTPAG*',
'Kandinsky5*', 'PRX*'])
```

One Model Series may contain many pipelines. cache-dit applies optimizations at the Transformer level; thus, any pipelines that include the supported transformer are already supported by cache-dit. ✅: supported now; ✖️: not supported now; **[🤖Q](https://github.com/nunchaku-tech/nunchaku)**: **[nunchaku](https://github.com/nunchaku-tech/nunchaku)** w/ SVDQ W4A4; **[C-P](./USER_GUIDE.md)**: Context Parallelism; **[T-P](./USER_GUIDE.md)**: Tensor Parallelism; **[TE-P](./USER_GUIDE.md)**: Text Encoder Parallelism; **[VAE-P](./USER_GUIDE.md)**: VAE Parallelism.

| 📚Supported DiTs: `🤗70+` | Cache  | C-P | T-P | TE-P | VAE-P |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen-Image-2512 | ✅ | ✅ | ✅ | ✅ | ✅ |
| Z-Image-Turbo `🤖Q` | ✅ | ✅ | ✖️ | ✅ | ✅ |
| Qwen-Image-Layered | ✅ | ✅ | ✅ | ✅ | ✅ |
| Qwen-Image-Edit-2511-Lightning | ✅ | ✅ | ✅ | ✅ | ✅ |
| Qwen-Image-Edit-2511 | ✅ | ✅ | ✅ | ✅ |  ✅ |
| LongCat-Image | ✅ | ✅ | ✅ | ✅ | ✅ |
| LongCat-Image-Edit | ✅ | ✅ | ✅ | ✅ | ✅ |
| Z-Image-Turbo | ✅ | ✅ | ✅ | ✅ | ✅ |
| Z-Image-Turbo-Fun-ControlNet-2.0 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Z-Image-Turbo-Fun-ControlNet-2.1 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Ovis-Image |✅ | ✅ | ✅ | ✅ | ✅ |
| FLUX.2-dev | ✅ | ✅ | ✅ | ✅ | ✅ |
| FLUX.1-dev | ✅ | ✅ | ✅ | ✅ | ✅ |
| FLUX.1-Fill-dev | ✅ | ✅ | ✅ | ✅ |✅ |
| FLUX.1-Kontext-dev | ✅ | ✅ | ✅ | ✅ |✅ |
| Qwen-Image | ✅ | ✅ | ✅ | ✅ |✅ |
| Qwen-Image-Edit | ✅ | ✅ | ✅ | ✅ |✅ |
| Qwen-Image-Edit-2509 | ✅ | ✅ | ✅ | ✅ |✅ |
| Qwen-Image-ControlNet | ✅ | ✅ | ✅ | ✅ |✅ |
| Qwen-Image-ControlNet-Inpainting | ✅ | ✅ | ✅ | ✅ |✅ |
| Qwen-Image-Lightning | ✅ | ✅ | ✅ | ✅ |✅ |
| Qwen-Image-Edit-Lightning | ✅ | ✅ | ✅ | ✅ |✅ |
| Qwen-Image-Edit-2509-Lightning | ✅ | ✅ | ✅ | ✅ |✅ |
| Wan-2.2-T2V  | ✅ | ✅ | ✅ | ✅ |✅ |
| Wan-2.2-ITV  | ✅ | ✅ | ✅ | ✅ |✅ |
| Wan-2.2-VACE-Fun | ✅ | ✅ | ✅ | ✅ |✅ |
| Wan-2.1-T2V |  ✅ | ✅ | ✅ | ✅ |✅ |
| Wan-2.1-ITV |  ✅ | ✅ | ✅ | ✅ |✅ |
| Wan-2.1-FLF2V |  ✅ | ✅ | ✅ | ✅ |✅ |
| Wan-2.1-VACE | ✅ | ✅ | ✅ | ✅ |✅ |
| HunyuanImage-2.1 | ✅ | ✅ | ✅ | ✅ | ✖️ |
| HunyuanVideo-1.5 | ✅ | ✖️ | ✖️ | ✅ | ✖️ |
| HunyuanVideo | ✅ | ✅ | ✅ | ✅ | ✅ |
| FLUX.1-dev `🤖Q` | ✅ | ✅ | ✖️ | ✅ | ✅ |
| FLUX.1-Fill-dev `🤖Q` | ✅ | ✅ | ✖️ | ✅ | ✅ |
| FLUX.1-Kontext-dev `🤖Q` | ✅ | ✅ | ✖️ | ✅ | ✅ |
| Qwen-Image `🤖Q` | ✅ | ✅ | ✖️ | ✅ | ✅ |
| Qwen-Image-Edit `🤖Q` | ✅ | ✅ | ✖️ | ✅ | ✅ |
| Qwen-Image-Edit-2509 `🤖Q` | ✅ | ✅ | ✖️ | ✅ | ✅ |
| Qwen-Image-Lightning `🤖Q` | ✅ | ✅ | ✖️ | ✅ | ✅ |
| Qwen-Image-Edit-Lightning `🤖Q` | ✅ | ✅ | ✖️ | ✅ | ✅ |
| Qwen-Image-Edit-2509-Lightning `🤖Q` | ✅ | ✅ | ✖️ | ✅ | ✅ |
| SkyReels-V2-T2V | ✅ | ✅  | ✅  | ✅ | ✅ |
| LongCat-Video | ✅ | ✖️ | ✖️ | ✖️ | ✖️ |
| ChronoEdit-14B | ✅ | ✅ | ✅ | ✅ | ✅ |
| Kandinsky-5.0-T2V-Lite | ✅ | ✅️ | ✅️ | ✅ | ✅ |
| PRX-512-t2i-sft | ✅ | ✖️ | ✖️ | ✅ | ✖️ | ✅ |
| LTX-Video-v0.9.8 | ✅ | ✅ | ✅ | ✅ | ✖️ |
| LTX-Video-v0.9.7 | ✅ | ✅ | ✅ | ✅ | ✖️ |
| CogVideoX | ✅ | ✅ | ✅ | ✅ | ✖️ |
| CogVideoX-1.5 | ✅ | ✅ | ✅ | ✅ | ✖️ |
| CogView-4 | ✅ | ✅ | ✅ | ✅ | ✅ |
| CogView-3-Plus | ✅ | ✅ | ✅ | ✅ | ✅ |
| Chroma1-HD | ✅ | ✅ | ✅ | ✅ | ✅ |
| PixArt-Sigma-XL-2-1024-MS | ✅ | ✅ | ✅ | ✅ | ✅ |
| PixArt-XL-2-1024-MS | ✅ | ✅ | ✅ | ✅ | ✅ |
| VisualCloze-512 | ✅ | ✅ | ✅ | ✅ | ✅ |
| ConsisID-preview | ✅ | ✅ | ✅ | ✅ | ✖️ |
| mochi-1-preview | ✅ | ✖️ | ✅ | ✅ | ✖️ |
| Lumina-Image-2.0 | ✅ | ✖️ | ✅ | ✅ | ✅ |
| HiDream-I1-Full | ✅ | ✖️ | ✖️ | ✅ | ✅ |
| HunyuanDiT | ✅ | ✖️ | ✅ | ✅ | ✖️ | ✅ |
| Sana-1600M-1024px | ✅ | ✖️ | ✖️ | ✅ | ✖️ |
| DiT-XL-2-256 | ✅ | ✅ | ✖️ | ✅ | ✅ |
| Allegro-T2V | ✅ | ✖️ | ✖️ | ✅ | ✖️ |
| OmniGen-2 | ✅ | ✖️ | ✖️ | ✅ | ✅ |
| stable-diffusion-3.5-large | ✅ | ✖️ | ✅ | ✖️ | ✅ |
| Amused-512 | ✅ | ✖️ | ✖️ | ✅ | ✖️ |
| AuraFlow | ✅ | ✖️ | ✖️ | ✅ | ✅ |
