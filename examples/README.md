# Examples for CacheDiT  

## ⚙️Install Requirements  

```bash
pip3 install -r requirements.txt
```

## 🚀Run Examples  

- Qwen-Image-Edit

```bash
python3 run_qwen_image_edit.py # baseline
python3 run_qwen_image_edit.py --cache
```

- Qwen-Image

```bash
python3 run_qwen_image.py # baseline
python3 run_qwen_image.py --cache
python3 run_qwen_image.py --cache --compile
python3 run_qwen_image.py --cache --compile --quantize
```

- FLUX.1-dev 

```bash
python3 run_flux.py # baseline
python3 run_flux.py --cache 
```

- FLUX.1-Fill-dev 

```bash
python3 run_flux_fill.py # baseline
python3 run_flux_fill.py --cache 
```

- FLUX.1-Kontext-dev 

```bash
python3 run_flux_kontext.py # baseline
python3 run_flux_kontext.py --cache 
```

- CogVideoX 

```bash
python3 run_cogvideox.py # baseline
python3 run_cogvideox.py --cache 
```

- Wan2.2 T2V

```bash
python3 run_wan_2.2.py # baseline
python3 run_wan_2.2.py --cache 
python3 run_wan_2.2.py --cache --compile
python3 run_wan_2.2.py --cache --compile --quantize
```

- Wan2.1 T2V

```bash
python3 run_wan.py # baseline
python3 run_wan.py --cache 
```

- Wan2.1 FLF2V

```bash
python3 run_wan_flf2v.py # baseline
python3 run_wan_flf2v.py --cache 
```

- mochi-1-preview

```bash
python3 run_mochi.py # baseline
python3 run_mochi.py --cache 
```

- HunyuanVideo

```bash
python3 run_hunyuan_video.py # baseline
python3 run_hunyuan_video.py --cache 
```
