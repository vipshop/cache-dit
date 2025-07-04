# Examples for CacheDiT  

## Install requirements  

```bash
pip3 install -r requirements.txt
```

## Run examples  

- FLUX.1-dev 

```bash
python3 run_flux.py # baseline
python3 run_flux.py --cache --Fn 8 --Bn 8
python3 run_flux.py --cache --Fn 8 --Bn 0 --taylorseer
```

- FLUX.1-Fill-dev 

```bash
python3 run_flux_fill.py # baseline
python3 run_flux_fill.py --cache --Fn 8 --Bn 8
python3 run_flux_fill.py --cache --Fn 8 --Bn 0 --taylorseer
```

- CogVideoX 

```bash
python3 run_cogvideox.py # baseline
python3 run_cogvideox.py --cache --Fn 8 --Bn 8
python3 run_cogvideox.py --cache --Fn 8 --Bn 0 --taylorseer
```

- Wan2.1 T2V

```bash
python3 run_wan.py # baseline
python3 run_wan.py --cache --Fn 8 --Bn 8
python3 run_wan.py --cache --Fn 8 --Bn 0 --taylorseer
```

- Wan2.1 FLF2V

```bash
python3 run_wan_flf2v.py # baseline
python3 run_wan_flf2v.py --cache --Fn 8 --Bn 8
python3 run_wan_flf2v.py --cache --Fn 8 --Bn 0 --taylorseer
```

- mochi-1-preview

```bash
python3 run_mochi.py # baseline
python3 run_mochi.py --cache --Fn 8 --Bn 8
python3 run_mochi.py --cache --Fn 8 --Bn 0 --taylorseer
```

- HunyuanVideo

```bash
python3 run_hunyuan_video.py # baseline
python3 run_hunyuan_video.py --cache --Fn 8 --Bn 8
python3 run_hunyuan_video.py --cache --Fn 8 --Bn 0 --taylorseer
```
