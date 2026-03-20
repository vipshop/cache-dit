# Metrics

You can utilize the APIs provided by cache-dit to quickly evaluate the accuracy losses caused by different cache configurations. 

## Metrics Functions

Use the metrics functions evaluate the accuracy losses:

<div id="metrics"></div>

```python
# pip3 install "cache-dit[metrics]"
from cache_dit.metrics import compute_psnr
from cache_dit.metrics import compute_ssim
from cache_dit.metrics import compute_fid

psnr,   n = compute_psnr("true.png", "test.png") # Num: n
psnr,   n = compute_psnr("true_dir", "test_dir")
ssim,   n = compute_ssim("true_dir", "test_dir")
fid,    n = compute_fid("true_dir", "test_dir")
```

## Metrics Command Line

Or, you can use <span style="color:hotpink;">cache-dit-metrics</span> tool. For examples: 

```bash
cache-dit-metrics -h  # show usage
# all: PSNR, FID, SSIM, MSE, ..., etc.
cache-dit-metrics psnr -i1 true.png -i2 test.png  # image
cache-dit-metrics psnr -i1 true_dir -i2 test_dir  # image dir
```
