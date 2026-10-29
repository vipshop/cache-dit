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

Or, you can use <span style="color:#c77dff;">cache-dit-metrics</span> tool. For examples: 

```bash
cache-dit-metrics -h  # show usage
# all: PSNR, FID, SSIM, MSE, ..., etc.
cache-dit-metrics psnr -i1 true.png -i2 test.png  # image
cache-dit-metrics psnr -i1 true_dir -i2 test_dir  # image dir
```

## Audio Metrics

Audio models (e.g. LongCat-AudioDiT) need a different measurement set: a waveform PSNR is destroyed by a phase flip or a one-sample shift that nobody can hear.

```python
from cache_dit.metrics import compute_si_sdr, compute_snr, compute_mel_psnr, compute_mel_ssim

si_sdr,    n = compute_si_sdr("true.wav", "test.wav")     # dB, inf when identical
snr,       n = compute_snr("true.wav", "test.wav")        # dB, no gain compensation
mel_psnr,  n = compute_mel_psnr("true.wav", "test.wav")   # dB, on 80-dim log-mel
mel_ssim,  n = compute_mel_ssim("true_dir", "test_dir")   # on the log-mel spectrogram
```

| Metric | Domain | Ignores | Sensitive to |
|---|---|---|---|
| `si_sdr` | waveform | a global gain (least-squares rescale) | sample-level error / phase |
| `snr` | waveform | nothing | gain change and sample-level error |
| `mel_psnr` | 80-dim log-mel | phase, absolute time shift | spectral envelope distance |
| `mel_ssim` | 80-dim log-mel | global gain, small shifts | time-frequency structure (harmonics, formants) |

Read them together. A diffusion TTS run is chaotic: a different kernel changes the sampling trajectory, so `si_sdr` collapses (often below 0 dB) while the audio still says the same words. Use `si_sdr` / `snr` as a bit-exactness gate for cache and offload (which should be identical), and `mel_psnr` / `mel_ssim` for parallel / quantized runs — always against a same-configuration rerun, since that rerun is the numerical noise floor.

```bash
cache-dit-metrics si_sdr mel_ssim -a1 true.wav -a2 test.wav   # audio
cache-dit-metrics all -a1 true_dir -a2 test_dir               # audio dir, 24 kHz defaults
```
