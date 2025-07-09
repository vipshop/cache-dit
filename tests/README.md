# Tests

## Taylorseer, Order=2  

```bash
python3 test_taylorseer.py --order 2
```

![](./taylorseer_approximation_order_2.png)

## Taylorseer, Order=4

```bash
python3 test_taylorseer.py --order 4
```

![](./taylorseer_approximation_order_4.png)

## Metrics

```bash
python3 test_metrics.py --img-true U0_C0_NONE_R0.08_S0_T24.82s.png --img-test U0_C0_DBCACHE_F1B0S1W0T1O2_R0.08_S10_T16.30s.png
``` 
output:

```bash
Namespace(img_true='U0_C0_NONE_R0.08_S0_T24.82s.png', img_test='U0_C0_DBCACHE_F1B0S1W0T1O2_R0.08_S10_T16.30s.png', video_true=None, video_test=None)
U0_C0_NONE_R0.08_S0_T24.82s.png vs U0_C0_DBCACHE_F1B0S1W0T1O2_R0.08_S10_T16.30s.png, PSNR: 24.68392809867634
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.34it/s]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  8.22it/s]
U0_C0_NONE_R0.08_S0_T24.82s.png vs U0_C0_DBCACHE_F1B0S1W0T1O2_R0.08_S10_T16.30s.png, FID: 75.76806327295184
```

```bash
python3 test_metrics.py --img-true U0_C0_NONE_R0.08_S0_T24.82s.png --img-test U0_C0_DBCACHE_F1B0S1W0T0O2_R0.08_S11_T15.43s.png
```

output: 
```bash
Namespace(img_true='U0_C0_NONE_R0.08_S0_T24.82s.png', img_test='U0_C0_DBCACHE_F1B0S1W0T0O2_R0.08_S11_T15.43s.png', video_true=None, video_test=None)
U0_C0_NONE_R0.08_S0_T24.82s.png vs U0_C0_DBCACHE_F1B0S1W0T0O2_R0.08_S11_T15.43s.png, PSNR: 21.240280356949647
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.36it/s]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  8.32it/s]
U0_C0_NONE_R0.08_S0_T24.82s.png vs U0_C0_DBCACHE_F1B0S1W0T0O2_R0.08_S11_T15.43s.png, FID: 136.1958835812449
```
