# b3_floor_raw_binary
_generated 2026-08-31T06:39:38+00:00_

## Config snapshot
```yaml
windows:
  length_sec: 60.0
  stride_sec: 15.0
  max_undetected_frac: 0.3
  ffill_max_gap_sec: 0.5
  calibration_sec: 60.0
normalized: false
target: binary
```

## Extras
- **n_windows**: 3758
- **identity_probe_accuracy**: 0.08994708994708994

## Per-fold results

| fold | window acc | window macro-F1 | video acc | video macro-F1 |
|---|---|---|---|---|
| 1 | 0.530 | 0.347 | 0.500 | 0.333 |
| 2 | 0.525 | 0.344 | 0.500 | 0.333 |
| 3 | 0.441 | 0.306 | 0.429 | 0.300 |
| 4 | 0.507 | 0.337 | 0.476 | 0.323 |
| 5 | 0.543 | 0.352 | 0.526 | 0.345 |

**Overall (pooled)**: window macro-F1 0.337, video macro-F1 0.327 (per-fold mean 0.327 ± 0.015)

## Confusion matrix (per-video, pooled)
```
[[ 0 55]
 [ 0 52]]
```
