# b3_floor_norm_binary
_generated 2026-08-31T06:40:14+00:00_

## Config snapshot
```yaml
windows:
  length_sec: 60.0
  stride_sec: 15.0
  max_undetected_frac: 0.3
  ffill_max_gap_sec: 0.5
  calibration_sec: 60.0
normalized: true
target: binary
```

## Extras
- **n_windows**: 3735
- **identity_probe_accuracy**: 0.10543130990415335

## Per-fold results

| fold | window acc | window macro-F1 | video acc | video macro-F1 |
|---|---|---|---|---|
| 1 | 0.470 | 0.320 | 0.500 | 0.333 |
| 2 | 0.475 | 0.322 | 0.500 | 0.333 |
| 3 | 0.441 | 0.306 | 0.429 | 0.300 |
| 4 | 0.507 | 0.337 | 0.476 | 0.323 |
| 5 | 0.526 | 0.345 | 0.500 | 0.333 |

**Overall (pooled)**: window macro-F1 0.478, video macro-F1 0.480 (per-fold mean 0.325 ± 0.013)

## Confusion matrix (per-video, pooled)
```
[[23 32]
 [23 28]]
```
