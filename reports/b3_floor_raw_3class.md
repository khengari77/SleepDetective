# b3_floor_raw_3class
_generated 2026-08-31T06:37:29+00:00_

## Config snapshot
```yaml
windows:
  length_sec: 60.0
  stride_sec: 15.0
  max_undetected_frac: 0.3
  ffill_max_gap_sec: 0.5
  calibration_sec: 60.0
normalized: false
target: 3class
```

## Extras
- **n_windows**: 5890
- **identity_probe_accuracy**: 0.09959486833220797

## Per-fold results

| fold | window acc | window macro-F1 | video acc | video macro-F1 |
|---|---|---|---|---|
| 1 | 0.303 | 0.155 | 0.294 | 0.152 |
| 2 | 0.367 | 0.179 | 0.353 | 0.174 |
| 3 | 0.372 | 0.181 | 0.364 | 0.178 |
| 4 | 0.379 | 0.183 | 0.364 | 0.178 |
| 5 | 0.389 | 0.187 | 0.367 | 0.179 |

**Overall (pooled)**: window macro-F1 0.177, video macro-F1 0.172 (per-fold mean 0.172 ± 0.010)

## Confusion matrix (per-video, pooled)
```
[[ 0 55  0]
 [ 0 57  0]
 [ 0 52  0]]
```
