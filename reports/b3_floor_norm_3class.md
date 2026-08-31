# b3_floor_norm_3class
_generated 2026-08-31T06:38:42+00:00_

## Config snapshot
```yaml
windows:
  length_sec: 60.0
  stride_sec: 15.0
  max_undetected_frac: 0.3
  ffill_max_gap_sec: 0.5
  calibration_sec: 60.0
normalized: true
target: 3class
```

## Extras
- **n_windows**: 5682
- **identity_probe_accuracy**: 0.13690476190476192

## Per-fold results

| fold | window acc | window macro-F1 | video acc | video macro-F1 |
|---|---|---|---|---|
| 1 | 0.303 | 0.155 | 0.294 | 0.152 |
| 2 | 0.347 | 0.172 | 0.333 | 0.167 |
| 3 | 0.277 | 0.144 | 0.273 | 0.143 |
| 4 | 0.359 | 0.176 | 0.344 | 0.171 |
| 5 | 0.326 | 0.164 | 0.308 | 0.157 |

**Overall (pooled)**: window macro-F1 0.229, video macro-F1 0.222 (per-fold mean 0.158 ± 0.010)

## Confusion matrix (per-video, pooled)
```
[[ 0 43 12]
 [ 0 40 12]
 [ 0 42  9]]
```
