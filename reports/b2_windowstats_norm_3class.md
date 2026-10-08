# b2_windowstats_norm_3class
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
| 1 | 0.508 | 0.501 | 0.588 | 0.572 |
| 2 | 0.605 | 0.596 | 0.697 | 0.682 |
| 3 | 0.509 | 0.497 | 0.545 | 0.532 |
| 4 | 0.547 | 0.546 | 0.625 | 0.617 |
| 5 | 0.598 | 0.598 | 0.577 | 0.564 |

**Overall (pooled)**: window macro-F1 0.546, video macro-F1 0.595 (per-fold mean 0.594 ± 0.052)

## Confusion matrix (per-video, pooled)
```
[[46  8  1]
 [16 27  9]
 [17 11 23]]
```
