# b2_windowstats_norm_binary
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
| 1 | 0.645 | 0.634 | 0.667 | 0.644 |
| 2 | 0.851 | 0.850 | 0.864 | 0.861 |
| 3 | 0.723 | 0.715 | 0.762 | 0.753 |
| 4 | 0.798 | 0.798 | 0.857 | 0.856 |
| 5 | 0.769 | 0.769 | 0.778 | 0.775 |

**Overall (pooled)**: window macro-F1 0.753, video macro-F1 0.777 (per-fold mean 0.778 ± 0.079)

## Confusion matrix (per-video, pooled)
```
[[50  5]
 [18 33]]
```
