# b1_perclos_binary
_generated 2026-08-31T06:39:05+00:00_

## Config snapshot
```yaml
windows:
  length_sec: 60.0
  stride_sec: 15.0
  max_undetected_frac: 0.3
  ffill_max_gap_sec: 0.5
  calibration_sec: 60.0
target: binary
note: single calibrated variant; raw/normalized axis N/A for B1
```

## Extras
- **n_windows**: 3735
- **subjects_excluded_no_calibration**: ['57']

## Per-fold results

| fold | window acc | window macro-F1 | video acc | video macro-F1 |
|---|---|---|---|---|
| 1 | 0.655 | 0.654 | 0.708 | 0.708 |
| 2 | 0.786 | 0.785 | 0.818 | 0.812 |
| 3 | 0.789 | 0.788 | 0.762 | 0.760 |
| 4 | 0.790 | 0.789 | 0.857 | 0.857 |
| 5 | 0.626 | 0.626 | 0.667 | 0.667 |

**Overall (pooled)**: window macro-F1 0.732, video macro-F1 0.764 (per-fold mean 0.761 ± 0.069)

## Confusion matrix (per-video, pooled)
```
[[41 14]
 [11 40]]
```
