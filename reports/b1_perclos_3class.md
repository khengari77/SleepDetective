# b1_perclos_3class
_generated 2026-08-31T06:36:28+00:00_

## Config snapshot
```yaml
windows:
  length_sec: 60.0
  stride_sec: 15.0
  max_undetected_frac: 0.3
  ffill_max_gap_sec: 0.5
  calibration_sec: 60.0
target: 3class
note: single calibrated variant; raw/normalized axis N/A for B1
```

## Extras
- **n_windows**: 5682
- **subjects_excluded_no_calibration**: ['21', '43', '49', '50', '57']

## Per-fold results

| fold | window acc | window macro-F1 | video acc | video macro-F1 |
|---|---|---|---|---|
| 1 | 0.442 | 0.442 | 0.471 | 0.471 |
| 2 | 0.451 | 0.437 | 0.485 | 0.463 |
| 3 | 0.534 | 0.521 | 0.485 | 0.471 |
| 4 | 0.482 | 0.489 | 0.469 | 0.469 |
| 5 | 0.412 | 0.396 | 0.385 | 0.363 |

**Overall (pooled)**: window macro-F1 0.467, video macro-F1 0.460 (per-fold mean 0.448 ± 0.042)

## Confusion matrix (per-video, pooled)
```
[[35 19  1]
 [18 18 16]
 [ 8 23 20]]
```
