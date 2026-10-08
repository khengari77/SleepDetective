# b2_windowstats_raw_binary
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
| 1 | 0.614 | 0.614 | 0.625 | 0.619 |
| 2 | 0.577 | 0.540 | 0.591 | 0.545 |
| 3 | 0.658 | 0.657 | 0.714 | 0.708 |
| 4 | 0.637 | 0.637 | 0.667 | 0.667 |
| 5 | 0.618 | 0.618 | 0.684 | 0.683 |

**Overall (pooled)**: window macro-F1 0.620, video macro-F1 0.654 (per-fold mean 0.644 ± 0.058)

## Confusion matrix (per-video, pooled)
```
[[35 20]
 [17 35]]
```
