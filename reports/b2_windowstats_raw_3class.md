# b2_windowstats_raw_3class
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
| 1 | 0.377 | 0.380 | 0.412 | 0.412 |
| 2 | 0.356 | 0.329 | 0.324 | 0.298 |
| 3 | 0.437 | 0.436 | 0.424 | 0.425 |
| 4 | 0.399 | 0.398 | 0.424 | 0.416 |
| 5 | 0.357 | 0.360 | 0.400 | 0.404 |

**Overall (pooled)**: window macro-F1 0.387, video macro-F1 0.397 (per-fold mean 0.391 ± 0.047)

## Confusion matrix (per-video, pooled)
```
[[20 20 15]
 [15 19 23]
 [ 8 18 26]]
```
