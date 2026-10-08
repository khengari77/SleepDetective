# Excluded videos
_generated 2026-10-07T14:06:23+00:00_

- extractor_version: 2
- min_face_detection_confidence: 0.5
- min_face_presence_confidence: 0.5
- min_tracking_confidence: 0.5
- verification gate: face_detected rate >= 0.5

## Still-failed videos

| video_id | subject | class | fold | archive | face rate | attempts | reason |
|---|---|---|---|---|---|---|---|
| f1_s05_c05 | 05 | 5 | 1 | fold1_part1 | 0.42 | 3 | face_detected rate 0.42 < 0.5 |
| f2_s21_c00 | 21 | 0 | 2 | fold2_part2 | 0.39 | 3 | face_detected rate 0.39 < 0.5 |
| f3_s33_c10 | 33 | 10 | 3 | fold3_part2 | 0.23 | 3 | face_detected rate 0.23 < 0.5 |
| f4_s43_c00 | 43 | 0 | 4 | fold4_part2 | 0.39 | 3 | face_detected rate 0.39 < 0.5 |
| f4_s43_c10 | 43 | 10 | 4 | fold4_part2 | 0.39 | 3 | face_detected rate 0.39 < 0.5 |
| f5_s49_c00 | 49 | 0 | 5 | fold5_part1 | 0.40 | 1 | face_detected rate 0.40 < 0.5 |
| f5_s50_c00 | 50 | 0 | 5 | fold5_part1 | 0.42 | 1 | face_detected rate 0.42 < 0.5 |
| f5_s50_c10 | 50 | 10 | 5 | fold5_part1 | 0.33 | 1 | face_detected rate 0.33 < 0.5 |
| f5_s53_c05 | 53 | 5 | 5 | fold5_part1 | 0.07 | 1 | face_detected rate 0.07 < 0.5 |
| f5_s57_c00 | 57 | 0 | 5 | fold5_part2 | 0.36 | 1 | face_detected rate 0.36 < 0.5 |

## Unregistered / absent videos

| video_id | expected source | status |
|---|---|---|
| f3_s32_c10 | Fold3_part?/32/10.mov | absent from manifest |
| f5_s49_c10 | Fold5_part?/49/10.mov | absent from manifest |

## Subjects without a done class-0 (calibration) video

These subjects are skipped entirely in normalized experiments (`build_dataset` requires a class-0 video for baseline stats): 21, 43, 49, 50, 57.

## Impact

- 10 video(s) failed verification and are excluded from training.
- 2 expected video(s) are not in the manifest.
- 5 subject(s) lack calibration and drop out of normalized runs.
- Done videos available for training: 168.
