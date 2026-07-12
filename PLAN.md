# SleepDetective v2 — Agent Execution Plan
## Learned drowsiness estimation from MediaPipe landmarks on UTA-RLDD, under a hard storage constraint

---

## 0. Mission

Replace the hand-tuned heuristic stage of SleepDetective (EAR→PERCLOS + head-pitch thresholding, fixed 50/50 fusion, hard 0.75 threshold — see `modules/PERCLOS.py`, `modules/HeadPose.py`, `modules/AwarenessTracker.py`) with a small learned temporal model trained on the UTA Real-Life Drowsiness Dataset (UTA-RLDD).

**Pipeline to build:**

```
video frame → MediaPipe Face Landmarker → per-frame feature vector
→ per-subject baseline normalization → sliding temporal window
→ tiny temporal model → drowsiness score
```

**Success criteria (in priority order):**
1. A fully resumable, storage-bounded extraction pipeline that converts all 180 RLDD videos into compact per-video feature files, never holding more than ONE video on disk at a time.
2. Subject-independent evaluation on the official 5 folds, reported honestly (macro-F1 + confusion matrices), for both 3-class and binary (alert vs. drowsy) formulations.
3. A trained model that beats a faithfully reimplemented PERCLOS heuristic baseline under the SAME evaluation protocol. If it does not beat it, report that honestly — a negative result with a clean comparison is an acceptable outcome.
4. An exported model + inference wrapper that drops into `AwarenessTracker` as a replacement for the heuristic fusion.

---

## 1. Hard constraints — read before writing any code

**C1. Storage.** Total dataset is ~111 GB; individual videos are roughly 0.3–2 GB. At no point may more than one video exist on disk. The loop is strictly: download one file → extract features → verify feature file → delete video → next. Before every download, check free disk space; refuse to download if free space < (expected file size × 2 + 5 GB safety margin). Extracted features for the entire dataset should total well under 1 GB.

**C2. Resumability.** The extraction job will take many hours and WILL be interrupted. Maintain a manifest with per-video status (`pending | downloading | extracting | done | failed`). On restart, skip `done`, retry `failed` up to 3 times, treat `downloading`/`extracting` as `pending` (delete any partial artifacts first). Never re-download a video whose feature file exists and passes verification.

**C3. Extract once, iterate forever.** The feature schema must be rich enough that NO experiment later in the project requires re-downloading videos. When in doubt, store the extra column — features are tiny, videos are not. This is the single most important design decision in the plan.

**C4. Subject-wise evaluation only.** Never split frames, windows, or videos randomly. Use the dataset's official 5 folds (12 participants each): train on 4 folds, test on the held-out fold, rotate, average. Any per-subject normalization statistics must be computed WITHOUT using label information that would be unavailable at deployment (see Phase 4 for the exact rule).

**C5. Licensing/citation.** The dataset requires citing Ghoddoosian, Galib & Athitsos, CVPRW 2019 (arXiv:1904.07312). Put the BibTeX in the README from day one.

**C6. Packaging: uv only.** Migrate the project from its current mixed state (stale `requirements.txt` + Poetry `pyproject.toml`/`poetry.lock`) to `uv` as the single tool for environments and dependencies. Concretely: rewrite `pyproject.toml` with a standard PEP 621 `[project]` table (dropping the `[tool.poetry]` sections and poetry-core build backend), regenerate the lock as `uv.lock` via `uv lock`, then delete `requirements.txt` and `poetry.lock`. The Poetry dependency list is the source of truth for existing deps (it is newer and includes `picamera2`, `typer`, `pynmea2` that requirements.txt lacks). All new v2 dependencies are added with `uv add`; all scripts/docs invoke tooling through `uv run`. No `pip install` instructions anywhere in the repo.

---

## 2. Repository layout

Create a new repo (or `v2/` branch of SleepDetective):

```
sleepdetective-v2/
├── configs/
│   └── default.yaml          # fps, window sizes, model hparams, paths
├── data/
│   ├── manifest.csv          # one row per video: ids, fold, subject, class, status, checksums
│   ├── features/             # per-video parquet files (the only durable data artifact)
│   └── tmp/                  # the single in-flight video lives here; wiped on startup
├── src/
│   ├── acquire.py            # download/delete streaming loop
│   ├── extract.py            # video → per-frame features (MediaPipe)
│   ├── verify.py             # feature-file sanity checks
│   ├── windows.py            # feature files → normalized windows + labels
│   ├── baselines.py          # PERCLOS heuristic + classical ML baselines
│   ├── model.py              # temporal model (GRU/TCN)
│   ├── train.py              # 5-fold training loop
│   ├── evaluate.py           # metrics, confusion matrices, per-fold reports
│   └── export.py             # ONNX export + inference wrapper
├── tests/                    # unit tests for EAR math, windowing, normalization, resume logic
├── reports/                  # generated markdown/plots per experiment
└── README.md
```

Use `uv` exclusively (see C6): `uv init`-style PEP 621 `pyproject.toml` + `uv.lock`, deps managed with `uv add`, everything executed via `uv run`. Core deps: `mediapipe` (tasks API, ≥0.10), `opencv-python-headless`, `numpy`, `pandas`, `pyarrow`, `scikit-learn`, `torch` (CPU is fine), `onnx`/`onnxruntime`, and the download tool chosen in Phase 1.

---

## 3. Phase 1 — Dataset acquisition manifest

**Goal:** a `manifest.csv` enumerating all 180 videos with enough metadata to download each one individually, before downloading anything.

The dataset is distributed two ways; support the first, fall back to the second:

1. **Kaggle mirror** (`rishab260/uta-reallife-drowsiness-dataset`). The Kaggle API supports listing files (`kaggle datasets files <slug>`) and downloading a single file (`kaggle datasets download <slug> -f <path>`), which fits the one-file-at-a-time constraint perfectly. Requires the user's `kaggle.json` credentials — ask the user to provide them; do not proceed without.
2. **Official Google Drive folders** (linked from the UTA-RLDD site, organized as Fold1..Fold5). Use `gdown` with individual file IDs; enumerate folder contents once to harvest IDs into the manifest. Google Drive imposes quota limits on large public files — if downloads start failing with quota errors, back off (hours) and resume; this is exactly why C2 exists.

Manifest columns: `video_id, source_path, download_ref (kaggle path or drive file id), fold (1–5), subject_id, class_label (0/5/10), expected_size_bytes (if listable), status, attempts, feature_file, extracted_at, error_msg`.

Parse `fold`, `subject_id`, and `class_label` from the directory/file naming (videos are named by class: 0, 5, 10 per subject). **Validate the manifest before proceeding:** exactly 60 subjects, exactly 3 videos per subject (flag known gaps rather than crashing — mirrors are occasionally missing a file), 12 subjects per fold, no subject in two folds. Print the validation report and stop for user confirmation if counts are off.

Also record per-video container format — the dataset mixes `.mov`/`.mp4` from phones and webcams. Some phone videos carry rotation metadata that OpenCV ignores; Phase 3 must handle this.

---

## 4. Phase 2 — Streaming extraction loop

**Goal:** `data/features/<video_id>.parquet` for all 180 videos, one video on disk at a time.

Loop skeleton (in `acquire.py`):

```
wipe data/tmp/
for row in manifest where status != done:
    check disk space (C1); abort with clear message if insufficient
    download → data/tmp/<video_id>.<ext>          [status: downloading]
    verify download (size matches if known; file opens in cv2/ffprobe)
    extract features → data/features/<video_id>.parquet   [status: extracting]
    run verify.py checks on the parquet
    delete video from tmp                          [status: done]
    flush manifest to disk after EVERY status change
```

Wrap each step in try/except; on failure, record `error_msg`, increment `attempts`, delete partial artifacts, continue with the next video. Log a running summary every video: n done / n failed / ETA / total feature bytes.

### 4.1 Extraction details (`extract.py`)

- Use the **MediaPipe Face Landmarker task** (the current tasks API, not the legacy `FaceMesh` solution used in SleepDetective v1) with `output_face_blendshapes=True` and `output_facial_transformation_matrixes=True`. Download the `.task` model file once at setup.
- **Sampling rate:** decode every frame but PROCESS at a target 10 fps (sample by timestamp, not frame index — source fps varies across videos). 10 fps preserves blink dynamics (blinks last 100–400 ms) while cutting compute ~3× and keeping sequences manageable. Record the true source fps in the parquet metadata.
- **Rotation:** probe rotation metadata with `ffprobe`; apply the corresponding `cv2.rotate` before inference. Sanity heuristic: if face detection fails on >80% of the first 300 processed frames, retry those frames at 90/180/270° and lock in whichever orientation detects best; log the decision.
- **Read frames sequentially** with `cv2.VideoCapture` — never seek per frame, and never dump frames to disk as images.

### 4.2 Per-frame feature schema (one parquet row per processed frame)

| Group | Columns | Notes |
|---|---|---|
| Bookkeeping | `t_sec`, `frame_idx`, `face_detected` (bool), `n_faces` | when no face: row present, features NaN |
| Blendshapes | all 52 coefficients, `bs_*` | already roughly identity-normalized; the workhorse features |
| Geometry | `ear_left`, `ear_right`, `mar` | reuse v1's EAR landmark indices so the heuristic baseline is faithful; compute in normalized-landmark space scaled by inter-ocular distance (NOT frame pixels — v1's pixel-space EAR is resolution-dependent, a genuine v1 bug worth noting) |
| Head pose | `pitch`, `yaw`, `roll` | decompose from the facial transformation matrix — do NOT port v1's solvePnP with its guessed camera intrinsics |
| Landmark subset | eye contours, iris, mouth outer, brows — ~60 points × (x,y,z), normalized: centered on nose tip, scaled by inter-ocular distance, rotation-canonicalized using the transformation matrix | insurance for future feature ideas (C3) |
| Detection quality | face bounding-box area fraction, landmarker confidence if exposed | lets later stages filter junk frames |

Parquet metadata: video_id, subject, class, fold, source fps, rotation applied, extractor version, mediapipe version. Use float32 throughout. Expected size: ~6,000 rows × ~250 columns ≈ 3–6 MB per video, ≈ 0.6–1 GB total — acceptable; if it matters, drop the landmark subset to get well under 300 MB, but only as a last resort (C3).

### 4.3 Verification (`verify.py`)

A feature file passes iff: row count ≈ video duration × 10 fps (±10%); `face_detected` rate ≥ 50% (below that, mark video `failed`, keep the parquet, flag for manual review — some RLDD videos are dark/blurry); EAR/MAR/pose values within sane physical ranges on detected frames; timestamps strictly increasing. Write per-video verification stats into the manifest.

---

## 5. Phase 3 — Windowing and normalization (`windows.py`)

**Goal:** a function that turns feature parquets into `(X, y, subject, fold)` arrays, parameterized by config — recomputed cheaply per experiment, never cached as a giant tensor blob.

- **Windows:** sliding windows over each video; default 60 s @ 10 fps = 600 timesteps, stride 15 s. Make window length and stride config parameters — window length is a key ablation. Drop windows with >30% undetected frames; forward-fill short gaps (≤0.5 s), mask longer ones.
- **Per-subject baseline normalization (the ablation that matters):** for each subject, compute baseline statistics (median + IQR of each feature) from the FIRST 60 seconds of their class-0 (alert) video, then z-score all of that subject's features against their own baseline. This mirrors v1's calibration mode and is deployment-honest: the real system calibrates on the driver at journey start, when they are presumed alert. It uses the class-0 label only to pick the calibration video, which corresponds exactly to deployment reality. Exclude the calibration segment's windows from train/test to avoid trivial leakage.
- Every experiment runs in two variants: **raw features** vs. **baseline-normalized features**. This ablation is a headline result.
- **Labels:** window inherits its video's label. Build both target versions: 3-class {0,5,10} and binary {0 vs 10, class-5 windows dropped}. Report both; expect binary to be the deployable one.

---

## 6. Phase 4 — Baselines first, model second

Do not train the neural model until all three baselines below have numbers. They define "added value."

**B1 — Faithful PERCLOS heuristic (v1 reimplementation).** Per window: calibrate threshold = subject baseline EAR mean − 1 std (as v1 does), PERCLOS = fraction of frames below threshold, head-pose deviation fraction, 50/50 average, threshold 0.75. Map its continuous score to classes via a threshold fitted on training folds only. This is the incumbent — the whole project is judged against it.

**B2 — Window-statistics + classical ML.** Aggregate each window into summary statistics (mean/std/percentiles of EAR, MAR, pitch; blink rate and mean blink duration from a simple EAR-valley blink detector; PERCLOS; yawn count from MAR threshold crossings) → gradient-boosted trees (or logistic regression). Strong, hard-to-beat baseline; if the temporal net can't beat this, ship this — it's smaller and more interpretable.

**B3 — Trivial floor:** majority class + a per-subject nearest-centroid check to quantify how much identity alone predicts (this exposes leakage if any normalization is broken).

**M1 — Temporal model.** Small GRU (1–2 layers, hidden 32–64) or TCN over the per-frame feature sequence (blendshapes + geometry + pose; NOT the raw landmark subset in the first iteration). Target < 100k parameters. Input masking for undetected frames. Train per fold with early stopping on a validation split carved out of training folds BY SUBJECT (e.g., 2 of the 48 training subjects). Standard cross-entropy; also try ordinal regression for the 3-class case since 0 < 5 < 10 is ordered — cheap experiment, mention in report.

Keep training CPU-feasible: with ~thousands of windows and <100k params this trains in minutes per fold. No GPU dependency anywhere in the project.

---

## 7. Phase 5 — Evaluation protocol (`evaluate.py`)

- 5-fold cross-validation on the official folds; metrics: accuracy, macro-F1, per-class F1, confusion matrix — averaged across folds AND reported per fold (variance across folds is large on this dataset; hiding it is dishonest).
- Two granularities: **per-window** and **per-video** (majority vote / mean score over the video's windows). Per-video is what comparable papers usually report.
- Also report a **latency-style metric** relevant to the product: using the drowsy videos, score windows chronologically and report how early the model's rolling score crosses the alarm threshold vs. the heuristic. (Alarm threshold fitted on training folds.)
- Expectations to calibrate against, not to chase: published subject-independent 3-class results on RLDD sit roughly in the 60s–70s%; the original paper's blink-feature HM-LSTM reported ~65% 3-class. Low-vigilant (5) is the confusion sink. Do not "fix" a bad number by weakening the split.
- Produce `reports/<experiment_name>.md` per run with config snapshot, per-fold tables, confusion matrices (matplotlib PNGs), and a short auto-written summary. The final report must include the results matrix: {B1, B2, M1} × {raw, baseline-normalized} × {3-class, binary}.

---

## 8. Phase 6 — Export and integration

- Export the winning model to ONNX; verify onnxruntime output matches torch to 1e-5.
- Write `DrowsinessEstimator` — an inference class mirroring the training-time pipeline exactly: ring buffer of per-frame features (600 × n_features), the same baseline-normalization code path fed by a calibration routine, masked inference, EMA smoothing of the output score. Unit-test that a features-parquet replayed through `DrowsinessEstimator` reproduces the offline evaluation scores for that video.
- Integration into SleepDetective: replace the body of `AwarenessTracker.take()` — FaceMesh result → feature row → estimator → `awareness_level`/`drowsy`. Keep v1's heuristic behind a config flag as fallback for the first N seconds before calibration completes.

---

## 9. Failure modes the agent must handle explicitly

1. **Download quota/auth failures** (Drive quota, Kaggle credentials): exponential backoff, mark `failed` after 3 attempts, keep going; never let one file wedge the loop.
2. **Corrupt/unreadable videos:** mark failed with the decoder error; the dataset has known rough files.
3. **No-face videos** (dark, extreme angle): keep the parquet, exclude from training via the `face_detected` rate, list them in the final report.
4. **Disk fills mid-download:** the pre-download space check plus tmp-wipe-on-startup must make this recoverable with zero manual cleanup.
5. **Identity leakage regression:** B3's identity-probe number goes in every report; if a model's gain over B2 coincides with high identity predictability, investigate before believing it.
6. **MediaPipe API drift:** pin the mediapipe version and record it in every parquet's metadata; extractor version bump ⇒ re-extraction is a NEW manifest status, not silent overwrite.

---

## 10. Milestone order and definition-of-done

| # | Milestone | Done when |
|---|---|---|
| 0 | uv migration (C6) | PEP 621 `pyproject.toml` + `uv.lock` committed; `requirements.txt` and `poetry.lock` deleted; `uv run python main.py --help` works |
| 1 | Repo + config + manifest | manifest.csv validates: 60 subjects × 3 videos, 5 folds × 12 subjects (gaps explicitly flagged) |
| 2 | Extraction pipeline on a 3-video smoke test | 3 parquets pass verify.py; tmp/ empty; manifest statuses correct after a simulated mid-run kill + restart |
| 3 | Full extraction | ≥ 95% of videos `done`; failures documented; total features < 1.5 GB; zero videos on disk |
| 4 | Windowing + normalization | unit tests pass; window counts logged per video; calibration segments excluded |
| 5 | Baselines B1–B3 | per-fold numbers in a report; B1 reproduces heuristic behavior on sanity clips |
| 6 | M1 + ablations | full results matrix; best config identified |
| 7 | Export + integration | ONNX parity test passes; replay test passes; SleepDetective runs with the new estimator |

Work strictly in milestone order. Commit at every milestone with the report artifacts. Ask the user before: (a) starting the full 180-video download, (b) any decision that would violate C1–C4, (c) declaring a final model when M1 does not beat B2.

---

## 11. Citation block for README

```bibtex
@inproceedings{ghoddoosian2019realistic,
  title={A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection},
  author={Ghoddoosian, Reza and Galib, Marnim and Athitsos, Vassilis},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  year={2019}
}
```
