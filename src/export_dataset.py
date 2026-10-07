"""Package extracted features as a standalone dataset and upload it PRIVATELY.

Builds `release.dir` with:
  features/<video_id>.parquet  per-frame features, identity columns prepended
  metadata.csv                 one row per video (fold, label, QC flags, provenance)
  README.md                    Hugging Face dataset card (also the Kaggle description)
  dataset-metadata.json        Kaggle dataset metadata

Every video with a parquet is shipped — including those that failed the
verification gate — flagged by `passed_verification` so users pick their own
filter. Uploads are private only: UTA-RLDD has no stated license, so nothing
goes public until redistribution of derived features is confirmed.

Usage:
    uv run python -m src.export_dataset build
    uv run python -m src.export_dataset upload-hf
    uv run python -m src.export_dataset upload-kaggle --owner <kaggle-user>
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import typer

from src.manifest import load_config

app = typer.Typer(add_completion=False)

CLASS_NAMES = {0: "alert", 5: "low_vigilant", 10: "drowsy"}

METADATA_COLUMNS = [
    "video_id", "file_name", "subject_id", "fold", "class_label", "class_name",
    "passed_verification", "face_detected_rate", "n_frames", "duration_sec",
    "source_fps", "target_fps", "rotation_applied", "extractor_version",
    "mediapipe_version", "min_face_detection_confidence",
    "min_face_presence_confidence", "min_tracking_confidence",
]


@app.callback()
def _cli():
    """Build and privately upload the UTA-RLDD feature dataset."""


def with_identity_columns(table: pa.Table, video_id: str, subject_id: str,
                          fold: int, class_label: int) -> pa.Table:
    """Prepend constant identity columns so concatenated files stay attributable."""
    n = table.num_rows
    ids = [
        ("video_id", pa.array([video_id] * n, pa.string())),
        ("subject_id", pa.array([subject_id] * n, pa.string())),
        ("fold", pa.array([fold] * n, pa.int8())),
        ("class_label", pa.array([class_label] * n, pa.int8())),
    ]
    out = table
    for i, (name, arr) in enumerate(ids):
        out = out.add_column(i, name, arr)
    # drop pandas metadata (it describes the old column layout); keep provenance
    meta = {k: v for k, v in (table.schema.metadata or {}).items()
            if not k.startswith(b"pandas")}
    return out.replace_schema_metadata(meta)


def metadata_row(row: pd.Series, table: pa.Table) -> dict:
    meta = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()
            if not k.startswith(b"pandas")}
    detected = table.column("face_detected").to_pandas()
    rate = float(detected.mean()) if len(detected) else 0.0
    label = int(row["class_label"])
    return {
        "video_id": row["video_id"],
        "file_name": f"features/{row['video_id']}.parquet",
        "subject_id": row["subject_id"],
        "fold": int(row["fold"]),
        "class_label": label,
        "class_name": CLASS_NAMES[label],
        "passed_verification": row["status"] == "done",
        "face_detected_rate": round(rate, 4),
        "n_frames": table.num_rows,
        "duration_sec": float(meta.get("duration_sec", "nan")),
        "source_fps": float(meta.get("source_fps", "nan")),
        "target_fps": float(meta.get("target_fps", "nan")),
        "rotation_applied": int(meta.get("rotation_applied", 0)),
        "extractor_version": int(meta.get("extractor_version", 1)),
        "mediapipe_version": meta.get("mediapipe_version", ""),
        # v1 parquets predate threshold recording; v1 always used the 0.5 defaults
        "min_face_detection_confidence": float(meta.get("min_face_detection_confidence", 0.5)),
        "min_face_presence_confidence": float(meta.get("min_face_presence_confidence", 0.5)),
        "min_tracking_confidence": float(meta.get("min_tracking_confidence", 0.5)),
    }


def render_card(meta: pd.DataFrame, n_columns: int, cfg: dict) -> str:
    rel = cfg["release"]
    passed = int(meta["passed_verification"].sum())
    by_class = meta.groupby("class_label").size()
    class_counts = ", ".join(f"{n} {CLASS_NAMES[c]}" for c, n in by_class.items())
    retried = meta[meta["extractor_version"] > 1]
    failed = meta[~meta["passed_verification"]]
    failed_rows = "\n".join(
        f"| {r.video_id} | {r.subject_id} | {r.class_name} | {r.face_detected_rate:.2f} |"
        for r in failed.itertuples())
    hours = meta["duration_sec"].sum() / 3600
    gate = cfg["verify"]["min_face_detected_rate"]
    return f"""---
pretty_name: {rel['title']}
license: other
license_name: uta-rldd-derived
task_categories:
  - video-classification
  - tabular-classification
tags:
  - drowsiness-detection
  - driver-monitoring
  - mediapipe
  - facial-landmarks
  - blendshapes
  - time-series
size_categories:
  - 1M<n<10M
configs:
  - config_name: frames
    data_files: features/*.parquet
    default: true
  - config_name: videos
    data_files: metadata.csv
---

# {rel['title']}

Per-frame facial features extracted from the
[UTA Real-Life Drowsiness Dataset (UTA-RLDD)](https://sites.google.com/view/utarldd/home)
with the MediaPipe Face Landmarker (tasks API, blendshapes + transformation
matrices). **No video or image data is included**, only derived numeric features.

- **{len(meta)} videos** from {meta['subject_id'].nunique()} subjects
  ({class_counts}), ~{hours:.0f} h of footage
- Sampled at **{meta['target_fps'].iloc[0]:g} fps**, {int(meta['n_frames'].sum()):,} frames total
- {n_columns} columns per frame
- Official UTA-RLDD **5-fold split** preserved (`fold` column, 12 subjects per fold)

Produced by the [SleepDetective](https://github.com/khengari77/SleepDetective)
project (`src/extract.py`).

## Labels

UTA-RLDD labels whole videos by self-reported state (Karolinska Sleepiness Scale
bands): `0` = alert, `5` = low vigilant, `10` = drowsy. Every frame of a video
carries the video's label.

## Files

| path | content |
|---|---|
| `features/<video_id>.parquet` | one row per sampled frame (zstd parquet) |
| `metadata.csv` | one row per video: fold, subject, label, QC flags, provenance |

`video_id` is `f<fold>_s<subject>_c<label>`, e.g. `f1_s01_c05`.

### Per-frame columns

| columns | description |
|---|---|
| `video_id`, `subject_id`, `fold`, `class_label` | identity (constant per file) |
| `t_sec`, `frame_idx` | timestamp and source-frame index |
| `face_detected`, `n_faces` | detection flag; all feature columns are null when no face |
| `bs_*` (52) | MediaPipe ARKit-style blendshape scores in [0, 1] |
| `ear_left`, `ear_right`, `mar` | eye/mouth aspect ratios in normalized-landmark space |
| `pitch`, `yaw`, `roll` | head pose in degrees from the facial transformation matrix |
| `lm<i>_x/y/z` | 72 landmarks (eye contours, iris, outer mouth, brows): nose-tip centered, inter-ocular scaled, rotation-canonicalized |
| `bbox_area_frac` | face bounding-box area as a fraction of the frame |

Per-file provenance (source fps, rotation applied, extractor version, MediaPipe
version, detection thresholds) is stored in the parquet schema metadata and
mirrored in `metadata.csv`.

## Quality control

A video passes verification when its row count matches duration × fps (±10%),
pose/EAR/MAR are within physical ranges, timestamps are monotonic, and the face
is detected in **≥ {gate:.0%}** of frames. **{passed}/{len(meta)} pass.** The
rest are included, flagged with `passed_verification = false`; filter them out
for a protocol-faithful setup:

| video_id | subject | class | face rate |
|---|---|---|---|
{failed_rows}

Missing entirely: `f3_s32_c10` and `f5_s49_c10` were not found in the
downloaded official archives.

Detection thresholds: {len(retried)} videos that were initially below the gate
were re-extracted with MediaPipe confidence thresholds lowered from 0.5 to
{retried['min_face_detection_confidence'].min():g} (`extractor_version = 2`); all
others use the 0.5 defaults. Thresholds are recorded per video in `metadata.csv`.

## Usage

```python
from datasets import load_dataset
frames = load_dataset("{rel['hf_repo']}", "frames", split="train")
videos = load_dataset("{rel['hf_repo']}", "videos", split="train")
```

or with pandas:

```python
import pandas as pd
meta = pd.read_csv("metadata.csv")
ok = meta[meta.passed_verification]
df = pd.concat(pd.read_parquet(f) for f in ok.file_name)
```

Evaluate **subject-wise** using the provided folds: never split frames or
windows randomly, since adjacent frames are near-duplicates.

## License and citation

These features are derived from UTA-RLDD, which is distributed for research
use. Redistribution terms for derived data are being confirmed with the
dataset authors; until then, use for non-commercial research only and cite the
original dataset:

```bibtex
@inproceedings{{ghoddoosian2019realistic,
  title={{A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection}},
  author={{Ghoddoosian, Reza and Galib, Marnim and Athitsos, Vassilis}},
  booktitle={{Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops}},
  year={{2019}}
}}
```
"""


def kaggle_metadata(owner: str, cfg: dict) -> dict:
    rel = cfg["release"]
    return {
        "title": rel["title"],
        "id": f"{owner}/{rel['slug']}",
        "subtitle": "Per-frame MediaPipe face features from UTA-RLDD drowsiness videos",
        "licenses": [{"name": "other"}],
        "keywords": ["health", "computer vision", "time series analysis"],
    }


@app.command()
def build(config: str = "configs/default.yaml"):
    """Write the release directory from manifest + feature parquets."""
    cfg = load_config(config)
    out = Path(cfg["release"]["dir"])
    (out / "features").mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(cfg["paths"]["manifest"], dtype={"subject_id": str})
    features_dir = Path(cfg["paths"]["features_dir"])

    rows, n_columns = [], 0
    for _, row in manifest.sort_values("video_id").iterrows():
        src = features_dir / f"{row['video_id']}.parquet"
        if not src.exists():
            print(f"  skip {row['video_id']}: no parquet ({row['status']})")
            continue
        table = pq.read_table(src)
        rows.append(metadata_row(row, table))
        table = with_identity_columns(table, row["video_id"], row["subject_id"],
                                      int(row["fold"]), int(row["class_label"]))
        n_columns = table.num_columns
        pq.write_table(table, out / "features" / src.name, compression="zstd")

    meta = pd.DataFrame(rows, columns=METADATA_COLUMNS)
    meta.to_csv(out / "metadata.csv", index=False)
    (out / "README.md").write_text(render_card(meta, n_columns, cfg))
    print(f"wrote {len(meta)} videos "
          f"({int(meta['passed_verification'].sum())} passed) → {out}")


@app.command("upload-hf")
def upload_hf(config: str = "configs/default.yaml"):
    """Create/update the Hugging Face dataset repo as PRIVATE."""
    from huggingface_hub import HfApi

    cfg = load_config(config)
    rel = cfg["release"]
    api = HfApi()
    api.create_repo(rel["hf_repo"], repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(rel["hf_repo"], repo_type="dataset").private:
        raise SystemExit(f"{rel['hf_repo']} already exists and is PUBLIC; refusing to push")
    api.upload_folder(folder_path=rel["dir"], repo_id=rel["hf_repo"],
                      repo_type="dataset", ignore_patterns=["dataset-metadata.json"],
                      commit_message="Upload UTA-RLDD face features")
    print(f"uploaded (private) → https://huggingface.co/datasets/{rel['hf_repo']}")


@app.command("upload-kaggle")
def upload_kaggle(owner: str = typer.Option(..., help="Kaggle username"),
                  config: str = "configs/default.yaml"):
    """Create (private) or version the Kaggle dataset. Needs ~/.kaggle/kaggle.json."""
    cfg = load_config(config)
    out = Path(cfg["release"]["dir"])
    (out / "dataset-metadata.json").write_text(
        json.dumps(kaggle_metadata(owner, cfg), indent=2))
    ref = f"{owner}/{cfg['release']['slug']}"
    exists = subprocess.run(["kaggle", "datasets", "status", ref],
                            capture_output=True).returncode == 0
    # no --public flag: `create` makes the dataset private by default
    cmd = (["kaggle", "datasets", "version", "-m", "Update features"] if exists
           else ["kaggle", "datasets", "create"])
    subprocess.run([*cmd, "-p", str(out), "-r", "zip"], check=True)
    print(f"uploaded (private) → https://www.kaggle.com/datasets/{ref}")


if __name__ == "__main__":
    app()
