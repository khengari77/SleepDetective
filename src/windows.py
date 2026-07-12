"""Phase 3 — feature parquets → normalized windows + labels (PLAN.md §5).

Turns per-video feature files into (X, y, subject, fold) arrays, parameterized
by config — recomputed cheaply per experiment, never cached as a tensor blob.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.extract import BLENDSHAPE_NAMES, LANDMARK_SUBSET

EPS = 1e-6

CORE_FEATURES = ([f"bs_{n}" for n in BLENDSHAPE_NAMES]
                 + ["ear_left", "ear_right", "mar", "pitch", "yaw", "roll"])
LANDMARK_FEATURES = [f"lm{i}_{ax}" for i in LANDMARK_SUBSET for ax in "xyz"]

FEATURE_SETS = {
    "core": CORE_FEATURES,                       # M1's first-iteration input (§6)
    "core+landmarks": CORE_FEATURES + LANDMARK_FEATURES,
}


def load_features(path: str | Path) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()


def prepare_frames(df: pd.DataFrame, feature_cols: list[str],
                   ffill_max_steps: int) -> tuple[pd.DataFrame, np.ndarray]:
    """NaN-out undetected frames, forward-fill short gaps (≤ ffill_max_steps),
    return (features, valid_mask). Longer gaps stay NaN and are masked."""
    feat = df[feature_cols].copy()
    feat.loc[~df["face_detected"].to_numpy(), :] = np.nan
    filled = feat.ffill(limit=ffill_max_steps)
    valid = filled.notna().all(axis=1).to_numpy()
    return filled, valid


def make_windows(df: pd.DataFrame, feature_cols: list[str], cfg: dict,
                 skip_first_sec: float = 0.0):
    """Sliding windows over one video.

    Returns (X[n, T, F] float32, mask[n, T] bool, t_start[n] float32).
    Windows with more than `max_undetected_frac` invalid frames are dropped;
    remaining NaNs are zeroed (the mask tells the model which steps are real).
    """
    w = cfg["windows"]
    fps = cfg["extract"]["target_fps"]
    length = int(round(w["length_sec"] * fps))
    stride = int(round(w["stride_sec"] * fps))
    ffill_max = int(round(w["ffill_max_gap_sec"] * fps))

    filled, valid = prepare_frames(df, feature_cols, ffill_max)
    values = filled.to_numpy(dtype=np.float32)
    t_sec = df["t_sec"].to_numpy()

    xs, masks, starts = [], [], []
    for start in range(0, len(df) - length + 1, stride):
        if t_sec[start] < skip_first_sec:
            continue
        window_valid = valid[start:start + length]
        if 1.0 - window_valid.mean() > w["max_undetected_frac"]:
            continue
        window = values[start:start + length].copy()
        window[~window_valid] = 0.0
        window = np.nan_to_num(window, nan=0.0)
        xs.append(window)
        masks.append(window_valid)
        starts.append(t_sec[start])

    if not xs:
        n_feat = len(feature_cols)
        return (np.empty((0, length, n_feat), np.float32),
                np.empty((0, length), bool), np.empty(0, np.float32))
    return np.stack(xs), np.stack(masks), np.asarray(starts, np.float32)


def compute_baseline(df: pd.DataFrame, feature_cols: list[str],
                     calibration_sec: float) -> pd.DataFrame:
    """Per-feature median + IQR from the first `calibration_sec` of DETECTED
    frames of a subject's alert (class-0) video — deployment-honest stats (§5)."""
    seg = df[(df["t_sec"] < calibration_sec) & df["face_detected"]]
    if len(seg) < 10:
        raise ValueError(f"calibration segment too short: {len(seg)} detected frames")
    q1, q3 = seg[feature_cols].quantile(0.25), seg[feature_cols].quantile(0.75)
    return pd.DataFrame({"median": seg[feature_cols].median(), "iqr": q3 - q1})


def apply_baseline(df: pd.DataFrame, feature_cols: list[str],
                   baseline: pd.DataFrame) -> pd.DataFrame:
    """Robust z-score against the subject's own alert baseline."""
    out = df.copy()
    med = baseline["median"].reindex(feature_cols)
    iqr = baseline["iqr"].reindex(feature_cols).clip(lower=EPS)
    out[feature_cols] = (out[feature_cols] - med) / iqr
    return out


def build_dataset(manifest: pd.DataFrame, cfg: dict, *, normalized: bool,
                  target: str = "3class", feature_set: str = "core") -> dict:
    """All done videos → window arrays with per-window labels/subject/fold.

    target: "3class" (0/5/10 → 0/1/2) or "binary" (0 vs 10; class-5 dropped).
    normalized: apply per-subject baseline z-scoring (the headline ablation).
    Calibration-segment windows of each class-0 video are excluded either way,
    so the two variants train on identical window sets (no trivial leakage).
    """
    feature_cols = FEATURE_SETS[feature_set]
    label_map = {"3class": {0: 0, 5: 1, 10: 2}, "binary": {0: 0, 10: 1}}[target]
    calib_sec = cfg["windows"]["calibration_sec"]

    done = manifest[manifest["status"] == "done"]
    xs, masks, ys, subjects, folds, video_ids = [], [], [], [], [], []
    skipped_subjects = []

    for subject_id, group in done.groupby("subject_id"):
        alert = group[group["class_label"] == 0]
        baseline = None
        if normalized:
            if len(alert) == 0:
                skipped_subjects.append(subject_id)
                continue
            baseline = compute_baseline(
                load_features(alert.iloc[0]["feature_file"]), feature_cols, calib_sec)

        for _, row in group.iterrows():
            if int(row["class_label"]) not in label_map:
                continue
            df = load_features(row["feature_file"])
            if baseline is not None:
                df = apply_baseline(df, feature_cols, baseline)
            skip = calib_sec if int(row["class_label"]) == 0 else 0.0
            X, mask, _ = make_windows(df, feature_cols, cfg, skip_first_sec=skip)
            if len(X) == 0:
                continue
            xs.append(X)
            masks.append(mask)
            ys.append(np.full(len(X), label_map[int(row["class_label"])], np.int64))
            subjects += [subject_id] * len(X)
            folds += [int(row["fold"])] * len(X)
            video_ids += [row["video_id"]] * len(X)

    if skipped_subjects:
        print(f"warning: no alert video for subjects {skipped_subjects} — skipped")
    if not xs:
        raise ValueError("no windows produced — is the manifest empty?")
    return {
        "X": np.concatenate(xs),
        "mask": np.concatenate(masks),
        "y": np.concatenate(ys),
        "subject": np.asarray(subjects),
        "fold": np.asarray(folds, np.int64),
        "video_id": np.asarray(video_ids),
        "feature_cols": feature_cols,
    }
