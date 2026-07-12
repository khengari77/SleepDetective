"""Phase 4 — baselines that define "added value" (PLAN.md §6).

B1: faithful v1 PERCLOS heuristic (modules/PERCLOS.py + HeadPose.py semantics):
    eye closed when EAR < calibration mean − 1·std; pose deviant when
    |pitch − calibration mean| > 10°; drowsiness = 0.5·closed_frac +
    0.5·deviant_frac; class thresholds fitted on training folds only.
B2: per-window summary statistics → gradient-boosted trees.
B3: majority class + per-subject nearest-centroid identity probe
    (quantifies how much identity alone predicts — leakage alarm).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestCentroid

from src.windows import load_features

POSE_DEV_DEG = 10.0          # v1 HeadPose threshold
YAWN_MAR_THRESHOLD = 0.6
BLINK_EAR_FACTOR = 0.8       # blink when EAR < factor × window median


# ---------------------------------------------------------------- B1 heuristic

def calibration_stats(manifest: pd.DataFrame, calibration_sec: float) -> pd.DataFrame:
    """Per-subject EAR mean/std + pitch mean from the alert video's first
    `calibration_sec` (mirrors v1 calibration mode)."""
    rows = {}
    alert = manifest[(manifest["status"] == "done") & (manifest["class_label"] == 0)]
    for _, row in alert.iterrows():
        df = load_features(row["feature_file"])
        seg = df[(df["t_sec"] < calibration_sec) & df["face_detected"]]
        if len(seg) < 10:
            continue
        ear = (seg["ear_left"] + seg["ear_right"]) / 2
        rows[row["subject_id"]] = {"ear_mean": ear.mean(), "ear_std": ear.std(ddof=0),
                                   "pitch_mean": seg["pitch"].mean()}
    return pd.DataFrame.from_dict(rows, orient="index")


def b1_window_scores(X: np.ndarray, mask: np.ndarray, feature_cols: list[str],
                     subjects: np.ndarray, calib: pd.DataFrame) -> np.ndarray:
    """Drowsiness score per window (0 = alert, 1 = drowsy), v1 semantics.

    X must be RAW (unnormalized) windows containing ear_left/ear_right/pitch.
    Windows of subjects without calibration stats get NaN (caller drops them).
    """
    i_l, i_r = feature_cols.index("ear_left"), feature_cols.index("ear_right")
    i_p = feature_cols.index("pitch")
    scores = np.full(len(X), np.nan, np.float32)
    for subject in np.unique(subjects):
        if subject not in calib.index:
            continue
        c = calib.loc[subject]
        sel = subjects == subject
        ear = (X[sel][:, :, i_l] + X[sel][:, :, i_r]) / 2
        closed = ear < (c["ear_mean"] - c["ear_std"])
        deviant = np.abs(X[sel][:, :, i_p] - c["pitch_mean"]) > POSE_DEV_DEG
        m = mask[sel]
        n_valid = np.maximum(m.sum(axis=1), 1)
        closed_frac = np.where(m, closed, False).sum(axis=1) / n_valid
        deviant_frac = np.where(m, deviant, False).sum(axis=1) / n_valid
        scores[sel] = 0.5 * closed_frac + 0.5 * deviant_frac
    return scores


def fit_score_thresholds(scores: np.ndarray, y: np.ndarray, n_classes: int,
                         grid_size: int = 40) -> tuple[float, ...]:
    """Ordinal cut points on a continuous score maximizing macro-F1 (train only)."""
    candidates = np.unique(np.quantile(scores, np.linspace(0.02, 0.98, grid_size)))
    best, best_f1 = None, -1.0
    for cuts in combinations(candidates, n_classes - 1):
        pred = np.digitize(scores, cuts)
        f1 = f1_score(y, pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best, best_f1 = cuts, f1
    return best


def apply_thresholds(scores: np.ndarray, cuts: tuple[float, ...]) -> np.ndarray:
    return np.digitize(scores, cuts)


class B1Perclos:
    """sklearn-style wrapper: 'fit' = fit ordinal cuts on train scores."""

    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        self.cuts_: tuple[float, ...] | None = None

    def fit(self, scores: np.ndarray, y: np.ndarray):
        self.cuts_ = fit_score_thresholds(scores, y, self.n_classes)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return apply_thresholds(scores, self.cuts_)


# ------------------------------------------------------- B2 window statistics

def _run_lengths(closed: np.ndarray) -> list[int]:
    runs, count = [], 0
    for c in closed:
        if c:
            count += 1
        elif count:
            runs.append(count)
            count = 0
    if count:
        runs.append(count)
    return runs


def window_stats(X: np.ndarray, mask: np.ndarray, feature_cols: list[str],
                 fps: float) -> pd.DataFrame:
    """Aggregate each window into B2's summary statistics (§6)."""
    idx = {name: feature_cols.index(name)
           for name in ("ear_left", "ear_right", "mar", "pitch")}
    rows = []
    for w in range(len(X)):
        m = mask[w]
        if m.sum() == 0:
            rows.append({})
            continue
        ear = (X[w][:, idx["ear_left"]] + X[w][:, idx["ear_right"]])[m] / 2
        mar = X[w][:, idx["mar"]][m]
        pitch = X[w][:, idx["pitch"]][m]
        row = {}
        for name, series in (("ear", ear), ("mar", mar), ("pitch", pitch)):
            row[f"{name}_mean"] = series.mean()
            row[f"{name}_std"] = series.std()
            for p in (10, 50, 90):
                row[f"{name}_p{p}"] = np.percentile(series, p)
        blink_thr = BLINK_EAR_FACTOR * np.median(ear)
        runs = _run_lengths(ear < blink_thr)
        window_sec = m.sum() / fps
        row["blink_rate"] = len(runs) / window_sec
        row["blink_dur_mean"] = np.mean(runs) / fps if runs else 0.0
        row["perclos"] = float((ear < blink_thr).mean())
        crossings = np.diff((mar > YAWN_MAR_THRESHOLD).astype(int)) == 1
        row["yawn_count"] = int(crossings.sum())
        row["valid_frac"] = float(m.mean())
        rows.append(row)
    return pd.DataFrame(rows).fillna(0.0)


class B2WindowStats:
    def __init__(self, seed: int = 0):
        self.clf = HistGradientBoostingClassifier(random_state=seed)

    def fit(self, stats: pd.DataFrame, y: np.ndarray):
        self.clf.fit(stats, y)
        return self

    def predict(self, stats: pd.DataFrame) -> np.ndarray:
        return self.clf.predict(stats)


# ------------------------------------------------------------------ B3 floors

class B3Majority:
    def fit(self, _X, y: np.ndarray):
        self.majority_ = np.bincount(y).argmax()
        return self

    def predict(self, X) -> np.ndarray:
        return np.full(len(X), self.majority_)


def identity_probe_accuracy(stats: pd.DataFrame, subjects: np.ndarray) -> float:
    """Nearest-centroid SUBJECT classification: fit on the first half of each
    subject's windows, predict the second half.

    Chance is 1/n_subjects; high values mean windows encode identity strongly —
    treat any model gain coincident with a high probe number as suspect
    (PLAN.md §9.5).
    """
    train_idx, test_idx = [], []
    for subject in np.unique(subjects):
        idx = np.flatnonzero(subjects == subject)
        if len(idx) < 2:
            continue
        half = len(idx) // 2
        train_idx += list(idx[:half])
        test_idx += list(idx[half:])
    probe = NearestCentroid()
    probe.fit(stats.iloc[train_idx], subjects[train_idx])
    return float((probe.predict(stats.iloc[test_idx]) == subjects[test_idx]).mean())
