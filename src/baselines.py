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
import typer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestCentroid

from src.evaluate import cross_validate, save_report
from src.manifest import load_config, load_video_manifest
from src.windows import build_dataset, load_features

app = typer.Typer(add_completion=False)


@app.callback()
def _cli():
    """SleepDetective v2 baselines: B1 PERCLOS heuristic, B2 window-stats
    GBT, B3 majority/identity-probe floor (PLAN.md §6)."""


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


# --------------------------------------------------------------- Phase 5 CLI

def filter_to_calibrated(ds: dict, calib: pd.DataFrame) -> dict:
    """Drop windows whose subject has no B1 calibration (missing/failed alert
    video) — B1 cannot score them, so they're excluded rather than guessed."""
    keep = np.isin(ds["subject"], calib.index.to_numpy())
    return {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in ds.items()}


def _b1_runner(calib: pd.DataFrame, n_classes: int):
    def runner(train_idx, test_idx, ds):
        scores = b1_window_scores(ds["X"], ds["mask"], ds["feature_cols"],
                                  ds["subject"], calib)
        model = B1Perclos(n_classes).fit(scores[train_idx], ds["y"][train_idx])
        return model.predict(scores[test_idx])
    return runner


def _b2_runner(stats: pd.DataFrame, n_classes: int):
    def runner(train_idx, test_idx, ds):
        model = B2WindowStats().fit(stats.iloc[train_idx], ds["y"][train_idx])
        return model.predict(stats.iloc[test_idx])
    return runner


def _b3_runner(n_classes: int):
    def runner(train_idx, test_idx, ds):
        model = B3Majority().fit(train_idx, ds["y"][train_idx])
        return model.predict(test_idx)
    return runner


def run_b1(manifest: pd.DataFrame, cfg: dict, *, target: str) -> None:
    """B1 has one calibrated variant per target — no raw/normalized axis:
    its calibration IS the per-subject baseline, computed the v1 way
    (mean/std, not build_dataset's median/IQR), so re-scoring it against
    already baseline-normalized windows would just re-apply a second,
    incompatible normalization on top."""
    n_classes = 3 if target == "3class" else 2
    calib = calibration_stats(manifest, cfg["windows"]["calibration_sec"])
    ds_raw = build_dataset(manifest, cfg, normalized=False, target=target)
    ds = filter_to_calibrated(ds_raw, calib)
    excluded = sorted(set(ds_raw["subject"]) - set(ds["subject"]))

    results = cross_validate(ds, _b1_runner(calib, n_classes), n_classes)
    save_report(
        f"b1_perclos_{target}", results,
        {"windows": cfg["windows"], "target": target,
         "note": "single calibrated variant; raw/normalized axis N/A for B1"},
        cfg["paths"]["reports_dir"],
        extras={"n_windows": len(ds["X"]),
               "subjects_excluded_no_calibration": excluded})


def run_b2_b3(manifest: pd.DataFrame, cfg: dict, *, normalized: bool, target: str) -> None:
    n_classes = 3 if target == "3class" else 2
    variant = "norm" if normalized else "raw"
    ds = build_dataset(manifest, cfg, normalized=normalized, target=target)
    stats = window_stats(ds["X"], ds["mask"], ds["feature_cols"], cfg["extract"]["target_fps"])
    probe_acc = identity_probe_accuracy(stats, ds["subject"])  # PLAN §9.5: in every report

    b2_results = cross_validate(ds, _b2_runner(stats, n_classes), n_classes)
    save_report(f"b2_windowstats_{variant}_{target}", b2_results,
               {"windows": cfg["windows"], "normalized": normalized, "target": target},
               cfg["paths"]["reports_dir"],
               extras={"n_windows": len(ds["X"]), "identity_probe_accuracy": probe_acc})

    b3_results = cross_validate(ds, _b3_runner(n_classes), n_classes)
    save_report(f"b3_floor_{variant}_{target}", b3_results,
               {"windows": cfg["windows"], "normalized": normalized, "target": target},
               cfg["paths"]["reports_dir"],
               extras={"n_windows": len(ds["X"]), "identity_probe_accuracy": probe_acc})


@app.command()
def matrix(config: str = "configs/default.yaml"):
    """The B1/B2/B3 quadrants of the results matrix (PLAN §7)."""
    cfg = load_config(config)
    manifest = load_video_manifest(cfg)
    for target in ("3class", "binary"):
        run_b1(manifest, cfg, target=target)
        for normalized in (False, True):
            run_b2_b3(manifest, cfg, normalized=normalized, target=target)
    print(f"wrote baseline reports to {cfg['paths']['reports_dir']}")


if __name__ == "__main__":
    app()
