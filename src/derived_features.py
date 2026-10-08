"""Window-level drowsiness features from per-frame parquets (literature-derived).

Twenty signals chosen by strength of published evidence, all computable from
the extracted columns (no video needed). Computed on the same sliding windows
as src/windows.py and left UNnormalized — callers can z-score against the
subject's alert-video calibration segment (PLAN.md §5) themselves.

Closure events use the MediaPipe eyeBlink blendshapes (a calibrated 0–1
closure score, robust to camera angle); blink amplitude/velocity follow the
UTA-RLDD paper's EAR definitions (Ghoddoosian et al. 2019, eqs. 3–4).

At 10 fps a blink spans 1–4 frames: durations are quantized to 100 ms, short
blinks are under-counted, and velocity-based features (velocities, AVR) are
coarse. They are kept because they track the published signals, not as
precise oculometrics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# MediaPipe eyeBlink saturates around 0.6–0.8 on a fully closed eye (EAR is
# already at its closed plateau from 0.5 up; <0.5% of frames exceed 0.8), so
# P70/P80 cannot be resolved from it: ≥0.5 is treated as "closed" and
# eye_closure_mean keeps the partial-closure information instead.
BLINK_CLOSED = 0.5
LONG_CLOSURE_SEC = 0.5       # microsleep threshold (closures > 500 ms)
VERY_LONG_CLOSURE_SEC = 1.0
NOD_DEG = 10.0               # pitch excursion from the window median
NOD_MAX_SEC = 3.0            # ...that returns within this time counts as a nod
YAWN_JAW_OPEN = 0.3          # ~96th percentile of jawOpen; sustained ≥2 s excludes speech
YAWN_MIN_SEC = 2.0

# MediaPipe iris centers and the eye corners they sit between
IRIS_EYES = ((468, 33, 133), (473, 362, 263))

FEATURE_NAMES = [
    # eyelid closure (PERCLOS / long closures: strongest, most replicated)
    "perclos", "eye_closure_mean",
    "long_closures_per_min", "very_long_closures_per_min", "max_closure_sec",
    # blink dynamics
    "blink_dur_mean_sec", "blink_rate_per_min", "ibi_cv",
    "blink_amplitude_mean", "eye_opening_velocity_mean",
    "eye_closing_velocity_mean", "avr_mean_sec",
    # resting eyelid opening; mouth-over-eye ratio (MAR/EAR, arXiv:2208.08091)
    "open_ear_median", "moe_mean",
    # head (Vural et al. 2007: roll second only to eye closure)
    "pitch_std", "roll_std", "head_motion_deg_per_sec", "nods_per_min",
    # mouth / gaze
    "yawns_per_min", "gaze_dispersion",
]

# (expected direction when drowsy, description) — direction per the literature
FEATURE_DOCS = {
    "perclos": (+1, "fraction of frames with eyes closed (eyeBlink ≥ 0.5)"),
    "eye_closure_mean": (+1, "mean eyeBlink score; keeps partial closures"),
    "long_closures_per_min": (+1, "closures ≥ 0.5 s (microsleep threshold) per minute"),
    "very_long_closures_per_min": (+1, "closures ≥ 1 s per minute"),
    "max_closure_sec": (+1, "longest closure in the window"),
    "blink_dur_mean_sec": (+1, "mean closure duration (100 ms resolution)"),
    "blink_rate_per_min": (+1, "closure events per minute (short blinks under-counted at 10 fps)"),
    "ibi_cv": (+1, "coefficient of variation of inter-blink intervals"),
    "blink_amplitude_mean": (-1, "UTA-RLDD blink amplitude on EAR (paper eq. 3)"),
    "eye_opening_velocity_mean": (-1, "UTA-RLDD eye-opening velocity, EAR/s (paper eq. 4)"),
    "eye_closing_velocity_mean": (-1, "eye-closing velocity, EAR/s"),
    "avr_mean_sec": (+1, "Johns amplitude-velocity ratio (amplitude / closing velocity)"),
    "open_ear_median": (-1, "median EAR while eyes open (lid droop)"),
    "moe_mean": (+1, "mean mouth-over-eye ratio, MAR / EAR"),
    "pitch_std": (+1, "head pitch standard deviation, degrees"),
    "roll_std": (+1, "head roll standard deviation, degrees"),
    "head_motion_deg_per_sec": (+1, "mean angular head speed"),
    "nods_per_min": (+1, "pitch excursions > 10° from window median lasting ≤ 3 s"),
    "yawns_per_min": (+1, "jawOpen > 0.3 sustained ≥ 2 s, per minute"),
    "gaze_dispersion": (+1, "spread of iris position within the eye (eye widths)"),
}
assert list(FEATURE_DOCS) == FEATURE_NAMES

WINDOW_COLUMNS = (["video_id", "subject_id", "fold", "class_label",
                   "passed_verification", "t_start", "t_end", "valid_frac"]
                  + FEATURE_NAMES)


def runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(start indices, lengths) of consecutive True runs."""
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    starts = np.flatnonzero(edges == 1)
    return starts, np.flatnonzero(edges == -1) - starts


def _per_min(count: int, n_valid: int, fps: float) -> float:
    return count / (n_valid / fps / 60) if n_valid else np.nan


def _mean(values) -> float:
    return float(np.mean(values)) if len(values) else np.nan


def blink_dynamics(ear: np.ndarray, starts: np.ndarray, lengths: np.ndarray,
                   fps: float) -> dict:
    """UTA-RLDD amplitude (eq. 3), opening velocity (eq. 4), plus closing
    velocity and Johns' amplitude-velocity ratio, averaged over blinks whose
    surrounding open frames are both observed."""
    amps, open_v, close_v, avr = [], [], [], []
    for s, n in zip(starts, lengths):
        start, end = s - 1, s + n
        if start < 0 or end >= len(ear) or np.isnan(ear[start]) or np.isnan(ear[end]):
            continue
        seg = ear[s:end]
        if np.isnan(seg).all():
            continue
        bottom = s + int(np.nanargmin(seg))
        amp = (ear[start] - 2 * ear[bottom] + ear[end]) / 2
        v_close = (ear[start] - ear[bottom]) / ((bottom - start) / fps)
        amps.append(amp)
        open_v.append((ear[end] - ear[bottom]) / ((end - bottom) / fps))
        close_v.append(v_close)
        if v_close > 0:
            avr.append(amp / v_close)
    return {"blink_amplitude_mean": _mean(amps),
            "eye_opening_velocity_mean": _mean(open_v),
            "eye_closing_velocity_mean": _mean(close_v),
            "avr_mean_sec": _mean(avr)}


def gaze_offsets(w: pd.DataFrame) -> np.ndarray:
    """Iris center relative to its eye-corner midpoint, in eye widths.
    Shape (frames, 2 eyes, 2 axes)."""
    out = []
    for iris, a, b in IRIS_EYES:
        mid_x = (w[f"lm{a}_x"] + w[f"lm{b}_x"]) / 2
        mid_y = (w[f"lm{a}_y"] + w[f"lm{b}_y"]) / 2
        width = np.hypot(w[f"lm{a}_x"] - w[f"lm{b}_x"], w[f"lm{a}_y"] - w[f"lm{b}_y"])
        out.append(np.stack([(w[f"lm{iris}_x"] - mid_x) / width,
                             (w[f"lm{iris}_y"] - mid_y) / width], axis=1))
    return np.stack(out, axis=1)


def window_features(w: pd.DataFrame, fps: float) -> dict:
    """All FEATURE_NAMES for one window of frames (undetected rows are NaN)."""
    det = w["face_detected"].to_numpy(bool)
    n_valid = int(det.sum())
    if n_valid == 0:
        return dict.fromkeys(FEATURE_NAMES, np.nan)

    closure = ((w["bs_eyeBlinkLeft"] + w["bs_eyeBlinkRight"]) / 2).to_numpy()
    ear = ((w["ear_left"] + w["ear_right"]) / 2).to_numpy()
    closed = np.nan_to_num(closure, nan=0.0) > BLINK_CLOSED   # gaps break events
    starts, lengths = runs(closed)
    durations = lengths / fps

    f = {
        "perclos": float(np.mean(closure[det] >= BLINK_CLOSED)),
        "eye_closure_mean": float(np.mean(closure[det])),
        "long_closures_per_min": _per_min(int((durations >= LONG_CLOSURE_SEC).sum()), n_valid, fps),
        "very_long_closures_per_min": _per_min(int((durations >= VERY_LONG_CLOSURE_SEC).sum()), n_valid, fps),
        "max_closure_sec": float(durations.max()) if len(durations) else 0.0,
        "blink_dur_mean_sec": _mean(durations),
        "blink_rate_per_min": _per_min(len(starts), n_valid, fps),
    }
    ibi = np.diff(starts) / fps
    f["ibi_cv"] = float(ibi.std() / ibi.mean()) if len(ibi) >= 2 else np.nan
    f.update(blink_dynamics(ear, starts, lengths, fps))

    open_frames = det & (np.nan_to_num(closure, nan=1.0) < BLINK_CLOSED)
    f["open_ear_median"] = float(np.median(ear[open_frames])) if open_frames.any() else np.nan
    f["moe_mean"] = float(np.nanmean(w["mar"].to_numpy() / ear))

    pitch, yaw, roll = (w[c].to_numpy() for c in ("pitch", "yaw", "roll"))
    f["pitch_std"] = float(np.nanstd(pitch))
    f["roll_std"] = float(np.nanstd(roll))
    step = np.sqrt(np.diff(pitch) ** 2 + np.diff(yaw) ** 2 + np.diff(roll) ** 2)
    f["head_motion_deg_per_sec"] = float(np.nanmean(step) * fps) if np.isfinite(step).any() else np.nan
    # sign-agnostic: a pitch excursion away from the window median that returns
    excursion = np.abs(pitch - np.nanmedian(pitch)) > NOD_DEG
    _, nod_lengths = runs(excursion)
    f["nods_per_min"] = _per_min(int((nod_lengths / fps <= NOD_MAX_SEC).sum()), n_valid, fps)

    _, yawn_lengths = runs(np.nan_to_num(w["bs_jawOpen"].to_numpy(), nan=0.0) > YAWN_JAW_OPEN)
    f["yawns_per_min"] = _per_min(int((yawn_lengths / fps >= YAWN_MIN_SEC).sum()), n_valid, fps)

    gaze = gaze_offsets(w)  # (frames, eyes, axes)
    f["gaze_dispersion"] = float(np.nanmean(np.sqrt(np.nanvar(gaze[:, :, 0], axis=0)
                                                    + np.nanvar(gaze[:, :, 1], axis=0))))
    return f


def video_windows(df: pd.DataFrame, cfg: dict) -> list[dict]:
    """Feature rows for every sliding window of one video (no window is
    dropped; `valid_frac` lets callers apply their own detection filter)."""
    fps = cfg["extract"]["target_fps"]
    length = int(round(cfg["windows"]["length_sec"] * fps))
    stride = int(round(cfg["windows"]["stride_sec"] * fps))
    t_sec = df["t_sec"].to_numpy()
    rows = []
    for start in range(0, len(df) - length + 1, stride):
        w = df.iloc[start:start + length]
        rows.append({"t_start": float(t_sec[start]),
                     "t_end": float(t_sec[start + length - 1]),
                     "valid_frac": float(w["face_detected"].mean()),
                     **window_features(w, fps)})
    return rows


def subject_agreement(windows: pd.DataFrame) -> pd.DataFrame:
    """Per feature: how many subjects' drowsy video (mean over windows) moves
    in the literature's expected direction vs their own alert video."""
    v = windows.groupby(["subject_id", "class_label"])[FEATURE_NAMES].mean()
    rows = []
    for f, (sign, _) in FEATURE_DOCS.items():
        pair = v[f].unstack().reindex(columns=[0, 10]).dropna()
        d = np.sign(pair[10] - pair[0]) * sign
        rows.append({"feature": f, "agree": int((d > 0).sum()),
                     "disagree": int((d < 0).sum()), "tie": int((d == 0).sum())})
    return pd.DataFrame(rows)
