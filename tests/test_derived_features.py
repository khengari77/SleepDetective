"""Tests for literature-derived window features."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.derived_features import (FEATURE_NAMES, IRIS_EYES, runs, subject_agreement,
                                  video_windows, window_features)

FPS = 10.0


def frames(n=600, closed=(), jaw_open=(), pitch=None, undetected=()):
    """Synthetic frames: eyes open (EAR 0.25) except `closed` index ranges
    (blink 0.7, EAR 0.10); still head unless `pitch` is given."""
    blink = np.full(n, 0.1)
    ear = np.full(n, 0.25)
    for a, b in closed:
        blink[a:b], ear[a:b] = 0.7, 0.10
    jaw = np.zeros(n)
    for a, b in jaw_open:
        jaw[a:b] = 0.6
    df = pd.DataFrame({
        "t_sec": np.arange(n) / FPS, "face_detected": True,
        "bs_eyeBlinkLeft": blink, "bs_eyeBlinkRight": blink, "bs_jawOpen": jaw,
        "ear_left": ear, "ear_right": ear, "mar": 0.05,
        "pitch": np.zeros(n) if pitch is None else pitch,
        "yaw": np.zeros(n), "roll": np.zeros(n),
    })
    for iris, a, b in IRIS_EYES:
        df[f"lm{a}_x"], df[f"lm{b}_x"] = -1.0, 1.0
        df[f"lm{a}_y"] = df[f"lm{b}_y"] = 0.0
        df[f"lm{iris}_x"], df[f"lm{iris}_y"] = 0.0, 0.0
    df.loc[list(undetected), df.columns.difference(["t_sec", "face_detected"])] = np.nan
    df.loc[list(undetected), "face_detected"] = False
    return df


def test_runs():
    starts, lengths = runs(np.array([0, 1, 1, 0, 1, 0, 0, 1], bool))
    assert starts.tolist() == [1, 4, 7] and lengths.tolist() == [2, 1, 1]


def test_blink_counts_durations_and_perclos():
    # three 2-frame blinks + one 8-frame (0.8 s) closure in a 60 s window
    f = window_features(frames(closed=[(50, 52), (200, 202), (350, 352), (500, 508)]), FPS)
    assert f["blink_rate_per_min"] == pytest.approx(4)
    assert f["long_closures_per_min"] == pytest.approx(1)
    assert f["very_long_closures_per_min"] == 0
    assert f["max_closure_sec"] == pytest.approx(0.8)
    assert f["blink_dur_mean_sec"] == pytest.approx((0.2 * 3 + 0.8) / 4)
    assert f["perclos"] == pytest.approx(14 / 600)


def test_rldd_amplitude_and_velocities():
    f = window_features(frames(closed=[(100, 102)]), FPS)
    # start/end EAR 0.25, bottom 0.10 → amplitude (0.25 - 0.2 + 0.25)/2
    assert f["blink_amplitude_mean"] == pytest.approx(0.15)
    assert f["eye_closing_velocity_mean"] == pytest.approx(0.15 / 0.1)   # 1 frame to bottom
    assert f["eye_opening_velocity_mean"] == pytest.approx(0.15 / 0.2)   # 2 frames back up
    assert f["avr_mean_sec"] == pytest.approx(0.1)
    assert f["open_ear_median"] == pytest.approx(0.25)


def test_closure_at_window_edge_is_counted_but_not_measured():
    f = window_features(frames(closed=[(0, 3)]), FPS)
    assert f["blink_rate_per_min"] == pytest.approx(1)
    assert np.isnan(f["blink_amplitude_mean"])


def test_undetected_frames_break_closures_and_shrink_time_base():
    f = window_features(frames(closed=[(100, 110)], undetected=range(104, 106)), FPS)
    assert f["long_closures_per_min"] == 0          # 10-frame closure split in two
    assert f["blink_rate_per_min"] == pytest.approx(2 / (598 / FPS / 60))


def test_nods_and_yawns():
    pitch = np.zeros(600)
    pitch[100:120] = 20        # 2 s excursion → nod
    pitch[300:400] = 20        # 10 s → posture change, not a nod
    f = window_features(frames(pitch=pitch, jaw_open=[(200, 230), (400, 405)]), FPS)
    assert f["nods_per_min"] == pytest.approx(1)
    assert f["yawns_per_min"] == pytest.approx(1)   # 0.5 s mouth-open is not a yawn


def test_empty_window_is_all_nan():
    f = window_features(frames(undetected=range(600)), FPS)
    assert set(f) == set(FEATURE_NAMES) and all(np.isnan(v) for v in f.values())


def test_video_windows_follow_config():
    cfg = {"extract": {"target_fps": FPS}, "windows": {"length_sec": 60.0, "stride_sec": 15.0}}
    rows = video_windows(frames(n=900), cfg)
    assert [r["t_start"] for r in rows] == [0.0, 15.0, 30.0]
    assert rows[0]["valid_frac"] == 1.0


def test_subject_agreement_uses_expected_direction():
    w = pd.DataFrame({"subject_id": ["a", "a", "b", "b"], "class_label": [0, 10, 0, 10]})
    for f in FEATURE_NAMES:
        w[f] = [0.0, 1.0, 0.0, 1.0]
    out = subject_agreement(w).set_index("feature")
    assert out.at["perclos", "agree"] == 2              # expected up, went up
    assert out.at["open_ear_median", "disagree"] == 2   # expected down, went up
