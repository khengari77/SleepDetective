"""Tests for baselines + evaluation protocol (PLAN.md milestone 5)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines import (
    B1Perclos,
    B2WindowStats,
    B3Majority,
    b1_window_scores,
    filter_to_calibrated,
    fit_score_thresholds,
    identity_probe_accuracy,
    window_stats,
    _run_lengths,
)
from src.evaluate import compute_metrics, cross_validate, per_video_vote

FEATURE_COLS = ["ear_left", "ear_right", "mar", "pitch"]


def make_window(ear=0.3, mar=0.2, pitch=0.0, T=50):
    w = np.zeros((T, len(FEATURE_COLS)), np.float32)
    w[:, 0] = w[:, 1] = ear
    w[:, 2] = mar
    w[:, 3] = pitch
    return w


class TestB1:
    CALIB = pd.DataFrame({"ear_mean": {"01": 0.30}, "ear_std": {"01": 0.05},
                          "pitch_mean": {"01": 0.0}})

    def test_alert_window_scores_zero(self):
        X = np.stack([make_window(ear=0.30, pitch=0.0)])
        mask = np.ones((1, 50), bool)
        s = b1_window_scores(X, mask, FEATURE_COLS, np.array(["01"]), self.CALIB)
        assert s[0] == 0.0

    def test_closed_eyes_and_pitch_deviation_score_one(self):
        X = np.stack([make_window(ear=0.10, pitch=45.0)])  # closed + deviant
        mask = np.ones((1, 50), bool)
        s = b1_window_scores(X, mask, FEATURE_COLS, np.array(["01"]), self.CALIB)
        assert s[0] == 1.0

    def test_eyes_closed_half_the_time_scores_quarter(self):
        w = make_window(ear=0.30)
        w[:25, 0] = w[:25, 1] = 0.10           # closed half the window
        mask = np.ones((1, 50), bool)
        s = b1_window_scores(np.stack([w]), mask, FEATURE_COLS,
                             np.array(["01"]), self.CALIB)
        assert s[0] == pytest.approx(0.25)     # 0.5·0.5 + 0.5·0
    def test_unknown_subject_is_nan(self):
        X = np.stack([make_window()])
        s = b1_window_scores(X, np.ones((1, 50), bool), FEATURE_COLS,
                             np.array(["99"]), self.CALIB)
        assert np.isnan(s[0])

    def test_masked_steps_do_not_count(self):
        w = make_window(ear=0.30)
        w[:25, 0] = w[:25, 1] = 0.0            # invalid steps zeroed like make_windows
        mask = np.ones((1, 50), bool)
        mask[0, :25] = False
        s = b1_window_scores(np.stack([w]), mask, FEATURE_COLS,
                             np.array(["01"]), self.CALIB)
        assert s[0] == 0.0                     # zeros excluded via mask

    def test_threshold_fitting_separates_ordered_scores(self):
        scores = np.concatenate([np.full(30, 0.1), np.full(30, 0.5), np.full(30, 0.9)])
        y = np.repeat([0, 1, 2], 30)
        model = B1Perclos(n_classes=3).fit(scores, y)
        assert (model.predict(scores) == y).all()


class TestB2:
    def test_window_stats_columns_and_blinks(self):
        w = make_window(ear=0.30, mar=0.2)
        w[10:13, 0] = w[10:13, 1] = 0.05       # one 3-step blink
        w[30:32, 2] = 0.9                      # one yawn crossing
        stats = window_stats(np.stack([w]), np.ones((1, 50), bool),
                             FEATURE_COLS, fps=10.0)
        row = stats.iloc[0]
        assert row["yawn_count"] == 1
        assert row["blink_rate"] == pytest.approx(1 / 5.0)   # 1 blink in 5 s
        assert row["blink_dur_mean"] == pytest.approx(0.3)   # 3 steps @ 10 fps
        assert row["ear_p50"] == pytest.approx(0.30)

    def test_b2_learns_separable_stats(self):
        rng = np.random.default_rng(0)
        stats = pd.DataFrame({"a": np.r_[rng.normal(0, .1, 50), rng.normal(3, .1, 50)]})
        y = np.repeat([0, 1], 50)
        assert (B2WindowStats().fit(stats, y).predict(stats) == y).mean() > 0.95


class TestB3:
    def test_majority(self):
        y = np.array([0, 0, 0, 1])
        assert (B3Majority().fit(None, y).predict(np.zeros(2)) == 0).all()

    def test_identity_probe_high_when_identity_leaks(self):
        rng = np.random.default_rng(1)
        stats = pd.DataFrame({
            "f": np.r_[rng.normal(0, .05, 40), rng.normal(5, .05, 40)]})
        subjects = np.repeat(["01", "02"], 40)
        assert identity_probe_accuracy(stats, subjects) > 0.95

    def test_identity_probe_chance_when_no_identity_signal(self):
        rng = np.random.default_rng(2)
        stats = pd.DataFrame({"f": rng.normal(0, 1, 200)})
        subjects = np.repeat([f"{i:02d}" for i in range(10)], 20)
        assert identity_probe_accuracy(stats, subjects) < 0.35


class TestFilterToCalibrated:
    def test_drops_windows_of_uncalibrated_subjects(self):
        ds = {"X": np.arange(4).reshape(4, 1, 1).astype(np.float32),
              "mask": np.ones((4, 1), bool),
              "y": np.array([0, 1, 0, 1]),
              "subject": np.array(["01", "01", "99", "99"]),
              "fold": np.array([1, 1, 2, 2]),
              "video_id": np.array(["a", "a", "b", "b"]),
              "feature_cols": ["ear_left"]}
        calib = pd.DataFrame({"ear_mean": {"01": 0.3}, "ear_std": {"01": 0.05},
                              "pitch_mean": {"01": 0.0}})

        out = filter_to_calibrated(ds, calib)

        assert list(out["subject"]) == ["01", "01"]
        assert out["feature_cols"] == ["ear_left"]  # non-array fields pass through


def test_run_lengths():
    assert _run_lengths(np.array([1, 1, 0, 1, 0, 0, 1, 1, 1], bool)) == [2, 1, 3]


class TestEvaluate:
    def synth_ds(self, n_per_fold=30):
        folds = np.repeat([1, 2, 3, 4, 5], n_per_fold)
        video_id = np.array([f"v{f}_{i % 6}" for i, f in enumerate(folds)])
        # label is a property of the video (as in the real dataset)
        y = np.array([hash(v) % 3 for v in video_id])
        return {"y": y, "fold": folds, "video_id": video_id}

    def test_oracle_runner_scores_perfectly(self):
        ds = self.synth_ds()
        res = cross_validate(ds, lambda tr, te, d: d["y"][te], n_classes=3)
        assert res["overall"]["window"]["macro_f1"] == 1.0
        assert res["overall"]["video"]["macro_f1"] == 1.0
        assert set(res["per_fold"]) == {1, 2, 3, 4, 5}

    def test_per_video_vote_majority(self):
        y_true = np.array([1, 1, 1, 0])
        y_pred = np.array([1, 1, 0, 0])
        vids = np.array(["a", "a", "a", "b"])
        vt, vp = per_video_vote(y_true, y_pred, vids)
        assert list(vt) == [1, 0] and list(vp) == [1, 0]

    def test_metrics_shape(self):
        m = compute_metrics(np.array([0, 1, 2]), np.array([0, 1, 1]), 3)
        assert len(m["per_class_f1"]) == 3
        assert np.asarray(m["confusion"]).shape == (3, 3)
