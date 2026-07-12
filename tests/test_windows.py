"""Unit tests for windowing + baseline normalization (PLAN.md milestone 4)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.windows import (
    CORE_FEATURES,
    apply_baseline,
    build_dataset,
    compute_baseline,
    make_windows,
)

CFG = {
    "extract": {"target_fps": 10.0},
    "windows": {"length_sec": 2.0, "stride_sec": 1.0, "max_undetected_frac": 0.30,
                "ffill_max_gap_sec": 0.5, "calibration_sec": 3.0},
}
COLS = ["f1", "f2"]


def synth_df(n=100, detected=None):
    detected = np.ones(n, bool) if detected is None else detected
    return pd.DataFrame({
        "t_sec": np.arange(n) / 10.0,
        "face_detected": detected,
        "f1": np.arange(n, dtype=float),
        "f2": np.full(n, 2.0),
    })


class TestMakeWindows:
    def test_counts_and_shapes(self):
        X, mask, t0 = make_windows(synth_df(100), COLS, CFG)
        assert X.shape == (9, 20, 2)          # starts 0,10,...,80
        assert mask.shape == (9, 20)
        assert mask.all()
        assert t0[0] == 0.0 and t0[-1] == pytest.approx(8.0)

    def test_short_gap_forward_filled(self):
        det = np.ones(100, bool)
        det[30:34] = False                    # 4 steps ≤ 5-step ffill limit
        X, mask, _ = make_windows(synth_df(100, det), COLS, CFG)
        assert X.shape[0] == 9                # nothing dropped
        assert mask.all()                     # filled frames count as valid
        w3 = X[3]                             # window starting at idx 30
        assert w3[0, 0] == 29.0               # ffilled from last detected frame

    def test_long_gap_masked_and_window_dropped(self):
        det = np.ones(100, bool)
        det[30:45] = False                    # 15-step gap: 5 ffilled, 10 invalid
        X, mask, t0 = make_windows(synth_df(100, det), COLS, CFG)
        # window @3.0s (rows 30–49): 10/20 invalid = 50% > 30% → dropped
        assert 3.0 not in t0
        # window @2.0s (rows 20–39): 5/20 invalid = 25% ≤ 30% → kept, masked
        w2 = list(t0).index(2.0)
        assert mask[w2].sum() == 15
        assert (X[w2][~mask[w2]] == 0.0).all()  # invalid steps zeroed

    def test_skip_first_sec_excludes_calibration_windows(self):
        X, _, t0 = make_windows(synth_df(100), COLS, CFG, skip_first_sec=3.0)
        assert (t0 >= 3.0).all()

    def test_all_undetected_returns_empty(self):
        X, mask, t0 = make_windows(synth_df(100, np.zeros(100, bool)), COLS, CFG)
        assert len(X) == 0


class TestBaseline:
    def test_compute_uses_only_calibration_detected_frames(self):
        df = synth_df(100)
        df.loc[df.index[:30], "f1"] = 5.0     # calibration segment (t<3s)
        df.loc[df.index[30:], "f1"] = 999.0   # after calibration — ignored
        base = compute_baseline(df, COLS, calibration_sec=3.0)
        assert base.loc["f1", "median"] == 5.0
        assert base.loc["f2", "iqr"] == 0.0

    def test_apply_is_robust_zscore(self):
        df = synth_df(10)
        base = pd.DataFrame({"median": {"f1": 4.0, "f2": 2.0},
                             "iqr": {"f1": 2.0, "f2": 0.0}})
        out = apply_baseline(df, COLS, base)
        assert out["f1"].iloc[6] == pytest.approx((6.0 - 4.0) / 2.0)
        assert np.isfinite(out["f2"]).all()   # zero IQR clipped, no inf

    def test_too_short_calibration_raises(self):
        with pytest.raises(ValueError, match="calibration segment too short"):
            compute_baseline(synth_df(5), COLS, calibration_sec=0.3)


@pytest.fixture
def tiny_dataset(tmp_path):
    """2 subjects × 3 videos of 400 frames (40 s) with the real core schema."""
    rng = np.random.default_rng(7)
    cfg = {
        "extract": {"target_fps": 10.0},
        "windows": {"length_sec": 10.0, "stride_sec": 5.0, "max_undetected_frac": 0.30,
                    "ffill_max_gap_sec": 0.5, "calibration_sec": 5.0},
    }
    rows = []
    for subj, fold in (("01", 1), ("13", 2)):
        for cls in (0, 5, 10):
            n = 400
            df = pd.DataFrame(
                rng.normal(0, 1, (n, len(CORE_FEATURES))).astype(np.float32),
                columns=CORE_FEATURES)
            df.insert(0, "t_sec", np.arange(n) / 10.0)
            df.insert(1, "face_detected", True)
            path = tmp_path / f"f{fold}_s{subj}_c{cls:02d}.parquet"
            df.to_parquet(path)
            rows.append({"video_id": path.stem, "subject_id": subj,
                         "class_label": cls, "fold": fold, "status": "done",
                         "feature_file": str(path)})
    return pd.DataFrame(rows), cfg


class TestBuildDataset:
    def test_3class_labels_and_folds(self, tiny_dataset):
        manifest, cfg = tiny_dataset
        ds = build_dataset(manifest, cfg, normalized=False, target="3class")
        assert set(ds["y"]) == {0, 1, 2}
        assert set(ds["fold"]) == {1, 2}
        assert len(ds["X"]) == len(ds["y"]) == len(ds["subject"])

    def test_binary_drops_class5(self, tiny_dataset):
        manifest, cfg = tiny_dataset
        ds = build_dataset(manifest, cfg, normalized=False, target="binary")
        assert set(ds["y"]) == {0, 1}
        assert not any("c05" in v for v in ds["video_id"])

    def test_calibration_windows_excluded_from_class0_only(self, tiny_dataset):
        manifest, cfg = tiny_dataset
        ds = build_dataset(manifest, cfg, normalized=False, target="3class")
        per_video = pd.Series(ds["video_id"]).value_counts()
        # 40s video, 10s windows, 5s stride → 7 windows; class-0 skips t<5s → 6
        assert per_video["f1_s01_c05"] == 7
        assert per_video["f1_s01_c00"] == 6

    def test_raw_and_normalized_have_identical_window_sets(self, tiny_dataset):
        manifest, cfg = tiny_dataset
        raw = build_dataset(manifest, cfg, normalized=False, target="3class")
        norm = build_dataset(manifest, cfg, normalized=True, target="3class")
        assert len(raw["X"]) == len(norm["X"])
        assert (raw["video_id"] == norm["video_id"]).all()
        assert np.isfinite(norm["X"]).all()
        assert not np.allclose(raw["X"], norm["X"])  # normalization did something
