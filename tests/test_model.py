"""Tests for the temporal model + training loop (PLAN.md milestone 6)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import DrowsinessGRU, FeatureScaler, make_model
from src.train import split_val_subjects, train_one_fold

CFG = {
    "model": {"kind": "gru", "hidden_size": 64, "num_layers": 1, "dropout": 0.2},
    "train": {"batch_size": 16, "lr": 1e-2, "max_epochs": 30,
              "early_stop_patience": 5, "val_subjects_per_fold": 2, "seed": 0},
}


class TestModel:
    def test_under_100k_params_at_full_feature_width(self):
        model = make_model(58, 3, CFG)  # core feature set width
        assert model.n_parameters() < 100_000

    def test_forward_shape(self):
        model = DrowsinessGRU(10, 3)
        x = torch.zeros(4, 50, 10)
        mask = torch.ones(4, 50, dtype=torch.bool)
        assert model(x, mask).shape == (4, 3)

    def test_mask_channel_distinguishes_missing_from_zero(self):
        model = DrowsinessGRU(4, 2)
        model.eval()
        x = torch.zeros(1, 20, 4)
        with torch.no_grad():
            out_valid = model(x, torch.ones(1, 20, dtype=torch.bool))
            out_masked = model(x, torch.zeros(1, 20, dtype=torch.bool))
        assert not torch.allclose(out_valid, out_masked)

    def test_rejects_unknown_kind(self):
        bad = {"model": {**CFG["model"], "kind": "transformer"}}
        with pytest.raises(NotImplementedError):
            make_model(10, 3, bad)


class TestScaler:
    def test_fits_on_valid_steps_only(self):
        X = np.ones((2, 10, 1), np.float32)
        X[0, :5] = 100.0                      # invalid garbage
        mask = np.ones((2, 10), bool)
        mask[0, :5] = False
        scaler = FeatureScaler().fit(X, mask)
        assert scaler.mean_[0] == pytest.approx(1.0)
        out = scaler.transform(X, mask)
        assert (out[~mask] == 0.0).all()      # masked steps stay zero


class TestTraining:
    def test_val_split_is_subject_wise(self):
        subjects = np.repeat(["01", "02", "03", "04"], 10)
        val = split_val_subjects(subjects, 2, seed=0)
        assert len(val) == 2
        is_val = np.isin(subjects, list(val))
        assert set(subjects[is_val]) & set(subjects[~is_val]) == set()

    def test_learns_separable_synthetic_windows(self):
        rng = np.random.default_rng(0)
        n, T, F = 120, 30, 6
        y = np.repeat([0, 1], n // 2).astype(np.int64)
        X = rng.normal(0, 0.3, (n, T, F)).astype(np.float32)
        X[y == 1] += 2.0
        mask = np.ones((n, T), bool)
        subjects = np.array([f"{i % 8:02d}" for i in range(n)])
        ds = {"X": X, "mask": mask, "y": y, "subject": subjects}
        order = rng.permutation(n)
        train_idx, test_idx = order[:90], order[90:]
        y_pred = train_one_fold(train_idx, test_idx, ds, CFG, n_classes=2)
        assert (y_pred == y[test_idx]).mean() > 0.9
