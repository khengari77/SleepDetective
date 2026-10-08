"""Phase 4 — 5-fold training loop for the temporal model (PLAN.md §6–7).

Validation is carved out of the training folds BY SUBJECT (never by window),
early stopping on validation macro-F1.

Usage:
    uv run python -m src.train run --target 3class --normalized
    uv run python -m src.train matrix        # full {raw,norm}×{3class,binary}
"""
from __future__ import annotations

import numpy as np
import torch
import typer
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.evaluate import cross_validate, save_report
from src.manifest import load_config, load_video_manifest
from src.model import FeatureScaler, make_model
from src.windows import build_dataset

app = typer.Typer(add_completion=False)


@app.callback()
def _cli():
    """SleepDetective v2 temporal-model training."""


def split_val_subjects(train_subjects: np.ndarray, n_val: int, seed: int) -> set:
    rng = np.random.default_rng(seed)
    unique = np.unique(train_subjects)
    return set(rng.choice(unique, size=min(n_val, len(unique)), replace=False))


def _loader(X, mask, y, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(mask),
                       torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_one_fold(train_idx: np.ndarray, test_idx: np.ndarray, ds: dict,
                   cfg: dict, n_classes: int, desc: str = "fold") -> np.ndarray:
    t = cfg["train"]
    torch.manual_seed(t["seed"])

    val_subjects = split_val_subjects(ds["subject"][train_idx],
                                      t["val_subjects_per_fold"], t["seed"])
    is_val = np.isin(ds["subject"][train_idx], list(val_subjects))
    tr_idx, va_idx = train_idx[~is_val], train_idx[is_val]

    scaler = FeatureScaler().fit(ds["X"][tr_idx], ds["mask"][tr_idx])
    Xtr = scaler.transform(ds["X"][tr_idx], ds["mask"][tr_idx])
    Xva = scaler.transform(ds["X"][va_idx], ds["mask"][va_idx])
    Xte = scaler.transform(ds["X"][test_idx], ds["mask"][test_idx])

    model = make_model(Xtr.shape[-1], n_classes, cfg)
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"])
    loss_fn = torch.nn.CrossEntropyLoss()
    train_loader = _loader(Xtr, ds["mask"][tr_idx], ds["y"][tr_idx],
                           t["batch_size"], shuffle=True)

    best_f1, best_state, patience_left = -1.0, None, t["early_stop_patience"]
    bar = tqdm(range(t["max_epochs"]), desc=desc, leave=False, unit="epoch")
    for epoch in bar:
        model.train()
        for xb, mb, yb in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(xb, mb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(torch.from_numpy(Xva),
                             torch.from_numpy(ds["mask"][va_idx])).argmax(1).numpy()
        val_f1 = f1_score(ds["y"][va_idx], val_pred, average="macro",
                          zero_division=0)
        if val_f1 > best_f1:
            best_f1, patience_left = val_f1, t["early_stop_patience"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
        bar.set_postfix(val_f1=f"{val_f1:.3f}", best=f"{best_f1:.3f}",
                        patience=patience_left)
        if patience_left == 0:
            break
    bar.close()

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(Xte),
                     torch.from_numpy(ds["mask"][test_idx])).argmax(1).numpy()


def run_experiment(cfg: dict, *, normalized: bool, target: str,
                   feature_set: str = "core", report_name: str | None = None) -> dict:
    manifest = load_video_manifest(cfg)
    ds = build_dataset(manifest, cfg, normalized=normalized, target=target,
                       feature_set=feature_set)
    n_classes = 3 if target == "3class" else 2
    name = report_name or f"m1_gru_{'norm' if normalized else 'raw'}_{target}"
    print(f"=== {name}: {len(ds['X'])} windows, {len(np.unique(ds['fold']))} folds ===")
    runner = lambda tr, te, d: train_one_fold(
        tr, te, d, cfg, n_classes, desc=f"{name} fold{d['fold'][te][0]}")
    results = cross_validate(ds, runner, n_classes)
    path = save_report(name, results, {"model": cfg["model"], "train": cfg["train"],
                                       "windows": cfg["windows"],
                                       "normalized": normalized, "target": target},
                       cfg["paths"]["reports_dir"],
                       extras={"n_windows": len(ds["X"]),
                               "n_features": len(ds["feature_cols"])})
    print(f"wrote {path} — video macro-F1 "
          f"{results['overall']['video']['macro_f1']:.3f}")
    return results


@app.command()
def run(config: str = "configs/default.yaml",
        normalized: bool = typer.Option(False, "--normalized/--raw"),
        target: str = "3class"):
    run_experiment(load_config(config), normalized=normalized, target=target)


@app.command()
def matrix(config: str = "configs/default.yaml"):
    """The M1 quadrant of the results matrix: {raw,norm} × {3class,binary}."""
    cfg = load_config(config)
    for normalized in (False, True):
        for target in ("3class", "binary"):
            run_experiment(cfg, normalized=normalized, target=target)


if __name__ == "__main__":
    app()
