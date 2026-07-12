"""Phase 5 — evaluation protocol (PLAN.md §7).

5-fold cross-validation on the official folds, per-window AND per-video
(majority vote), reported per fold and averaged — variance across folds is
large on RLDD; hiding it is dishonest.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    labels = list(range(n_classes))
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels,
                                   average="macro", zero_division=0)),
        "per_class_f1": [float(v) for v in f1_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0)],
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def per_video_vote(y_true: np.ndarray, y_pred: np.ndarray,
                   video_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Majority vote over each video's windows → (y_true_video, y_pred_video)."""
    trues, preds = [], []
    for vid in pd.unique(video_ids):
        sel = video_ids == vid
        trues.append(y_true[sel][0])
        preds.append(np.bincount(y_pred[sel]).argmax())
    return np.asarray(trues), np.asarray(preds)


Runner = Callable[[np.ndarray, np.ndarray, dict], np.ndarray]
# runner(train_idx, test_idx, dataset) -> y_pred for test_idx


def cross_validate(ds: dict, runner: Runner, n_classes: int) -> dict:
    """Rotate over the official folds present in the dataset."""
    folds = sorted(np.unique(ds["fold"]))
    per_fold = {}
    all_true_w, all_pred_w, all_true_v, all_pred_v = [], [], [], []
    for fold in folds:
        test_idx = np.flatnonzero(ds["fold"] == fold)
        train_idx = np.flatnonzero(ds["fold"] != fold)
        y_pred = np.asarray(runner(train_idx, test_idx, ds))
        y_true = ds["y"][test_idx]
        vt, vp = per_video_vote(y_true, y_pred, ds["video_id"][test_idx])
        per_fold[int(fold)] = {
            "window": compute_metrics(y_true, y_pred, n_classes),
            "video": compute_metrics(vt, vp, n_classes),
        }
        all_true_w.append(y_true); all_pred_w.append(y_pred)
        all_true_v.append(vt); all_pred_v.append(vp)

    overall = {
        "window": compute_metrics(np.concatenate(all_true_w),
                                  np.concatenate(all_pred_w), n_classes),
        "video": compute_metrics(np.concatenate(all_true_v),
                                 np.concatenate(all_pred_v), n_classes),
        "macro_f1_video_mean": float(np.mean(
            [f["video"]["macro_f1"] for f in per_fold.values()])),
        "macro_f1_video_std": float(np.std(
            [f["video"]["macro_f1"] for f in per_fold.values()])),
    }
    return {"per_fold": per_fold, "overall": overall}


def format_report(name: str, results: dict, config_snapshot: dict,
                  extras: dict | None = None) -> str:
    lines = [f"# {name}",
             f"_generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}_",
             "", "## Config snapshot", "```yaml"]
    import yaml
    lines += [yaml.safe_dump(config_snapshot, sort_keys=False).rstrip(), "```", ""]
    if extras:
        lines += ["## Extras"] + [f"- **{k}**: {v}" for k, v in extras.items()] + [""]

    lines += ["## Per-fold results", "",
              "| fold | window acc | window macro-F1 | video acc | video macro-F1 |",
              "|---|---|---|---|---|"]
    for fold, r in sorted(results["per_fold"].items()):
        lines.append(f"| {fold} | {r['window']['accuracy']:.3f} "
                     f"| {r['window']['macro_f1']:.3f} "
                     f"| {r['video']['accuracy']:.3f} "
                     f"| {r['video']['macro_f1']:.3f} |")
    ov = results["overall"]
    lines += ["",
              f"**Overall (pooled)**: window macro-F1 {ov['window']['macro_f1']:.3f}, "
              f"video macro-F1 {ov['video']['macro_f1']:.3f} "
              f"(per-fold mean {ov['macro_f1_video_mean']:.3f} "
              f"± {ov['macro_f1_video_std']:.3f})", "",
              "## Confusion matrix (per-video, pooled)", "```",
              str(np.asarray(ov["video"]["confusion"])), "```", ""]
    return "\n".join(lines)


def save_report(name: str, results: dict, config_snapshot: dict,
                reports_dir: str | Path = "reports",
                extras: dict | None = None) -> Path:
    path = Path(reports_dir) / f"{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_report(name, results, config_snapshot, extras))
    return path
