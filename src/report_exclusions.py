"""Write reports/excluded_videos.md listing every video excluded from training.

Per PLAN.md §9.3: no-face videos are kept as parquets, excluded from training,
and listed in the final report. Also records subjects without a done class-0
video (skipped in normalized experiments) and any unregistered expected ids.

Usage:
    uv run python -m src.report_exclusions
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import typer

from src.extract import EXTRACTOR_VERSION
from src.manifest import load_config, load_video_manifest
from src.verify import verify_feature_file

app = typer.Typer(add_completion=False)


@app.callback()
def _cli():
    """Generate the exclusion report."""


def expected_video_ids(cfg: dict) -> set[str]:
    """All 60×3 ids implied by the dataset config (fold f holds subjects
    (f-1)*subjects_per_fold+1 .. f*subjects_per_fold)."""
    ds = cfg["dataset"]
    per_fold = ds["subjects_per_fold"]
    return {
        f"f{fold}_s{subj:02d}_c{cls:02d}"
        for fold in range(1, ds["n_folds"] + 1)
        for subj in range((fold - 1) * per_fold + 1, fold * per_fold + 1)
        for cls in ds["class_labels"]
    }


def build_report(manifest: pd.DataFrame, cfg: dict) -> str:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ex = cfg["extract"]
    gate = cfg["verify"]["min_face_detected_rate"]
    lines = [
        "# Excluded videos",
        f"_generated {now}_",
        "",
        f"- extractor_version: {EXTRACTOR_VERSION}",
        f"- min_face_detection_confidence: {ex.get('min_face_detection_confidence', 0.5)}",
        f"- min_face_presence_confidence: {ex.get('min_face_presence_confidence', 0.5)}",
        f"- min_tracking_confidence: {ex.get('min_tracking_confidence', 0.5)}",
        f"- verification gate: face_detected rate >= {gate}",
        "",
    ]

    failed = manifest[manifest["status"] == "failed"].copy()
    lines.append("## Still-failed videos")
    lines.append("")
    if len(failed) == 0:
        lines.append("None — all registered videos passed verification.")
    else:
        lines.append("| video_id | subject | class | fold | archive | face rate | attempts | reason |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, row in failed.sort_values("video_id").iterrows():
            rate, reason = "?", str(row.get("error_msg", "")) or "?"
            ff = row.get("feature_file", "")
            if ff and Path(ff).exists():
                _, vstats = verify_feature_file(ff, cfg)
                rate = f"{vstats['face_detected_rate']:.2f}"
                if vstats["problems"]:
                    reason = "; ".join(vstats["problems"])
            lines.append(
                f"| {row['video_id']} | {row['subject_id']} | {row['class_label']} "
                f"| {row['fold']} | {row.get('archive_id', '?')} | {rate} "
                f"| {int(row['attempts'])} | {reason} |")
    lines.append("")

    registered = set(manifest["video_id"])
    missing = sorted(expected_video_ids(cfg) - registered)
    lines.append("## Unregistered / absent videos")
    lines.append("")
    if not missing:
        lines.append("None — all expected ids are registered.")
    else:
        lines.append("| video_id | expected source | status |")
        lines.append("|---|---|---|")
        for vid in missing:
            # f{fold}_s{subj}_c{cls} → Fold{fold}_part?/{subj}/{cls}.mov
            parts = vid.split("_")
            fold = parts[0][1:]
            subj = parts[1][1:]
            cls = str(int(parts[2][1:]))
            lines.append(f"| {vid} | Fold{fold}_part?/{subj}/{cls}.mov | absent from manifest |")
    lines.append("")

    done = manifest[manifest["status"] == "done"]
    no_alert = []
    for sid, group in manifest.groupby("subject_id"):
        done_alert = group[(group["status"] == "done") & (group["class_label"] == 0)]
        if len(done_alert) == 0:
            no_alert.append(sid)
    lines.append("## Subjects without a done class-0 (calibration) video")
    lines.append("")
    if not no_alert:
        lines.append("None — every subject has a usable alert video for normalization.")
    else:
        lines.append(
            f"These subjects are skipped entirely in normalized experiments "
            f"(`build_dataset` requires a class-0 video for baseline stats): "
            f"{', '.join(sorted(no_alert))}.")
    lines.append("")

    lines.append("## Impact")
    lines.append("")
    n_failed = len(failed)
    n_missing = len(missing)
    lines.append(
        f"- {n_failed} video(s) failed verification and are excluded from training.")
    lines.append(
        f"- {n_missing} expected video(s) are not in the manifest.")
    lines.append(
        f"- {len(no_alert)} subject(s) lack calibration and drop out of normalized runs.")
    lines.append(
        f"- Done videos available for training: {len(done)}.")
    lines.append("")
    return "\n".join(lines)


@app.command()
def write(config: str = "configs/default.yaml"):
    cfg = load_config(config)
    manifest = load_video_manifest(cfg)
    manifest["error_msg"] = manifest["error_msg"].fillna("")
    manifest["feature_file"] = manifest["feature_file"].fillna("")
    report = build_report(manifest, cfg)
    out = Path(cfg["paths"]["reports_dir"]) / "excluded_videos.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(f"wrote {out}")


if __name__ == "__main__":
    app()
