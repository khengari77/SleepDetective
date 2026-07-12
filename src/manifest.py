"""Phase 1 — build and validate the UTA-RLDD acquisition manifest (PLAN.md §3).

The manifest is the single source of truth for the streaming extraction loop:
one row per video, with enough metadata to download each file individually.

Usage:
    uv run python -m src.manifest build              # list files from Kaggle
    uv run python -m src.manifest build --listing f  # offline: CSV of path,size
    uv run python -m src.manifest validate
"""
from __future__ import annotations

import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import typer
import yaml

app = typer.Typer(add_completion=False)

MANIFEST_COLUMNS = [
    "video_id", "source_path", "download_ref", "fold", "subject_id",
    "class_label", "expected_size_bytes", "container", "status", "attempts",
    "feature_file", "extracted_at", "error_msg",
]

VIDEO_EXTENSIONS = {".mov", ".mp4", ".m4v", ".avi"}
CLASS_LABELS = {"0", "5", "10"}

# The official Drive distribution ships 10 zip archives, one per fold-part
# (Fold1_part1.zip .. Fold5_part2.zip), ~11 GB each — not individual videos.
ARCHIVE_COLUMNS = ["archive_id", "download_ref", "fold", "part",
                   "expected_size_bytes", "status", "attempts", "error_msg"]
ARCHIVE_RE = re.compile(r"fold\s*_?(\d)_part\s*_?(\d)\.zip$", re.IGNORECASE)


def build_archive_manifest(files: list[tuple[str, int | None, str]]) -> tuple[pd.DataFrame, list[str]]:
    """Archive-level manifest for the official Drive zip distribution."""
    rows, skipped = [], []
    for path, size, ref in files:
        m = ARCHIVE_RE.search(path)
        if not m:
            skipped.append(f"not a fold archive: {path}")
            continue
        rows.append({
            "archive_id": f"fold{m.group(1)}_part{m.group(2)}",
            "download_ref": ref,
            "fold": int(m.group(1)),
            "part": int(m.group(2)),
            "expected_size_bytes": size,
            "status": "pending",
            "attempts": 0,
            "error_msg": "",
        })
    archives = pd.DataFrame(rows, columns=ARCHIVE_COLUMNS)
    archives = archives.sort_values(["fold", "part"]).reset_index(drop=True)
    return archives, skipped


def validate_archive_manifest(archives: pd.DataFrame, cfg: dict) -> tuple[bool, str]:
    ds = cfg["dataset"]
    expected = {(f, p) for f in range(1, ds["n_folds"] + 1) for p in (1, 2)}
    present = set(zip(archives["fold"], archives["part"]))
    problems = []
    if missing := expected - present:
        problems.append(f"missing archives: {sorted(missing)}")
    if extra := present - expected:
        problems.append(f"unexpected archives: {sorted(extra)}")
    if len(archives) != len(present):
        problems.append("duplicate archive entries")
    ok = not problems
    report = "\n".join(
        ["ARCHIVE MANIFEST VALIDATION", "=" * 40,
         f"archives: {len(archives)} (expected {len(expected)})"]
        + (["OK — all fold parts present"] if ok else [f"PROBLEM: {p}" for p in problems])
    )
    return ok, report


@dataclass
class ParsedVideo:
    source_path: str
    fold: int | None
    subject_id: str | None
    class_label: int | None
    size_bytes: int | None

    @property
    def ok(self) -> bool:
        return None not in (self.fold, self.subject_id, self.class_label)


def parse_video_path(path: str, size_bytes: int | None = None) -> ParsedVideo:
    """Parse fold/subject/class from an RLDD-style path.

    Expected shapes (Kaggle mirror and official Drive folders):
        Fold1_part1/07/0.MOV
        Fold3_part2/31/10.mp4
    Tolerates extra leading directories and case differences. Fields that
    cannot be parsed come back None — the validator flags them, per the
    plan's "flag known gaps rather than crashing" rule.
    """
    parts = Path(path).parts
    stem = Path(path).stem
    class_label = int(stem) if stem in CLASS_LABELS else None

    fold = None
    for part in parts:
        m = re.search(r"fold\s*_?(\d)", part, re.IGNORECASE)
        if m:
            fold = int(m.group(1))
            break

    # subject = the directory component that is a bare integer (subject dirs
    # are numbered; fold dirs contain "Fold", filenames carry the class)
    subject_id = None
    for part in parts[:-1]:
        if re.fullmatch(r"\d{1,3}", part):
            subject_id = f"{int(part):02d}"
            break

    return ParsedVideo(path, fold, subject_id, class_label, size_bytes)


def list_gdrive_files(folder_url: str) -> list[tuple[str, int | None, str]]:
    """Enumerate the official RLDD Google Drive folder without downloading.

    Returns (relative_path, size_bytes_or_None, drive_file_id) tuples.
    Enumerates one Fold*_part* subfolder at a time to stay under gdown's
    50-files-per-listing limit (each part holds ~18 files).
    """
    import gdown

    top = gdown.download_folder(url=folder_url, skip_download=True, quiet=True)
    if top is None:
        raise RuntimeError(f"could not enumerate Drive folder {folder_url}")
    files: list[tuple[str, int | None, str]] = []
    for entry in top:
        files.append((entry.path, None, entry.id))
    return files


def list_kaggle_files(slug: str) -> list[tuple[str, int | None, str]]:
    """List all files in a Kaggle dataset as (path, size_bytes, ref) tuples.

    Uses the CLI (`kaggle datasets files -v`) with page-token pagination.
    Requires ~/.kaggle/kaggle.json.
    """
    files: list[tuple[str, int | None, str]] = []
    page_token = None
    while True:
        cmd = ["kaggle", "datasets", "files", "-v", slug, "--page-size", "1000"]
        if page_token:
            cmd += ["--page-token", page_token]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        page_token = None
        lines = []
        for line in out.splitlines():
            m = re.match(r"Next Page Token = (\S+)", line)
            if m:
                page_token = m.group(1)
            elif line.strip():
                lines.append(line)
        rows = list(csv.DictReader(lines))
        for row in rows:
            name = row.get("name") or row.get("ref")
            size = _parse_size(row.get("totalBytes") or row.get("size"))
            if name:
                files.append((name, size, name))
        if not page_token:
            return files


def _parse_size(value: str | None) -> int | None:
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return int(value)
    m = re.fullmatch(r"([\d.]+)\s*(B|KB|MB|GB)", value, re.IGNORECASE)
    if not m:
        return None
    mult = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}[m.group(2).upper()]
    return int(float(m.group(1)) * mult)


def build_manifest(files: list[tuple[str, int | None, str]]) -> tuple[pd.DataFrame, list[str]]:
    """Turn a raw (path, size, download_ref) listing into the manifest DataFrame.

    Returns (manifest, skipped) where skipped lists non-video or unparseable
    paths for the validation report.
    """
    rows, skipped = [], []
    for path, size, ref in files:
        if Path(path).suffix.lower() not in VIDEO_EXTENSIONS:
            skipped.append(f"non-video: {path}")
            continue
        parsed = parse_video_path(path, size)
        if not parsed.ok:
            skipped.append(f"unparseable: {path}")
            continue
        rows.append({
            "video_id": f"f{parsed.fold}_s{parsed.subject_id}_c{parsed.class_label:02d}",
            "source_path": path,
            "download_ref": ref,
            "fold": parsed.fold,
            "subject_id": parsed.subject_id,
            "class_label": parsed.class_label,
            "expected_size_bytes": size,
            "container": Path(path).suffix.lower().lstrip("."),
            "status": "pending",
            "attempts": 0,
            "feature_file": "",
            "extracted_at": "",
            "error_msg": "",
        })
    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    manifest = manifest.sort_values(["fold", "subject_id", "class_label"]).reset_index(drop=True)
    return manifest, skipped


def validate_manifest(manifest: pd.DataFrame, cfg: dict) -> tuple[bool, str]:
    """Check the dataset invariants from PLAN.md §3; returns (ok, report)."""
    ds = cfg["dataset"]
    problems: list[str] = []
    lines = [f"videos: {len(manifest)} (expected {ds['n_subjects'] * ds['videos_per_subject']})"]

    dupes = manifest[manifest.duplicated("video_id", keep=False)]
    if len(dupes):
        problems.append(f"{dupes['video_id'].nunique()} duplicate video_ids: "
                        f"{sorted(dupes['video_id'].unique())[:5]}...")

    subjects = manifest.groupby("subject_id")
    lines.append(f"subjects: {subjects.ngroups} (expected {ds['n_subjects']})")
    if subjects.ngroups != ds["n_subjects"]:
        problems.append(f"subject count {subjects.ngroups} != {ds['n_subjects']}")

    incomplete = {s: sorted(g["class_label"]) for s, g in subjects
                  if sorted(g["class_label"]) != sorted(ds["class_labels"])}
    if incomplete:
        problems.append(f"{len(incomplete)} subjects missing videos: {incomplete}")

    multi_fold = {s: sorted(g["fold"].unique()) for s, g in subjects
                  if g["fold"].nunique() > 1}
    if multi_fold:
        problems.append(f"subjects in multiple folds: {multi_fold}")

    fold_sizes = manifest.groupby("fold")["subject_id"].nunique()
    for fold, n in fold_sizes.items():
        lines.append(f"fold {fold}: {n} subjects (expected {ds['subjects_per_fold']})")
        if n != ds["subjects_per_fold"]:
            problems.append(f"fold {fold} has {n} subjects, expected {ds['subjects_per_fold']}")
    if set(fold_sizes.index) != set(range(1, ds["n_folds"] + 1)):
        problems.append(f"folds present: {sorted(fold_sizes.index)}, expected 1..{ds['n_folds']}")

    ok = not problems
    report = "\n".join(
        ["MANIFEST VALIDATION", "=" * 40, *lines, "-" * 40]
        + (["OK — all invariants hold"] if ok else [f"PROBLEM: {p}" for p in problems])
    )
    return ok, report


def load_config(path: str = "configs/default.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


@app.command()
def build(
    config: str = "configs/default.yaml",
    source: str = typer.Option("gdrive", help="gdrive (official Drive folder) or kaggle"),
    listing: str = typer.Option(None, help="Offline CSV (path,size_bytes,ref) instead of a remote listing"),
):
    """Build data/manifest.csv from the official Drive folder (or Kaggle / offline CSV)."""
    cfg = load_config(config)
    if listing:
        with open(listing) as f:
            files = [(r["path"],
                      int(r["size_bytes"]) if r.get("size_bytes") else None,
                      r.get("ref") or r["path"])
                     for r in csv.DictReader(f)]
    elif source == "gdrive":
        files = list_gdrive_files(cfg["dataset"]["gdrive_folder"])
        archives, skipped = build_archive_manifest(files)
        ok, report = validate_archive_manifest(archives, cfg)
        for s in skipped:
            print(f"skipped {s}")
        print(report)
        out = Path(cfg["paths"]["archive_manifest"])
        out.parent.mkdir(parents=True, exist_ok=True)
        archives.to_csv(out, index=False)
        print(f"\nwrote {out} ({len(archives)} archives)")
        print("Per-video manifest rows are registered as each archive is opened "
              "during extraction (Drive distributes zips, not videos).")
        raise typer.Exit(code=0 if ok else 1)
    elif source == "kaggle":
        files = list_kaggle_files(cfg["dataset"]["kaggle_slug"])
    else:
        raise typer.BadParameter(f"unknown source {source!r}")
    manifest, skipped = build_manifest(files)
    ok, report = validate_manifest(manifest, cfg)
    for s in skipped:
        print(f"skipped {s}")
    print(report)
    out = Path(cfg["paths"]["manifest"])
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out, index=False)
    print(f"\nwrote {out} ({len(manifest)} rows)")
    if not ok:
        print("Validation FAILED — review the problems above before extraction (PLAN.md §3).")
        raise typer.Exit(code=1)


@app.command()
def validate(config: str = "configs/default.yaml"):
    """Re-validate an existing data/manifest.csv."""
    cfg = load_config(config)
    manifest = pd.read_csv(cfg["paths"]["manifest"], dtype={"subject_id": str})
    ok, report = validate_manifest(manifest, cfg)
    print(report)
    raise typer.Exit(code=0 if ok else 1)


if __name__ == "__main__":
    app()
