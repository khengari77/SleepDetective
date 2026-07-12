"""Unit tests for manifest parsing and validation (PLAN.md milestone 1)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from src.manifest import (
    build_archive_manifest,
    build_manifest,
    load_config,
    parse_video_path,
    validate_archive_manifest,
    validate_manifest,
    _parse_size,
)

CFG = load_config(str(Path(__file__).resolve().parents[1] / "configs/default.yaml"))


def synthetic_listing(missing: set[str] = frozenset()):
    """A complete 60-subject × 3-video listing shaped like the RLDD mirrors."""
    files = []
    for fold in range(1, 6):
        for i in range(12):
            subject = (fold - 1) * 12 + i + 1
            part = 1 if i < 6 else 2
            for label in (0, 5, 10):
                path = f"Fold{fold}_part{part}/{subject:02d}/{label}.MOV"
                if f"s{subject:02d}_c{label:02d}" not in missing:
                    files.append((path, 500_000_000, f"driveid_{subject:02d}_{label}"))
    return files


class TestParseVideoPath:
    def test_standard_path(self):
        p = parse_video_path("Fold1_part1/07/0.MOV")
        assert (p.fold, p.subject_id, p.class_label) == (1, "07", 0)

    def test_mp4_class_10(self):
        p = parse_video_path("Fold3_part2/31/10.mp4")
        assert (p.fold, p.subject_id, p.class_label) == (3, "31", 10)

    def test_leading_directory(self):
        p = parse_video_path("uta-rldd/Fold5_part1/55/5.mov")
        assert (p.fold, p.subject_id, p.class_label) == (5, "55", 5)

    def test_unparseable_flags_not_crashes(self):
        p = parse_video_path("README.txt")
        assert not p.ok

    def test_non_class_filename(self):
        assert not parse_video_path("Fold1_part1/07/7_extra.MOV").ok


class TestBuildManifest:
    def test_complete_listing(self):
        manifest, skipped = build_manifest(synthetic_listing())
        assert len(manifest) == 180
        assert skipped == []
        assert manifest["video_id"].is_unique
        assert set(manifest["status"]) == {"pending"}

    def test_skips_non_video_files(self):
        manifest, skipped = build_manifest(
            synthetic_listing() + [("labels.txt", 100, "x"), ("Fold1_part1/junk/readme.md", 1, "y")]
        )
        assert len(manifest) == 180
        assert len(skipped) == 2


class TestValidateManifest:
    def test_complete_manifest_passes(self):
        manifest, _ = build_manifest(synthetic_listing())
        ok, report = validate_manifest(manifest, CFG)
        assert ok, report
        assert "OK" in report

    def test_missing_video_fails_with_flag(self):
        manifest, _ = build_manifest(synthetic_listing(missing={"s07_c05"}))
        ok, report = validate_manifest(manifest, CFG)
        assert not ok
        assert "missing videos" in report and "07" in report

    def test_subject_in_two_folds_fails(self):
        manifest, _ = build_manifest(synthetic_listing())
        manifest.loc[manifest["subject_id"] == "01", "fold"] = [1, 1, 2]
        ok, report = validate_manifest(manifest, CFG)
        assert not ok
        assert "multiple folds" in report

    def test_duplicate_video_id_fails(self):
        manifest, _ = build_manifest(synthetic_listing())
        manifest = pd.concat([manifest, manifest.iloc[[0]]], ignore_index=True)
        ok, report = validate_manifest(manifest, CFG)
        assert not ok
        assert "duplicate" in report


class TestArchiveManifest:
    ZIPS = [(f"Fold{f}_part{p}.zip", None, f"driveid_f{f}p{p}")
            for f in range(1, 6) for p in (1, 2)]

    def test_complete_archives_pass(self):
        archives, skipped = build_archive_manifest(self.ZIPS + [("readme.txt", 1, "x")])
        assert len(archives) == 10
        assert skipped == ["not a fold archive: readme.txt"]
        ok, report = validate_archive_manifest(archives, CFG)
        assert ok, report

    def test_missing_archive_fails(self):
        archives, _ = build_archive_manifest(self.ZIPS[:-1])
        ok, report = validate_archive_manifest(archives, CFG)
        assert not ok
        assert "missing archives" in report and "(5, 2)" in report


@pytest.mark.parametrize("raw,expected", [
    ("1234", 1234),
    ("1.5GB", int(1.5 * 1024**3)),
    ("500MB", 500 * 1024**2),
    ("", None),
    ("n/a", None),
])
def test_parse_size(raw, expected):
    assert _parse_size(raw) == expected
