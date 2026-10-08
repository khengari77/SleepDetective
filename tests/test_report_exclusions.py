"""Tests for the exclusion report generator."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.manifest import MANIFEST_COLUMNS
from src.report_exclusions import build_report, expected_video_ids

CFG = {
    "extract": {
        "extractor_version": 2,
        "min_face_detection_confidence": 0.3,
        "min_face_presence_confidence": 0.3,
        "min_tracking_confidence": 0.3,
    },
    "verify": {"min_face_detected_rate": 0.50},
    "dataset": {
        "n_subjects": 3,
        "videos_per_subject": 3,
        "n_folds": 1,
        "subjects_per_fold": 3,
        "class_labels": [0, 5, 10],
    },
}


def make_manifest(rows):
    df = pd.DataFrame(rows)
    for col in MANIFEST_COLUMNS + ["archive_id"]:
        if col not in df.columns:
            df[col] = ""
    return df


class TestExpectedIds:
    def test_full_grid(self):
        ids = expected_video_ids(CFG)
        assert len(ids) == 3 * 3  # 3 subjects × 3 classes, 1 fold
        assert "f1_s01_c00" in ids
        assert "f1_s03_c10" in ids


class TestBuildReport:
    def test_lists_failed_video(self):
        manifest = make_manifest([
            {"video_id": "f1_s01_c00", "subject_id": "01", "class_label": 0,
             "fold": 1, "status": "done", "attempts": 0,
             "archive_id": "fold1_part1", "feature_file": "", "error_msg": ""},
            {"video_id": "f1_s01_c05", "subject_id": "01", "class_label": 5,
             "fold": 1, "status": "failed", "attempts": 1,
             "archive_id": "fold1_part1", "feature_file": "",
             "error_msg": "face_detected rate 0.30 < 0.5"},
        ])
        report = build_report(manifest, CFG)
        assert "f1_s01_c05" in report
        assert "face_detected rate 0.30" in report

    def test_lists_missing_id(self):
        manifest = make_manifest([
            {"video_id": "f1_s01_c00", "subject_id": "01", "class_label": 0,
             "fold": 1, "status": "done", "attempts": 0,
             "archive_id": "fold1_part1", "feature_file": "", "error_msg": ""},
        ])
        report = build_report(manifest, CFG)
        assert "f1_s01_c05" in report  # unregistered
        assert "absent from manifest" in report

    def test_lists_subject_without_calibration(self):
        # s02 has only class-5 done — no class-0 → skipped in normalized runs
        manifest = make_manifest([
            {"video_id": "f1_s02_c05", "subject_id": "02", "class_label": 5,
             "fold": 1, "status": "done", "attempts": 0,
             "archive_id": "fold1_part1", "feature_file": "", "error_msg": ""},
        ])
        report = build_report(manifest, CFG)
        assert "without a done class-0" in report
        assert "02" in report

    def test_clean_manifest_reports_none(self):
        rows = []
        for subj in (1, 2, 3):
            for cls in (0, 5, 10):
                rows.append({
                    "video_id": f"f1_s{subj:02d}_c{cls:02d}",
                    "subject_id": f"{subj:02d}", "class_label": cls,
                    "fold": 1, "status": "done", "attempts": 0,
                    "archive_id": "fold1_part1", "feature_file": "",
                    "error_msg": "",
                })
        report = build_report(make_manifest(rows), CFG)
        assert "None — all registered videos passed verification." in report
        assert "None — all expected ids are registered." in report
        assert "None — every subject has a usable alert video" in report
