"""Tests for the dataset release packager."""
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.derived_features import FEATURE_NAMES
from src.export_dataset import (METADATA_COLUMNS, kaggle_metadata, metadata_row,
                                render_card, with_identity_columns)

CFG = {
    "verify": {"min_face_detected_rate": 0.50},
    "windows": {"length_sec": 60.0, "stride_sec": 15.0, "max_undetected_frac": 0.30},
    "release": {"title": "T", "slug": "t-slug", "hf_repo": "me/t-slug"},
}


def make_table(detected, meta=None):
    t = pa.table({"t_sec": pa.array([0.1 * i for i in range(len(detected))], pa.float32()),
                  "face_detected": pa.array(detected)})
    meta = {"duration_sec": "0.4", "target_fps": "10.0", **(meta or {})}
    return t.replace_schema_metadata({**{k.encode(): v.encode() for k, v in meta.items()},
                                      b"pandas": b"{}"})


def row(video_id="f1_s01_c05", status="done", label=5):
    return pd.Series({"video_id": video_id, "subject_id": "01", "fold": 1,
                      "class_label": label, "status": status})


def test_identity_columns_prepended_and_provenance_kept():
    out = with_identity_columns(make_table([True, False]), "f1_s01_c05", "01", 1, 5)
    assert out.column_names[:4] == ["video_id", "subject_id", "fold", "class_label"]
    assert out.column("subject_id").to_pylist() == ["01", "01"]
    assert out.schema.field("fold").type == pa.int8()
    assert b"duration_sec" in out.schema.metadata
    assert b"pandas" not in out.schema.metadata


def test_metadata_row_flags_and_v1_threshold_defaults():
    r = metadata_row(row(status="failed"), make_table([True, False, False, False]))
    assert r["passed_verification"] is False
    assert r["face_detected_rate"] == 0.25
    assert r["class_name"] == "low_vigilant"
    assert r["extractor_version"] == 1
    assert r["min_face_detection_confidence"] == 0.5
    assert set(r) == set(METADATA_COLUMNS)


def test_metadata_row_reads_recorded_thresholds():
    t = make_table([True], {"extractor_version": "2", "min_face_detection_confidence": "0.3"})
    r = metadata_row(row(), t)
    assert r["extractor_version"] == 2
    assert r["min_face_detection_confidence"] == 0.3


def test_card_lists_failed_videos_and_counts():
    rows = [metadata_row(row("f1_s01_c00", label=0), make_table([True])),
            metadata_row(row("f1_s01_c10", "failed", 10), make_table([False, True, False]))]
    card = render_card(pd.DataFrame(rows, columns=METADATA_COLUMNS), 10, CFG)
    assert card.startswith("---\n")
    assert "**1/2 pass.**" in card
    assert "| f1_s01_c10 | 01 | drowsy | 0.33 |" in card
    assert "1 alert, 1 drowsy" in card


def test_kaggle_metadata_id():
    assert kaggle_metadata("someone", CFG)["id"] == "someone/t-slug"


def test_card_windows_section_only_when_windows_given():
    meta = pd.DataFrame([metadata_row(row(), make_table([True]))], columns=METADATA_COLUMNS)
    assert "windows.parquet" not in render_card(meta, 10, CFG)
    w = pd.DataFrame({"subject_id": ["01", "01"], "class_label": [0, 10],
                      "passed_verification": True, "valid_frac": 1.0})
    for f in FEATURE_NAMES:
        w[f] = [0.0, 1.0]
    card = render_card(meta, 10, CFG, w)
    assert "config_name: windows" in card
    assert "| `perclos` |" in card and "| 1 / 0 |" in card
