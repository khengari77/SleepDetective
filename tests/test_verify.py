"""Tests for verify_feature_file gate behaviour (synthetic parquets)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.verify import verify_feature_file

CFG = {
    "extract": {"target_fps": 10.0},
    "verify": {"row_count_tolerance": 0.10, "min_face_detected_rate": 0.50},
}


def write_parquet(path: Path, face_rate: float, n_rows: int = 100):
    n_det = int(round(n_rows * face_rate))
    face = np.array([True] * n_det + [False] * (n_rows - n_det))
    df = pd.DataFrame({
        "t_sec": np.arange(n_rows, dtype=np.float32) / 10.0,
        "face_detected": face,
        "ear_left": np.where(face, 0.3, np.nan),
        "ear_right": np.where(face, 0.3, np.nan),
        "mar": np.where(face, 0.5, np.nan),
        "pitch": np.where(face, 10.0, np.nan),
        "yaw": np.where(face, 0.0, np.nan),
        "roll": np.where(face, -5.0, np.nan),
    })
    table = pa.Table.from_pandas(df, preserve_index=False)
    table = table.replace_schema_metadata({
        b"duration_sec": b"10.000",
        b"target_fps": b"10.0",
    })
    pq.write_table(table, path)


class TestFaceRateGate:
    def test_rate_below_gate_fails(self, tmp_path):
        p = tmp_path / "v.parquet"
        write_parquet(p, face_rate=0.49)
        ok, stats = verify_feature_file(p, CFG)
        assert not ok
        assert any("face_detected rate" in prob for prob in stats["problems"])

    def test_rate_at_gate_passes(self, tmp_path):
        p = tmp_path / "v.parquet"
        write_parquet(p, face_rate=0.50)
        ok, stats = verify_feature_file(p, CFG)
        assert ok
        assert stats["problems"] == []
        assert stats["face_detected_rate"] == 0.50

    def test_rate_well_above_gate_passes(self, tmp_path):
        p = tmp_path / "v.parquet"
        write_parquet(p, face_rate=1.0)
        ok, stats = verify_feature_file(p, CFG)
        assert ok
        assert stats["face_detected_rate"] == 1.0


class TestStatsShape:
    def test_stats_keys(self, tmp_path):
        p = tmp_path / "v.parquet"
        write_parquet(p, face_rate=0.6)
        _, stats = verify_feature_file(p, CFG)
        assert set(stats) >= {"rows", "duration_sec", "face_detected_rate", "problems"}
        assert stats["rows"] == 100
