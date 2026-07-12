"""Phase 2 — feature-file sanity checks (PLAN.md §4.3).

A parquet passes iff: row count ≈ duration × target_fps (±tolerance);
face_detected rate ≥ threshold; EAR/MAR/pose within physical ranges on
detected frames; timestamps strictly increasing.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

PHYSICAL_RANGES = {
    "ear_left": (0.0, 2.0),
    "ear_right": (0.0, 2.0),
    "mar": (0.0, 5.0),
    "pitch": (-180.0, 180.0),
    "yaw": (-180.0, 180.0),
    "roll": (-180.0, 180.0),
}


def verify_feature_file(path: str | Path, cfg: dict) -> tuple[bool, dict]:
    """Returns (ok, stats). `stats['problems']` lists every failed check."""
    v = cfg["verify"]
    problems: list[str] = []

    table = pq.read_table(path)
    meta = {k.decode(): val.decode() for k, val in (table.schema.metadata or {}).items()
            if not k.startswith(b"pandas")}
    df: pd.DataFrame = table.to_pandas()

    duration = float(meta.get("duration_sec", 0))
    target_fps = float(meta.get("target_fps", cfg["extract"]["target_fps"]))
    expected_rows = duration * target_fps
    if expected_rows > 0:
        ratio = len(df) / expected_rows
        if abs(ratio - 1.0) > v["row_count_tolerance"]:
            problems.append(
                f"row count {len(df)} vs expected {expected_rows:.0f} (ratio {ratio:.3f})")
    else:
        problems.append("missing/zero duration_sec metadata")

    detected_rate = float(df["face_detected"].mean()) if len(df) else 0.0
    if detected_rate < v["min_face_detected_rate"]:
        problems.append(f"face_detected rate {detected_rate:.2f} "
                        f"< {v['min_face_detected_rate']}")

    detected = df[df["face_detected"]]
    for col, (lo, hi) in PHYSICAL_RANGES.items():
        vals = detected[col].dropna()
        if len(vals) and ((vals < lo) | (vals > hi)).any():
            problems.append(f"{col} outside [{lo}, {hi}]: "
                            f"min={vals.min():.3f} max={vals.max():.3f}")

    if not df["t_sec"].is_monotonic_increasing or df["t_sec"].duplicated().any():
        problems.append("t_sec not strictly increasing")

    stats = {
        "rows": len(df),
        "duration_sec": duration,
        "face_detected_rate": round(detected_rate, 4),
        "problems": problems,
    }
    return not problems, stats
