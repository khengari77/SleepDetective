"""Resume/manifest-state tests for the acquisition loop (PLAN.md C2, milestone 2)."""
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.acquire import Manifests, check_disk_space, select_archives
from src.manifest import ARCHIVE_COLUMNS, MANIFEST_COLUMNS


def make_cfg(tmp_path: Path) -> dict:
    return {
        "paths": {
            "archive_manifest": str(tmp_path / "manifest_archives.csv"),
            "manifest": str(tmp_path / "manifest.csv"),
            "features_dir": str(tmp_path / "features"),
            "tmp_dir": str(tmp_path / "tmp"),
        },
        "acquire": {"max_attempts": 3, "free_space_factor": 1.5,
                    "free_space_margin_gb": 5.0},
    }


def seed_archives(cfg, statuses):
    rows = [{"archive_id": f"fold1_part{i+1}", "download_ref": f"ref{i}",
             "fold": 1, "part": i + 1, "expected_size_bytes": None,
             "status": s, "attempts": 0, "error_msg": ""}
            for i, s in enumerate(statuses)]
    pd.DataFrame(rows, columns=ARCHIVE_COLUMNS).to_csv(
        cfg["paths"]["archive_manifest"], index=False)


def zip_members(names):
    infos = []
    for name in names:
        zi = zipfile.ZipInfo(name)
        zi.file_size = 1000
        infos.append(zi)
    return infos


class TestResumeLogic:
    def test_stale_statuses_reset_to_pending(self, tmp_path):
        cfg = make_cfg(tmp_path)
        seed_archives(cfg, ["done", "downloading", "extracting"])
        m = Manifests(cfg)
        m.videos = pd.DataFrame(
            [{**dict.fromkeys(MANIFEST_COLUMNS, ""), "video_id": "f1_s01_c00",
              "status": "extracting", "attempts": 0, "archive_id": "fold1_part1"}])
        m.reset_stale()

        reloaded = Manifests(cfg)
        assert list(reloaded.archives["status"]) == ["done", "pending", "pending"]
        assert list(reloaded.videos["status"]) == ["pending"]

    def test_done_videos_survive_reregistration(self, tmp_path):
        """Restart re-downloads a zip; register_videos must not reset done rows."""
        cfg = make_cfg(tmp_path)
        seed_archives(cfg, ["pending"])
        m = Manifests(cfg)
        members = zip_members(["Fold1_part1/01/0.mp4", "Fold1_part1/01/5.mp4"])

        rows = m.register_videos("fold1_part1", members)
        assert len(rows) == 2
        m.set_video("f1_s01_c00", status="done", feature_file="x.parquet")

        rows2 = Manifests(cfg).register_videos("fold1_part1", members)
        assert len(rows2) == 2  # no duplicates
        done = rows2[rows2["video_id"] == "f1_s01_c00"].iloc[0]
        assert done["status"] == "done"
        assert done["feature_file"] == "x.parquet"

    def test_status_changes_flushed_to_disk_immediately(self, tmp_path):
        cfg = make_cfg(tmp_path)
        seed_archives(cfg, ["pending"])
        m = Manifests(cfg)
        m.set_archive("fold1_part1", status="downloading", attempts=1)
        # a fresh reader (≈ restart) sees the change without any explicit save
        assert Manifests(cfg).archives.iloc[0]["status"] == "downloading"

    def test_non_videos_in_zip_are_skipped(self, tmp_path):
        cfg = make_cfg(tmp_path)
        seed_archives(cfg, ["pending"])
        m = Manifests(cfg)
        rows = m.register_videos(
            "fold1_part1", zip_members(["Fold1_part1/01/0.mp4", "__MACOSX/x", "a.txt"]))
        assert len(rows) == 1


class TestRetrySelection:
    def test_done_archive_with_retryable_failed_member_is_reselected(self, tmp_path):
        """C2: retry `failed` up to max_attempts, even after the archive
        itself was marked `done` (a `done` archive may hold `failed` members)."""
        cfg = make_cfg(tmp_path)
        seed_archives(cfg, ["done", "done"])
        m = Manifests(cfg)
        m.videos = pd.DataFrame([
            {**dict.fromkeys(MANIFEST_COLUMNS, ""), "video_id": "f1_s01_c00",
             "status": "failed", "attempts": 1, "archive_id": "fold1_part1"},
            {**dict.fromkeys(MANIFEST_COLUMNS, ""), "video_id": "f1_s02_c00",
             "status": "done", "attempts": 0, "archive_id": "fold1_part2"},
        ])

        todo = select_archives(m, max_attempts=3)

        assert list(todo["archive_id"]) == ["fold1_part1"]

    def test_failed_member_at_max_attempts_is_not_retried(self, tmp_path):
        cfg = make_cfg(tmp_path)
        seed_archives(cfg, ["done"])
        m = Manifests(cfg)
        m.videos = pd.DataFrame([
            {**dict.fromkeys(MANIFEST_COLUMNS, ""), "video_id": "f1_s01_c00",
             "status": "failed", "attempts": 3, "archive_id": "fold1_part1"},
        ])

        todo = select_archives(m, max_attempts=3)

        assert len(todo) == 0

    def test_pending_archive_still_selected_without_any_videos(self, tmp_path):
        cfg = make_cfg(tmp_path)
        seed_archives(cfg, ["pending"])
        m = Manifests(cfg)

        todo = select_archives(m, max_attempts=3)

        assert list(todo["archive_id"]) == ["fold1_part1"]

    def test_done_archive_with_pending_member_is_reselected(self, tmp_path):
        """Late-registered pending members (e.g. a video missing from the
        first pass) must reopen a done archive — otherwise resume strands
        them forever."""
        cfg = make_cfg(tmp_path)
        seed_archives(cfg, ["done"])
        m = Manifests(cfg)
        m.videos = pd.DataFrame([
            {**dict.fromkeys(MANIFEST_COLUMNS, ""), "video_id": "f1_s01_c00",
             "status": "pending", "attempts": 0, "archive_id": "fold1_part1"},
        ])

        todo = select_archives(m, max_attempts=3)

        assert list(todo["archive_id"]) == ["fold1_part1"]


class TestDiskSpaceGate:
    def test_passes_with_reasonable_requirement(self, tmp_path):
        cfg = make_cfg(tmp_path)
        # tiny margin: tmp_path may sit on a small tmpfs; this tests the
        # formula, not the host's disk
        cfg["acquire"]["free_space_margin_gb"] = 0.001
        check_disk_space(tmp_path, 1024**2, cfg)  # 1 MB archive — fine

    def test_refuses_when_insufficient(self, tmp_path):
        cfg = make_cfg(tmp_path)
        with pytest.raises(RuntimeError, match="insufficient disk space"):
            check_disk_space(tmp_path, 10**15, cfg)  # 1 PB archive
