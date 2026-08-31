"""Phase 2 — streaming acquisition loop (PLAN.md §4, zip-archive variant).

Strictly one archive + one unzipped member video on disk at any time (C1).
Fully resumable via manifest statuses flushed after every change (C2).

Usage:
    uv run python -m src.acquire run                 # everything pending
    uv run python -m src.acquire run --max-videos 3  # milestone-2 smoke test
"""
from __future__ import annotations

import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import typer

from src.extract import extract_video
from src.manifest import MANIFEST_COLUMNS, build_manifest, load_config
from src.verify import verify_feature_file

app = typer.Typer(add_completion=False)


@app.callback()
def _cli():
    """SleepDetective v2 dataset acquisition (keeps `run` a named subcommand)."""

DEFAULT_ARCHIVE_SIZE = 15 * 1024**3  # assume 15 GB when Drive won't tell us


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Manifests:
    """Both manifest levels, flushed to disk after every status change (C2)."""

    def __init__(self, cfg: dict):
        self.archive_path = Path(cfg["paths"]["archive_manifest"])
        self.video_path = Path(cfg["paths"]["manifest"])
        self.archives = pd.read_csv(self.archive_path)
        self.archives["error_msg"] = self.archives["error_msg"].fillna("")
        if self.video_path.exists():
            self.videos = pd.read_csv(self.video_path, dtype={"subject_id": str})
            self.videos["error_msg"] = self.videos["error_msg"].fillna("")
            self.videos["feature_file"] = self.videos["feature_file"].fillna("")
        else:
            self.videos = pd.DataFrame(columns=MANIFEST_COLUMNS + ["archive_id"])

    def flush(self):
        self.archives.to_csv(self.archive_path, index=False)
        self.videos.to_csv(self.video_path, index=False)

    def reset_stale(self):
        """downloading/extracting are crash leftovers → pending (C2)."""
        for df in (self.archives, self.videos):
            stale = df["status"].isin(["downloading", "extracting"])
            df.loc[stale, "status"] = "pending"
        self.flush()

    def set_archive(self, archive_id: str, **fields):
        idx = self.archives["archive_id"] == archive_id
        for key, value in fields.items():
            self.archives.loc[idx, key] = value
        self.flush()

    def set_video(self, video_id: str, **fields):
        idx = self.videos["video_id"] == video_id
        for key, value in fields.items():
            self.videos.loc[idx, key] = value
        self.flush()

    def register_videos(self, archive_id: str, members: list[zipfile.ZipInfo]):
        listing = [(m.filename, m.file_size, m.filename)
                   for m in members if not m.is_dir()]
        new, skipped = build_manifest(listing)
        for s in skipped:
            print(f"  skipped {s}")
        new["archive_id"] = archive_id
        known = set(self.videos["video_id"])
        additions = new[~new["video_id"].isin(known)]
        if len(additions):
            self.videos = pd.concat([self.videos, additions], ignore_index=True)
        self.flush()
        return self.videos[self.videos["archive_id"] == archive_id]


def select_archives(manifests: "Manifests", max_attempts: int) -> pd.DataFrame:
    """Archives to (re)open: not yet done, or `done` but still holding a
    retryable `failed` member — C2 promises retrying failures up to
    `max_attempts` on restart, and a `done` archive is allowed to have
    `failed` members (see the `not_done` check in `run`)."""
    retry_ids = set(manifests.videos.loc[
        (manifests.videos["status"] == "failed")
        & (manifests.videos["attempts"] < max_attempts),
        "archive_id"])
    return manifests.archives[
        ((manifests.archives["status"] != "done")
         | manifests.archives["archive_id"].isin(retry_ids))
        & (manifests.archives["attempts"] < max_attempts)]


def check_disk_space(tmp_dir: Path, expected_size: int | None, cfg: dict):
    a = cfg["acquire"]
    expected = expected_size or DEFAULT_ARCHIVE_SIZE
    needed = expected * a["free_space_factor"] + a["free_space_margin_gb"] * 1024**3
    free = shutil.disk_usage(tmp_dir).free
    if free < needed:
        raise RuntimeError(
            f"insufficient disk space: {free / 1e9:.1f} GB free, "
            f"need {needed / 1e9:.1f} GB (C1) — refusing to download")


def download_archive(download_ref: str, dest: Path) -> Path:
    import gdown

    out = gdown.download(id=download_ref, output=str(dest), quiet=False)
    if out is None or not dest.exists():
        raise RuntimeError("gdown download failed (Drive quota? see PLAN.md §3)")
    return dest


def wait_for_prefetch(prefetched: Path, max_wait_sec: int = 3600) -> bool:
    """If the prefetcher is mid-download for this archive (gdown .part file
    present), wait for it to finish instead of double-downloading."""
    import time

    waited = 0
    while waited < max_wait_sec:
        if prefetched.exists():
            return True
        if not list(prefetched.parent.glob(f"{prefetched.name}*part*")):
            return prefetched.exists()
        time.sleep(30)
        waited += 30
    return prefetched.exists()


def process_video(row: pd.Series, zf: zipfile.ZipFile, tmp: Path,
                  cfg: dict, manifests: Manifests) -> bool:
    """Unzip one member → extract features → verify → delete video."""
    video_id = row["video_id"]
    manifests.set_video(video_id, status="extracting")
    extracted = Path(zf.extract(row["source_path"], tmp))
    feature_file = Path(cfg["paths"]["features_dir"]) / f"{video_id}.parquet"
    try:
        meta = {k: str(row[k]) for k in
                ("video_id", "subject_id", "class_label", "fold", "source_path")}
        stats = extract_video(extracted, feature_file, cfg, meta)
        ok, vstats = verify_feature_file(feature_file, cfg)
        if ok:
            manifests.set_video(video_id, status="done",
                                feature_file=str(feature_file), extracted_at=_now())
            print(f"  ✓ {video_id}: {vstats['rows']} rows, "
                  f"face rate {vstats['face_detected_rate']}")
            return True
        # low face-detection rate: keep the parquet, flag for manual review (§4.3)
        only_face_rate = all("face_detected rate" in p for p in vstats["problems"])
        if not only_face_rate:
            feature_file.unlink(missing_ok=True)
        manifests.set_video(
            video_id, status="failed",
            attempts=int(row["attempts"]) + 1,
            feature_file=str(feature_file) if only_face_rate else "",
            error_msg="; ".join(vstats["problems"]))
        print(f"  ✗ {video_id}: {vstats['problems']}")
        return False
    except Exception as exc:  # corrupt video, decoder error, ... (§9.2)
        feature_file.unlink(missing_ok=True)
        manifests.set_video(video_id, status="failed",
                            attempts=int(row["attempts"]) + 1, error_msg=str(exc)[:500])
        print(f"  ✗ {video_id}: {exc}")
        return False
    finally:
        extracted.unlink(missing_ok=True)


@app.command()
def run(
    config: str = "configs/default.yaml",
    max_videos: int = typer.Option(0, help="Stop after N successful videos (0 = no limit)"),
    max_archives: int = typer.Option(0, help="Stop after N archives (0 = no limit)"),
):
    cfg = load_config(config)
    tmp = Path(cfg["paths"]["tmp_dir"])
    shutil.rmtree(tmp, ignore_errors=True)  # wiped on startup (C1)
    tmp.mkdir(parents=True, exist_ok=True)

    manifests = Manifests(cfg)
    manifests.reset_stale()
    max_attempts = cfg["acquire"]["max_attempts"]
    done_now = archives_now = 0

    todo = select_archives(manifests, max_attempts)
    archives_dir = Path(cfg["paths"].get("archives_dir", "data/archives"))
    for _, arc in todo.iterrows():
        archive_id = arc["archive_id"]
        prefetched = archives_dir / f"{archive_id}.zip"
        zip_path = tmp / f"{archive_id}.zip"
        try:
            if wait_for_prefetch(prefetched, max_wait_sec=0) or wait_for_prefetch(prefetched):
                zip_path = prefetched
                print(f"using prefetched {zip_path}", flush=True)
                manifests.set_archive(archive_id, status="extracting",
                                      expected_size_bytes=zip_path.stat().st_size)
            else:
                size = arc["expected_size_bytes"]
                check_disk_space(tmp, int(size) if pd.notna(size) else None, cfg)
                # attempts counts FAILURES only — interrupted runs must not
                # burn retry budget (a restart is not a failure)
                manifests.set_archive(archive_id, status="downloading")
                print(f"downloading {archive_id} …", flush=True)
                download_archive(arc["download_ref"], zip_path)
                manifests.set_archive(archive_id, status="extracting",
                                      expected_size_bytes=zip_path.stat().st_size)

            with zipfile.ZipFile(zip_path) as zf:
                rows = manifests.register_videos(archive_id, zf.infolist())
                pending = rows[(rows["status"] != "done")
                               & (rows["attempts"] < max_attempts)]
                print(f"{archive_id}: {len(rows)} videos, {len(pending)} to process")
                for _, row in pending.iterrows():
                    if process_video(row, zf, tmp, cfg, manifests):
                        done_now += 1
                    _summary(manifests)
                    if max_videos and done_now >= max_videos:
                        print("max-videos reached — stopping (archive left non-done)")
                        return

            not_done = manifests.videos[
                (manifests.videos["archive_id"] == archive_id)
                & (~manifests.videos["status"].isin(["done", "failed"]))]
            if len(not_done) == 0:
                manifests.set_archive(archive_id, status="done", error_msg="")
        except Exception as exc:  # quota/auth/corrupt zip — never wedge the loop (§9.1)
            manifests.set_archive(archive_id, status="failed",
                                  attempts=int(arc["attempts"]) + 1,
                                  error_msg=str(exc)[:500])
            print(f"✗ {archive_id}: {exc}")
        finally:
            zip_path.unlink(missing_ok=True)

        archives_now += 1
        if max_archives and archives_now >= max_archives:
            print("max-archives reached — stopping")
            return


@app.command()
def download(config: str = "configs/default.yaml"):
    """Prefetch ALL pending archives to archives_dir (user-relaxed C1,
    2026-07-12: bulk download OK; zips still deleted after extraction).

    Touches no manifest state, so it can run alongside `run`.
    """
    cfg = load_config(config)
    archives_dir = Path(cfg["paths"].get("archives_dir", "data/archives"))
    archives_dir.mkdir(parents=True, exist_ok=True)
    manifests = Manifests(cfg)

    todo = manifests.archives[manifests.archives["status"] != "done"]
    remaining = [a for _, a in todo.iterrows()
                 if not (archives_dir / f"{a['archive_id']}.zip").exists()]
    total_expected = sum(int(a["expected_size_bytes"]) if pd.notna(a["expected_size_bytes"])
                         else DEFAULT_ARCHIVE_SIZE for a in remaining)
    free = shutil.disk_usage(archives_dir).free
    needed = total_expected * 1.1 + cfg["acquire"]["free_space_margin_gb"] * 1024**3
    if free < needed:
        raise RuntimeError(f"bulk prefetch needs {needed / 1e9:.0f} GB free, "
                           f"have {free / 1e9:.0f} GB")

    print(f"prefetching {len(remaining)} archives "
          f"(~{total_expected / 1e9:.0f} GB) → {archives_dir}", flush=True)
    failures = 0
    for arc in remaining:
        dest = archives_dir / f"{arc['archive_id']}.zip"
        try:
            print(f"prefetch {arc['archive_id']} …", flush=True)
            download_archive(arc["download_ref"], dest)
        except Exception as exc:
            failures += 1
            print(f"✗ prefetch {arc['archive_id']}: {exc}", flush=True)
    print(f"prefetch finished, {failures} failures "
          f"(run will self-download any missing archive)", flush=True)


def _summary(manifests: Manifests):
    v = manifests.videos["status"].value_counts().to_dict()
    features_dir = Path(load_config()["paths"]["features_dir"])
    total_mb = sum(f.stat().st_size for f in features_dir.glob("*.parquet")) / 1024**2
    print(f"  [summary] videos done={v.get('done', 0)} failed={v.get('failed', 0)} "
          f"pending={v.get('pending', 0)} | features {total_mb:.1f} MB")


if __name__ == "__main__":
    app()
