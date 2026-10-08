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
        (manifests.videos["status"] != "done")
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


def _read_parquet_rotation(path: Path) -> int | None:
    """Rotation recorded in an existing parquet's metadata, if any."""
    if not path.exists():
        return None
    try:
        import pyarrow.parquet as pq
        meta = pq.read_schema(path).metadata or {}
        rot = meta.get(b"rotation_applied")
        return int(rot.decode()) if rot is not None else None
    except Exception:
        return None


def _verify_and_record(video_id: str, extracted: Path, feature_file: Path,
                       row: pd.Series, cfg: dict, manifests: Manifests,
                       rotation: int | None) -> bool:
    """Extract features → verify → update manifest. Shared by zip and Kaggle paths."""
    try:
        meta = {k: str(row[k]) for k in
                ("video_id", "subject_id", "class_label", "fold", "source_path")}
        stats = extract_video(extracted, feature_file, cfg, meta, rotation=rotation)
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


def process_video(row: pd.Series, zf: zipfile.ZipFile, tmp: Path,
                  cfg: dict, manifests: Manifests) -> bool:
    """Unzip one member → extract features → verify → delete video."""
    video_id = row["video_id"]
    manifests.set_video(video_id, status="extracting")
    extracted = Path(zf.extract(row["source_path"], tmp))
    feature_file = Path(cfg["paths"]["features_dir"]) / f"{video_id}.parquet"
    try:
        # reuse recorded rotation on retry — skips the 300-frame probe
        rotation = _read_parquet_rotation(feature_file)
        return _verify_and_record(video_id, extracted, feature_file, row,
                                  cfg, manifests, rotation)
    finally:
        extracted.unlink(missing_ok=True)


@app.command()
def run(
    config: str = "configs/default.yaml",
    max_videos: int = typer.Option(0, help="Stop after N successful videos (0 = no limit)"),
    max_archives: int = typer.Option(0, help="Stop after N archives (0 = no limit)"),
    face_confidence: float = typer.Option(
        None, help="Override MediaPipe confidence thresholds (all three)"),
):
    cfg = load_config(config)
    if face_confidence is not None:
        for key in ("min_face_detection_confidence",
                    "min_face_presence_confidence", "min_tracking_confidence"):
            cfg["extract"][key] = face_confidence
        print(f"confidence thresholds set to {face_confidence}", flush=True)
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
                members = [m for m in zf.infolist() if not m.is_dir()]
                rows = manifests.register_videos(archive_id, zf.infolist())
                if len(members) != len(rows):
                    print(f"  note: {len(members)} zip members vs "
                          f"{len(rows)} registered rows — check skips above")
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


def _kaggle_listing(slug: str) -> dict[tuple[int, int], list[str]]:
    """Map (subject, class) → Kaggle file paths (handles odd names like 10_1.mp4)."""
    import csv as _csv
    import re as _re
    import subprocess as _sp

    files: list[str] = []
    page_token = None
    while True:
        cmd = ["uv", "run", "python", "-m", "kaggle", "datasets", "files", "-v",
               slug, "--page-size", "1000"]
        if page_token:
            cmd += ["--page-token", page_token]
        out = _sp.run(cmd, capture_output=True, text=True, check=True).stdout
        page_token = None
        lines = []
        for line in out.splitlines():
            m = _re.match(r"Next Page Token = (\S+)", line)
            if m:
                page_token = m.group(1)
            elif line.strip():
                lines.append(line)
        for row in _csv.DictReader(lines):
            name = row.get("name") or row.get("ref")
            if name:
                files.append(name)
        if not page_token:
            break

    by_key: dict[tuple[int, int], list[str]] = {}
    for f in files:
        m = _re.search(r"/(\d{1,3})/(0|5|10)[^/]*\.(\w+)$", f, _re.IGNORECASE)
        if m:
            key = (int(m.group(1)), int(m.group(2)))
            by_key.setdefault(key, []).append(f)
    return by_key


def _force_ipv4_for_gcs():
    """Local resolver returns AAAA-only for storage.googleapis.com; IPv6 is
    unreachable here, so pin the hostname to a known GCS IPv4 (PLAN.md §3)."""
    import socket
    gcs_v4 = "142.250.181.123"
    orig = socket.getaddrinfo

    def patched(host, port, *a, **kw):
        if host == "storage.googleapis.com":
            return orig(gcs_v4, port, *a, **kw)
        return orig(host, port, *a, **kw)

    socket.getaddrinfo = patched


def _download_kaggle_file(slug: str, kaggle_path: str, dest_dir: Path) -> Path:
    """Download one Kaggle file into dest_dir; returns the extracted video path."""
    import zipfile as _zip

    _force_ipv4_for_gcs()
    from kaggle.api.kaggle_api_extended import KaggleApi

    dest_dir.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_file(slug, kaggle_path, path=str(dest_dir), quiet=True)

    # CLI drops a single .zip named after the remote basename
    zips = list(dest_dir.glob("*.zip"))
    if zips:
        with _zip.ZipFile(zips[0]) as zf:
            zf.extractall(dest_dir)
        zips[0].unlink(missing_ok=True)

    videos = [p for p in dest_dir.rglob("*")
              if p.suffix.lower() in (".mov", ".mp4", ".m4v", ".avi") and p.is_file()]
    if not videos:
        raise RuntimeError(f"no video file found after download of {kaggle_path}")
    return videos[0]


@app.command()
def kaggle_retry(
    config: str = "configs/default.yaml",
    face_confidence: float = typer.Option(
        None, help="Override MediaPipe confidence thresholds (all three)"),
    video_ids: str = typer.Option(
        "", help="Comma-separated video_ids (default: all retryable failed)"),
):
    """Retry failed/missing videos via the Kaggle mirror (per-video downloads).

    PLAN.md §3 fallback when Drive quota blocks archive downloads. Uses the
    same extract → verify → manifest path as the zip loop.
    """
    cfg = load_config(config)
    if face_confidence is not None:
        for key in ("min_face_detection_confidence",
                    "min_face_presence_confidence", "min_tracking_confidence"):
            cfg["extract"][key] = face_confidence
        print(f"confidence thresholds set to {face_confidence}", flush=True)

    slug = cfg["dataset"]["kaggle_slug"]
    tmp = Path(cfg["paths"]["tmp_dir"])
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)

    manifests = Manifests(cfg)
    max_attempts = cfg["acquire"]["max_attempts"]

    if video_ids:
        want = [v.strip() for v in video_ids.split(",") if v.strip()]
    else:
        want = list(manifests.videos.loc[
            (manifests.videos["status"] != "done")
            & (manifests.videos["attempts"] < max_attempts), "video_id"])

    if not want:
        print("nothing to retry")
        return

    print(f"listing Kaggle dataset {slug} …", flush=True)
    by_key = _kaggle_listing(slug)
    print(f"found {sum(len(v) for v in by_key.values())} video files", flush=True)

    done_now = 0
    for video_id in want:
        row = manifests.videos[manifests.videos["video_id"] == video_id]
        if len(row) == 0:
            print(f"  ? {video_id}: not in manifest — register manually")
            continue
        row = row.iloc[0]
        if int(row["attempts"]) >= max_attempts:
            print(f"  - {video_id}: attempts exhausted ({row['attempts']})")
            continue

        subject = int(row["subject_id"])
        cls = int(row["class_label"])
        matches = by_key.get((subject, cls), [])
        if not matches:
            print(f"  ? {video_id}: subject {subject} class {cls} not on Kaggle")
            manifests.set_video(video_id, status="failed",
                                attempts=int(row["attempts"]) + 1,
                                error_msg="not present on Kaggle mirror")
            continue

        # prefer a single canonical file (exact stem 0/5/10); else first match
        exact = [m for m in matches
                 if m.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower() in ("0", "5", "10")]
        kaggle_path = (exact or matches)[0]
        if len(exact or matches) > 1:
            print(f"  note {video_id}: multiple Kaggle parts {matches} — using {kaggle_path}")

        print(f"→ {video_id} from {kaggle_path}", flush=True)
        manifests.set_video(video_id, status="extracting")
        dest = tmp / video_id
        try:
            video_path = _download_kaggle_file(slug, kaggle_path, dest)
            feature_file = Path(cfg["paths"]["features_dir"]) / f"{video_id}.parquet"
            rotation = _read_parquet_rotation(feature_file)
            ok = _verify_and_record(video_id, video_path, feature_file, row,
                                    cfg, manifests, rotation)
            if ok:
                done_now += 1
        except Exception as exc:
            manifests.set_video(video_id, status="failed",
                                attempts=int(row["attempts"]) + 1,
                                error_msg=str(exc)[:500])
            print(f"  ✗ {video_id}: {exc}")
        finally:
            shutil.rmtree(dest, ignore_errors=True)
        _summary(manifests)

    print(f"kaggle retry finished: {done_now}/{len(want)} recovered")


def _summary(manifests: Manifests):
    v = manifests.videos["status"].value_counts().to_dict()
    features_dir = Path(load_config()["paths"]["features_dir"])
    total_mb = sum(f.stat().st_size for f in features_dir.glob("*.parquet")) / 1024**2
    print(f"  [summary] videos done={v.get('done', 0)} failed={v.get('failed', 0)} "
          f"pending={v.get('pending', 0)} | features {total_mb:.1f} MB")


if __name__ == "__main__":
    app()
