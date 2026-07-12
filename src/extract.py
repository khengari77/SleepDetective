"""Phase 2 — video → per-frame feature parquet (PLAN.md §4.1–4.2).

Uses the MediaPipe Face Landmarker *tasks* API (not v1's legacy FaceMesh)
with blendshapes and facial transformation matrices enabled.
"""
from __future__ import annotations

import json
import math
import subprocess
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

EXTRACTOR_VERSION = 1

# Pinned model asset (C7: exact Google weights, versioned URL — not "latest").
LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# v1 EAR landmark indices (modules/PERCLOS.py), reused verbatim so the B1
# heuristic baseline is faithful. v2 computes distances in normalized-landmark
# space scaled by inter-ocular distance — v1's pixel-space EAR was
# resolution-dependent (a genuine v1 bug, noted in PLAN.md §4.2).
RIGHT_EYE_EAR = ((160, 144), (158, 153), (33, 133))
LEFT_EYE_EAR = ((387, 373), (384, 381), (362, 263))
MOUTH_VERTICAL = (13, 14)     # inner-lip midpoints
MOUTH_HORIZONTAL = (61, 291)  # mouth corners
NOSE_TIP = 1
INTER_OCULAR = (33, 263)      # outer eye corners

# Landmark subset stored as insurance (C3): eye contours, iris, mouth outer, brows.
RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
IRIS = list(range(468, 478))
MOUTH_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
BROWS = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]
LANDMARK_SUBSET = RIGHT_EYE_CONTOUR + LEFT_EYE_CONTOUR + IRIS + MOUTH_OUTER + BROWS

BLENDSHAPE_NAMES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft",
    "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker",
    "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
    "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft",
    "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
]

ROTATIONS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
             270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def ensure_landmarker_model(path: str | Path) -> Path:
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading landmarker model → {path}")
        urllib.request.urlretrieve(LANDMARKER_URL, path)
    return path


def make_landmarker(model_path: str | Path) -> mp_vision.FaceLandmarker:
    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def probe_rotation_metadata(video_path: str | Path) -> int:
    """Rotation in degrees from container metadata via ffprobe (0 if none)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams",
         "-show_entries", "side_data", "-of", "json", str(video_path)],
        capture_output=True, text=True,
    ).stdout
    try:
        streams = json.loads(out).get("streams", [])
    except json.JSONDecodeError:
        return 0
    for stream in streams:
        rot = stream.get("tags", {}).get("rotate")
        if rot is None:
            for sd in stream.get("side_data_list", []):
                if "rotation" in sd:
                    rot = sd["rotation"]
                    break
        if rot is not None:
            return int(float(rot)) % 360
    return 0


def euler_from_matrix(m: np.ndarray) -> tuple[float, float, float]:
    """(pitch, yaw, roll) degrees from the 4x4 facial transformation matrix."""
    r = np.asarray(m)[:3, :3]
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    yaw = math.degrees(math.atan2(-r[2, 0], math.hypot(r[2, 1], r[2, 2])))
    roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return pitch, yaw, roll


def _dist(a, b) -> float:
    return math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))


def frame_features(result) -> dict:
    """One feature row from a FaceLandmarkerResult (face present)."""
    lms = result.face_landmarks[0]
    row: dict = {"face_detected": True, "n_faces": len(result.face_landmarks)}

    for cat in result.face_blendshapes[0]:
        row[f"bs_{cat.category_name}"] = cat.score

    scale = _dist(lms[INTER_OCULAR[0]], lms[INTER_OCULAR[1]])
    for side, ((v1a, v1b), (v2a, v2b), (ha, hb)) in (
        ("right", RIGHT_EYE_EAR), ("left", LEFT_EYE_EAR)):
        v = _dist(lms[v1a], lms[v1b]) + _dist(lms[v2a], lms[v2b])
        row[f"ear_{side}"] = v / (2 * _dist(lms[ha], lms[hb]))
    row["mar"] = (_dist(lms[MOUTH_VERTICAL[0]], lms[MOUTH_VERTICAL[1]])
                  / _dist(lms[MOUTH_HORIZONTAL[0]], lms[MOUTH_HORIZONTAL[1]]))

    if result.facial_transformation_matrixes:
        rot = np.asarray(result.facial_transformation_matrixes[0])[:3, :3]
        row["pitch"], row["yaw"], row["roll"] = euler_from_matrix(
            result.facial_transformation_matrixes[0])
    else:
        rot = np.eye(3)
        row["pitch"] = row["yaw"] = row["roll"] = np.nan

    # normalized landmark subset: nose-tip centered, inter-ocular scaled,
    # rotation-canonicalized (C3 insurance features)
    pts = np.array([(lms[i].x, lms[i].y, lms[i].z) for i in LANDMARK_SUBSET],
                   dtype=np.float64)
    nose = np.array([lms[NOSE_TIP].x, lms[NOSE_TIP].y, lms[NOSE_TIP].z])
    canon = ((pts - nose) / max(scale, 1e-9)) @ rot
    for idx, (x, y, z) in zip(LANDMARK_SUBSET, canon):
        row[f"lm{idx}_x"], row[f"lm{idx}_y"], row[f"lm{idx}_z"] = x, y, z

    xs = [lm.x for lm in lms]
    ys = [lm.y for lm in lms]
    row["bbox_area_frac"] = max(0.0, (max(xs) - min(xs)) * (max(ys) - min(ys)))
    return row


def feature_columns() -> list[str]:
    cols = ["t_sec", "frame_idx", "face_detected", "n_faces"]
    cols += [f"bs_{n}" for n in BLENDSHAPE_NAMES]
    cols += ["ear_left", "ear_right", "mar", "pitch", "yaw", "roll"]
    for idx in LANDMARK_SUBSET:
        cols += [f"lm{idx}_x", f"lm{idx}_y", f"lm{idx}_z"]
    cols += ["bbox_area_frac"]
    return cols


class FrameSampler:
    """Sequential reader yielding (frame_idx, t_sec, rotated_bgr_frame) at
    ~target_fps, sampled by timestamp (source fps varies across RLDD videos).
    After iteration, `src_fps` and `n_frames` hold the source stats."""

    def __init__(self, video_path, rotation_deg: int, target_fps: float):
        self.video_path = str(video_path)
        self.rotate_flag = ROTATIONS.get(rotation_deg % 360)
        self.target_fps = target_fps
        self.src_fps: float = 0.0
        self.n_frames: int = 0

    def __iter__(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"cv2 cannot open {self.video_path}")
        # be deterministic: we apply rotation ourselves from ffprobe metadata
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.src_fps = fps if fps and not math.isnan(fps) and fps > 0 else 30.0
        next_t, frame_idx = 0.0, 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                t = frame_idx / self.src_fps
                if t >= next_t:
                    if self.rotate_flag is not None:
                        frame = cv2.rotate(frame, self.rotate_flag)
                    yield frame_idx, t, frame
                    next_t += 1.0 / self.target_fps
                frame_idx += 1
                self.n_frames = frame_idx
        finally:
            cap.release()


def detection_rate(video_path, model_path, rotation_deg: int, target_fps: float,
                   max_processed: int) -> float:
    """Fraction of the first `max_processed` sampled frames with a face."""
    detected = processed = 0
    with make_landmarker(model_path) as lm:
        last_ts = -1
        for _, t, frame in FrameSampler(video_path, rotation_deg, target_fps):
            ts = max(int(t * 1000), last_ts + 1)
            last_ts = ts
            image = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if lm.detect_for_video(image, ts).face_landmarks:
                detected += 1
            processed += 1
            if processed >= max_processed:
                break
    return detected / max(processed, 1)


def choose_rotation(video_path, model_path, cfg) -> tuple[int, str]:
    """ffprobe metadata first; if detection mostly fails, probe 90/180/270 (§4.1)."""
    ex = cfg["extract"]
    rotation = probe_rotation_metadata(video_path)
    rate = detection_rate(video_path, model_path, rotation, ex["target_fps"],
                          ex["rotation_probe_frames"])
    if rate >= 1 - ex["rotation_fail_rate"]:
        return rotation, f"metadata rotation {rotation}° (probe detect rate {rate:.2f})"
    rates = {rotation: rate}
    for cand in (0, 90, 180, 270):
        if cand not in rates:
            rates[cand] = detection_rate(video_path, model_path, cand,
                                         ex["target_fps"], ex["rotation_probe_frames"])
    best = max(rates, key=rates.get)
    return best, f"probed rotations {rates} → {best}°"


def extract_video(video_path: str | Path, out_path: str | Path, cfg: dict,
                  meta: dict) -> dict:
    """Extract per-frame features for one video → parquet. Returns stats."""
    model_path = ensure_landmarker_model(cfg["paths"]["landmarker_task"])
    target_fps = cfg["extract"]["target_fps"]

    rotation, rotation_note = choose_rotation(video_path, model_path, cfg)
    print(f"  {Path(video_path).name}: {rotation_note}")

    rows = []
    sampler = FrameSampler(video_path, rotation, target_fps)
    with make_landmarker(model_path) as lm:
        last_ts = -1
        for frame_idx, t, frame in sampler:
            ts = max(int(t * 1000), last_ts + 1)
            last_ts = ts
            image = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = lm.detect_for_video(image, ts)
            if result.face_landmarks:
                row = frame_features(result)
            else:
                row = {"face_detected": False, "n_faces": 0}
            row["t_sec"] = t
            row["frame_idx"] = frame_idx
            rows.append(row)

    src_fps = sampler.src_fps
    n_source_frames = sampler.n_frames
    df = pd.DataFrame(rows, columns=feature_columns())
    df["face_detected"] = df["face_detected"].astype(bool)
    df["n_faces"] = df["n_faces"].fillna(0).astype(np.int8)
    df["frame_idx"] = df["frame_idx"].astype(np.int32)
    float_cols = df.columns.difference(["face_detected", "n_faces", "frame_idx"])
    df[float_cols] = df[float_cols].astype(np.float32)

    metadata = {
        **meta,
        "source_fps": f"{src_fps:.4f}",
        "n_source_frames": str(n_source_frames),
        "duration_sec": f"{n_source_frames / src_fps:.3f}",
        "rotation_applied": str(rotation),
        "rotation_note": rotation_note,
        "target_fps": str(target_fps),
        "extractor_version": str(EXTRACTOR_VERSION),
        "mediapipe_version": mp.__version__,
    }
    table = pa.Table.from_pandas(df, preserve_index=False)
    table = table.replace_schema_metadata(
        {**(table.schema.metadata or {}),
         **{k.encode(): str(v).encode() for k, v in metadata.items()}})
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # zstd: ~35% smaller than default snappy, keeps the 180-video total
    # under milestone 3's 1.5 GB budget
    pq.write_table(table, out_path, compression="zstd")

    return {
        "rows": len(df),
        "duration_sec": n_source_frames / src_fps,
        "face_detected_rate": float(df["face_detected"].mean()) if len(df) else 0.0,
        "rotation": rotation,
    }
