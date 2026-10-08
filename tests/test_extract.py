"""Tests for configurable MediaPipe thresholds (no MediaPipe/video needed)."""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extract import EXTRACTOR_VERSION, landmarker_thresholds


class TestLandmarkerThresholds:
    def test_defaults_are_half(self):
        assert landmarker_thresholds(None) == {
            "min_face_detection_confidence": 0.5,
            "min_face_presence_confidence": 0.5,
            "min_tracking_confidence": 0.5,
        }

    def test_defaults_when_keys_absent(self):
        assert landmarker_thresholds({}) == landmarker_thresholds(None)

    def test_overrides_all_three(self):
        ex = {"min_face_detection_confidence": 0.3,
              "min_face_presence_confidence": 0.3,
              "min_tracking_confidence": 0.3}
        assert landmarker_thresholds(ex) == ex

    def test_partial_override_keeps_defaults(self):
        t = landmarker_thresholds({"min_face_detection_confidence": 0.2})
        assert t["min_face_detection_confidence"] == 0.2
        assert t["min_face_presence_confidence"] == 0.5
        assert t["min_tracking_confidence"] == 0.5


class TestConfig:
    def test_default_yaml_has_threshold_keys(self):
        cfg = yaml.safe_load(Path("configs/default.yaml").read_text())
        ex = cfg["extract"]
        for key in ("min_face_detection_confidence",
                    "min_face_presence_confidence", "min_tracking_confidence"):
            assert key in ex
            assert isinstance(ex[key], float)

    def test_extractor_version_matches_code(self):
        cfg = yaml.safe_load(Path("configs/default.yaml").read_text())
        assert cfg["extract"]["extractor_version"] == EXTRACTOR_VERSION
