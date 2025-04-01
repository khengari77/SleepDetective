from abc import ABC, abstractmethod
from threading import Thread, Lock
import cv2

# Abstract base class for video streams
class VideoStream(ABC):
    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def get_fps(self):
        pass

# OpenCV stream implementation
class OpenCVStream(VideoStream):
    def __init__(self, src, fps=None):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            raise RuntimeError(f"Failed to open video source: {src}")
        if fps and isinstance(src, int):  # For cameras, attempt to set FPS
            self.stream.set(cv2.CAP_PROP_FPS, fps)
        # Use provided fps, fallback to stream FPS, or default to 30
        self.fps = fps if fps else self.stream.get(cv2.CAP_PROP_FPS) or 30

    def read(self):
        return self.stream.read()

    def stop(self):
        self.stream.release()

    def get_fps(self):
        return self.fps

# Picamera2 stream implementation
class PicameraStream(VideoStream):
    def __init__(self, fps=None):
        from picamera2 import Picamera2
        self.stream = Picamera2()
        frame_duration = int(1e6 / (fps or 30))  # Convert FPS to microseconds
        config = self.stream.create_video_configuration(
            main={"size": (640, 480), "format": "BGR888"},
            controls={"FrameDurationLimits": (frame_duration, frame_duration)}
        )
        self.stream.configure(config)
        self.stream.start()
        self.fps = fps or 30

    def read(self):
        frame = self.stream.capture_array()
        return (not (frame is None), frame)

    def stop(self):
        self.stream.stop()

    def get_fps(self):
        return self.fps

# Optimized VideoCapture class
class VideoCapture:
    def __init__(self, src=0, use_picamera=False, fps=None):
        """Initialize the video capture with a source and optional FPS."""
        if use_picamera:
            self.stream = PicameraStream(fps)
        else:
            self.stream = OpenCVStream(src, fps)
        self._grabbed, self._frame = self.stream.read()
        if not self._grabbed:
            raise RuntimeError("Failed to capture initial frame")
        self.fps = self.stream.get_fps()
        self.stopped = False
        self.lock = Lock()

    @property
    def frame(self):
        """Get the latest frame with thread-safe access."""
        with self.lock:
            return self._frame

    @frame.setter
    def frame(self, value):
        """Set the latest frame with thread-safe access."""
        with self.lock:
            self._frame = value

    @property
    def grabbed(self):
        """Get the grabbed status with thread-safe access."""
        with self.lock:
            return self._grabbed

    @grabbed.setter
    def grabbed(self, value):
        """Set the grabbed status with thread-safe access."""
        with self.lock:
            self._grabbed = value

    def start(self):
        """Start the frame capture thread."""
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        """Continuously update the frame in a separate thread."""
        while not self.stopped:
            grabbed, frame = self.stream.read()
            with self.lock:
                self._grabbed = grabbed
                self._frame = frame
            if not grabbed:
                self.stop()

    def stop(self):
        """Stop the frame capture and release resources."""
        self.stopped = True
        self.stream.stop()

    def __enter__(self):
        """Context manager support: start capturing."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager support: stop capturing."""
        self.stop()
