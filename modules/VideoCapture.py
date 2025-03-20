from threading import Thread
from picamera2 import Picamera2
import cv2


class VideoCapture:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0, use_picamera=False):
        if use_picamera:
            self.stream = Picamera2()

            camera_config = self.stream.create_preview_configuration()
            self.stream.configure(camera_config)
            self.stream.start()
        else:
            self.stream = cv2.VideoCapture(src)
        if use_picamera:
            self.frame = self.stream.capture_array()
            self.grabbed = True
        else:
            (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        self.stopped = True
