from threading import Thread
import cv2


class VideoCapture:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0, use_picamera=False):
        if use_picamera:
            from picamera2 import Picamera2
            self.stream = Picamera2()
            config = self.stream.create_video_configuration(main={"size": (640, 480)}, lores={"size": (320, 240)}, display="lores")
            self.stream.configure(config)
            self.stream.start()
        else:
            self.stream = cv2.VideoCapture(src)
        if use_picamera:
            self.frame = self.stream.capture_array()
            self.grabbed = not (self.frame is None)
        else:
            (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.use_picamera = use_picamera

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                if self.use_picamera:
                    self.frame = self.stream.capture_array()
                    self.grabbed = not (self.frame is None)
                else:
                    (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        self.stopped = True
