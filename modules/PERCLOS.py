from collections import deque
import numpy as np

class PERCLOS:

    def __init__(self, frame_width, frame_height, window_size):
        self.window = deque([1]*window_size, maxlen=window_size)
        self.ear_window = deque([0]*window_size, maxlen=window_size)
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.eye_aspect_ratio = 0
        self.awareness_level = 1
        self.calibration_mode = False
        self.average_eye_aspect_ratio = 0
        self.std_eye_aspect_ratio = 0


    def take(self, landmarks):
        self.eye_aspect_ratio = self.calculate_ear(landmarks)
        if self.calibration_mode:
            self.ear_window.append(self.eye_aspect_ratio)
            self.average_eye_aspect_ratio = np.mean(self.ear_window)
            self.std_eye_aspect_ratio = np.std(self.ear_window)
        self.awareness_level = self.calculate_perclos(self.eye_aspect_ratio)

    def calculate_perclos(self, eye_aspect_ratio):
        if eye_aspect_ratio < (self.average_eye_aspect_ratio - self.std_eye_aspect_ratio):
            self.window.append(0)
        else:
            self.window.append(1)

        return np.mean(self.window)

    def calculate_ear(self, coords):
        # Calculate EAR
        right_eye_v1 = self.euclidean_distance(coords[160], coords[144])
        right_eye_v2 = self.euclidean_distance(coords[158], coords[153])
        right_eye_h = self.euclidean_distance(coords[33], coords[133]) * 2
        left_eye_v1 = self.euclidean_distance(coords[387], coords[373])
        left_eye_v2 = self.euclidean_distance(coords[384], coords[381])
        left_eye_h = self.euclidean_distance(coords[362], coords[263]) * 2

        right_eye = (right_eye_v1 + right_eye_v2) / right_eye_h
        left_eye = (left_eye_v1 + left_eye_v2) / left_eye_h

        return (right_eye + left_eye) / 2

    def euclidean_distance(self, point1, point2):
        x1, y1, *_ = point1
        x2, y2, *_ = point2

        return (((x1 - x2) * self.frame_width) ** 2
                + ((y1 - y2) * self.frame_height) ** 2) ** 0.5
