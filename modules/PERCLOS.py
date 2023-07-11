from collections import deque


class PERCLOS:

    def __init__(self, frame_width, frame_height, window_size):
        self.window = deque(maxlen=window_size)
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.eye_aspect_ratio = 0
        self.awareness_level = 1

    def take(self, landmarks):
        self.eye_aspect_ratio = self.calculate_ear(landmarks)
        self.awareness_level = self.calculate_perclos(self.eye_aspect_ratio)

    def calculate_perclos(self, eye_aspect_ratio):
        if eye_aspect_ratio < 0.22:
            self.window.append(0)
        else:
            self.window.append(1)

        return sum(self.window) / len(self.window)

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
