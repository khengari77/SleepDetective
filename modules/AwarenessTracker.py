import mediapipe as mp
from .PERCLOS import PERCLOS
from .HeadPose import HeadPose
import cv2
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

THRESHOLD = 0.75

class AwarenessTracker:
    def __init__(self, frame, window_size=200):
        self.frame_height, self.frame_width, *_ = frame.shape
        self.window_size = window_size
        self.perclos = PERCLOS(self.frame_width, self.frame_height,
                               self.window_size)
        self.head_pose = HeadPose(self.frame_width, self.frame_height,
                                  self.window_size)
        self._awareness_level = 1
        self.drowsy = False

    @property
    def awareness_level(self):
        return self._awareness_level

    @property
    def data(self):
        return {'eye aspect ratio': self.perclos.eye_aspect_ratio,
                'average eye aspect ratio': self.perclos.average_eye_aspect_ratio,
                'std eye aspect ratio': self.perclos.std_eye_aspect_ratio,
                'perclos': self.perclos.awareness_level,
                'head angle': self.head_pose.head_angle,
                'average head angle': self.head_pose.average_angle,
                'std head angle': self.head_pose.std_angle,
                'head pose': self.head_pose.awareness_level,
                'average awareness level': self.awareness_level,
                }

    @property
    def calibration_mode(self):
        return self.perclos.calibration_mode and self.head_pose.calibration_mode

    @calibration_mode.setter
    def calibration_mode(self, value : bool):
        self.perclos.calibration_mode = value
        self.head_pose.calibration_mode = value

    def take(self, mesh_result):
        if mesh_result is None:
            return
        if mesh_result.multi_face_landmarks is not None:
            landmarks = self.get_landmarks(mesh_result)
            self.perclos.take(landmarks)
            self.head_pose.take(landmarks)
            if self.perclos.calibration_mode or self.head_pose.calibration_mode:
                return
            self._awareness_level = self.perclos.awareness_level * 0.5 +\
                    self.head_pose.awareness_level * 0.5
            if self.awareness_level < THRESHOLD:
                self.drowsy = True
            else:
                self.drowsy = False


    @ staticmethod
    def get_landmarks(mesh_result):
        return [(point.x, point.y, point.z) for point
                in mesh_result.multi_face_landmarks[0].landmark]

