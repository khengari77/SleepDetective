import mediapipe as mp
from .PERCLOS import PERCLOS
from .HeadPose import HeadPose
import cv2

THRESHOLD = 0.75

class AwarenessTracker:
    def __init__(self, frame, window_size=200):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh()
        self.frame_height, self.frame_width, *_ = frame.shape
        self.window_size = window_size
        self.perclos = PERCLOS(self.frame_width, self.frame_height,
                               self.window_size)
        self.head_pose = HeadPose(self.frame_width, self.frame_height,
                                  self.window_size)
        self.awareness_level = 1
        self.drowsy = False
        self.data = self.get_data()

    def take(self, frame):
        frame.flags.writeable = False
        mesh_result = self.face_mesh.process(cv2.cvtColor(frame,
                                                         cv2.COLOR_RGB2BGR))
        frame.flags.writeable = True
        if mesh_result.multi_face_landmarks is not None:
            landmarks = self.get_landmarks(mesh_result)
            self.perclos.take(landmarks)
            self.head_pose.take(landmarks)
            self.awareness_level = self.perclos.awareness_level * 0.5 +\
                    self.head_pose.awareness_level * 0.5
            if self.awareness_level < THRESHOLD:
                self.drowsy = True
            else:
                self.drowsy = False
        self.data = self.get_data()

    def get_data(self):
        return {'Eye Aspect Ratio': self.perclos.eye_aspect_ratio,
                'PERCLOS': self.perclos.awareness_level,
                'Head Angle': self.head_pose.head_angle,
                'Head Pose': self.head_pose.awareness_level,
                'Average Awareness Level': self.awareness_level}

    @ staticmethod
    def get_landmarks(mesh_result):
        return [(point.x, point.y, point.z) for point
                in mesh_result.multi_face_landmarks[0].landmark]

