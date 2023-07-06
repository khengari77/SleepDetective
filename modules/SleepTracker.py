import mediapipe as mp
from .PERCLOS import PERCLOS
import cv2


class SleepTracker:
    def __init__(self, frame):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh()
        self.frame_height, self.frame_width, *_ = frame.shape
        self.perclos = PERCLOS(self.frame_width, self.frame_height)

    def take(self, frame):
        frame.flags.writeable = False
        mesh_result = self.face_mesh.process(cv2.cvtColor(frame,
                                                         cv2.COLOR_RGB2BGR))
        frame.flags.writeable = True
        if mesh_result.multi_face_landmarks is not None:
            landmarks = self.get_landmarks(mesh_result)
            self.perclos.take(landmarks)
        cv2.putText(frame, f"PERCLOS: {self.perclos.drowsiness_level:.2f}",
                    (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
        cv2.putText(frame, f"EAR: {self.perclos.eye_aspect_ratio:.2f}",
                    (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
        return frame

    @ staticmethod
    def get_landmarks(mesh_result):
        return [(point.x, point.y, point.z) for point
                in mesh_result.multi_face_landmarks[0].landmark]
