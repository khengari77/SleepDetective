from collections import deque
import cv2
import numpy as np


class HeadPose:
    def __init__(self, frame_width, frame_height, window_size):
        self.window = deque([1]*window_size, maxlen=window_size)
        self.angle_window = deque([0]*window_size, maxlen=window_size)
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.indices = [33, 263, 1, 61, 291, 199]
        self.focal_length = self.frame_width * 1
        self.cam_matrix = np.array([
                    [self.focal_length, 0, self.frame_height / 2],
                    [0, self.focal_length, self.frame_width / 2],
                    [0, 0, 1]])
        self.dist_matrix = np.zeros((4, 1), dtype=np.float64)
        self.awareness_level = 1 
        self.head_angle = 0
        self.average_angle = 0
        self.std_angle = 0
        self.calibration_mode = False

    def take(self, landmarks):
        x, y, z = self.get_angles(landmarks)
        self.head_angle = x
        if self.calibration_mode:
            self.angle_window.append(x)
            self.average_angle = np.mean(self.angle_window)
            self.std_angle = np.std(self.angle_window)
        self.awareness_level = self.get_awareness_level(x)

# Returns 2D & 3D meshes of the face.
    def get_faces(self, landmarks):
        face_2d = []
        face_3d = []
        for idx, (x, y, z) in enumerate(landmarks):
            if idx in self.indices:
                point = [self.frame_width * x, self.frame_height * y, z]
                face_3d += [point]
                face_2d += [point[:-1]]
        return np.array(face_2d, dtype=np.float64), np.array(face_3d,
                                                             dtype=np.float64)

    def get_angles(self, landmarks):
        face_2d, face_3d = self.get_faces(landmarks)

        # Solve PnP
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d,
                                                   self.cam_matrix,
                                                   self.dist_matrix)
        # Get rotational matrix
        rmat, jac = cv2.Rodrigues(rot_vec)

        # Get angles
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        return [angle * 360 for angle in angles]

    def get_awareness_level(self, x):
        if abs(x - self.average_angle) > 10:
            self.window.append(0)
        else:
            self.window.append(1)
        return np.mean(self.window)
