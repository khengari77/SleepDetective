from threading import Thread
import cv2 
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class FacialFeatures:

    def __init__(self, frame=None, show_landmarks=False):
        self.frame = frame
        self.stopped = False
        self.face_mesh = mp_face_mesh.FaceMesh()
        self.new_frame = False
        self.mesh_result = None
        self.show_landmarks = show_landmarks

    def take(self, frame):
        self.frame = frame
        self.new_frame = True

    def start(self):
        Thread(target=self.process, args=()).start()
        return self

    def process(self):
        while not self.stopped:
            if self.new_frame:
                self.frame.flags.writeable = False
                self.mesh_result = self.face_mesh.process(cv2.cvtColor(self.frame,
                                                         cv2.COLOR_RGB2BGR))
                self.frame.flags.writeable = True
                if self.mesh_result.multi_face_landmarks is not None \
                and self.show_landmarks:
                    mp_drawing.draw_landmarks(
                        image=self.frame,
                        landmark_list=self.mesh_result.multi_face_landmarks[0],
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())


    def stop(self):
        self.stopped = True


