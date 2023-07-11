import cv2 as cv
import mediapipe as mp
import numpy as np
import time
from collections import deque

DEBUG = True

class PERCLOS:

  def __init__(self, video_capture, window_size = 100):
    self.video_capture = video_capture
    self.frame_height, self.frame_width, *_ = self.video_capture.read()[1].shape
    self.face_mesh = mp.solutions.face_mesh.FaceMesh()
    self.frame_counter = 0  
    self.window = deque([], maxlen = window_size)
    self.drowsy = False

  def run(self):

    start_time = time.time()

    while self.video_capture.isOpened():
      # Capture frame-by-frame
      ret, frame = self.video_capture.read()
    
      if not ret: break
      
      eye_aspect_ratio, awareness_level = self.detect_awareness(frame)

      if DEBUG:
        # Display awareness level
        cv.putText(frame, f"EAR: {eye_aspect_ratio:.3f}", (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
        
        # Display awareness level
        cv.putText(frame, f"Drowsiness Level: {awareness_level:.3f}", (10, 80), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        
        # Draw left eye outline
        cv.polylines(frame, [np.array([landmarks[p] for p in LEFT_EYE ], 
                                                       dtype=np.int32)], 
                     True, (0,255,0), 1, cv.LINE_AA) 
        
        # Draw right eye outline
        cv.polylines(frame, [np.array([landmarks[p] for p in RIGHT_EYE ], 
                                                       dtype=np.int32)], 
                     True, (0,255,0), 1, cv.LINE_AA) 
        if self.drowsy:
             
          # Display awareness level
          cv.putText(frame, "DROWSY!!!", (100, 250), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255))
      
        # Display frame
        cv.imshow("Drowsiness Detection", frame)
        
        time.sleep(0.1)  
    
      # Calculate FPS
      self.frame_counter += 1
      if cv.waitKey(1) & 0xFF == ord('q'):
        break
  def detect_awareness(self, frame):
    
    rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    
    
    # Detect facial features
    results = self.face_mesh.process(rgb_frame)
    
    # no face detected 
    if results.multi_face_landmarks:
      
      # get landmarks from results
      landmarks = self.get_landmarks(results)
     
      # Calculate EAR
      eye_aspect_ratio = self.calculate_ear(landmarks)
      
      # Calculate PERCLOS
      awareness_level = self.calculate_perclos(landmarks)
      
      # Change state to drowsy
      if awareness_level < .8: self.drowsy = True
      else: self.drowsy = False

    return eye_aspect_ratio, awareness_level

  def calculate_perclos(self, results):
    # Average EAR
    ear = self.calculate_ear(results)
    
    if ear < 0.22:
      self.window.append(0)
    else:
      self.window.append(1)

    return sum(self.window) / len(self.window)

  def calculate_head_angle(self, coords):
    # Calculate EAR
    right_eye_v1 = self.euclidean_distance(coords[160] , coords[144]) 
    right_eye_v2 = self.euclidean_distance(coords[158] , coords[153]) 
    right_eye_h  = self.euclidean_distance(coords[33 ] , coords[133])*2
    left_eye_v1  = self.euclidean_distance(coords[387] , coords[373]) 
    left_eye_v2  = self.euclidean_distance(coords[384] , coords[381]) 
    left_eye_h   = self.euclidean_distance(coords[362] , coords[263])*2
    
    right_eye = (right_eye_v1 + right_eye_v2) / right_eye_h
    left_eye = (left_eye_v1 + left_eye_v2) / left_eye_h
    
    return (right_eye + left_eye) / 2
    
    
  @staticmethod
  def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
  
  def get_landmarks(self, results):
    return [(int(point.x * self.frame_width), int(point.y * self.frame_height)) for
            point in  results.multi_face_landmarks[0].landmark]
    
    

if __name__ == "__main__":
  # Create video capture object
  video_capture = cv.VideoCapture(0)

  # Create PERCLOS object
  perclos = PERCLOS(video_capture)

  # Detect awareness
  perclos.detect_awareness()

