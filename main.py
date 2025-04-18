import os
from threading import Thread
import cv2
import argparse
import time
import threading
from flask import Flask, render_template, Response, jsonify

from modules.VideoCapture import VideoCapture
from modules.AwarenessTracker import AwarenessTracker
from modules.FacialFeatures import FacialFeatures
from modules.ActionTaker import ActionTaker

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--use-gpio", action=argparse.BooleanOptionalAction)
parser.add_argument("--use-gsm", action=argparse.BooleanOptionalAction)
parser.add_argument("--number", type=str, default="+218922653284")
parser.add_argument("--gsm-port", type=str, default="/dev/ttyUSB2")
parser.add_argument("--use-picamera", action=argparse.BooleanOptionalAction)
parser.add_argument("--draw-landmarks", action=argparse.BooleanOptionalAction)
parser.add_argument("--fps", type=int, default=30, help="Video capture FPS")
args = parser.parse_args()


# Initialize Flask app
app = Flask(__name__)

# Global variables to store the latest frame and data, protected by a lock
latest_frame = None
latest_data = None
lock = threading.Lock()

# Initialize and start the modules
capture = VideoCapture(0, use_picamera=args.use_picamera, fps=args.fps).start()
while capture.frame is None:
    pass
features = FacialFeatures(capture.frame, show_landmarks=args.draw_landmarks).start()
sleep_tracker = AwarenessTracker(capture.frame)

# Initialize and start ActionTaker if GPIO is enabled
print("GPIO enabled" if args.use_gpio else "GPIO disabled")
print("GSM enabled" if args.use_gsm else "GSM disabled")
action_taker = ActionTaker(use_gpio=args.use_gpio, use_gsm=args.use_gsm)
action_taker.start(args.gsm_port, args.number)

# Processing loop to handle video and data updates
def processing_loop():
    global latest_frame, latest_data
    # calibration loop.
    sleep_tracker.calibration_mode = True
    for i in range(10*args.fps):
        frame = capture.frame  # Get the latest frame from the camera
        if frame is not None:
            features.take(frame)  # Process facial features
            sleep_tracker.take(features.mesh_result)  # Update awareness data
            with lock:  # Safely update global variables
                latest_frame = features.frame.copy() if features.frame is not None else None
                latest_data = sleep_tracker.data
                latest_data["calibration_frame"] = i
        if capture.stopped:  # Exit condition
            break
    sleep_tracker.calibration_mode = False 
    while True:
        frame = capture.frame  # Get the latest frame from the camera
        if frame is not None:
            features.take(frame)  # Process facial features
            sleep_tracker.take(features.mesh_result)  # Update awareness data
            action_taker.take(sleep_tracker.awareness_level)
            with lock:  # Safely update global variables
                latest_frame = features.frame.copy() if features.frame is not None else None
                latest_data = sleep_tracker.data
                latest_data['location'] = action_taker.gps.location
                latest_data['time'] = action_taker.gps.time
        if capture.stopped:  # Exit condition
            break

# Start the processing loop in a separate thread
processing_thread = Thread(target=processing_loop)
processing_thread.daemon = True  # Allow thread to exit when main program exits
processing_thread.start()

# Generator function to stream video frames
def generate_frames():
    while True:
        with lock:
            if latest_frame is not None:
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', latest_frame)
                frame_bytes = buffer.tobytes()
                # Yield frame in the multipart format for streaming
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                # Yield an empty frame if no frame is available yet
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n')

# Flask route for video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Flask route for awareness data
@app.route('/data')
def get_data():
    with lock:
        return jsonify(latest_data if latest_data is not None else {})

# Flask route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Main entry point
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
