from modules.VideoCapture import VideoCapture
from modules.AwarenessTracker import AwarenessTracker
from modules.ActionTaker import ActionTaker

from flask import Flask, render_template, Response
import cv2
import os
from threading import Thread

app = Flask(__name__)

capture = VideoCapture(0).start()
sleep_tracker = AwarenessTracker(capture.frame)
action_taker = ActionTaker().start()

@app.route('/video_feed')
def video_feed():
    return Response(get_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/', methods=["POST", "GET"])
def index():
    data = sleep_tracker.get_data()
    data.update(action_taker.get_data())
    return render_template('index.html', data=data)

@app.route('/shutdown')
def shutdown():
    os.system("shutdown 0")
    return render_template('index.html', data=sleep_tracker.get_data())


def get_frame():
    while True:
            _, buffer = cv2.imencode('.jpg', capture.frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
             b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
def update():
    while True:
        try:
            sleep_tracker.take(capture.frame)
            action_taker.take(sleep_tracker.drowsy)
        except KeyboardInterrupt:
            capture.stop()
            action_taker.stop()
            break

def main():
    Thread(target=update).start()
    app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False, debug=True)
    

if __name__ == '__main__':
    main()
