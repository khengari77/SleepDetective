from threading import Thread
#from flask import Flask, Response
import cv2

class VideoViewer:
    """
    Class that continuously shows a frame using a dedicated thread.
    """

    def __init__(self, frame=None):
        self.frame = frame
        self.stopped = False

    def take(self, frame, data):
        self.frame = frame

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        while not self.stopped:
#            cv2.imshow("Video", self.frame)
            if cv2.waitKey(1) == ord("q"):
                self.stopped = True

    def stop(self):
        self.stopped = True


#class VideoViewer:
#    """
#    Class that continuously shows a frame using a dedicated thread.
#    """
#
#    def __init__(self, frame=None):
#        self.frame = frame
#        self.stopped = False
#
#    def take(self, frame, data):
#        self.frame = frame
#
#    def start(self):
#        Thread(target=self.show, args=()).start()
#        return self
#
#    def show(self):
#        while not self.stopped:
#            cv2.imshow("Video", self.frame)
#            if cv2.waitKey(1) == ord("q"):
#                self.stopped = True
#
#    def stop(self):
#        self.stopped = True
#
#
#class VideoViewer:
#    """
#    Class that continuously shows a frame using a dedicated thread.
#    """
#
#    def __init__(self, frame=None):
#        self.frame = frame
#        self.flask_app = Flask(__name__)
#        self.data = {}
#        self.stopped = False
#
#    def take(self, frame, data):
#        self.frame = frame
#        self.data = data
#
#    def start(self):
#        Thread(target=self.show, args=()).start()
#        self.flask_app.run(host='0.0.0.0', port=2204, threaded=True)
#        return self
#
#    def show(self):
#        def gen():
#            ret, jpeg = cv2.imencode('.jpg', self.frame)
#            frame = jpeg.tobytes()
#            yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
#
#        @self.flask_app.route('/')
#        def video_feed():
#            return Response(gen(),
#                        mimetype='multipart/x-mixed-replace; boundary=frame')
#
#    def stop(self):
#        self.stopped = True
