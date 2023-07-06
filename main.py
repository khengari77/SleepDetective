from modules.VideoCapture import VideoCapture
from modules.VideoViewer import VideoViewer
from modules.SleepTracker import SleepTracker


def main():
    capture = VideoCapture("./test.mp4").start()
    viewer = VideoViewer(capture.frame).start()
    sleep_tracker = SleepTracker(capture.frame)
    while True:
        frame = sleep_tracker.take(capture.frame)
        viewer.frame = frame
        if capture.stopped or viewer.stopped:
            viewer.stop()
            capture.stop()
            break

if __name__ == '__main__':
    main()
