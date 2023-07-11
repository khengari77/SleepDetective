from modules.VideoCapture import VideoCapture
from modules.VideoViewer import VideoViewer
from modules.SleepTracker import SleepTracker


def main():
    capture = VideoCapture(0).start()
    viewer = VideoViewer(capture.frame).start()
    sleep_tracker = SleepTracker(capture.frame)
    while True:
        sleep_tracker.take(capture.frame)
        viewer.frame = capture.frame
        print(f"awareness_level: {sleep_tracker.awareness_level:.3f}")
        if capture.stopped or viewer.stopped:
            viewer.stop()
            capture.stop()
            break

if __name__ == '__main__':
    main()
