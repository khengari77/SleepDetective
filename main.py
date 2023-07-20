from modules.VideoCapture import VideoCapture
from modules.VideoViewer import VideoViewer
from modules.AwarenessTracker import AwarenessTracker
from modules.ActionTaker import ActionTaker

def main():
    capture = VideoCapture(0).start()
    viewer = VideoViewer(capture.frame).start()
    sleep_tracker = AwarenessTracker(capture.frame)
    action_taker = ActionTaker().start()
    while True:
        try:
            sleep_tracker.take(capture.frame)
            viewer.take(capture.frame, {}) 
            action_taker.take(sleep_tracker.drowsy)
            print(f"awareness_level: {sleep_tracker.awareness_level:.3f}")
        except KeyboardInterrupt:
            viewer.stop()
            capture.stop()
            action_taker.stop()
            break

if __name__ == '__main__':
    main()
