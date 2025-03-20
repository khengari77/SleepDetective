import os
from threading import Thread
import cv2
import argparse
import json

from modules.VideoCapture import VideoCapture
from modules.AwarenessTracker import AwarenessTracker
from modules.VideoViewer import VideoViewer
from modules.FacialFeatures import FacialFeatures
from modules.ActionTaker import ActionTaker

parser = argparse.ArgumentParser()
parser.add_argument("--gpio", action=argparse.BooleanOptionalAction) 
args = parser.parse_args()

capture = VideoCapture(0).start()
features = FacialFeatures(capture.frame, show_landmarks=True).start()
sleep_tracker = AwarenessTracker(capture.frame)
viewer = VideoViewer(capture.frame).start()

action_taker = ActionTaker()
print("GPIO enabled" if args.gpio else "GPIO disabled")
if args.gpio:
    action_taker.start()


def main():
    while True:
        features.take(capture.frame)
        sleep_tracker.take(features.mesh_result)
        viewer.take(features.frame, {})
        print(json.dumps(sleep_tracker.get_data()))
        if capture.stopped or viewer.stopped:
            viewer.stop()
            capture.stop()
            break

if __name__ == '__main__':
    main()
