import cv2
import numpy as np

class MotionDetector:
    def __init__(self):
        self.prev_frame = None

    def detect_motion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_frame is None:
            self.prev_frame = gray
            return False

        # Compute absolute difference between frames
        diff = cv2.absdiff(self.prev_frame, gray)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        # Count non-zero pixels (motion detection)
        motion_detected = np.count_nonzero(thresh) > 5000
        self.prev_frame = gray  # Update previous frame
        return motion_detected
