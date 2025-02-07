import cv2
from camera import Camera
from motion_detection import MotionDetector
from face_detection import FaceDetector

# Initialize components
camera = Camera()
motion_detector = MotionDetector()
face_detector = FaceDetector()

while True:
    frame = camera.get_frame()
    if frame is None:
        break

    # Detect motion
    if motion_detector.detect_motion(frame):
        cv2.putText(frame, "Motion Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Detect faces
    faces = face_detector.detect_faces(frame)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Face Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Show the video feed
    cv2.imshow("Surveillance Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
