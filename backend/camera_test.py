
import cv2
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera unavailable!")
    raise SystemExit

print("Camera started. Get ready...")

# Allow camera to adjust lighting and focus
time.sleep(2)

# Capture a fresh frame
success, frame = cap.read()

if success:
    filename = "backend/camera_test_frame.jpg"

    cv2.imwrite(filename, frame)

    print("New image captured!")
    print("Saved as:", filename)
else:
    print("Could not capture image!")

cap.release()
