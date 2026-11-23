import numpy as np
import cv2
import mediapipe as mp
import joblib
import pandas as pd
import time
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# File paths
MODEL_PATH = "backend/models/signbridge_model.pkl"
HAND_MODEL_PATH = "backend/models/hand_landmarker.task"

# Check whether files exist
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Trained model not found!")

if not os.path.exists(HAND_MODEL_PATH):
    raise FileNotFoundError("MediaPipe hand model not found!")

# Load trained ML model
model = joblib.load(MODEL_PATH)

# Configure MediaPipe
options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=HAND_MODEL_PATH
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam!")

print("\nSignBridge Live Prediction Started!")
print("Show one hand sign in front of your webcam.")
print("Press CTRL + C in the terminal to stop.\n")

last_print_time = 0

try:
    with vision.HandLandmarker.create_from_options(options) as detector:

        while True:
            success, frame = cap.read()

            if not success:
                print("Could not read webcam frame.")
                break

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Detect hand
            timestamp_ms = int(time.monotonic() * 1000)

            result = detector.detect_for_video(
                mp_image, timestamp_ms
            )

            current_time = time.time()

            if current_time - last_print_time >= 1:

                if result.hand_landmarks:
                    landmarks = result.hand_landmarks[0]

                    coordinates = []

                    for point in landmarks:
                        coordinates.extend([
                            point.x,
                            point.y,
                            point.z
                        ])

                    # Prepare features for prediction
                    
                    # Convert landmarks into a NumPy array
                    landmarks_array = np.array(coordinates).reshape(21, 3)

                    # Make coordinates wrist-relative
                    landmarks_array = landmarks_array - landmarks_array[0]

                    # Normalize hand size
                    hand_size = np.linalg.norm(
                        landmarks_array, axis=1
                    ).max()

                    if hand_size > 0:
                        landmarks_array = landmarks_array / hand_size

                        # Convert to 63 features
                        input_data = landmarks_array.flatten().reshape(1, 63)

                        # Predict sign
                        prediction = model.predict(input_data)[0]

                        probabilities = model.predict_proba(input_data)[0]
                        confidence = max(probabilities) * 100

                        print(
                            f"Predicted Sign: {prediction} | "
                            f"Confidence: {confidence:.2f}%"
                        )
                    else:
                        print("Unable to detect a clear hand shape.")

except KeyboardInterrupt:
    print("\nPrediction stopped.")

finally:
    cap.release()
    print("Webcam released.")
