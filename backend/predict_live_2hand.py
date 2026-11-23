
from collections import deque, Counter
import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load trained model
MODEL_PATH = "backend/models/signbridge_model_2hand.pkl"
model = joblib.load(MODEL_PATH)

# Initialize MediaPipe
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path="backend/models/hand_landmarker.task"
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=2
)

def normalize_landmarks(detected_hands):
    ordered_hands = sorted(
        detected_hands,
        key=lambda hand: np.mean([point.x for point in hand])
    )

    features = []

    for landmarks in ordered_hands:
        hand_data = np.array(
            [[point.x, point.y, point.z] for point in landmarks],
            dtype=float
        )

        hand_data -= hand_data[0]

        hand_size = np.linalg.norm(hand_data, axis=1).max()

        if hand_size > 0:
            hand_data /= hand_size

        features.extend(hand_data.flatten())

    if len(ordered_hands) == 1:
        features.extend([0.0] * 63)

    return np.array(features).reshape(1, 126)


# Prediction history
prediction_history = deque(maxlen=10)

# Start webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Could not open webcam.")
    raise SystemExit

print("SignBridge Live Prediction Started!")
print("Show a hand sign. Press Q to quit.")

try:
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:

        while True:
            success, frame = cap.read()

            if not success:
                print("Could not read webcam frame.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp = int(
                cv2.getTickCount() * 1000 / cv2.getTickFrequency()
            )

            result = detector.detect_for_video(mp_image, timestamp)
            detected_hands = result.hand_landmarks

            prediction_text = "Show a hand sign"

            if len(detected_hands) in [1, 2]:
                features = normalize_landmarks(detected_hands)

                prediction = model.predict(features)[0]
                prediction_history.append(prediction)

                stable_prediction = Counter(
                    prediction_history
                ).most_common(1)[0][0]

                prediction_text = f"Prediction: {stable_prediction}"

            else:
                prediction_history.clear()

            display_frame = cv2.flip(frame, 1)

            cv2.putText(
                display_frame,
                prediction_text,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2
            )

            cv2.imshow("SignBridge - Live Recognition", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()
