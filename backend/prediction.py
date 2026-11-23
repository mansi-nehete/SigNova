
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
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=2
)

detector = mp.tasks.vision.HandLandmarker.create_from_options(options)


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


def predict_sign(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_image
    )

    result = detector.detect(mp_image)
    detected_hands = result.hand_landmarks

    if len(detected_hands) == 0:
        return {
            "prediction": None,
            "message": "No hands detected"
        }

    features = normalize_landmarks(detected_hands)
    prediction = model.predict(features)[0]

    return {
        "prediction": str(prediction),
        "message": "Sign detected",
        "hands_detected": len(detected_hands)
    }
