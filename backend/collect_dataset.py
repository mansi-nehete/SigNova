
import time
import cv2
import mediapipe as mp
import csv
import os
import numpy as np
from datetime import datetime

# =====================================
# 1. FILE PATHS
# =====================================

MODEL_PATH = "backend/models/hand_landmarker.task"
DATASET_FOLDER = "backend/dataset"

RAW_FOLDER = os.path.join(DATASET_FOLDER, "raw_2hand")
CSV_PATH = os.path.join(DATASET_FOLDER, "landmarks_2hand.csv")

os.makedirs(RAW_FOLDER, exist_ok=True)

# =====================================
# 2. MEDIAPIPE SETUP
# =====================================

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=2
)

# =====================================
# 3. CSV HEADERS
# =====================================

headers = []

for hand in range(2):
    for i in range(21):
        headers.extend([
            f"hand{hand+1}_x{i}",
            f"hand{hand+1}_y{i}",
            f"hand{hand+1}_z{i}"
        ])

headers.append("label")

# =====================================
# 4. SELECT LABEL AND HAND COUNT
# =====================================

label = input("Enter sign label: ").strip().upper()

if not label:
    print("Label cannot be empty!")
    raise SystemExit

expected_hands = int(
    input("How many hands does this sign use? (1 or 2): ")
)

if expected_hands not in [1, 2]:
    print("Please enter either 1 or 2.")
    raise SystemExit

samples = int(
    input("How many samples do you want to collect? ")
)

if samples <= 0:
    print("Enter a positive sample count!")
    raise SystemExit

# =====================================
# 5. OPEN WEBCAM
# =====================================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Could not access webcam!")
    raise SystemExit

print("\nSignBridge Two-Hand Dataset Collector")
print(f"Sign: {label}")
print(f"Expected hands: {expected_hands}")
print("Keep the required hand(s) completely visible.")
print("Press ENTER in the terminal to capture.")
print("Type q and press ENTER to stop.\n")

# =====================================
# 6. COLLECT LANDMARK DATA
# =====================================

successful_samples = 0

try:
    with mp.tasks.vision.HandLandmarker.create_from_options(
        options
    ) as detector:

        while successful_samples < samples:

            sample_number = successful_samples + 1

            command = input(
                f"Sample {sample_number}/{samples} - "
                "Press ENTER to capture: "
            )

            if command.lower() == "q":
                print("Collection stopped by user.")
                break

            time.sleep(0.5)

            success = False
            frame = None

            for _ in range(5):
                success, frame = cap.read()

            if not success:
                print("Could not capture frame. Try again.")
                continue

            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            result = detector.detect(mp_image)

            detected_hands = result.hand_landmarks

            if len(detected_hands) != expected_hands:
                print(
                    f"Expected {expected_hands} hand(s), "
                    f"but detected {len(detected_hands)}. "
                    "Try again."
                )
                continue

            # Sort hands from left to right in the image
            ordered_hands = sorted(
                detected_hands,
                key=lambda hand: np.mean(
                    [point.x for point in hand]
                )
            )

            coordinates = []

            # Extract landmarks for each detected hand
            for landmarks in ordered_hands:

                for point in landmarks:
                    coordinates.extend([
                        point.x,
                        point.y,
                        point.z
                    ])

            # Add zeros if only one hand is present
            if expected_hands == 1:
                coordinates.extend([0.0] * 63)

            if len(coordinates) != 126:
                print("Incorrect feature count. Try again.")
                continue

            # Save original image
            filename = (
                f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
            )

            image_path = os.path.join(RAW_FOLDER, filename)

            cv2.imwrite(image_path, frame)

            # Save coordinates to CSV
            file_exists = os.path.exists(CSV_PATH)

            with open(CSV_PATH, "a", newline="") as file:
                writer = csv.writer(file)

                if not file_exists:
                    writer.writerow(headers)

                writer.writerow(coordinates + [label])

            successful_samples += 1

            print(
                f"Sample saved successfully! "
                f"({successful_samples}/{samples})"
            )

finally:
    cap.release()

print(
    f"\nCollection completed! "
    f"Successfully saved {successful_samples} samples."
)
