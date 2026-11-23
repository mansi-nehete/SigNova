
import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time

# MediaPipe setup
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path="backend/models/hand_landmarker.task"
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=2
)

# Ask for word label
word = input("Enter the word/sign label: ").strip().upper()

if not word:
    print("Word label cannot be empty.")
    raise SystemExit

# Create folder for this word
save_dir = f"backend/word_dataset/raw/{word}"
os.makedirs(save_dir, exist_ok=True)

# Recording settings
recording_count = 20
sequence_duration = 3
sequence_length = 30

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam.")
    raise SystemExit

print(f"\nSignBridge Word Dataset Collector")
print(f"Word: {word}")
print("Press SPACE to start a recording.")
print("Press Q to quit.")

try:
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:

        sample_number = len([
            name for name in os.listdir(save_dir)
            if name.endswith(".csv")
        ]) + 1

        while sample_number <= recording_count:
            success, frame = cap.read()

            if not success or frame is None:
                print("Camera frame not received.")
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp = int(
                cv2.getTickCount() * 1000 / cv2.getTickFrequency()
            )

            result = detector.detect_for_video(mp_image, timestamp)

            display = cv2.flip(frame, 1)

            cv2.putText(
                display,
                f"Word: {word} | Sample: {sample_number}/{recording_count}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2
            )

            cv2.putText(
                display,
                "SPACE: Record | Q: Quit",
                (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

            cv2.imshow("SignBridge Word Collector", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == 32:
                frames = []
                start_time = time.time()

                while time.time() - start_time < sequence_duration:
                    success, frame = cap.read()

                    if not success:
                        break

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb
                    )

                    timestamp = int(
                        cv2.getTickCount() * 1000 / cv2.getTickFrequency()
                    )

                    result = detector.detect_for_video(mp_image, timestamp)

                    hands = result.hand_landmarks
                    frame_data = []

                    for hand in hands[:2]:
                        for point in hand:
                            frame_data.extend([
                                point.x, point.y, point.z
                            ])

                    while len(frame_data) < 126:
                        frame_data.extend([0.0] * (126 - len(frame_data)))

                    frames.append(frame_data[:126])

                    display = cv2.flip(frame, 1)
                    cv2.putText(
                        display,
                        "RECORDING...",
                        (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )

                    cv2.imshow("SignBridge Word Collector", display)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if len(frames) == 0:
                    print("No frames captured. Try again.")
                    continue

                # Convert recording to fixed length
                indices = np.linspace(
                    0, len(frames) - 1,
                    sequence_length
                ).astype(int)

                sequence = np.array(frames)[indices]

                file_path = os.path.join(
                    save_dir,
                    f"sample_{sample_number}.csv"
                )

                np.savetxt(
                    file_path,
                    sequence,
                    delimiter=","
                )

                print(f"Saved sample {sample_number}")

                sample_number += 1

finally:
    cap.release()
    cv2.destroyAllWindows()

print("Recording session finished.")
