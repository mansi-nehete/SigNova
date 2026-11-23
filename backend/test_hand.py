
import cv2
import mediapipe as mp

# =====================================
# 1. LOAD MEDIAPIPE TASKS
# =====================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Model file location
MODEL_PATH = "backend/models/hand_landmarker.task"


# =====================================
# 2. CONFIGURE HAND DETECTION
# =====================================

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# =====================================
# 3. OPEN WEBCAM
# =====================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not access webcam!")
    raise SystemExit

print("SignBridge hand detection started!")
print("Press Q to close the webcam.")


# =====================================
# 4. CREATE HAND DETECTOR
# =====================================

# Use a monotonic timestamp for video frames
import time

start_time = time.monotonic()

with HandLandmarker.create_from_options(options) as detector:

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            print("Could not read webcam frame!")
            break

        # =====================================
        # 5. IMAGE PREPROCESSING
        # =====================================

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Generate increasing timestamp in milliseconds
        timestamp = int((time.monotonic() - start_time) * 1000)

        # =====================================
        # 6. DETECT HANDS
        # =====================================

        result = detector.detect_for_video(
            mp_image,
            timestamp
        )

        # =====================================
        # 7. EXTRACT LANDMARK COORDINATES
        # =====================================

        if result.hand_landmarks:

            height, width, _ = frame.shape

            for hand_index, hand_landmarks in enumerate(
                result.hand_landmarks
            ):

                # Store X, Y, Z values
                landmark_data = []

                for landmark in hand_landmarks:

                    # Pixel coordinates for drawing
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    # Store normalized coordinates
                    landmark_data.extend([
                        landmark.x,
                        landmark.y,
                        landmark.z
                    ])

                    # Draw landmark points
                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )

                # =====================================
                # 8. DISPLAY LANDMARK INFORMATION
                # =====================================

                print("\nHand detected!")
                print("Hand number:", hand_index + 1)
                print("Total landmarks:", len(hand_landmarks))
                print("Total coordinate values:", len(landmark_data))
                print("First landmark (X, Y, Z):", landmark_data[:3])

            cv2.putText(
                frame,
                "HAND DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        else:
            cv2.putText(
                frame,
                "NO HAND DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        # =====================================
        # 9. DISPLAY WEBCAM
        # =====================================

        cv2.imshow(
            "SignBridge - Hand Detection",
            frame
        )

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# =====================================
# 10. RELEASE RESOURCES
# =====================================

cap.release()
cv2.destroyAllWindows()

print("\nSignBridge hand detection stopped!")
