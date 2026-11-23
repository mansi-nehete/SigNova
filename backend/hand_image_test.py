
import cv2
import mediapipe as mp

# Image and model paths
IMAGE_PATH = "backend/camera_test_frame.jpg"
MODEL_PATH = "backend/models/hand_landmarker.task"

# Load the image
image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Could not load image!")
    raise SystemExit

# Convert BGR to RGB
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create MediaPipe image
mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=rgb_image
)

# Configure MediaPipe for image detection
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=2
)

# Detect hands
with mp.tasks.vision.HandLandmarker.create_from_options(
    options
) as detector:

    result = detector.detect(mp_image)

    if result.hand_landmarks:

        height, width, _ = image.shape

        for hand_number, landmarks in enumerate(
            result.hand_landmarks
        ):

            coordinates = []

            for point in landmarks:

                x = int(point.x * width)
                y = int(point.y * height)

                coordinates.extend([
                    point.x,
                    point.y,
                    point.z
                ])

                cv2.circle(
                    image,
                    (x, y),
                    6,
                    (0, 255, 0),
                    -1
                )

            print("\nHand:", hand_number + 1)
            print("Landmarks:", len(landmarks))
            print("Coordinate values:", len(coordinates))
            print("First X, Y, Z:", coordinates[:3])

        output_path = "backend/hand_landmarks_result.jpg"

        cv2.imwrite(output_path, image)

        print("\nHand detection successful!")
        print("Result saved at:", output_path)

    else:
        print("No hands detected in the image.")

