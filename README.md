# Sign Language Recognition System

A real-time Sign Language Recognition System that uses computer vision and machine learning to recognize hand signs through a webcam.

## About the Project

The project aims to make communication easier by recognizing hand signs and displaying the predicted sign on the screen.

It uses hand landmark detection and a trained machine learning model to identify signs from camera input.

## Features

- Real-time webcam input
- Hand landmark detection
- Two-hand sign recognition
- Machine learning-based prediction
- React-based user interface
- Python backend API

## Technologies Used

**Frontend**
- React.js
- JavaScript
- Vite
- CSS

**Backend**
- Python
- FastAPI
- OpenCV
- MediaPipe
- NumPy
- Joblib

## Project Structure

```text
Sign Language/
│
├── backend/
│   ├── dataset/
│   ├── models/
│   ├── main.py
│   ├── prediction.py
│   ├── predict_live.py
│   └── predict_live_2hand.py
│
├── signbridge-react/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── index.html
│
└── README.md
```

## How It Works

1. The webcam captures hand movements.
2. The backend receives camera frames.
3. MediaPipe detects hand landmarks.
4. The trained machine learning model predicts the sign.
5. The recognized prediction is displayed on the frontend.

## Current Status

The project is under development. Improvements in recognition accuracy and continuous sign-to-word translation are ongoing.

## Future Improvements

- Improve recognition accuracy.
- Support continuous sign recognition.
- Display complete words and sentences.
- Improve the user interface.
- Enhance real-time performance.
