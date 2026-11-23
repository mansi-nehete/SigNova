
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2

from backend.prediction import predict_sign

app = FastAPI(title="SignBridge API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {
        "message": "Welcome to SignBridge!",
        "status": "Backend is running successfully"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/test")
def test():
    return {
        "message": "SignBridge prediction API is ready!"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    image_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "Invalid image"}

    result = predict_sign(image)

    return result
