
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load cleaned dataset
df = pd.read_csv("backend/dataset/cleaned_landmarks.csv")

# Separate labels and coordinates
y = df["label"]
X = df.drop("label", axis=1).to_numpy(dtype=float)

# Reshape: 21 landmarks x 3 coordinates
X = X.reshape(-1, 21, 3)

# Make coordinates relative to wrist
X = X - X[:, 0:1, :]

# Normalize using maximum landmark distance
hand_sizes = np.linalg.norm(X, axis=2).max(axis=1)
hand_sizes[hand_sizes == 0] = 1

X = X / hand_sizes[:, None, None]

# Flatten back to 63 features
X = X.reshape(-1, 63)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(model, "backend/models/signbridge_model.pkl")

print("\nNormalized model trained and saved!")
