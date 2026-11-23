
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load the two-hand dataset
DATASET_PATH = "backend/dataset/landmarks_2hand.csv"

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nSamples per sign:")
print(df["label"].value_counts())

# 2. Separate features and labels
X = df.drop("label", axis=1).to_numpy(dtype=float)
y = df["label"].to_numpy()

# 3. Reshape into 2 hands, 21 landmarks, 3 coordinates
X = X.reshape(-1, 2, 21, 3)

# 4. Normalize each hand separately
for hand in range(2):
    hand_data = X[:, hand, :, :]

    # Subtract wrist coordinates
    hand_data -= hand_data[:, 0:1, :]

    # Calculate hand size
    hand_sizes = np.linalg.norm(hand_data, axis=2).max(axis=1)
    hand_sizes[hand_sizes == 0] = 1

    # Scale coordinates
    X[:, hand, :, :] = hand_data / hand_sizes[:, None, None]

# 5. Flatten features
X = X.reshape(-1, 126)

# 6. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 7. Train Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# 8. Evaluate model
y_pred = model.predict(X_test)

print("\nModel Evaluation")
print("----------------")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 9. Save model
MODEL_PATH = "backend/models/signbridge_model_2hand.pkl"

joblib.dump(model, MODEL_PATH)

print("\nModel saved successfully at:", MODEL_PATH)
