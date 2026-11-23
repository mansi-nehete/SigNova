
import pandas as pd

# Load dataset
DATASET_PATH = "backend/dataset/landmarks.csv"

df = pd.read_csv(DATASET_PATH)

# Remove old testing samples
df = df[df["label"].str.lower() != "test_pose"].copy()

# Make labels consistent
df["label"] = df["label"].str.upper()

# Check missing values
print("Missing values:", df.isnull().sum().sum())

# Display samples per sign
print("\nSamples per sign:")
print(df["label"].value_counts())

# Check dataset shape
print("\nDataset shape:", df.shape)

# Save cleaned dataset
df.to_csv("backend/dataset/cleaned_landmarks.csv", index=False)

print("\nDataset cleaned and saved successfully!")

