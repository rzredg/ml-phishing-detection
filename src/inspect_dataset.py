import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_PATH = "data/CEAS_08.csv"

# Load dataset
df = pd.read_csv(DATASET_PATH)

# Handle missing subjects
df["subject"] = df["subject"].fillna("")

# Combine subject and body
df["text"] = df["subject"] + " " + df["body"]

# Separate features and labels
X = df["text"]
y = df["label"]

# First split: 70% training, 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

# Second split: divide temporary set equally
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)

print("=== Dataset Split ===")
print(f"Total:      {len(df)}")
print(f"Training:   {len(X_train)}")
print(f"Validation: {len(X_val)}")
print(f"Test:       {len(X_test)}")

print("\n=== Label Distribution ===")

print("\nTraining:")
print(y_train.value_counts(normalize=True))

print("\nValidation:")
print(y_val.value_counts(normalize=True))

print("\nTest:")
print(y_test.value_counts(normalize=True))