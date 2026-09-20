import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

DATASET_PATH = "data/CEAS_08.csv"

# Load dataset
df = pd.read_csv(DATASET_PATH)

# Handle missing subjects
df["subject"] = df["subject"].fillna("")

# Create normalized subject groups
df["group"] = (
    df["subject"]
    .str.lower()
    .str.strip()
)

# Give emails without subjects unique groups
empty_subjects = df["group"] == ""

df.loc[empty_subjects, "group"] = (
    "no_subject_" + df.index[empty_subjects].astype(str)
)

X = df["subject"] + " " + df["body"]
y = df["label"]
groups = df["group"]

# Create 5 stratified, non-overlapping groups
cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

folds = list(cv.split(X, y, groups))

# Use:
# Fold 0 -> validation
# Fold 1 -> test
# Folds 2, 3, 4 -> training

val_idx = folds[0][1]
test_idx = folds[1][1]

train_idx = pd.concat([
    pd.Series(folds[2][1]),
    pd.Series(folds[3][1]),
    pd.Series(folds[4][1])
]).values

print("=== Stratified Grouped Dataset Split ===")
print(f"Training emails:   {len(train_idx)}")
print(f"Validation emails: {len(val_idx)}")
print(f"Test emails:       {len(test_idx)}")

print("\n=== Label Distribution ===")

for name, indices in [
    ("Training", train_idx),
    ("Validation", val_idx),
    ("Test", test_idx)
]:
    distribution = y.iloc[indices].value_counts(normalize=True)

    print(f"\n{name}:")
    print(distribution)

print("\n=== Group Overlap Check ===")

train_groups = set(groups.iloc[train_idx])
val_groups = set(groups.iloc[val_idx])
test_groups = set(groups.iloc[test_idx])

print("Train/Validation overlap:",
      len(train_groups & val_groups))

print("Train/Test overlap:",
      len(train_groups & test_groups))

print("Validation/Test overlap:",
      len(val_groups & test_groups))