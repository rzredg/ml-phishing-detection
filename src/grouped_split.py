import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

DATASET_PATH = "data/CEAS_08.csv"

# Load dataset
df = pd.read_csv(DATASET_PATH)

# Handle missing subjects
df["subject"] = df["subject"].fillna("")

# Use normalized subject as a grouping variable
df["group"] = (
    df["subject"]
    .str.lower()
    .str.strip()
)

# Empty subjects should not all belong to one group
empty_subjects = df["group"] == ""

df.loc[empty_subjects, "group"] = (
    "no_subject_" + df.index[empty_subjects].astype(str)
)

X = df["text"] if "text" in df.columns else (
    df["subject"] + " " + df["body"]
)

y = df["label"]
groups = df["group"]

# First split: 70% training, 30% temporary
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=42
)

train_idx, temp_idx = next(
    splitter.split(X, y, groups=groups)
)

# Second split: divide temporary set into validation/test
temp_groups = groups.iloc[temp_idx]

splitter2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=42
)

val_relative_idx, test_relative_idx = next(
    splitter2.split(
        X.iloc[temp_idx],
        y.iloc[temp_idx],
        groups=temp_groups
    )
)

val_idx = temp_idx[val_relative_idx]
test_idx = temp_idx[test_relative_idx]

print("=== Grouped Dataset Split ===")
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