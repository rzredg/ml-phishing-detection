import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold

DATASET_PATH = "data/CEAS_08.csv"

# Load dataset
df = pd.read_csv(DATASET_PATH)

# Handle missing subjects
df["subject"] = df["subject"].fillna("")

# Create combined text
df["text"] = df["subject"] + " " + df["body"]

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

X = df["text"]
y = df["label"]
groups = df["group"]

# Recreate the exact same split as train_baseline.py
cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

folds = list(cv.split(X, y, groups))

val_idx = folds[0][1]
test_idx = folds[1][1]

# Create validation and test dataframes
val_df = df.iloc[val_idx].copy()
test_df = df.iloc[test_idx].copy()


# --------------------------------------------------
# Basic dataset statistics
# --------------------------------------------------

print("=== Validation vs Test ===")

print(f"\nValidation emails: {len(val_df)}")
print(f"Test emails:       {len(test_df)}")


# --------------------------------------------------
# Email length
# --------------------------------------------------

val_df["body_length"] = val_df["body"].str.len()
test_df["body_length"] = test_df["body"].str.len()

val_df["subject_length"] = val_df["subject"].str.len()
test_df["subject_length"] = test_df["subject"].str.len()

print("\n=== Body Length ===")

print(
    f"Validation mean: {val_df['body_length'].mean():.1f}"
)
print(
    f"Test mean:       {test_df['body_length'].mean():.1f}"
)

print(
    f"Validation median: {val_df['body_length'].median():.1f}"
)
print(
    f"Test median:       {test_df['body_length'].median():.1f}"
)


print("\n=== Subject Length ===")

print(
    f"Validation mean: {val_df['subject_length'].mean():.1f}"
)
print(
    f"Test mean:       {test_df['subject_length'].mean():.1f}"
)

print(
    f"Validation median: {val_df['subject_length'].median():.1f}"
)
print(
    f"Test median:       {test_df['subject_length'].median():.1f}"
)


# --------------------------------------------------
# URL presence
# --------------------------------------------------

print("\n=== URL Presence ===")

print(
    "Validation:",
    val_df["urls"].value_counts(normalize=True).sort_index()
)

print(
    "Test:",
    test_df["urls"].value_counts(normalize=True).sort_index()
)


# --------------------------------------------------
# Label distribution
# --------------------------------------------------

print("\n=== Label Distribution ===")

print(
    "Validation:",
    val_df["label"].value_counts(normalize=True).sort_index()
)

print(
    "Test:",
    test_df["label"].value_counts(normalize=True).sort_index()
)


# --------------------------------------------------
# HTML presence
# --------------------------------------------------

val_df["has_html"] = val_df["body"].str.contains(
    r"<[^>]+>",
    regex=True,
    na=False
)

test_df["has_html"] = test_df["body"].str.contains(
    r"<[^>]+>",
    regex=True,
    na=False
)

print("\n=== HTML Presence ===")

print(
    "Validation:",
    val_df["has_html"].mean()
)

print(
    "Test:",
    test_df["has_html"].mean()
)


# --------------------------------------------------
# Very short emails
# --------------------------------------------------

print("\n=== Very Short Emails ===")

for threshold in [50, 100, 200]:

    val_percentage = (
        (val_df["body_length"] < threshold).mean()
    )

    test_percentage = (
        (test_df["body_length"] < threshold).mean()
    )

    print(f"\nUnder {threshold} characters:")
    print(f"Validation: {val_percentage:.4f}")
    print(f"Test:       {test_percentage:.4f}")


# --------------------------------------------------
# URL count
# --------------------------------------------------

import re

url_pattern = r"https?://\S+|www\.\S+"

val_df["url_count"] = val_df["body"].str.count(url_pattern)
test_df["url_count"] = test_df["body"].str.count(url_pattern)

print("\n=== URL Count ===")

print(
    f"Validation mean: {val_df['url_count'].mean():.2f}"
)

print(
    f"Test mean:       {test_df['url_count'].mean():.2f}"
)

print(
    f"Validation median: {val_df['url_count'].median():.1f}"
)

print(
    f"Test median:       {test_df['url_count'].median():.1f}"
)