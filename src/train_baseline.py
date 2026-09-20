import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

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

# Create groups based on normalized email subjects
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

groups = df["group"]

# Create 5 stratified, non-overlapping groups
cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

folds = list(cv.split(X, y, groups))

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

# Create the actual datasets
X_train = X.iloc[train_idx]
y_train = y.iloc[train_idx]

X_val = X.iloc[val_idx]
y_val = y.iloc[val_idx]

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

# Learn vocabulary ONLY from training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform validation and test data
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

print("=== TF-IDF Feature Extraction ===")
print(f"Training emails:   {X_train_tfidf.shape[0]}")
print(f"Validation emails: {X_val_tfidf.shape[0]}")
print(f"Test emails:       {X_test_tfidf.shape[0]}")
print(f"Number of features: {X_train_tfidf.shape[1]}")

# Create Logistic Regression classifier
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

# Train on training data
print("\n=== Training Logistic Regression ===")
model.fit(X_train_tfidf, y_train)

# Evaluate on validation data
y_val_pred = model.predict(X_val_tfidf)

accuracy = accuracy_score(y_val, y_val_pred)
precision = precision_score(y_val, y_val_pred)
recall = recall_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)

print("\n=== Validation Results ===")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# Confusion matrix
cm = confusion_matrix(y_val, y_val_pred)

print("\n=== Confusion Matrix ===")
print(cm)

print("\nRows = Actual")
print("Columns = Predicted")
print("             Legitimate  Phishing")
print(f"Legitimate   {cm[0][0]:10d}  {cm[0][1]:8d}")
print(f"Phishing     {cm[1][0]:10d}  {cm[1][1]:8d}")

# Final evaluation on the held-out test set
y_test_pred = model.predict(X_test_tfidf)

test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print("\n=== Final Test Results ===")
print(f"Accuracy:  {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall:    {test_recall:.4f}")
print(f"F1 Score:  {test_f1:.4f}")

test_cm = confusion_matrix(y_test, y_test_pred)

print("\n=== Test Confusion Matrix ===")
print(test_cm)

print("\nRows = Actual")
print("Columns = Predicted")
print("             Legitimate  Phishing")
print(f"Legitimate   {test_cm[0][0]:10d}  {test_cm[0][1]:8d}")
print(f"Phishing     {test_cm[1][0]:10d}  {test_cm[1][1]:8d}")

# Inspect validation false negatives
val_false_negative_indices = X_val.index[
    (y_val == 1) & (y_val_pred == 0)
]

print("\n=== Validation False Negatives ===")
print(f"Number of false negatives: {len(val_false_negative_indices)}")

for i, index in enumerate(val_false_negative_indices[:10]):
    print(f"\n--- Validation False Negative {i + 1} ---")
    print("Subject:", df.loc[index, "subject"])
    print("Body:")
    print(df.loc[index, "body"][:1000])

# --------------------------------------------------
# Performance by email characteristics
# --------------------------------------------------

analysis_df = df.loc[X_test.index].copy()

analysis_df["body_length"] = analysis_df["body"].str.len()
analysis_df["prediction"] = y_test_pred
analysis_df["actual"] = y_test

print("\n=== Test Performance by Body Length ===")

length_ranges = [
    ("< 100", 0, 100),
    ("100-199", 100, 200),
    ("200-499", 200, 500),
    ("500-999", 500, 1000),
    ("1000+", 1000, float("inf"))
]

for name, lower, upper in length_ranges:

    subset = analysis_df[
        (analysis_df["body_length"] >= lower) &
        (analysis_df["body_length"] < upper)
    ]

    if len(subset) == 0:
        continue

    subset_f1 = f1_score(
        subset["actual"],
        subset["prediction"]
    )

    subset_recall = recall_score(
        subset["actual"],
        subset["prediction"]
    )

    print(
        f"{name:10s} | "
        f"Emails: {len(subset):4d} | "
        f"Recall: {subset_recall:.4f} | "
        f"F1: {subset_f1:.4f}"
    )


# --------------------------------------------------
# Performance by URL presence
# --------------------------------------------------

print("\n=== Test Performance by URL Presence ===")

for url_value in [0, 1]:

    subset = analysis_df[
        analysis_df["urls"] == url_value
    ]

    subset_f1 = f1_score(
        subset["actual"],
        subset["prediction"]
    )

    subset_recall = recall_score(
        subset["actual"],
        subset["prediction"]
    )

    print(
        f"URLs = {url_value} | "
        f"Emails: {len(subset):4d} | "
        f"Recall: {subset_recall:.4f} | "
        f"F1: {subset_f1:.4f}"
    )

# --------------------------------------------------
# Analyze test false negatives
# --------------------------------------------------

test_false_negatives = analysis_df[
    (analysis_df["actual"] == 1) &
    (analysis_df["prediction"] == 0)
].copy()

print("\n=== Test False Negatives by Body Length ===")

for name, lower, upper in length_ranges:

    subset = test_false_negatives[
        (test_false_negatives["body_length"] >= lower) &
        (test_false_negatives["body_length"] < upper)
    ]

    print(
        f"{name:10s} | "
        f"False negatives: {len(subset):4d}"
    )


print("\n=== Test False Negatives by URL Presence ===")

for url_value in [0, 1]:

    subset = test_false_negatives[
        test_false_negatives["urls"] == url_value
    ]

    print(
        f"URLs = {url_value} | "
        f"False negatives: {len(subset):4d}"
    )


# --------------------------------------------------
# Show representative long-email false negatives
# --------------------------------------------------

print("\n=== Long Test False Negatives ===")

long_false_negatives = test_false_negatives[
    test_false_negatives["body_length"] >= 500
]

print(
    f"Long false negatives: {len(long_false_negatives)}"
)

for i, (index, row) in enumerate(
    long_false_negatives.head(10).iterrows()
):

    print(f"\n--- Long False Negative {i + 1} ---")
    print("Subject:", row["subject"])
    print("Body length:", row["body_length"])
    print("URLs:", row["urls"])
    print("Body:")
    print(row["body"][:1000])