import pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


DATASET_PATH = "data/CEAS_08.csv"


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv(DATASET_PATH)

df["subject"] = df["subject"].fillna("")
df["text"] = df["subject"] + " " + df["body"]

X = df["text"]
y = df["label"]


# --------------------------------------------------
# Create grouped stratified split
# --------------------------------------------------

df["group"] = df["subject"].str.lower().str.strip()

empty_subjects = df["group"] == ""

df.loc[empty_subjects, "group"] = (
    "no_subject_" + df.index[empty_subjects].astype(str)
)

groups = df["group"]


cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

folds = list(cv.split(X, y, groups))

val_idx = folds[0][1]
test_idx = folds[1][1]

train_idx = pd.concat([
    pd.Series(folds[2][1]),
    pd.Series(folds[3][1]),
    pd.Series(folds[4][1])
]).values


X_train = X.iloc[train_idx]
y_train = y.iloc[train_idx]

X_val = X.iloc[val_idx]
y_val = y.iloc[val_idx]

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]


# --------------------------------------------------
# Word-level TF-IDF
# --------------------------------------------------

word_vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_word = word_vectorizer.fit_transform(X_train)
X_val_word = word_vectorizer.transform(X_val)
X_test_word = word_vectorizer.transform(X_test)


# --------------------------------------------------
# Character-level TF-IDF
# --------------------------------------------------

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=3,
    sublinear_tf=True
)

X_train_char = char_vectorizer.fit_transform(X_train)
X_val_char = char_vectorizer.transform(X_val)
X_test_char = char_vectorizer.transform(X_test)


# --------------------------------------------------
# Combine word + character features
# --------------------------------------------------

X_train_combined = hstack([
    X_train_word,
    X_train_char
])

X_val_combined = hstack([
    X_val_word,
    X_val_char
])

X_test_combined = hstack([
    X_test_word,
    X_test_char
])


print("=== Feature Extraction ===")
print(f"Word features:       {X_train_word.shape[1]}")
print(f"Character features:  {X_train_char.shape[1]}")
print(f"Combined features:   {X_train_combined.shape[1]}")


# --------------------------------------------------
# Train model
# --------------------------------------------------

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

print("\n=== Training Logistic Regression ===")

model.fit(
    X_train_combined,
    y_train
)


# --------------------------------------------------
# Validation
# --------------------------------------------------

y_val_pred = model.predict(X_val_combined)

val_accuracy = accuracy_score(y_val, y_val_pred)
val_precision = precision_score(y_val, y_val_pred)
val_recall = recall_score(y_val, y_val_pred)
val_f1 = f1_score(y_val, y_val_pred)

print("\n=== Validation Results ===")
print(f"Accuracy:  {val_accuracy:.4f}")
print(f"Precision: {val_precision:.4f}")
print(f"Recall:    {val_recall:.4f}")
print(f"F1 Score:  {val_f1:.4f}")

val_cm = confusion_matrix(y_val, y_val_pred)

print("\n=== Validation Confusion Matrix ===")
print(val_cm)


# --------------------------------------------------
# Final test evaluation
# --------------------------------------------------

y_test_pred = model.predict(X_test_combined)

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

# --------------------------------------------------
# Analyze test performance by body length
# --------------------------------------------------

analysis_df = df.iloc[test_idx].copy()

analysis_df["prediction"] = y_test_pred
analysis_df["actual"] = y_test.values
analysis_df["body_length"] = analysis_df["body"].str.len()

length_ranges = [
    ("< 100", 0, 100),
    ("100-199", 100, 200),
    ("200-499", 200, 500),
    ("500-999", 500, 1000),
    ("1000+", 1000, float("inf"))
]

print("\n=== Test Performance by Body Length ===")

for name, lower, upper in length_ranges:

    subset = analysis_df[
        (analysis_df["body_length"] >= lower) &
        (analysis_df["body_length"] < upper)
    ]

    if len(subset) == 0:
        continue

    recall = recall_score(
        subset["actual"],
        subset["prediction"],
        zero_division=0
    )

    f1 = f1_score(
        subset["actual"],
        subset["prediction"],
        zero_division=0
    )

    print(
        f"{name:10s} | "
        f"Emails: {len(subset):5d} | "
        f"Recall: {recall:.4f} | "
        f"F1: {f1:.4f}"
    )


# --------------------------------------------------
# Analyze test performance by URL presence
# --------------------------------------------------

print("\n=== Test Performance by URL Presence ===")

for url_value in [0, 1]:

    subset = analysis_df[
        analysis_df["urls"] == url_value
    ]

    recall = recall_score(
        subset["actual"],
        subset["prediction"],
        zero_division=0
    )

    f1 = f1_score(
        subset["actual"],
        subset["prediction"],
        zero_division=0
    )

    print(
        f"URLs = {url_value} | "
        f"Emails: {len(subset):5d} | "
        f"Recall: {recall:.4f} | "
        f"F1: {f1:.4f}"
    )