import pandas as pd
import numpy as np
import re
from scipy.sparse import hstack, csr_matrix

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
from sklearn.preprocessing import StandardScaler


DATASET_PATH = "data/CEAS_08.csv"


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv(DATASET_PATH)

df["subject"] = df["subject"].fillna("")
df["body"] = df["body"].fillna("")
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
# Structural features
# --------------------------------------------------

def extract_structural_features(dataframe):

    features = pd.DataFrame(index=dataframe.index)

    body = dataframe["body"]
    subject = dataframe["subject"]
    text = dataframe["text"]

    # Length features
    features["body_length"] = body.str.len()
    features["subject_length"] = subject.str.len()

    # URL features
    features["url_count"] = (
        text.str.count(r"http[s]?://")
    )

    # Character composition
    features["digit_count"] = (
        text.str.count(r"\d")
    )

    features["uppercase_count"] = (
        text.str.count(r"[A-Z]")
    )

    features["exclamation_count"] = (
        text.str.count(r"!")
    )

    features["question_count"] = (
        text.str.count(r"\?")
    )

    # Special characters
    features["special_char_count"] = (
        text.str.count(r"[^a-zA-Z0-9\s]")
    )

    # HTML indicator
    features["html_present"] = (
        text.str.contains(
            r"<[^>]+>",
            regex=True,
            na=False
        ).astype(int)
    )

    return features


train_structural = extract_structural_features(
    df.iloc[train_idx]
)

val_structural = extract_structural_features(
    df.iloc[val_idx]
)

test_structural = extract_structural_features(
    df.iloc[test_idx]
)


# --------------------------------------------------
# Standardize structural features
# --------------------------------------------------

scaler = StandardScaler()

X_train_structural = scaler.fit_transform(
    train_structural
)

X_val_structural = scaler.transform(
    val_structural
)

X_test_structural = scaler.transform(
    test_structural
)


# Convert to sparse matrices
X_train_structural = csr_matrix(X_train_structural)
X_val_structural = csr_matrix(X_val_structural)
X_test_structural = csr_matrix(X_test_structural)


# --------------------------------------------------
# Combine all features
# --------------------------------------------------

X_train_combined = hstack([
    X_train_word,
    X_train_char,
    X_train_structural
])

X_val_combined = hstack([
    X_val_word,
    X_val_char,
    X_val_structural
])

X_test_combined = hstack([
    X_test_word,
    X_test_char,
    X_test_structural
])


print("=== Feature Extraction ===")
print(f"Word features:        {X_train_word.shape[1]}")
print(f"Character features:   {X_train_char.shape[1]}")
print(f"Structural features:  {X_train_structural.shape[1]}")
print(f"Combined features:    {X_train_combined.shape[1]}")


# --------------------------------------------------
# Train model
# --------------------------------------------------

model = LogisticRegression(
    max_iter=1000,
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
# Analyze remaining test false negatives
# --------------------------------------------------

analysis_df = df.iloc[test_idx].copy()

analysis_df["prediction"] = y_test_pred
analysis_df["actual"] = y_test.values
analysis_df["body_length"] = analysis_df["body"].str.len()

false_negatives = analysis_df[
    (analysis_df["actual"] == 1) &
    (analysis_df["prediction"] == 0)
].copy()

print("\n=== Test False Negatives ===")
print("Number of false negatives:", len(false_negatives))


def clean_email_text(text):
    # Convert all whitespace runs into a single space
    text = re.sub(r"\s+", " ", str(text))

    # Remove excessive repeated punctuation
    text = re.sub(r"([^\w\s])\1{4,}", r"\1\1\1", text)

    return text.strip()


for i, (index, row) in enumerate(false_negatives.iterrows()):

    cleaned_body = clean_email_text(row["body"])

    print(f"\n--- False Negative {i + 1} ---")
    print("Dataset index:", index)
    print("Subject:", row["subject"])
    print("Body length:", row["body_length"])
    print("URLs:", row["urls"])
    print("Body:")
    print(cleaned_body[:1500])