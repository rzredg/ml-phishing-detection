import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

DATASET_PATH = "data/CEAS_08.csv"

# Load dataset
df = pd.read_csv(DATASET_PATH)

# Handle missing subjects
df["subject"] = df["subject"].fillna("")

# Combine subject and body
df["text"] = df["subject"] + " " + df["body"]

# Use a sample so this analysis stays manageable
sample = df.sample(n=5000, random_state=42).reset_index(drop=True)

# Convert emails to TF-IDF vectors
vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=(1, 2),
    min_df=2,
    max_features=100000
)

X = vectorizer.fit_transform(sample["text"])

# Find the nearest neighboring email for each email
neighbors = NearestNeighbors(
    n_neighbors=2,
    metric="cosine",
    n_jobs=-1
)

neighbors.fit(X)

distances, indices = neighbors.kneighbors(X)

# Convert cosine distance to cosine similarity
similarities = 1 - distances[:, 1]

print("=== Near-Duplicate Analysis ===")
print(f"Emails analyzed: {len(sample)}")

print("\nSimilarity statistics:")
print(f"Highest similarity:  {similarities.max():.4f}")
print(f"Average similarity:  {similarities.mean():.4f}")
print(f"95th percentile:     {pd.Series(similarities).quantile(0.95):.4f}")

# Show highly similar pairs
threshold = 0.90

similar_pairs = []

for i, similarity in enumerate(similarities):
    if similarity >= threshold:
        neighbor = indices[i, 1]

        similar_pairs.append(
            (similarity, i, neighbor)
        )

similar_pairs.sort(reverse=True)

print(f"\nPairs with similarity >= {threshold}: {len(similar_pairs)}")

for similarity, i, neighbor in similar_pairs[:10]:
    print("\n--- Similar Pair ---")
    print(f"Similarity: {similarity:.4f}")

    print("\nEmail A:")
    print("Label:", sample.iloc[i]["label"])
    print("Subject:", sample.iloc[i]["subject"])
    print(sample.iloc[i]["body"][:500])

    print("\nEmail B:")
    print("Label:", sample.iloc[neighbor]["label"])
    print("Subject:", sample.iloc[neighbor]["subject"])
    print(sample.iloc[neighbor]["body"][:500])