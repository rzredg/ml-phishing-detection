import os
import psutil
import joblib
from scipy.sparse import hstack

process = psutil.Process(os.getpid())

def memory_mb():
    return process.memory_info().rss / (1024 * 1024)

print(f"Starting memory: {memory_mb():.2f} MB")

word_vectorizer = joblib.load("models/word_vectorizer.joblib")
char_vectorizer = joblib.load("models/char_vectorizer.joblib")
model = joblib.load("models/phishing_model.joblib")

print(f"After loading model: {memory_mb():.2f} MB")

subject = "Your account has been suspended"
body = """
Dear customer,

Your account has been suspended due to suspicious activity.
Please click the link below and verify your account immediately.

Thank you.
"""

text = subject + " " + body

print(f"Before transformation: {memory_mb():.2f} MB")

word_features = word_vectorizer.transform([text])
print(f"After word transformation: {memory_mb():.2f} MB")

char_features = char_vectorizer.transform([text])
print(f"After char transformation: {memory_mb():.2f} MB")

combined_features = hstack([word_features, char_features])
print(f"After combining features: {memory_mb():.2f} MB")

prediction = model.predict(combined_features)
probabilities = model.predict_proba(combined_features)

print(f"After prediction: {memory_mb():.2f} MB")

print("\nPrediction:", "Phishing" if prediction[0] == 1 else "Legitimate")
print(f"Phishing probability: {probabilities[0][1]:.2%}")

print(f"\nPeak/current memory: {memory_mb():.2f} MB")