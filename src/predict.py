import joblib
from scipy.sparse import hstack


# --------------------------------------------------
# Load saved model and vectorizers
# --------------------------------------------------

word_vectorizer = joblib.load(
    "models/word_vectorizer.joblib"
)

char_vectorizer = joblib.load(
    "models/char_vectorizer.joblib"
)

model = joblib.load(
    "models/phishing_model.joblib"
)


# --------------------------------------------------
# Prediction function
# --------------------------------------------------

def predict_email(subject, body):
    text = subject + " " + body

    # Generate the same features used during training
    word_features = word_vectorizer.transform([text])
    char_features = char_vectorizer.transform([text])

    combined_features = hstack([
        word_features,
        char_features
    ])

    # Get prediction probabilities
    probabilities = model.predict_proba(combined_features)[0]

    legitimate_probability = probabilities[0]
    phishing_probability = probabilities[1]

    # Determine predicted class
    if phishing_probability >= legitimate_probability:
        label = "Phishing"
    else:
        label = "Legitimate"

    return {
        "label": label,
        "phishing_probability": phishing_probability,
        "legitimate_probability": legitimate_probability
    }


# --------------------------------------------------
# Command-line test
# --------------------------------------------------

if __name__ == "__main__":

    subject = input("Enter email subject: ")

    print("Enter email body. Press Enter on an empty line when finished:")

    body_lines = []

    while True:
        line = input()

        if line == "":
            break

        body_lines.append(line)

    body = "\n".join(body_lines)

    result = predict_email(subject, body)

    print("\n=== Prediction ===")
    print(f"Result: {result['label']}")
    print(
        f"Phishing probability: "
        f"{result['phishing_probability']:.2%}"
    )
    print(
        f"Legitimate probability: "
        f"{result['legitimate_probability']:.2%}"
    )