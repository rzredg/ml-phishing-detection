from flask import Flask, request, jsonify
from flask_cors import CORS
from src.predict import predict_email


app = Flask(__name__)
CORS(app)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok"
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({
            "error": "No JSON data provided."
        }), 400

    subject = data.get("subject", "")
    body = data.get("body", "")

    if not subject.strip() and not body.strip():
        return jsonify({
            "error": "Please provide an email subject or body."
        }), 400

    result = predict_email(subject, body)

    return jsonify({
        "label": result["label"],
        "phishing_probability": result["phishing_probability"],
        "legitimate_probability": result["legitimate_probability"]
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)