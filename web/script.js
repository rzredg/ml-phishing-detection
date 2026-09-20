const subjectInput = document.getElementById("subject");
const bodyInput = document.getElementById("body");

const analyzeButton = document.getElementById("analyze-button");

const resultCard = document.getElementById("result-card");
const resultLabel = document.getElementById("result-label");
const resultIcon = document.getElementById("result-icon");

const phishingProbability =
    document.getElementById("phishing-probability");

const legitimateProbability =
    document.getElementById("legitimate-probability");

const phishingBar =
    document.getElementById("phishing-bar");

const legitimateBar =
    document.getElementById("legitimate-bar");

const errorMessage =
    document.getElementById("error-message");


analyzeButton.addEventListener("click", analyzeEmail);


async function analyzeEmail() {

    const subject = subjectInput.value.trim();
    const body = bodyInput.value.trim();

    errorMessage.classList.add("hidden");
    resultCard.classList.add("hidden");

    if (!subject && !body) {
        showError("Please enter an email subject or body.");
        return;
    }

    analyzeButton.disabled = true;
    analyzeButton.textContent = "Analyzing...";

    try {

        const response = await fetch(
            "https://ml-phishing-detection-api.onrender.com/api/predict",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    subject: subject,
                    body: body
                })
            }
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.error || "Prediction failed."
            );
        }

        displayResult(data);

    } catch (error) {

        showError(
            "Could not connect to the prediction server. " +
            "Please try again in a moment."
        );

        console.error(error);

    } finally {

        analyzeButton.disabled = false;
        analyzeButton.textContent = "Analyze Email";
    }
}


function displayResult(data) {

    const phishingPercent =
        data.phishing_probability * 100;

    const legitimatePercent =
        data.legitimate_probability * 100;

    resultLabel.textContent = data.label;

    phishingProbability.textContent =
        `${phishingPercent.toFixed(2)}%`;

    legitimateProbability.textContent =
        `${legitimatePercent.toFixed(2)}%`;

    phishingBar.style.width =
        `${phishingPercent}%`;

    legitimateBar.style.width =
        `${legitimatePercent}%`;

    if (data.label === "Phishing") {
        resultIcon.textContent = "!";
    } else {
        resultIcon.textContent = "✓";
    }

    resultCard.classList.remove("hidden");
}


function showError(message) {

    errorMessage.textContent = message;
    errorMessage.classList.remove("hidden");
}