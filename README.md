# ML Phishing Detection

A machine learning system that classifies email messages as **phishing** or **legitimate** using natural language features extracted from the email subject and body.

The project uses TF-IDF features with both word-level and character-level n-grams and a Logistic Regression classifier. The trained model is exposed through a Flask API and a web interface for interactive predictions.

## Project Goals

This project explores how machine learning can be applied to cybersecurity, specifically the detection of potentially malicious email messages.

In addition to measuring classification performance, the project evaluates how the model responds to controlled modifications of email content. This helps identify situations where a model may be sensitive to changes that do not fundamentally alter the underlying message.

## Model

The classifier uses the subject and body of each email as its input.

### Feature Extraction

Two TF-IDF representations are used:

- **Word-level TF-IDF:** captures individual words and two-word phrases.
- **Character-level TF-IDF:** captures character patterns and helps represent variations within words and URLs.

The two feature sets are combined into a single sparse feature matrix.

### Classifier

A Logistic Regression model is trained on the combined TF-IDF features.

The training process uses a stratified group split based on normalized email subjects. This reduces the chance of highly similar email templates appearing across the training, validation, and test sets.

The dataset is divided into:

- **Training:** 23,492 emails
- **Validation:** 7,831 emails
- **Test:** 7,831 emails

## Results

The final model was evaluated on a held-out test set of 7,831 emails.

| Metric    | Validation |   Test |
| --------- | ---------: | -----: |
| Accuracy  |     99.60% | 99.18% |
| Precision |     99.72% | 99.49% |
| Recall    |     99.48% | 99.04% |
| F1 Score  |     99.60% | 99.27% |

The test-set confusion matrix was:

|                       | Predicted Legitimate | Predicted Phishing |
| --------------------- | -------------------: | -----------------: |
| **Actual Legitimate** |                3,441 |                 22 |
| **Actual Phishing**   |                   42 |              4,326 |

The model correctly identified 4,326 of 4,368 phishing emails in the test set, while 42 phishing emails were classified as legitimate.

## Robustness Testing

The model was also evaluated against controlled modifications to held-out test emails. These experiments were designed to measure the model's sensitivity to controlled changes in email content.

The following transformations were tested:

- Character substitutions
- Whitespace changes
- Punctuation changes
- Character spacing
- URL obfuscation
- Addition of benign text

The most significant effect came from adding benign text to phishing emails. Across the 7,831-email test set, this transformation caused 392 phishing-to-legitimate classification flips.

A more detailed analysis found that 352 of these emails were initially classified correctly as phishing before the modification. These cases were concentrated in 35 normalized email templates, indicating that repeated patterns in the dataset contributed substantially to the observed vulnerability.

The results demonstrate that high classification accuracy does not necessarily imply robustness to changes in input content. They also highlight the importance of evaluating machine learning systems against controlled perturbations rather than relying solely on standard classification metrics.

These experiments are dataset-specific and should not be interpreted as real-world attack success rates.

## Dataset

The model was trained and evaluated using the CEAS_08 email dataset.

The dataset contains **39,154 emails**, with labels indicating whether each message is classified as phishing/spam or legitimate:

- **21,842 phishing/spam emails**
- **17,312 legitimate emails**

Only the email subject and body are used as model inputs. Sender, receiver, date, and URL-count fields are not directly provided to the classifier.

The dataset contains both legitimate and malicious email examples, including repeated message templates. This repetition was considered when creating the stratified group split used for evaluation.

The original dataset is not included in this repository.

## Technologies

- Python
- scikit-learn
- pandas
- NumPy
- SciPy
- Flask
- HTML / CSS / JavaScript
- GitHub Pages
- Render (hosting)

## Web Demo

The project includes a web interface where users can enter an email subject and body and receive a phishing or legitimate prediction from the trained model.

The application consists of:

- **Frontend:** HTML, CSS, and JavaScript hosted with GitHub Pages
- **Backend:** Flask REST API
- **Model serving:** The trained scikit-learn model and TF-IDF vectorizers
- **Deployment:** GitHub Pages for the frontend and Render for the API

**Live demo:** https://rzredg.github.io/ml-phishing-detection/

The frontend sends email content to the Flask API, which processes the text and returns the predicted class along with the model's phishing and legitimate probabilities.

## Project Structure

```text
ML Phishing Detection/
├── models/
│   ├── word_vectorizer.joblib
│   ├── char_vectorizer.joblib
│   └── phishing_model.joblib
├── src/
│   ├── inspect_dataset.py
│   ├── stratified_group_split.py
│   ├── analyze_split.py
│   ├── train_final.py
│   ├── predict.py
│   └── app.py
├── web/
│   ├── index.html
│   ├── script.js
│   └── styles.css
├── .gitignore
├── requirements.txt
└── README.md
```
