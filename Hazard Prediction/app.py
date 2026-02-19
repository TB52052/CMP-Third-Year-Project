from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import pandas as pd

#utuilising flask to use python code on a webpage front end 
app = Flask(__name__)

# load models .pkl from previous file
model_severity = joblib.load("mlp_severity_model.pkl")
tfidf_severity = joblib.load("tfidf_severity.pkl")
mlb_severity = joblib.load("mlb_severity.pkl")
scaler_severity = joblib.load("scaler_severity.pkl")

model_likelihood = joblib.load("mlp_likelihood_model.pkl")
tfidf_likelihood = joblib.load("tfidf_likelihood.pkl")
scaler_likelihood = joblib.load("scaler_likelihood.pkl")

#lexicons
severity_lexicon = pd.read_csv("HAZARD–SEMANTIC LEXICON.csv")
severity_keywords = severity_lexicon['Keyword'].str.lower().tolist()
severity_dict = dict(zip(severity_lexicon['Keyword'].str.lower(), severity_lexicon['Severity']))

likelihood_lexicon = pd.read_csv("HAZARD–SEMANTIC LEXICON likelihood.csv")
likelihood_lexicon['Keyword'] = likelihood_lexicon['Keyword'].astype(str).str.lower().str.strip()
likelihood_lexicon = likelihood_lexicon[
    likelihood_lexicon['Keyword'].notna() &
    (likelihood_lexicon['Keyword'] != '') &
    (likelihood_lexicon['Keyword'] != 'nan')
]
likelihood_keywords = likelihood_lexicon['Keyword'].tolist()
likelihood_dict = dict(zip(likelihood_lexicon['Keyword'], likelihood_lexicon['Likelihood']))

# --------------------------
# Utility functions
# --------------------------
def find_keywords(description):
    if pd.isna(description):
        return []
    description_lower = description.lower()
    return [kw for kw in severity_keywords if kw in description_lower]

def compute_lexi_score(matched_keywords):
    if not matched_keywords:
        return 0.0
    severities = [severity_dict[kw] for kw in matched_keywords if kw in severity_dict]
    return float(np.mean(severities)) if severities else 0.0

def final_severity(description, original_severity):
    matched = find_keywords(description)
    if matched:
        severities = [severity_dict[kw] for kw in matched if kw in severity_dict]
        if severities:
            return max(severities)
    return original_severity

def find_likelihood_keywords(text):
    if pd.isna(text):
        return []
    text = text.lower()
    return [kw for kw in likelihood_keywords if kw in text]

def compute_mean_likelihood(matched):
    if not matched:
        return 0.0
    values = [likelihood_dict[k] for k in matched if k in likelihood_dict]
    return float(np.mean(values)) if values else 0.0

# --------------------------
# API route
# --------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    hazard_name = data.get("Hazard Name", "")
    hazard_description = data.get("Hazard Description", "")
    original_severity = data.get("Severity", 0)

    # Severity prediction
    matched_keywords = find_keywords(hazard_description)
    lexi_score = compute_lexi_score(matched_keywords)
    final_sev = final_severity(hazard_description, original_severity)

    X_tfidf = tfidf_severity.transform([hazard_description]).toarray()
    X_keywords = mlb_severity.transform([matched_keywords])
    X_num = scaler_severity.transform([[lexi_score]])
    X_input = np.hstack([X_tfidf, X_keywords, X_num])
    severity_pred = model_severity.predict(X_input)[0]

    # Likelihood prediction
    text_combined = f"{hazard_name} {hazard_description}"
    matched_likelihood = find_likelihood_keywords(text_combined)
    lexi_mean_likelihood = compute_mean_likelihood(matched_likelihood)
    X_num_lik = scaler_likelihood.transform([[0, lexi_mean_likelihood]])
    X_tfidf_lik = tfidf_likelihood.transform([text_combined]).toarray()
    X_input_lik = np.hstack([X_tfidf_lik, X_num_lik])
    likelihood_pred = model_likelihood.predict(X_input_lik)[0]

    return jsonify({
        "Predicted_Severity": int(severity_pred),
        "Predicted_Likelihood": int(likelihood_pred)
    })

#hosting the webpage
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        hazard_name = request.form.get("hazard_name", "")
        hazard_description = request.form.get("hazard_description", "")
        severity_input = int(request.form.get("severity", 0))

        # Severity prediction
        matched_keywords = find_keywords(hazard_description)
        lexi_score = compute_lexi_score(matched_keywords)
        X_tfidf = tfidf_severity.transform([hazard_description]).toarray()
        X_keywords = mlb_severity.transform([matched_keywords])
        X_num = scaler_severity.transform([[lexi_score]])
        X_input = np.hstack([X_tfidf, X_keywords, X_num])
        severity_pred = model_severity.predict(X_input)[0]

        # Likelihood prediction
        text_combined = f"{hazard_name} {hazard_description}"
        matched_likelihood = find_likelihood_keywords(text_combined)
        lexi_mean_likelihood = compute_mean_likelihood(matched_likelihood)
        X_num_lik = scaler_likelihood.transform([[0, lexi_mean_likelihood]])
        X_tfidf_lik = tfidf_likelihood.transform([text_combined]).toarray()
        X_input_lik = np.hstack([X_tfidf_lik, X_num_lik])
        likelihood_pred = model_likelihood.predict(X_input_lik)[0]

        prediction = {
            "severity": int(severity_pred),
            "likelihood": int(likelihood_pred)
        }

    # HTML form
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Hazard Prediction</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            input, textarea { width: 100%; padding: 8px; margin: 5px 0; }
            input[type="submit"], button { width: auto; padding: 10px 20px; }
            .result { margin-top: 20px; background-color: #f0f0f0; padding: 15px; border-radius: 5px; }
                     </style>
                     </head>
                     <body>
                     <h1>Hazard Prediction Form</h1>

        <form id="hazardForm">
            <label>Hazard Name:</label><br>
            <input type="text" id="hazard_name" required><br>
            
            <label>Hazard Description:</label><br>
            <textarea id="hazard_description" rows="4" required></textarea><br>
            
            <input type="submit" value="Predict">
            </form>
            
            <div id="result"></div>
            
            <script>
            const form = document.getElementById('hazardForm');
            form.addEventListener('submit', async (e) => {
                e.preventDefault(); 
                
                const payload = {
                    "Hazard Name": document.getElementById('hazard_name').value,
                    "Hazard Description": document.getElementById('hazard_description').value
                    };
                
                // Call Flask API
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                    });
                
                const data = await response.json();
                
                // Display result without clearing input fields
                document.getElementById('result').innerHTML = `
                <div class="result">
                    <h2>Prediction Result:</h2>
                    <p><strong>Predicted Severity:</strong> ${data.Predicted_Severity}</p>
                    <p><strong>Predicted Likelihood:</strong> ${data.Predicted_Likelihood}</p>
                    </div>
                    `;
                    });
            </script>
            </body>
            </html>
            """
    return render_template_string(html)


if __name__ == "__main__":
    app.run(debug=True)
