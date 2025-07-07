from flask import Flask, request, jsonify
import joblib
import re

# Load model and vectorizer
model = joblib.load('job_classifier.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Initialize Flask app
app = Flask(__name__)

# Visa-related sentence extractor
def extract_visa_related_sentences(text):
    text = str(text).lower()
    visa_keywords = [
        "sponsor", "sponsorship", "h1b", "opt", "cpt", "visa", "U.S. citizen", "polygraph", "clearance", "authorized", "authorization", "green card", "us citizen", "work permit", "work visa", "unrestricted access"
    ]
    sentences = re.split(r'(?<=[.!?]) +', text)
    relevant = [s for s in sentences if any(k in s for k in visa_keywords)]
    return " ".join(relevant)

# Route to analyze a job description
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    job_text = data.get('text', '')

    if not job_text.strip():
        return jsonify({'error': 'No job description provided.'}), 400

    # Filter and vectorize
    filtered_text = extract_visa_related_sentences(job_text)
    X_vec = vectorizer.transform([filtered_text])
    prediction = model.predict(X_vec)[0]

    return jsonify({
        'prediction': prediction,
        'filtered_text': filtered_text
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
