from flask import Flask, request, jsonify
import joblib

# Load model and vectorizer
model = joblib.load('job_classifier.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Start Flask app
app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    job_text = data.get('text', '')

    if not job_text.strip():
        return jsonify({'error': 'No job description provided.'}), 400

    X_vec = vectorizer.transform([job_text])
    prediction = model.predict(X_vec)[0]

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
