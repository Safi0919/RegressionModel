import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# ----------------------------
# Function to extract visa-related sentences
# ----------------------------
def extract_visa_related_sentences(text):
    text = str(text).lower()
    visa_keywords = [
        "sponsor", "sponsorship", "h1b", "opt", "cpt", "visa", "U.S. citizen", "polygraph", "clearance", "authorized", "authorization", "green card", "us citizen", "work permit", "work visa", "unrestricted access"
    ]
    sentences = re.split(r'(?<=[.!?]) +', text)
    relevant = [s for s in sentences if any(k in s for k in visa_keywords)]
    return " ".join(relevant)

# ----------------------------
# Load and preprocess data
# ----------------------------
df = pd.read_csv('job_data.csv')
df.dropna(subset=['description', 'tag'], inplace=True)

# Extract only visa-related sentences for training
df['filtered_text'] = df['description'].apply(extract_visa_related_sentences)

# ----------------------------
# Split the data
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['filtered_text'], df['tag'], test_size=0.2, random_state=42
)

# ----------------------------
# TF-IDF Vectorization
# ----------------------------
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ----------------------------
# Train the Logistic Regression model
# ----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# ----------------------------
# Evaluate the model
# ----------------------------
y_pred = model.predict(X_test_vec)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# ----------------------------
# Save the model and vectorizer
# ----------------------------
joblib.dump(model, 'job_classifier.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("Model and vectorizer saved.")

# ----------------------------
# Test the model on a sample job description
# ----------------------------
sample_text = """
We are looking for a backend engineer experienced in Node.js and MongoDB.
Applicants must be authorized to work in the U.S. without sponsorship.
"""

filtered = extract_visa_related_sentences(sample_text)
X_sample = vectorizer.transform([filtered])
prediction = model.predict(X_sample)[0]

print("\nTest Prediction:")
print("Original Text:", sample_text.strip())
print("Filtered Text:", filtered.strip())
print("Prediction:", prediction)
