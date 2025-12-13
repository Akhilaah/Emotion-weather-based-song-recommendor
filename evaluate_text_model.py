import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer #type: ignore
from sklearn.metrics import classification_report, confusion_matrix #type: ignore
import joblib

# ---- Load test data ----
test_df = pd.read_csv("test.csv")

X_test = test_df["text"]
y_test = test_df["label"]

# ---- Load trained model ----
model = joblib.load("text_emotion_model.pkl")

# ---- IMPORTANT: Load the same TF-IDF vectorizer used during training ----
vectorizer = joblib.load("text_vectorizer.pkl")

X_test_tfidf = vectorizer.transform(X_test)

# ---- Predict ----
y_pred = model.predict(X_test_tfidf)

# ---- Print classification report ----
print("\n📊 CLASSIFICATION REPORT:\n")
print(classification_report(y_test, y_pred))

# ---- Confusion matrix ----
print("\n🔢 CONFUSION MATRIX:\n")
print(confusion_matrix(y_test, y_pred))
