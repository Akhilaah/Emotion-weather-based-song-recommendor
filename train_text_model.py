import pandas as pd
from sklearn.model_selection import train_test_split #type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer #type: ignore
from sklearn.linear_model import LogisticRegression #type: ignore
import joblib

# Load CSV files
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Combine train + val if needed
#val_df = pd.read_csv("val.csv")
#train_df = pd.concat([train_df, val_df])

# Prepare data
X_train = train_df["text"]
y_train = train_df["label"]

X_test = test_df["text"]
y_test = test_df["label"]

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=2000)
model.fit(X_train_vec, y_train)

# Test accuracy
acc = model.score(X_test_vec, y_test)
print("Test Accuracy:", acc)

# Save vectorizer & model
joblib.dump(vectorizer, "text_vectorizer.pkl")
joblib.dump(model, "text_emotion_model.pkl")

print("Model saved!")
