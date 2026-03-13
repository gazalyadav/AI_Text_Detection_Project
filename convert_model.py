import joblib

# load model using joblib
model = joblib.load("models/combined_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# save again with joblib (clean format)
joblib.dump(model, "models/combined_model.joblib")
joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")

print("Model successfully converted to joblib format.")