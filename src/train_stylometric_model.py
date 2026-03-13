import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# load dataset with stylometric features
df = pd.read_csv("../data/processed/dataset_with_stylometry.csv")

# select stylometric features
X = df[[
    "avg_sentence_length",
    "sentence_variance",
    "type_token_ratio",
    "stopword_ratio",
    "punctuation_ratio"
]]

y = df["label"]

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# train classifier
model = LogisticRegression()
model.fit(X_train, y_train)

# predictions
y_pred = model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))