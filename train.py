import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

print("Loading dataset...")
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df.rename(columns={"v1":"label","v2":"message"})
df = df[["label","message"]]

df["label_num"] = df["label"].map({"ham":0, "spam":1})

X = df["message"]
y = df["label_num"]

model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))
])

print("Training...")
model.fit(X, y)

joblib.dump(model, "model.joblib")
print("Model saved successfully!")