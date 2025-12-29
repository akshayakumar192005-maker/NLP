import joblib
model = joblib.load("model.joblib")

text = ["URGENT! Your bank account has been locked. Verify details at this link"]
pred = model.predict(text)[0]
proba = model.predict_proba(text)[0][1]

print("Prediction:", "spam" if pred==1 else "ham")
print("Spam probability:", proba)