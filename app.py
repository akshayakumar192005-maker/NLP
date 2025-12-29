from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

model = joblib.load("model.joblib")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    msg = data["message"]

    # -----------------------------
    # 🔥 1. Keyword-based instant spam detection (Indian scam words)
    # -----------------------------
    fraud_keywords = [
        "aadhaar", "aadhar", "kyc", "loan approved", "loan", "account blocked",
        "pan card", "verify your account", "update your account", 
        "your bank account", "blocked", "free gift", "click the link",
        "urgent update", "offer", "prize", "winner","otp"
    ]

    for word in fraud_keywords:
        if word.lower() in msg.lower():
            return jsonify({
                "label": "spam",
                "confidence": 0.99
            })

    # -----------------------------
    # 🔥 2. Machine Learning prediction (your model)
    # -----------------------------
    proba = model.predict_proba([msg])[0][1]
    pred = model.predict([msg])[0]

    return jsonify({
        "label": "spam" if pred == 1 else "ham",
        "confidence": float(proba)
    })

if __name__ == "__main__":
    app.run(debug=True)