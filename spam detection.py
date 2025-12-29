import joblib

# Load the trained model
model = joblib.load("model.joblib")

print("Spam Detection System (type 'exit' to quit)")

while True:
    text = input("Enter a message: ")
    if text.lower() == "exit":
        print("Exiting...")
        break

    # Predict
    prediction = model.predict([text])[0]

    # Display result
    if prediction == 1:
        print("Result: Spam")
    else:
        print("Result: Not Spam")# Fixed parenthesis
