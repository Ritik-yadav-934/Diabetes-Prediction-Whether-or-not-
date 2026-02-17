from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("model/svm_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get values from form
    features = {
        "Pregnancies": float(request.form["Pregnancies"]),
        "Glucose": float(request.form["Glucose"]),
        "BloodPressure": float(request.form["BloodPressure"]),
        "SkinThickness": float(request.form["SkinThickness"]),
        "Insulin": float(request.form["Insulin"]),
        "BMI": float(request.form["BMI"]),
        "DiabetesPedigreeFunction": float(request.form["DiabetesPedigreeFunction"]),
        "Age": float(request.form["Age"])
    }

    # Convert to numpy array
    input_df = pd.DataFrame([features])

    # Scale
    final_features = scaler.transform(input_df)

    # Predict
    prediction = model.predict(final_features)[0]

    result = "Diabetic" if prediction == 1 else "Not Diabetic"

    return render_template("index.html", prediction_text=f"Prediction: {result}")

if __name__ == "__main__":
    app.run(debug=True)
