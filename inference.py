import os
import joblib
import pandas as pd

# ------------------------- Load Saved Artifacts -------------------------

base_path = os.getcwd()
models_path = os.path.join(base_path, "models")

# Load model
model = joblib.load(os.path.join(models_path, "rf_diabetes_model.pkl"))

# Load scaler
scaler = joblib.load(os.path.join(models_path, "rf_scaler.pkl"))

# Load saved columns
saved_columns = joblib.load(os.path.join(models_path, "rf_columns.pkl"))

# ------------------------- Prediction Function -------------------------

def predict_diabetes(age, gender, weight, height, glucose, family_diabetes, hypertensive):
    """
    Takes patient inputs and returns:
        - prediction: "Diabetic" / "Not Diabetic"
        - probability: float percentage
    """

    # Create dataframe for the new sample
    new_data = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "weight": [weight],
        "height": [height],
        "glucose": [glucose],
        "family_diabetes": [family_diabetes],
        "hypertensive": [hypertensive],
    })

    # Calculate BMI
    new_data["bmi"] = new_data["weight"] / ((new_data["height"] / 100) ** 2)

    # Drop unused features since model uses BMI
    new_data = new_data.drop(["weight", "height"], axis=1)

    # Convert categorical columns to dummies
    new_data = pd.get_dummies(new_data, drop_first=True)

    # Add missing columns not present in new_data
    for col in saved_columns:
        if col not in new_data.columns:
            new_data[col] = 0

    # Ensure same column order as training data
    new_data = new_data[saved_columns]

    # Scale the new data
    new_data_scaled = scaler.transform(new_data)

    # Predict
    pred = model.predict(new_data_scaled)[0]
    prob = model.predict_proba(new_data_scaled)[0][1] * 100

    output_label = "Diabetic" if pred == 1 else "Not Diabetic"

    return output_label, round(prob, 2)


# ------------------------- Manual Test -------------------------
if __name__ == "__main__":
    label, prob = predict_diabetes(
        age=45,
        gender="Male",
        weight=80,
        height=175,
        glucose=150,
        family_diabetes=1,
        hypertensive=0
    )

    print("Prediction:", label)
    print("Probability:", prob, "%")
