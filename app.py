# app.py

import streamlit as st
from inference import predict_diabetes
# Transformer-based recommendations will be used exclusively

st.set_page_config(
    page_title="Diabetes Risk & Smart Health Recommendations",
    page_icon="🩺",
    layout="centered",
)

st.title("🩺 Diabetes Risk Prediction & Personalized Advice")

st.write(
    """
A simple demo that estimates diabetes risk from basic inputs and gives
practical, non-medical lifestyle suggestions. This is educational only
and not a medical diagnosis — consult a healthcare professional for
medical decisions.
"""
)


st.header("1 — Enter your information")

age = st.number_input("Age (years)", min_value=10, max_value=120, value=30, step=1)
gender = st.selectbox("Gender", options=["Male", "Female"])
weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=70.0, step=0.1)
height = st.number_input("Height (cm)", min_value=80.0, max_value=250.0, value=170.0, step=0.1)
glucose = st.slider("Glucose (mg/dL)", min_value=40, max_value=400, value=110, step=1)
family_history = st.selectbox("Family history of diabetes?", options=["No", "Yes"])
hypertensive = st.selectbox("Known hypertension?", options=["No", "Yes"])

# Convert boolean-like inputs to integers expected by the model
family_diabetes = 1 if family_history == "Yes" else 0
hypertensive_flag = 1 if hypertensive == "Yes" else 0

st.header("2 — Predict")

if st.button("Predict My Diabetes Risk"):
    with st.spinner("Running model inference..."):
        # inference.predict_diabetes expects separate args
        label, prob_percent = predict_diabetes(
            age=age,
            gender=gender,
            weight=weight,
            height=height,
            glucose=glucose,
            family_diabetes=family_diabetes,
            hypertensive=hypertensive_flag,
        )

    # `predict_diabetes` returns probability as percentage (e.g. 42.5)
    st.subheader("Prediction Result")
    st.metric(label="Estimated risk (%)", value=f"{prob_percent:.1f}%")

    if label.lower().startswith("diab"):
        st.error("Model classification: Diabetic ")
    else:
        st.success("Model classification: Not Diabetic ")

    st.caption(
        "This estimate is based on a historical dataset and is not a medical diagnosis."
    )

    # Build a minimal patient dict for the recommendation systems
    bmi = weight / ((height / 100) ** 2) if height > 0 else None
    patient_data_for_rules = {
        "Age": age,
        "BMI": round(bmi, 1) if bmi is not None else None,
        "Glucose": glucose,
        "BloodPressure": None,
        "Pregnancies": 0,
        "Insulin": None,
        "DiabetesPedigreeFunction": None,
    }

    # Recommendation generation (transformer only)
    st.header("3 — Personalized recommendations")

    prob_decimal = prob_percent / 100.0

    with st.spinner("Generating personalized advice using transformer (may take a while)..."):
        try:
            # Import lazily so the app can still load if transformers aren't installed,
            # but we will show an error message if import or generation fails.
            from recommendation_nlp import generate_lifestyle_recommendations_transformer

            advice_text = generate_lifestyle_recommendations_transformer(
                prob_decimal, patient_data_for_rules
            )
        except Exception as e:
            advice_text = (
                "Transformer recommendation failed: "
                + str(e)
                + "\n\nEnsure you have the `transformers` (and `torch`) packages installed and network access to download the model."
            )

    st.text_area("Your personalized recommendations", value=advice_text, height=360)

else:
    st.info("Fill in your information and click **Predict My Diabetes Risk** to see results.")
