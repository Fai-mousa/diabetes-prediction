# 🩺 Diabetes Risk Prediction & Personalized Health Advice

A machine learning-powered web application that estimates diabetes risk from basic health information and generates personalized, detailed health recommendations using transformer-based NLP.

## Overview

This project combines:
- **RandomForest Classification** for diabetes risk prediction
- **Transformer NLP (Flan-T5)** for personalized, human-like health advice
- **Streamlit** for an interactive web interface

Users input basic health metrics, receive a diabetes risk estimate, and get tailored lifestyle recommendations based on their profile.

> **Disclaimer:** This tool is for educational purposes only and is **not** a medical diagnosis. Always consult a healthcare professional for medical decisions.

---

## Features

✅ **Fast Risk Estimation** – RandomForest model trained on Bangladesh diabetes dataset  
✅ **Personalized Advice** – Transformer-generated recommendations tailored to patient profile  
✅ **Detailed Guidance** – 4-week action plans, concrete examples, and monitoring tips  
✅ **Safety Filters** – Prevents medication/dosage recommendations; emphasizes professional consultation  
✅ **GPU Support** – Automatically uses CUDA if available for faster inference  
✅ **Lazy Loading** – Models only load when needed to reduce memory overhead  

---

## Project Structure

```
diabetes-prediction/
├── app.py                          # Streamlit UI (main entry point)
├── inference.py                    # ML model inference pipeline
├── model.py                        # Model training & hyperparameter tuning
├── recommendation.py               # Rule-based health recommendations
├── recommendation_nlp.py           # Transformer-based personalized advice
├── models/                         # Pre-trained model artifacts
│   ├── rf_diabetes_model.pkl       # Random Forest classifier
│   ├── rf_scaler.pkl               # Feature scaler
│   └── rf_columns.pkl              # Feature column names
├── dataset/                        # Training data
│   └── DiaBD_A Diabetes Dataset... # Bangladesh diabetes dataset (CSV)
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone or Download the Project

```bash
cd diabetes-prediction
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Key dependencies:**
- `streamlit` – Web interface
- `pandas`, `scikit-learn`, `joblib` – ML pipeline
- `transformers`, `torch` – NLP & transformer models
- `imbalanced-learn` – SMOTE for handling class imbalance

### Step 3: Verify Setup

Ensure the `models/` folder contains the three pickle files:
- `rf_diabetes_model.pkl`
- `rf_scaler.pkl`
- `rf_columns.pkl`

If missing, run:

```bash
python model.py
```

This will train a new model and save artifacts.

---

## Usage

### Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### User Workflow

1. **Enter Health Information**
   - Age, gender, weight, height
   - Glucose level, family history, hypertension status

2. **Get Risk Prediction**
   - Model estimates diabetes probability as a percentage
   - Classification: "Diabetic" or "Not Diabetic"

3. **Receive Personalized Advice**
   - Transformer generates a detailed 400–800 word recommendation
   - Includes: risk summary, 4-week action plan, concrete examples, monitoring tips
   - Always emphasizes consulting a healthcare professional

---

## Model Details

### Machine Learning Model
- **Algorithm:** Random Forest Classifier
- **Dataset:** Bangladesh Diabetes Dataset (DiaBD)
- **Features:** Age, Gender, BMI, Glucose, Family History, Hypertension
- **Class Handling:** SMOTE for imbalanced data
- **Hyperparameter Tuning:** GridSearchCV (5-fold CV)
- **Output:** Classification ("Diabetic" / "Not Diabetic") + Probability (%)

### NLP/Recommendation Model
- **Model:** Flan-T5-Small (768M parameters)
- **Task:** Sequence-to-Sequence text generation
- **Input:** Patient profile + risk score + rule-based explanation
- **Output:** Personalized health guidance (400–800 words)
- **Generation Strategy:** Beam search (6 beams) + sampling (temp=0.7, top_p=0.95)
- **Device:** Automatically uses CUDA (GPU) if available; falls back to CPU

### Minimum Output Guarantee
If the transformer generates output shorter than 3 lines (~60 words):
1. Retries generation up to 2 times
2. Falls back to rule-based + technical explanation if retries fail

---

## Training & Retraining

To retrain the model with updated data:

```bash
python model.py
```

This will:
1. Load the dataset from `dataset/`
2. Preprocess features and handle class imbalance
3. Perform grid search for optimal hyperparameters
4. Evaluate performance (accuracy, confusion matrix, classification report)
5. Save updated artifacts to `models/`

---

## Configuration

### Transformer Model
Edit `recommendation_nlp.py` to:
- Change model size: Update `MODEL_NAME` (e.g., `"google/flan-t5-base"` for larger model)
- Adjust generation length: Modify `max_new_tokens` (current: 600)
- Change generation strategy: Tune `num_beams`, `temperature`, `top_p`

### Risk Thresholds
Edit `recommendation.py` to adjust risk categorization:
- Low: < 25%
- Moderate: 25–60%
- High: ≥ 60%

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'transformers'`
**Solution:** Install required packages:
```bash
pip install transformers torch
```

### Issue: Model files not found
**Solution:** Ensure the `models/` folder contains the three pickle files, or retrain:
```bash
python model.py
```

### Issue: Streamlit app crashes or runs slowly
**Causes & Solutions:**
- **First-time transformer download:** Downloads ~1.5 GB; takes 5–10 minutes on first run
- **Memory issues:** Reduce batch size or use `flan-t5-small` (default) instead of larger models
- **GPU not detected:** Install `torch` with CUDA support: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

### Issue: Short or repetitive recommendations
**Solution:** Already handled! The app retries generation and appends rule-based text if output is too short.

---

## API Reference

### `inference.py`

```python
from inference import predict_diabetes

label, prob_percent = predict_diabetes(
    age=45,
    gender="Male",
    weight=80,
    height=175,
    glucose=150,
    family_diabetes=1,
    hypertensive=0
)
# Returns:
# label = "Diabetic" or "Not Diabetic"
# prob_percent = 72.5 (e.g., 72.5%)
```

### `recommendation.py`

```python
from recommendation import generate_lifestyle_recommendations

advice = generate_lifestyle_recommendations(probability=0.72, data={
    "Age": 45,
    "BMI": 26.0,
    "Glucose": 150,
    "BloodPressure": None,
    ...
})
# Returns: rule-based recommendation string
```

### `recommendation_nlp.py`

```python
from recommendation_nlp import generate_lifestyle_recommendations_transformer

advice = generate_lifestyle_recommendations_transformer(probability=0.72, data={...})
# Returns: transformer-generated personalized advice (400–800 words)
```

---

## Performance & Benchmarks

- **Inference Time (ML Model):** ~50–100 ms
- **Transformer Generation Time:** ~2–5 seconds (CPU); ~500 ms–1s (GPU)
- **Memory Usage:** ~500 MB (CPU mode); ~2 GB (GPU mode with transformer)
- **Model Accuracy:** ~85–92% (varies by dataset split; see `model.py` output)

---

## Future Enhancements

- [ ] Add medication history as a feature
- [ ] Incorporate genetic/family history severity levels
- [ ] Add multi-language support
- [ ] Deploy to cloud (Heroku, AWS, GCP)
- [ ] Add export functionality (PDF recommendations)
- [ ] Integrate real healthcare datasets (with privacy compliance)
- [ ] A/B test different NLP models for advice quality

---

## License

This project is for educational purposes. Use responsibly and always consult healthcare professionals.

---

## Contact & Support

For issues, questions, or contributions, please refer to the repository owner or create an issue in the GitHub repository.

---

## Acknowledgments

- **Dataset:** Bangladesh Diabetes Dataset (DiaBD)
- **ML Framework:** scikit-learn
- **NLP Model:** Hugging Face Transformers (Flan-T5)
- **UI Framework:** Streamlit

---

**Last Updated:** November 21, 2025  
**Status:** Active & Maintained
