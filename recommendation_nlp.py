# recommendation_nlp.py

from typing import Dict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from recommendation import generate_lifestyle_recommendations
import torch

MODEL_NAME = "google/flan-t5-small"

_tokenizer = None
_model = None
_device = None


def _load_model():
    """Lazy-load transformer model + tokenizer and move to available device."""
    global _tokenizer, _model, _device
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        _device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        try:
            _model.to(_device)
        except Exception:
            _device = next(_model.parameters()).device


def _build_patient_profile(probability: float, data: Dict) -> str:
    """Turn numeric features into a concise text profile for the transformer prompt."""
    risk_percent = round(probability * 100, 1)
    lines = [
        f"Estimated diabetes risk: {risk_percent}%",
        "",
        "Patient profile:",
        f"- Age: {data.get('Age', 'N/A')} years",
        f"- BMI: {data.get('BMI', 'N/A')}",
        f"- Glucose: {data.get('Glucose', 'N/A')} mg/dL",
        f"- Blood pressure: {data.get('BloodPressure', 'N/A')} mm Hg",
        f"- Family history (diabetes pedigree function): {data.get('DiabetesPedigreeFunction', 'N/A')}",
        f"- Number of pregnancies: {data.get('Pregnancies', 'N/A')}",
        f"- Insulin level: {data.get('Insulin', 'N/A')} IU/mL",
    ]
    return "\n".join(lines)


def _safety_filter(text: str) -> str:
    """Append disclaimer if text contains medication or treatment advice keywords."""
    risky_keywords = [
        "metformin", "insulin dose", "change your medication",
        "increase your dose", "reduce your dose", "tablets for diabetes",
        "pills for diabetes"
    ]
    if any(kw in text.lower() for kw in risky_keywords):
        text += (
            "\n\n⚠️ Note: For any questions about medications or treatment, "
            "please consult a licensed healthcare professional. "
            "This tool does not provide medication advice."
        )
    return text


def _clean_output(text: str) -> str:
    """Remove system prompts or instructions leaked into model output."""
    instruction_patterns = [
        "You are an empathetic", "Requirements:", "Patient profile:",
        "Base technical explanation:", "Now write the final",
        "Using the information below", "produce a detailed"
    ]
    lines = text.split("\n")
    cleaned_lines = []
    skip_mode = False

    for line in lines:
        line_lower = line.lower().strip()
        is_instruction = any(p.lower() in line_lower for p in instruction_patterns)
        if is_instruction:
            skip_mode = True
            continue
        if skip_mode and line.strip() and not any(k in line_lower for k in ["requirements:", "patient", "technical", "now write"]):
            skip_mode = False
        if not skip_mode and line.strip():
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip() or text


def generate_lifestyle_recommendations_transformer(probability: float, data: Dict) -> str:
    """
    Generate structured, transformer-enhanced lifestyle recommendations.
    Steps:
    1. Generate rule-based advice from recommendation.py
    2. Use transformer to rewrite & personalize it
    3. Ensure safe, well-structured output
    """
    _load_model()

    # Base rule-based advice
    base_text = generate_lifestyle_recommendations(probability, data)
    profile_text = _build_patient_profile(probability, data)

    # Prompt for the transformer model
    prompt = (
        "You are an empathetic, professional health assistant. "
        "Rewrite the following advice into a well-structured, detailed, and practical plan for an adult with no medical background.\n\n"
        "Instructions:\n"
        "- Start with a 1–2 sentence plain-language summary of risk.\n"
        "- Keep the original content but make it more actionable and easier to follow.\n"
        "- Structure as sections: Summary, Personalized Insights, 4-Week Action Plan, General Tips.\n"
        "- Use numbered steps or bullet points where appropriate.\n"
        "- Keep tone encouraging, non-judgmental.\n"
        "- Do NOT recommend medications or dosages; always advise consulting a healthcare professional.\n\n"
        f"Patient profile:\n{profile_text}\n\n"
        f"Base advice:\n{base_text}\n\n"
        "Now produce the final improved recommendation."
    )

    inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)

    if _device is not None:
        inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=600,
            num_beams=6,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            early_stopping=True,
            no_repeat_ngram_size=3
        )

    advice = _tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    advice = _clean_output(advice)
    advice = _safety_filter(advice)

    # Retry if output too short
    min_lines = 4
    min_words = 60
    if advice.count("\n") < (min_lines - 1) or len(advice.split()) < min_words:
        for _ in range(2):
            with torch.no_grad():
                alt_outputs = _model.generate(
                    **inputs,
                    max_new_tokens=800,
                    num_beams=4,
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.95,
                    early_stopping=True,
                    no_repeat_ngram_size=3
                )
                candidate = _tokenizer.decode(alt_outputs[0], skip_special_tokens=True).strip()
                candidate = _clean_output(candidate)
                candidate = _safety_filter(candidate)
                if candidate.count("\n") >= (min_lines - 1) and len(candidate.split()) >= min_words:
                    advice = candidate
                    break
        # Fallback to rule-based
        if advice.count("\n") < (min_lines - 1) or len(advice.split()) < min_words:
            advice = advice + "\n\n" + base_text

    return advice
