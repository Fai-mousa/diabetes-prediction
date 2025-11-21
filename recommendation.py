# recommendation.py (improved)

def risk_level(probability: float) -> str:
    if probability < 0.25:
        return "low"
    elif probability < 0.6:
        return "moderate"
    else:
        return "high"


def generate_general_message(probability: float) -> str:
    level = risk_level(probability)

    if level == "low":
        return (
            "Your predicted diabetes risk is *low* based on your current indicators. "
            "This suggests that your measurements fall within generally expected ranges. "
            "However, routine monitoring and healthy habits are still important."
        )

    elif level == "moderate":
        return (
            "Your predicted diabetes risk is *moderate*. "
            "It may be helpful to consult a healthcare professional for a more complete evaluation. "
            "Adopting preventive lifestyle habits now can significantly reduce your long-term risk."
        )

    else:
        return (
            "Your predicted diabetes risk is *high*. "
            "We strongly recommend discussing this with a qualified healthcare professional soon "
            "for proper testing, assessment, and guidance."
        )


def generate_feature_specific_advice(data: dict) -> list:
    advice = []

    glucose = data.get("Glucose")
    if glucose is not None:
        if glucose >= 200:
            advice.append(
                "Your glucose level is very high. Avoid sugary drinks and refined carbohydrates, "
                "and speak with a healthcare provider about diagnostic testing."
            )
        elif glucose >= 140:
            advice.append(
                "Your glucose level is above the normal range. Reducing added sugars and increasing "
                "fiber sources (vegetables, whole grains, legumes) may help improve levels over time."
            )
        else:
            advice.append(
                "Your glucose level appears within a typical range. Maintaining balanced meals and "
                "regular physical activity can help keep it stable."
            )

    bmi = data.get("BMI")
    if bmi is not None:
        if bmi >= 30:
            advice.append(
                "Your BMI falls in the obese range. Gradual, sustainable weight reduction through "
                "daily movement and nutrient-dense meals can significantly lower diabetes risk."
            )
        elif bmi >= 25:
            advice.append(
                "Your BMI is in the overweight range. Small lifestyle adjustments—like reducing "
                "processed snacks and adding 20–30 minutes of walking—can make a meaningful impact."
            )
        else:
            advice.append(
                "Your BMI is in the normal range. Continue prioritizing exercise and balanced nutrition."
            )

    bp = data.get("BloodPressure")
    if bp is not None:
        if bp >= 140:
            advice.append(
                "Your blood pressure reading is high. Consider reducing sodium intake, managing stress, "
                "and increasing activity. Consulting a clinician is recommended, especially if repeated readings are elevated."
            )
        elif bp >= 120:
            advice.append(
                "Your blood pressure is slightly elevated. Monitoring it regularly and adopting "
                "heart-healthy habits may help bring it back into range."
            )
        else:
            advice.append(
                "Your blood pressure appears within a normal range. Maintaining routine exercise and "
                "a balanced diet supports cardiovascular health."
            )

    age = data.get("Age")
    if age is not None:
        if age >= 45:
            advice.append(
                "Your age places you in a group with naturally increased diabetes risk. "
                "Regular screenings are especially important."
            )
        else:
            advice.append(
                "Building healthy habits at a younger age provides strong long-term protection against diabetes."
            )

    return advice


def four_week_plan() -> str:
    return (
        "**4-Week Action Plan**\n"
        "\n"
        "**Week 1: Awareness & Monitoring**\n"
        "- Track daily meals and note sugar intake.\n"
        "- Walk 15–20 minutes at least 4 days this week.\n"
        "- Measure blood pressure (if possible).\n"
        "\n"
        "**Week 2: Nutrition Foundations**\n"
        "- Replace sugary drinks with water or unsweetened beverages.\n"
        "- Add 1–2 servings of vegetables daily.\n"
        "- Reduce refined carbs (white bread, pastries).\n"
        "\n"
        "**Week 3: Physical Activity Boost**\n"
        "- Aim for 120–150 minutes of moderate exercise this week.\n"
        "- Include light strength activities (body-weight squats, bands).\n"
        "- Monitor energy levels and how your body responds.\n"
        "\n"
        "**Week 4: Long-Term Habits**\n"
        "- Maintain consistent meal timing.\n"
        "- Add more whole grains and fiber-rich foods.\n"
        "- Re-evaluate glucose/BP if you have the tools.\n"
    )


def generate_lifestyle_recommendations(probability: float, data: dict) -> str:
    parts = [
        "**Summary**",
        generate_general_message(probability),
        "",
        "**Personalized Insights**",
    ]

    feature_insights = generate_feature_specific_advice(data)
    for a in feature_insights:
        parts.append(f"- {a}")

    parts.append("")
    parts.append(four_week_plan())

    parts.append(
        "**General Tips**\n"
        "- Aim for 150 minutes of moderate exercise per week.\n"
        "- Prioritize vegetables, whole grains, lean proteins.\n"
        "- Limit sugary drinks and highly processed foods.\n"
        "- Avoid smoking; manage stress through sleep and movement."
    )

    parts.append(
        "\n**Disclaimer:** This tool provides educational guidance only and does not replace professional medical advice."
    )

    return "\n".join(parts)
