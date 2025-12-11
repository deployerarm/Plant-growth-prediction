# Models/llm_model.py
from langchain_groq import ChatGroq

# ---------------------------------------------------------
# Load Groq LLaMA-3.1 Model (Hardcoded API Key)
# ---------------------------------------------------------

def load_llm():

    groq_key = "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"  # <-- FIXED

    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.0,
        max_retries=2,
        groq_api_key=groq_key
    )


def explain_growth_llm(
        plant_name, height, leaf_area, health_score,
        forecast_60, forecast_120):

    llm = load_llm()

    prompt = f"""
You are an agricultural expert. Analyze the plant growth prediction.

Plant species: {plant_name}

Current values:
- Height: {height} cm
- Leaf area: {leaf_area} cm²
- Health score: {health_score}

Forecasted heights:
- 60 days: {forecast_60[-1]:.2f} cm
- 120 days: {forecast_120[-1]:.2f} cm

Write 5–7 rich paragraphs explaining:
1. Why this growth happened.
2. Any limiting factors.
3. How environment (temp, humidity, soil moisture, soil pH, light) affects the trends.
4. Long-term survival at 120d, 200d, and 1yr.
5. Actionable care instructions (watering, soil, nutrients, pests, light).
"""

    result = llm.invoke(prompt)
    return result.content
