import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from modules.column_detection import detect_column_types


load_dotenv()


def get_openai_api_key():
    try:
        return os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        return os.getenv("OPENAI_API_KEY")


def get_gemini_api_key():
    try:
        return os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)
    except Exception:
        return os.getenv("GEMINI_API_KEY")


def generate_openai_insights(prompt, api_key):
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You generate beginner-friendly data analysis insights "
                    "from compact dataset summaries."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=900
    )
    return response.choices[0].message.content


def generate_gemini_insights(prompt, api_key):
    try:
        from google import genai
    except ImportError:
        raise ImportError(
            "google-genai is not installed. Please install it with: pip install google-genai"
        )

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text


def generate_ai_insights(df, column_info, data_quality_summary, ml_task_info):
    openai_key = get_openai_api_key()
    gemini_key = get_gemini_api_key()

    if not openai_key and not gemini_key:
        return None, (
            "AI insights are optional. Add OPENAI_API_KEY or GEMINI_API_KEY "
            "to your environment to enable the AI Insight Generator."
        ), None

    numeric_cols, categorical_cols, id_cols = detect_column_types(df)
    numeric_summary = {}
    categorical_summary = {}

    if len(numeric_cols) > 0:
        numeric_summary = (
            df[numeric_cols[:10]]
            .describe()
            .round(3)
            .to_dict()
        )

    for col in categorical_cols[:10]:
        categorical_summary[col] = (
            df[col]
            .astype(str)
            .value_counts()
            .head(5)
            .to_dict()
        )

    compact_summary = {
        "dataset_shape": {
            "rows": df.shape[0],
            "columns": df.shape[1]
        },
        "column_information": column_info.head(50).to_dict(orient="records"),
        "numeric_columns": numeric_cols[:20],
        "categorical_columns": categorical_cols[:20],
        "id_like_columns": id_cols[:20],
        "numeric_summary": numeric_summary,
        "top_categorical_values": categorical_summary,
        "data_quality_summary": data_quality_summary,
        "machine_learning_summary": ml_task_info
    }

    prompt = f"""
You are an AI data analysis assistant helping a beginner understand a CSV dataset.

Use only the compact summary below. Do not assume access to the full dataset.
Write a concise Markdown insight report with these sections:

1. Dataset overview
2. Data quality insights
3. Important patterns
4. Possible machine learning direction
5. Recommended next steps

Keep the tone clear, practical, and suitable for a student portfolio project.

Compact dataset summary:
{compact_summary}
"""

    openai_error = None

    if openai_key:
        try:
            insight_text = generate_openai_insights(prompt, openai_key)
            return insight_text, None, "OpenAI"
        except Exception as error:
            openai_error = str(error)

    if gemini_key:
        try:
            insight_text = generate_gemini_insights(prompt, gemini_key)
            return insight_text, None, "Gemini"
        except Exception as error:
            gemini_error = str(error)
            if openai_error:
                return None, (
                    "AI insights could not be generated safely. "
                    f"OpenAI error: {openai_error}. Gemini error: {gemini_error}"
                ), None
            return None, (
                "AI insights could not be generated safely. "
                f"Gemini error: {gemini_error}"
            ), None

    return None, (
        "OpenAI insights could not be generated, and no Gemini API key is configured. "
        f"OpenAI error: {openai_error}"
    ), None
