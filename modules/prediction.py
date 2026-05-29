import pandas as pd
import streamlit as st


def create_prediction_input(df, model_results):
    feature_columns = model_results["feature_columns"]
    numeric_features = model_results["numeric_features"]
    categorical_features = model_results["categorical_features"]
    input_values = {}

    for col in feature_columns:
        if col in numeric_features:
            numeric_values = pd.to_numeric(df[col], errors="coerce")
            median_value = numeric_values.median()

            if pd.isna(median_value):
                median_value = 0.0

            input_values[col] = st.number_input(
                col,
                value=float(median_value),
                key=f"prediction_numeric_{col}"
            )

        elif col in categorical_features:
            options = (
                df[col]
                .dropna()
                .drop_duplicates()
                .head(50)
                .tolist()
            )

            if len(options) > 0:
                input_values[col] = st.selectbox(
                    col,
                    options,
                    key=f"prediction_categorical_{col}"
                )
            else:
                input_values[col] = st.text_input(
                    col,
                    key=f"prediction_text_{col}"
                )

    return pd.DataFrame([input_values], columns=feature_columns)


def make_single_prediction(model_results, input_df):
    pipeline = model_results["best_pipeline"]
    prediction = pipeline.predict(input_df)[0]

    probabilities = None
    if hasattr(pipeline, "predict_proba"):
        try:
            probability_values = pipeline.predict_proba(input_df)[0]
            class_labels = pipeline.classes_
            probabilities = pd.DataFrame({
                "Class": class_labels,
                "Probability": probability_values
            })
        except Exception:
            probabilities = None

    return prediction, probabilities
