import io
import os

import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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


def get_agent_workflow_steps(api_provider=None):
    return [
        {
            "step": "1. Read Dataset",
            "tool": "Dataset Reader",
            "description": "Load uploaded CSV or Excel files and convert them into a pandas DataFrame.",
            "output": "Raw dataset preview and dataset shape."
        },
        {
            "step": "2. Prepare Data",
            "tool": "Data Type Converter",
            "description": "Detect numeric-like text columns and convert them into numerical values when appropriate.",
            "output": "Cleaned dataset for analysis."
        },
        {
            "step": "3. Detect Column Types",
            "tool": "Column Type Detector",
            "description": "Identify numerical, categorical, and ID-like columns.",
            "output": "Feature groups used for visualization and modeling suggestions."
        },
        {
            "step": "4. Assess Data Quality",
            "tool": "Data Quality Scorer",
            "description": "Check missing values, duplicate rows, and ID-like columns to calculate a data quality score.",
            "output": "Data quality score and cleaning suggestions."
        },
        {
            "step": "5. Analyze Correlations",
            "tool": "Correlation Analyzer",
            "description": "Calculate pairwise correlations between suitable numerical columns.",
            "output": "Correlation matrix and top strongest correlations."
        },
        {
            "step": "6. Suggest ML Task",
            "tool": "ML Task Recommender",
            "description": "Use the selected target column to suggest regression, binary classification, or multi-class classification.",
            "output": "Recommended task type, models, and evaluation metrics."
        },
        {
            "step": "7. Train Baseline Model",
            "tool": "Baseline Model Trainer",
            "description": "Train simple baseline classification or regression models using sklearn pipelines.",
            "output": "Model comparison table, best baseline model, and evaluation metrics."
        },
        {
            "step": "8. Generate AI Insights",
            "tool": "LLM Insight Generator",
            "description": "Send only a compact dataset summary to the configured LLM provider. OpenAI is tried first, and Gemini is used as fallback.",
            "output": f"AI-generated insight report using {api_provider or 'OpenAI/Gemini fallback'}."
        },
        {
            "step": "9. Generate Report",
            "tool": "Markdown Report Generator",
            "description": "Combine rule-based analysis and optional AI insights into a downloadable Markdown report.",
            "output": "Final data analysis report."
        }
    ]


def load_dataset(uploaded_file):
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        return df, "CSV", None

    if file_name.endswith(".xlsx"):
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_name = st.selectbox(
            "Choose an Excel sheet",
            excel_file.sheet_names
        )
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        return df, "Excel", sheet_name

    raise ValueError("Unsupported file format. Please upload a CSV or XLSX file.")


def prepare_data(df):
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":
            converted_col = pd.to_numeric(df[col], errors="coerce")
            valid_ratio = converted_col.notna().sum() / len(df)

            if valid_ratio > 0.8:
                df[col] = converted_col

    return df


def detect_column_types(df):
    id_cols = []

    for col in df.columns:
        unique_ratio = df[col].nunique(dropna=True) / len(df)

        if "id" in col.lower() or unique_ratio > 0.95:
            id_cols.append(col)

    numeric_cols = []
    categorical_cols = []

    for col in df.columns:
        if col in id_cols:
            continue

        unique_count = df[col].nunique(dropna=True)

        if pd.api.types.is_numeric_dtype(df[col]) and unique_count > 10:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols, id_cols


def calculate_data_quality_score(df):
    rows, cols = df.shape
    total_cells = rows * cols
    missing_ratio = df.isnull().sum().sum() / total_cells if total_cells > 0 else 0
    duplicate_ratio = df.duplicated().sum() / rows if rows > 0 else 0
    _, _, id_cols = detect_column_types(df)
    id_ratio = len(id_cols) / cols if cols > 0 else 0

    score = 100
    suggestions = []

    if missing_ratio > 0:
        missing_deduction = min(40, missing_ratio * 100)
        score -= missing_deduction
        suggestions.append(
            f"Missing values affect {missing_ratio:.1%} of all cells. "
            "Consider filling or removing missing values before modeling."
        )
    else:
        suggestions.append("No missing values were detected.")

    if duplicate_ratio > 0:
        duplicate_deduction = min(30, duplicate_ratio * 100)
        score -= duplicate_deduction
        suggestions.append(
            f"Duplicate rows make up {duplicate_ratio:.1%} of the dataset. "
            "Review whether these records should be removed."
        )
    else:
        suggestions.append("No duplicate rows were detected.")

    if len(id_cols) > 1 or id_ratio > 0.2:
        id_deduction = min(10, len(id_cols) * 2)
        score -= id_deduction
        suggestions.append(
            f"{len(id_cols)} ID-like columns were detected. "
            "Exclude ID columns from charts and machine learning features."
        )
    elif len(id_cols) == 1:
        suggestions.append(
            f"1 ID-like column was detected: {id_cols[0]}. "
            "It should usually be excluded from modeling."
        )

    score = max(0, round(score))
    return score, suggestions


def analyze_correlations(df, numeric_cols):
    if len(numeric_cols) < 2:
        return None

    corr_matrix = df[numeric_cols].corr()
    top_correlations = []

    for i, col_a in enumerate(corr_matrix.columns):
        for col_b in corr_matrix.columns[i + 1:]:
            value = corr_matrix.loc[col_a, col_b]
            if pd.notna(value):
                top_correlations.append({
                    "column_a": col_a,
                    "column_b": col_b,
                    "correlation": value
                })

    top_correlations = sorted(
        top_correlations,
        key=lambda item: abs(item["correlation"]),
        reverse=True
    )[:3]

    return {
        "corr_matrix": corr_matrix,
        "top_correlations": top_correlations
    }


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


st.set_page_config(
    page_title="AI Data Analysis & ML Agent",
    page_icon="📊",
    layout="wide"
)

def suggest_ml_task(df, target_col, numeric_cols, categorical_cols, id_cols):
    target_unique = df[target_col].nunique(dropna=True)
    target_missing = df[target_col].isnull().sum()

    numeric_features = [
        col for col in numeric_cols
        if col != target_col and col not in id_cols
    ]

    categorical_features = [
        col for col in categorical_cols
        if col != target_col and col not in id_cols
    ]

    if pd.api.types.is_numeric_dtype(df[target_col]) and target_unique > 10:
        task_type = "Regression"
        reason = (
            "The selected target column is numerical and contains many unique values. "
            "Therefore, this dataset is more suitable for a regression task."
        )
        suggested_models = [
            "Linear Regression",
            "Random Forest Regressor",
            "XGBoost Regressor"
        ]
        suggested_metrics = [
            "MAE",
            "RMSE",
            "R² Score"
        ]

    elif target_unique == 2:
        task_type = "Binary Classification"
        reason = (
            "The selected target column has only two unique classes. "
            "Therefore, this dataset is suitable for a binary classification task."
        )
        suggested_models = [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "XGBoost Classifier"
        ]
        suggested_metrics = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
            "ROC-AUC"
        ]

    else:
        task_type = "Multi-class Classification"
        reason = (
            "The selected target column contains more than two classes. "
            "Therefore, this dataset is suitable for a multi-class classification task."
        )
        suggested_models = [
            "Decision Tree",
            "Random Forest",
            "XGBoost Classifier",
            "Support Vector Machine"
        ]
        suggested_metrics = [
            "Accuracy",
            "Macro F1-score",
            "Weighted F1-score",
            "Confusion Matrix"
        ]

    return {
        "task_type": task_type,
        "reason": reason,
        "target_unique": target_unique,
        "target_missing": target_missing,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "suggested_models": suggested_models,
        "suggested_metrics": suggested_metrics
    }


def train_baseline_model(df, selected_target, task_info, numeric_cols, categorical_cols, id_cols):
    df_model = df.copy()
    df_model = df_model.dropna(subset=[selected_target])

    valid_numeric_features = [
        col for col in numeric_cols
        if col in df_model.columns and col != selected_target and col not in id_cols
    ]
    valid_categorical_features = [
        col for col in categorical_cols
        if col in df_model.columns and col != selected_target and col not in id_cols
    ]
    feature_cols = valid_numeric_features + valid_categorical_features

    if len(df_model) < 5:
        raise ValueError("At least 5 rows with a non-missing target are needed to train a baseline model.")

    if len(feature_cols) == 0:
        raise ValueError("No valid numerical or categorical feature columns are available for model training.")

    X = df_model[feature_cols]
    y = df_model[selected_target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, valid_numeric_features),
        ("cat", categorical_transformer, valid_categorical_features)
    ])

    task_type = task_info["task_type"]
    safe_target_name = str(selected_target).replace(" ", "_")

    if task_type in ["Binary Classification", "Multi-class Classification"]:
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            raise ValueError(
                "The train/test split needs at least two target classes in both sets. "
                "Try using a larger or more balanced dataset."
            )

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest Classifier": RandomForestClassifier(random_state=42)
        }
        comparison_rows = []
        confusion_matrices = {}
        trained_pipelines = {}

        for model_name, model in models.items():
            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ])
            pipeline.fit(X_train, y_train)
            trained_pipelines[model_name] = pipeline
            predictions = pipeline.predict(X_test)

            comparison_rows.append({
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, predictions),
                "Precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
                "Recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
                "F1-score": f1_score(y_test, predictions, average="weighted", zero_division=0)
            })

            labels = sorted(y.unique(), key=lambda value: str(value))
            confusion_matrices[model_name] = {
                "labels": labels,
                "matrix": confusion_matrix(y_test, predictions, labels=labels).tolist()
            }

        comparison_df = pd.DataFrame(comparison_rows)
        best_model_name = comparison_df.sort_values("F1-score", ascending=False).iloc[0]["Model"]
        safe_model_name = best_model_name.replace(" ", "_")
        interpretation = (
            f"{best_model_name} performed best because it had the highest weighted F1-score "
            "among the baseline classification models."
        )

        return {
            "task_type": task_type,
            "target_column": selected_target,
            "comparison_table": comparison_df,
            "best_model_name": best_model_name,
            "best_pipeline": trained_pipelines[best_model_name],
            "best_metric_name": "Weighted F1-score",
            "interpretation": interpretation,
            "confusion_matrix": confusion_matrices[best_model_name],
            "feature_columns": feature_cols,
            "numeric_features": valid_numeric_features,
            "categorical_features": valid_categorical_features,
            "model_file_name": f"baseline_model_{safe_target_name}_{safe_model_name}.pkl"
        }

    if task_type == "Regression":
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(random_state=42)
        }
        comparison_rows = []
        trained_pipelines = {}

        for model_name, model in models.items():
            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ])
            pipeline.fit(X_train, y_train)
            trained_pipelines[model_name] = pipeline
            predictions = pipeline.predict(X_test)
            mse = mean_squared_error(y_test, predictions)

            comparison_rows.append({
                "Model": model_name,
                "MAE": mean_absolute_error(y_test, predictions),
                "MSE": mse,
                "RMSE": np.sqrt(mse),
                "R2 Score": r2_score(y_test, predictions)
            })

        comparison_df = pd.DataFrame(comparison_rows)
        best_model_name = comparison_df.sort_values("R2 Score", ascending=False).iloc[0]["Model"]
        safe_model_name = best_model_name.replace(" ", "_")
        interpretation = (
            f"{best_model_name} performed best because it had the highest R2 score "
            "among the baseline regression models."
        )

        return {
            "task_type": task_type,
            "target_column": selected_target,
            "comparison_table": comparison_df,
            "best_model_name": best_model_name,
            "best_pipeline": trained_pipelines[best_model_name],
            "best_metric_name": "R2 Score",
            "interpretation": interpretation,
            "feature_columns": feature_cols,
            "numeric_features": valid_numeric_features,
            "categorical_features": valid_categorical_features,
            "model_file_name": f"baseline_model_{safe_target_name}_{safe_model_name}.pkl"
        }

    raise ValueError(f"Unsupported task type for baseline training: {task_type}")


def create_model_download_bytes(model_results):
    buffer = io.BytesIO()
    model_package = {
        "model": model_results["best_pipeline"],
        "task_type": model_results["task_type"],
        "target_column": model_results["target_column"],
        "best_model_name": model_results["best_model_name"],
        "feature_columns": model_results["feature_columns"],
        "numeric_features": model_results["numeric_features"],
        "categorical_features": model_results["categorical_features"],
        "best_metric_name": model_results["best_metric_name"],
        "interpretation": model_results["interpretation"]
    }
    joblib.dump(model_package, buffer)
    buffer.seek(0)
    return buffer.getvalue()


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


def generate_markdown_report(
    df,
    data_quality_score=None,
    data_quality_suggestions=None,
    selected_target=None,
    task_info=None,
    correlation_summary=None,
    ai_insights=None,
    model_results=None
):
    rows, cols = df.shape
    missing_values = df.isnull().sum()
    duplicate_rows = df.duplicated().sum()
    data_quality_suggestions = data_quality_suggestions or []

    numeric_cols, categorical_cols, id_cols = detect_column_types(df)

    report = f"""
# AI Data Analysis & ML Report

## 1. Dataset Overview

- Number of rows: {rows}
- Number of columns: {cols}
- Number of duplicate rows: {duplicate_rows}
- Number of numerical columns: {len(numeric_cols)}
- Number of categorical columns: {len(categorical_cols)}
- Number of ID-like columns: {len(id_cols)}

## 2. Missing Value Summary

"""

    if missing_values.sum() == 0:
        report += "No missing values were found in the dataset.\n"
    else:
        report += "The dataset contains missing values in the following columns:\n\n"
        for col, value in missing_values.items():
            if value > 0:
                report += f"- {col}: {value} missing values\n"

    report += f"""

## 3. Data Quality Assessment

- Data quality score: {data_quality_score if data_quality_score is not None else "Not calculated"}/100

"""

    if data_quality_suggestions:
        report += "Suggestions:\n\n"
        for suggestion in data_quality_suggestions:
            report += f"- {suggestion}\n"
    else:
        report += "No data quality suggestions were generated.\n"

    report += """

## 4. Numerical Column Summary

"""

    if len(numeric_cols) == 0:
        report += "No numerical columns were found.\n"
    else:
        summary = df[numeric_cols].describe().T
        report += summary.to_markdown()

    report += """

## 5. Machine Learning Task Suggestion

"""

    if task_info is None or selected_target is None:
        report += "No suitable target column was selected, so no machine learning task suggestion was generated.\n"
    else:
        report += f"""- Selected target column: {selected_target}
- Suggested task type: {task_info["task_type"]}
- Reason: {task_info["reason"]}
- Target unique values: {task_info["target_unique"]}
- Target missing values: {task_info["target_missing"]}

Suggested models:
"""
        for model in task_info["suggested_models"]:
            report += f"- {model}\n"

        report += "\nSuggested evaluation metrics:\n"
        for metric in task_info["suggested_metrics"]:
            report += f"- {metric}\n"

    report += """

## 6. Model Training Results

"""

    if model_results:
        comparison_table = model_results["comparison_table"]
        report += f"""- Task type: {model_results["task_type"]}
- Target column: {model_results["target_column"]}
- Best model: {model_results["best_model_name"]}
- Best metric: {model_results["best_metric_name"]}
- Model download available: Yes
- Prediction demo available: Yes

Model comparison:

"""
        report += comparison_table.to_markdown(index=False)
        report += f"\n\nInterpretation: {model_results['interpretation']}\n"
    else:
        report += "Baseline model training was not performed.\n\n"
        report += "- Model download available: No\n"
        report += "- Prediction demo available: No\n"

    report += """

## 7. Correlation Analysis

"""

    if correlation_summary is None:
        report += "Correlation analysis was skipped because fewer than two suitable numerical columns were found.\n"
    elif len(correlation_summary["top_correlations"]) == 0:
        report += "No valid pairwise correlations were found among the suitable numerical columns.\n"
    else:
        report += "Top strongest correlations:\n\n"
        for item in correlation_summary["top_correlations"]:
            report += (
                f"- {item['column_a']} and {item['column_b']}: "
                f"{item['correlation']:.3f}\n"
            )

    report += """

## 8. AI Generated Insights

"""

    if ai_insights:
        report += ai_insights
        report += "\n"
    else:
        report += "AI insights were not generated for this report.\n"

    report += """

## 9. Initial Insights

Based on the automatic analysis, this dataset can be further explored by checking data
quality, understanding feature distributions, and identifying possible relationships
between variables.

## 10. Suggested Next Steps

- Handle missing values if necessary.
- Remove duplicate records if they are not meaningful.
- Explore relationships between numerical and categorical variables.
- If there is a target column, this dataset may be used for classification or regression modeling.
"""

    return report


st.title("📊 AI Data Analysis & ML Agent")

st.write(
    "Upload a CSV or Excel dataset to analyze data quality, generate reports, "
    "train baseline machine learning models, download the best model, and try "
    "a simple prediction demo."
)

st.subheader("Agent Workflow")
st.write(
    "This project follows an agent-style workflow. Each step acts like a tool "
    "that processes the dataset and passes useful information to the next step."
)
st.markdown(
    "CSV/Excel Upload → Data Profiling → Quality Check → ML Suggestion → "
    "Baseline Training → LLM Insights → Report"
)

for step_info in get_agent_workflow_steps():
    with st.expander(step_info["step"]):
        st.write(f"**Tool:** {step_info['tool']}")
        st.write(f"**Description:** {step_info['description']}")
        st.write(f"**Output:** {step_info['output']}")

uploaded_file = st.file_uploader(
    "Upload your dataset file",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:
    try:
        df, file_type, sheet_name = load_dataset(uploaded_file)
        df = prepare_data(df)

        file_signature = f"{uploaded_file.name}-{df.shape}-{list(df.columns)}"
        if st.session_state.get("ai_file_signature") != file_signature:
            st.session_state["ai_file_signature"] = file_signature
            st.session_state["ai_insights"] = None
            st.session_state["model_results"] = None

        st.success(f"{file_type} file uploaded successfully!")

        if sheet_name:
            st.info(f"Selected Excel sheet: {sheet_name}")

        st.subheader("1. Dataset Preview")
        st.dataframe(df.head())

        st.subheader("2. Dataset Shape")
        st.write(f"Rows: {df.shape[0]}")
        st.write(f"Columns: {df.shape[1]}")

        st.subheader("3. Column Information")
        column_info = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str),
            "Missing Values": df.isnull().sum().values,
            "Unique Values": df.nunique().values
        })
        st.dataframe(column_info)

        numeric_cols, categorical_cols, id_cols = detect_column_types(df)

        st.subheader("4. Duplicate Rows")
        duplicate_count = df.duplicated().sum()
        st.write(f"Number of duplicate rows: {duplicate_count}")

        st.subheader("5. Data Quality Assessment")
        data_quality_score, data_quality_suggestions = calculate_data_quality_score(df)
        st.metric("Data Quality Score", f"{data_quality_score}/100")

        for suggestion in data_quality_suggestions:
            if data_quality_score >= 80:
                st.info(suggestion)
            else:
                st.warning(suggestion)

        st.subheader("6. Descriptive Statistics")

        if len(numeric_cols) == 0:
            st.info("No suitable numerical columns found.")
        else:
            st.dataframe(df[numeric_cols].describe())

        st.subheader("7. Machine Learning Task Suggestion")

        target_options = [
            col for col in df.columns
            if col not in id_cols
        ]
        selected_target = None
        task_info = None

        if len(target_options) == 0:
            st.info("No suitable target column found.")
        else:
            default_index = 0

            if "Churn" in target_options:
                default_index = target_options.index("Churn")

            selected_target = st.selectbox(
                "Choose a target column",
                target_options,
                index=default_index
            )      

            task_info = suggest_ml_task(
                df,
                selected_target,
                numeric_cols,
                categorical_cols,
                id_cols
            )

            st.success(f"Suggested Machine Learning Task: {task_info['task_type']}")

            st.write("**Reason:**")
            st.write(task_info["reason"])

            st.write("**Target Column Summary:**")
            st.write(f"- Target column: {selected_target}")
            st.write(f"- Unique values: {task_info['target_unique']}")
            st.write(f"- Missing values: {task_info['target_missing']}")

            st.write("**Available Numerical Features:**")
            st.write(task_info["numeric_features"])

            st.write("**Available Categorical Features:**")
            st.write(task_info["categorical_features"])

            st.write("**Suggested Models:**")
            for model in task_info["suggested_models"]:
                st.write(f"- {model}")

            st.write("**Suggested Evaluation Metrics:**")
            for metric in task_info["suggested_metrics"]:
                st.write(f"- {metric}")

        st.subheader("8. Correlation Analysis")
        correlation_summary = analyze_correlations(df, numeric_cols)

        if correlation_summary is None:
            st.info("Correlation analysis requires at least two suitable numerical columns.")
        else:
            corr_matrix = correlation_summary["corr_matrix"]
            st.write("**Correlation Matrix:**")
            st.dataframe(corr_matrix)

            fig, ax = plt.subplots()
            heatmap = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr_matrix.columns)))
            ax.set_yticks(range(len(corr_matrix.columns)))
            ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
            ax.set_yticklabels(corr_matrix.columns)
            ax.set_title("Correlation Heatmap")
            fig.colorbar(heatmap, ax=ax)
            fig.tight_layout()
            st.pyplot(fig)

            st.write("**Top 3 Strongest Correlations:**")
            if len(correlation_summary["top_correlations"]) == 0:
                st.info("No valid pairwise correlations were found.")
            else:
                for item in correlation_summary["top_correlations"]:
                    st.write(
                        f"- {item['column_a']} and {item['column_b']}: "
                        f"{item['correlation']:.3f}"
                    )

        st.subheader("9. Numerical Column Visualization")

        if len(numeric_cols) == 0:
            st.info("No numerical columns found.")
        else:
            selected_num_col = st.selectbox(
                "Choose a numerical column",
                numeric_cols
            )

            fig, ax = plt.subplots()
            ax.hist(df[selected_num_col].dropna(), bins=20)
            ax.set_title(f"Distribution of {selected_num_col}")
            ax.set_xlabel(selected_num_col)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

        st.subheader("10. Categorical Column Visualization")

        if len(categorical_cols) == 0:
            st.info("No categorical columns found.")
        else:
            selected_cat_col = st.selectbox(
                "Choose a categorical column",
                categorical_cols
            )

            value_counts = df[selected_cat_col].value_counts().head(10)

            fig, ax = plt.subplots()
            ax.bar(value_counts.index.astype(str), value_counts.values)
            ax.set_title(f"Top Categories in {selected_cat_col}")
            ax.set_xlabel(selected_cat_col)
            ax.set_ylabel("Count")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)

        st.subheader("11. AI Generated Insights")
        openai_key = get_openai_api_key()
        gemini_key = get_gemini_api_key()
        api_key_available = bool(openai_key or gemini_key)

        data_quality_summary = {
            "score": data_quality_score,
            "suggestions": data_quality_suggestions,
            "duplicate_rows": int(duplicate_count),
            "missing_values_by_column": {
                col: int(value)
                for col, value in df.isnull().sum().items()
                if value > 0
            }
        }

        ml_task_summary = None
        if task_info is not None and selected_target is not None:
            ml_task_summary = {
                "selected_target": selected_target,
                "task_type": task_info["task_type"],
                "reason": task_info["reason"],
                "suggested_models": task_info["suggested_models"],
                "suggested_metrics": task_info["suggested_metrics"]
            }

        if not api_key_available:
            st.info(
                "AI insights are optional. Add OPENAI_API_KEY or GEMINI_API_KEY "
                "to your environment to enable the AI Insight Generator."
            )

        st.info(
            f"OpenAI configured: {'Yes' if openai_key else 'No'}\n\n"
            f"Gemini configured: {'Yes' if gemini_key else 'No'}"
        )

        if st.button("Generate AI Insights", disabled=not api_key_available):
            with st.spinner("Generating AI insights from the dataset summary..."):
                ai_insights, ai_error, provider_name = generate_ai_insights(
                    df,
                    column_info,
                    data_quality_summary,
                    ml_task_summary
                )

            if ai_error:
                st.warning(ai_error)
            else:
                st.session_state["ai_insights"] = ai_insights
                st.success(f"AI insights generated successfully using {provider_name}.")

        if st.session_state.get("ai_insights"):
            st.markdown(st.session_state["ai_insights"])

        st.subheader("12. Baseline Model Training")
        st.write(
            "This trains simple baseline models so you can compare a first modeling result. "
            "It is not a final production model."
        )

        can_train_model = task_info is not None and selected_target is not None

        if not can_train_model:
            st.info("Choose a suitable target column before training a baseline model.")

        if st.button("Train Baseline Model", disabled=not can_train_model):
            with st.spinner("Training baseline models..."):
                try:
                    model_results = train_baseline_model(
                        df,
                        selected_target,
                        task_info,
                        numeric_cols,
                        categorical_cols,
                        id_cols
                    )
                    st.session_state["model_results"] = model_results
                    st.success("Baseline model training completed.")
                except Exception as training_error:
                    st.session_state["model_results"] = None
                    st.warning(f"Baseline model training could not be completed: {training_error}")

        model_results = st.session_state.get("model_results")
        current_model_results = None

        if model_results and model_results["target_column"] == selected_target:
            current_model_results = model_results
            st.write("**Model Comparison:**")
            st.dataframe(model_results["comparison_table"])

            st.write(f"**Best model:** {model_results['best_model_name']}")
            st.write(model_results["interpretation"])

            if model_results["task_type"] in ["Binary Classification", "Multi-class Classification"]:
                confusion_info = model_results["confusion_matrix"]
                confusion_df = pd.DataFrame(
                    confusion_info["matrix"],
                    index=confusion_info["labels"],
                    columns=confusion_info["labels"]
                )
                st.write("**Confusion Matrix for Best Model:**")
                st.dataframe(confusion_df)
            elif model_results["task_type"] == "Regression":
                st.info(
                    f"The best regression baseline was selected using "
                    f"{model_results['best_metric_name']}."
                )

            if current_model_results and "best_pipeline" in current_model_results:
                st.write(
                    "The downloaded file contains the best baseline sklearn pipeline, "
                    "including preprocessing steps and the trained model."
                )
                st.download_button(
                    label="Download Best Baseline Model (.pkl)",
                    data=create_model_download_bytes(current_model_results),
                    file_name=current_model_results["model_file_name"],
                    mime="application/octet-stream"
                )
        elif model_results:
            st.info("Stored model results are for a different target column. Train again to update them.")

        st.subheader("13. Prediction Demo")

        if current_model_results and "best_pipeline" in current_model_results:
            st.write(
                "This is a simple demo using the best baseline model trained above."
            )

            prediction_input_df = create_prediction_input(df, current_model_results)

            if st.button("Predict with Best Baseline Model"):
                try:
                    prediction, probabilities = make_single_prediction(
                        current_model_results,
                        prediction_input_df
                    )
                    st.success(f"Predicted value: {prediction}")

                    if probabilities is not None:
                        st.write("**Class Probabilities:**")
                        st.dataframe(probabilities)

                    st.info(
                        "This is a baseline prediction and should not be used as a "
                        "final production decision."
                    )
                except Exception as prediction_error:
                    st.warning(f"Prediction could not be completed: {prediction_error}")
        else:
            st.info("Train a baseline model first to use the prediction demo.")

        st.subheader("14. Generated Report")

        report = generate_markdown_report(
            df,
            data_quality_score=data_quality_score,
            data_quality_suggestions=data_quality_suggestions,
            selected_target=selected_target,
            task_info=task_info,
            correlation_summary=correlation_summary,
            ai_insights=st.session_state.get("ai_insights"),
            model_results=current_model_results
        )
        st.markdown(report)

        st.download_button(
            label="Download Report as Markdown",
            data=report,
            file_name="data_analysis_report.md",
            mime="text/markdown"
        )

    except Exception as e:
        st.error("Something went wrong while reading the file.")
        st.write(e)
else:
    st.info("Please upload a CSV or Excel file to start the analysis.")
