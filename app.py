import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


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


st.set_page_config(
    page_title="Data Analysis Report Agent",
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


def generate_markdown_report(
    df,
    data_quality_score=None,
    data_quality_suggestions=None,
    selected_target=None,
    task_info=None,
    correlation_summary=None
):
    rows, cols = df.shape
    missing_values = df.isnull().sum()
    duplicate_rows = df.duplicated().sum()
    data_quality_suggestions = data_quality_suggestions or []

    numeric_cols, categorical_cols, id_cols = detect_column_types(df)

    report = f"""
# Data Analysis Report

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

## 6. Correlation Analysis

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

## 7. Initial Insights

Based on the automatic analysis, this dataset can be further explored by checking data
quality, understanding feature distributions, and identifying possible relationships
between variables.

## 8. Suggested Next Steps

- Handle missing values if necessary.
- Remove duplicate records if they are not meaningful.
- Explore relationships between numerical and categorical variables.
- If there is a target column, this dataset may be used for classification or regression modeling.
"""

    return report


st.title("📊 Data Analysis Report Agent")

st.write(
    "Upload a CSV file, and this agent will automatically analyze the dataset "
    "and generate a simple data analysis report."
)

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df = prepare_data(df)

        st.success("CSV file uploaded successfully!")

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

        st.subheader("11. Generated Report")

        report = generate_markdown_report(
            df,
            data_quality_score=data_quality_score,
            data_quality_suggestions=data_quality_suggestions,
            selected_target=selected_target,
            task_info=task_info,
            correlation_summary=correlation_summary
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
    st.info("Please upload a CSV file to start the analysis.")
