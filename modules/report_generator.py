from modules.column_detection import detect_column_types


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
