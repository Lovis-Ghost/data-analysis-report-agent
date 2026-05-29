# Technical Design

## Project Name

AI Data Analysis & ML Agent

## System Overview

AI Data Analysis & ML Agent is a Streamlit web application for first-pass dataset analysis and baseline machine learning. Users can upload CSV or Excel files, review automated profiling results, generate optional AI-assisted insights, train baseline models, test a single prediction, download the best model pipeline, and export reports in Markdown, Word, or PDF format.

The system is organized as a modular Python project. `app.py` controls the Streamlit interface, while reusable logic lives in the `modules/` package.

## Main Modules and Responsibilities

| Module | Responsibility |
|---|---|
| `modules/workflow.py` | Defines the visible agent-style workflow steps shown in the app. |
| `modules/data_loader.py` | Loads CSV and Excel files and prepares numeric-like text columns. |
| `modules/column_detection.py` | Detects numerical, categorical, and ID-like columns. |
| `modules/data_quality.py` | Calculates a simple data quality score and cleaning suggestions. |
| `modules/correlation.py` | Computes correlation matrices and strongest numerical relationships. |
| `modules/ai_insights.py` | Generates optional AI insights using OpenAI or Gemini fallback. |
| `modules/ml_task.py` | Suggests regression, binary classification, or multi-class classification. |
| `modules/ml_trainer.py` | Trains baseline models, evaluates metrics, selects the best model, and packages model downloads. |
| `modules/prediction.py` | Builds prediction demo inputs and runs single-sample predictions. |
| `modules/report_generator.py` | Creates the Markdown report content. |
| `modules/report_export.py` | Converts generated report text into Word and PDF download bytes. |

## End-to-End Data Flow

1. The user uploads a CSV or Excel dataset in the Streamlit interface.
2. The app loads the file into a pandas DataFrame.
3. Numeric-like text columns are converted when appropriate.
4. Column detection identifies numerical, categorical, and ID-like columns.
5. Data quality checks identify missing values, duplicate rows, and ID-like columns.
6. Correlation analysis summarizes relationships between numerical columns.
7. The user selects a target column for machine learning.
8. The app suggests a suitable machine learning task type.
9. Optional AI insights can be generated from a compact dataset summary.
10. Baseline models can be trained and compared.
11. The best model can be downloaded as a reusable pipeline.
12. The prediction demo can score one manually entered sample.
13. A report is generated and can be downloaded as Markdown, Word, or PDF.

## Machine Learning Workflow

The machine learning workflow is intentionally baseline-focused. It is designed to provide a fast first modeling result rather than a production-ready model.

For classification tasks:

- Logistic Regression
- Random Forest Classifier
- Metrics: accuracy, weighted precision, weighted recall, weighted F1-score, and confusion matrix
- Best model selection: weighted F1-score

For regression tasks:

- Linear Regression
- Random Forest Regressor
- Metrics: MAE, MSE, RMSE, and R2 score
- Best model selection: R2 score

All models use a scikit-learn Pipeline with preprocessing:

- Numerical features: median imputation and standard scaling
- Categorical features: most-frequent imputation and one-hot encoding
- ID-like columns and the selected target column are excluded from features

## Report Export Workflow

The report workflow starts with `generate_markdown_report()`, which produces a structured Markdown report string.

The app then supports three export paths:

- Markdown: downloaded directly as `.md`
- Word: converted with `python-docx`
- PDF: converted with ReportLab

The Word and PDF exporters use simple parsing rules for headings, bullets, normal text, and Markdown tables. The goal is a readable report format that remains stable on Streamlit Cloud.

## LLM Insight Workflow

AI insights are optional. If no API key is configured, the app continues to work normally.

When AI insights are enabled:

1. The app creates a compact dataset summary.
2. The summary includes dataset shape, column information, numerical summaries, categorical value counts, data quality notes, and machine learning suggestions.
3. The app tries OpenAI first when available.
4. If OpenAI fails and Gemini is configured, the app tries Gemini as a fallback.
5. The generated insights can be displayed in the app and included in the report.

The full raw dataset is not sent to the LLM. Only compact summary information is used.

## Current Limitations

- Models are baseline models, not final production models.
- Hyperparameter tuning is not yet included.
- SHAP or other model explainability methods are not yet included.
- PDF formatting is intentionally simple.
- The app does not persist analysis history between sessions.
- Very large datasets may require additional performance optimization.

## Future Improvements

- Add advanced model tuning.
- Add SHAP-based model explanation.
- Improve PDF formatting with richer tables and visual summaries.
- Add more example datasets from different domains.
- Add optional saved analysis history.
- Add stronger structured validation for AI-generated insights.
