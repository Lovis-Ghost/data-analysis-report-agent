# Changelog

## V1.8 - Baseline Model Training

### Added
- Added baseline ML model training.
- Added classification and regression model comparison.
- Added evaluation metrics.
- Added confusion matrix for classification.
- Added model training results to Markdown report.

## V1.7 - CSV and Excel File Support

### Added
- Added support for Excel `.xlsx` file uploads.
- Added Excel sheet selection before analysis.
- Added dataset file type display in the Streamlit app.
- Added `openpyxl` dependency for Excel reading.

### Changed
- Updated the dataset reader workflow from CSV-only to CSV/Excel support.

## V1.6 - Agent Workflow Explanation

### Added
- Added an Agent Workflow section to the Streamlit app.
- Added tool-style explanation of each analysis step.
- Added README documentation for the agent-style workflow.
- Improved project presentation for AI agent internship portfolio use.

## V1.5 - Multi-provider LLM Fallback

### Added
- Added Gemini API support using GEMINI_API_KEY.
- Added multi-provider LLM fallback from OpenAI to Gemini.
- Added provider status display in the Streamlit UI.
- Added safer AI insight generation when one provider has quota or billing issues.

### Changed
- AI Insight Generator now supports multiple providers instead of only OpenAI.

## V1.4 - AI Insight Generator

### Added
- Added optional OpenAI API support through the `OPENAI_API_KEY` environment variable.
- Added `generate_ai_insights(df, column_info, data_quality_summary, ml_task_info)`.
- Added an "AI Generated Insights" section with a "Generate AI Insights" button.
- Added AI-generated dataset overview, data quality insights, patterns, machine learning direction, and next steps.
- Added AI insights to the downloadable Markdown report when generated.
- Added `openai` and `python-dotenv` dependencies.

### Changed
- The AI feature sends compact dataset summaries only, not the full dataset.
- The app continues to work normally when `OPENAI_API_KEY` is not configured.

## V1.3 - Smart Insights and Enhanced Report

### Added
- Added a data quality score based on missing values, duplicate rows, and ID-like columns.
- Added automatic data quality suggestions for beginner-friendly cleaning guidance.
- Added correlation matrix analysis and a simple heatmap for suitable numerical columns.
- Added top 3 strongest correlation summary.
- Improved the downloaded Markdown report with data quality, machine learning, and correlation insights.

### Fixed
- Fixed the target-column edge case so the app does not use `task_info` when no suitable target column exists.

## V1.2 - Machine Learning Task Suggestion

### Added
- Added target column selection.
- Added automatic machine learning task suggestion.
- Supported regression, binary classification, and multi-class classification detection.
- Added suggested models based on the selected target column.
- Added suggested evaluation metrics for different task types.

## V1.1 - Smart Column Detection

### Added
- Added `prepare_data()` function to convert numeric-like text columns into numeric values.
- Added `detect_column_types()` function.
- Automatically detects and excludes ID-like columns such as `customerID`.
- Treats binary numerical columns such as `SeniorCitizen` as categorical features.
- Improved numerical and categorical visualization column selection.

### Fixed
- Fixed Markdown report generation error by adding the `tabulate` dependency.

## V1.0 - Basic Version

### Added
- Created Streamlit web application.
- Added CSV file upload.
- Added dataset preview.
- Added dataset shape summary.
- Added column information table.
- Added duplicate row checking.
- Added descriptive statistics.
- Added numerical column visualization.
- Added categorical column visualization.
- Added Markdown report generation and download.
