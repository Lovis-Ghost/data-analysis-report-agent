# Changelog

## V2.6A - Public Portfolio Polish

### Added
- Added public technical design documentation.
- Added a docs folder for public project documentation.

### Changed
- Updated README with a technical documentation section.
- Kept private resume and interview preparation materials outside the public repository.

## V2.5 - PDF Report Export

### Added
- Added PDF report export support.
- Added a PDF report download button in the Streamlit app.
- Added ReportLab as a project dependency.
- Extended the report_export module to support Markdown, Word, and PDF report downloads.

### Changed
- Updated README to describe Markdown, Word, and PDF report export support.

## V2.4B - Final Screenshots Added

### Added
- Added final Streamlit UI screenshots to README.
- Added screenshots for workflow, dataset preview, data quality, model training, prediction demo, and report download.
- Updated assets/screenshots documentation.

### Changed
- Replaced the planned screenshots checklist with real screenshot references.
- Improved README visual presentation for portfolio review.

## V2.4A - Architecture Diagram and Screenshot Preparation

### Added
- Added a Mermaid system architecture diagram to README.
- Added a planned screenshots checklist to README.
- Added screenshot naming guidance under assets/screenshots/README.md.

### Changed
- Updated README project presentation for portfolio review.

## V2.3 - Example Dataset and Demo Outputs

### Added
- Added an examples folder for portfolio demonstration.
- Added a synthetic customer churn sample dataset.
- Added an example Markdown report.
- Added an example prediction result document.
- Added an example usage guide for testing the app workflow.

### Changed
- Updated README to include example files and demo testing instructions.

## V2.2 - Word Report Export

### Added
- Added Word `.docx` report export support.
- Added a report_export module for converting generated Markdown reports into Word documents.
- Added a Word report download button in the Streamlit app.
- Added python-docx as a project dependency.

### Changed
- Updated README to describe Markdown and Word report export support.

## V2.1 - Modular Project Structure

### Changed
- Refactored reusable logic from app.py into separate modules.
- Added a modules package for data loading, column detection, data quality analysis, correlation analysis, AI insights, ML task suggestion, model training, prediction, workflow description, and report generation.
- Simplified app.py so it mainly controls the Streamlit UI flow.
- Updated README project structure for the modular layout.

## V2.0.2 - README Portfolio Polish

### Changed
- Rewrote the README for internship portfolio presentation.
- Added an end-to-end workflow summary.
- Added clearer feature groups for data analysis, AI insights, machine learning, model download, prediction demo, and report generation.
- Added portfolio highlights and a resume-ready project summary.

## V2.0.1 - Project Rename and Branding Update

### Changed
- Renamed the project branding from Data Analysis Report Agent to AI Data Analysis & ML Agent.
- Updated the Streamlit app title, README title, and Markdown report title.
- Updated project descriptions to better reflect the V2.0 machine learning and prediction demo features.

## V2.0 - Prediction Demo

### Added
- Added a prediction demo interface for the best trained baseline model.
- Added dynamic input fields based on numerical and categorical feature columns.
- Added single-sample prediction using the trained sklearn Pipeline.
- Added class probability display when supported by the model.
- Added prediction demo availability information to the Markdown report.

## V1.9 - Model Download

### Added
- Added download support for the best trained baseline model.
- Saved preprocessing steps and trained model together as an sklearn Pipeline.
- Added model metadata including task type, target column, feature columns, and best model name.
- Added model download availability information to the Markdown report.

## V1.8.1 - Documentation and Workflow Polish

### Changed
- Updated app text from CSV-only to CSV/Excel wording.
- Added Baseline Model Trainer to the agent workflow description.
- Updated README features and tech stack to reflect model training support.

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
- Improved the downloaded report with data quality, machine learning, and correlation insights.

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
