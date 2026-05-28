# Changelog

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
