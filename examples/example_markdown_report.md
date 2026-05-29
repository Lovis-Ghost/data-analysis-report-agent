# AI Data Analysis & ML Report

## Dataset Overview

- Dataset used: `examples/sample_churn.csv`
- Number of rows: 40
- Number of columns: 12
- Target column: `Churn`
- The dataset contains customer account, service, billing, and churn status fields.

## Missing Value Summary

No missing values were found in the example dataset.

## Data Quality Assessment

- Example data quality score: 96/100
- The dataset is small but clean enough for a baseline demo.
- `customerID` is detected as an ID-like column and should be excluded from model features.
- `TotalCharges`, `MonthlyCharges`, and `tenure` provide useful numerical signals for exploration.

## Machine Learning Task Suggestion

- Selected target column: `Churn`
- Suggested task type: Binary Classification
- Reason: The target column contains two classes: `Yes` and `No`.
- Suggested metrics: Accuracy, Precision, Recall, F1-score, and Confusion Matrix

## Model Training Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.88 | 0.89 | 0.88 | 0.87 |
| Random Forest Classifier | 0.75 | 0.78 | 0.75 | 0.74 |

- Example best baseline model: Logistic Regression
- Model download available: Yes
- Prediction demo available: Yes

Interpretation: Logistic Regression performed better in this example because it had the highest weighted F1-score on the test split.

## Prediction Demo Example

Example input:

- tenure: 24
- InternetService: Fiber optic
- Contract: One year
- MonthlyCharges: 72.50
- TotalCharges: 1740.00

Example output:

- Predicted value: No
- Probability for No: 0.79
- Probability for Yes: 0.21

This is a baseline prediction and should not be used as a final production decision.

## Suggested Next Steps

- Review class balance before using the model for real decisions.
- Try more feature engineering for contract type, service mix, and billing behavior.
- Compare performance on a larger dataset.
- Consider advanced tuning and explainability methods in future versions.
