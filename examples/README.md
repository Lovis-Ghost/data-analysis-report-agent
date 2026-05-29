# Example Files

This folder contains small demo files for testing the AI Data Analysis & ML Agent workflow.

## Files

- `sample_churn.csv`: A synthetic customer churn dataset with realistic but fake customer service and billing fields.
- `example_markdown_report.md`: A short example of what a generated Markdown report may look like.
- `example_prediction_result.md`: A sample prediction demo output for a customer churn use case.

## How To Test The App

1. Open the Streamlit app.
2. Upload `examples/sample_churn.csv`.
3. Select `Churn` as the target column.
4. Train the baseline model.
5. Download the Markdown report.
6. Download the Word report.
7. Try the prediction demo by changing one or more input values.

Outputs may differ slightly because baseline models depend on the train/test split and the selected sample values.
