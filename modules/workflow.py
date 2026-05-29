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
            "tool": "Report Generator",
            "description": "Combine rule-based analysis and optional AI insights into downloadable Markdown and Word reports.",
            "output": "Final data analysis reports."
        }
    ]
