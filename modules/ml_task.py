import pandas as pd


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
