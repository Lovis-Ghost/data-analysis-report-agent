import io

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def train_baseline_model(df, selected_target, task_info, numeric_cols, categorical_cols, id_cols):
    df_model = df.copy()
    df_model = df_model.dropna(subset=[selected_target])

    valid_numeric_features = [
        col for col in numeric_cols
        if col in df_model.columns and col != selected_target and col not in id_cols
    ]
    valid_categorical_features = [
        col for col in categorical_cols
        if col in df_model.columns and col != selected_target and col not in id_cols
    ]
    feature_cols = valid_numeric_features + valid_categorical_features

    if len(df_model) < 5:
        raise ValueError("At least 5 rows with a non-missing target are needed to train a baseline model.")

    if len(feature_cols) == 0:
        raise ValueError("No valid numerical or categorical feature columns are available for model training.")

    X = df_model[feature_cols]
    y = df_model[selected_target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, valid_numeric_features),
        ("cat", categorical_transformer, valid_categorical_features)
    ])

    task_type = task_info["task_type"]
    safe_target_name = str(selected_target).replace(" ", "_")

    if task_type in ["Binary Classification", "Multi-class Classification"]:
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            raise ValueError(
                "The train/test split needs at least two target classes in both sets. "
                "Try using a larger or more balanced dataset."
            )

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest Classifier": RandomForestClassifier(random_state=42)
        }
        comparison_rows = []
        confusion_matrices = {}
        trained_pipelines = {}

        for model_name, model in models.items():
            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ])
            pipeline.fit(X_train, y_train)
            trained_pipelines[model_name] = pipeline
            predictions = pipeline.predict(X_test)

            comparison_rows.append({
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, predictions),
                "Precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
                "Recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
                "F1-score": f1_score(y_test, predictions, average="weighted", zero_division=0)
            })

            labels = sorted(y.unique(), key=lambda value: str(value))
            confusion_matrices[model_name] = {
                "labels": labels,
                "matrix": confusion_matrix(y_test, predictions, labels=labels).tolist()
            }

        comparison_df = pd.DataFrame(comparison_rows)
        best_model_name = comparison_df.sort_values("F1-score", ascending=False).iloc[0]["Model"]
        safe_model_name = best_model_name.replace(" ", "_")
        interpretation = (
            f"{best_model_name} performed best because it had the highest weighted F1-score "
            "among the baseline classification models."
        )

        return {
            "task_type": task_type,
            "target_column": selected_target,
            "comparison_table": comparison_df,
            "best_model_name": best_model_name,
            "best_pipeline": trained_pipelines[best_model_name],
            "best_metric_name": "Weighted F1-score",
            "interpretation": interpretation,
            "confusion_matrix": confusion_matrices[best_model_name],
            "feature_columns": feature_cols,
            "numeric_features": valid_numeric_features,
            "categorical_features": valid_categorical_features,
            "model_file_name": f"baseline_model_{safe_target_name}_{safe_model_name}.pkl"
        }

    if task_type == "Regression":
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(random_state=42)
        }
        comparison_rows = []
        trained_pipelines = {}

        for model_name, model in models.items():
            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ])
            pipeline.fit(X_train, y_train)
            trained_pipelines[model_name] = pipeline
            predictions = pipeline.predict(X_test)
            mse = mean_squared_error(y_test, predictions)

            comparison_rows.append({
                "Model": model_name,
                "MAE": mean_absolute_error(y_test, predictions),
                "MSE": mse,
                "RMSE": np.sqrt(mse),
                "R2 Score": r2_score(y_test, predictions)
            })

        comparison_df = pd.DataFrame(comparison_rows)
        best_model_name = comparison_df.sort_values("R2 Score", ascending=False).iloc[0]["Model"]
        safe_model_name = best_model_name.replace(" ", "_")
        interpretation = (
            f"{best_model_name} performed best because it had the highest R2 score "
            "among the baseline regression models."
        )

        return {
            "task_type": task_type,
            "target_column": selected_target,
            "comparison_table": comparison_df,
            "best_model_name": best_model_name,
            "best_pipeline": trained_pipelines[best_model_name],
            "best_metric_name": "R2 Score",
            "interpretation": interpretation,
            "feature_columns": feature_cols,
            "numeric_features": valid_numeric_features,
            "categorical_features": valid_categorical_features,
            "model_file_name": f"baseline_model_{safe_target_name}_{safe_model_name}.pkl"
        }

    raise ValueError(f"Unsupported task type for baseline training: {task_type}")


def create_model_download_bytes(model_results):
    buffer = io.BytesIO()
    model_package = {
        "model": model_results["best_pipeline"],
        "task_type": model_results["task_type"],
        "target_column": model_results["target_column"],
        "best_model_name": model_results["best_model_name"],
        "feature_columns": model_results["feature_columns"],
        "numeric_features": model_results["numeric_features"],
        "categorical_features": model_results["categorical_features"],
        "best_metric_name": model_results["best_metric_name"],
        "interpretation": model_results["interpretation"]
    }
    joblib.dump(model_package, buffer)
    buffer.seek(0)
    return buffer.getvalue()
