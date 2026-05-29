import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from modules.ai_insights import (
    generate_ai_insights,
    get_gemini_api_key,
    get_openai_api_key,
)
from modules.column_detection import detect_column_types
from modules.correlation import analyze_correlations
from modules.data_loader import load_dataset, prepare_data
from modules.data_quality import calculate_data_quality_score
from modules.ml_task import suggest_ml_task
from modules.ml_trainer import create_model_download_bytes, train_baseline_model
from modules.prediction import create_prediction_input, make_single_prediction
from modules.report_export import create_docx_report_bytes, create_pdf_report_bytes
from modules.report_generator import generate_markdown_report
from modules.workflow import get_agent_workflow_steps


st.set_page_config(
    page_title="AI Data Analysis & ML Agent",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Data Analysis & ML Agent")

st.write(
    "Upload a CSV or Excel dataset to analyze data quality, generate reports, "
    "train baseline machine learning models, download the best model, and try "
    "a simple prediction demo."
)

st.subheader("Agent Workflow")
st.write(
    "This project follows an agent-style workflow. Each step acts like a tool "
    "that processes the dataset and passes useful information to the next step."
)
st.markdown(
    "CSV/Excel Upload → Data Profiling → Quality Check → ML Suggestion → "
    "Baseline Training → LLM Insights → Report"
)

for step_info in get_agent_workflow_steps():
    with st.expander(step_info["step"]):
        st.write(f"**Tool:** {step_info['tool']}")
        st.write(f"**Description:** {step_info['description']}")
        st.write(f"**Output:** {step_info['output']}")

uploaded_file = st.file_uploader(
    "Upload your dataset file",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:
    try:
        df, file_type, sheet_name = load_dataset(uploaded_file)
        df = prepare_data(df)

        file_signature = f"{uploaded_file.name}-{df.shape}-{list(df.columns)}"
        if st.session_state.get("ai_file_signature") != file_signature:
            st.session_state["ai_file_signature"] = file_signature
            st.session_state["ai_insights"] = None
            st.session_state["model_results"] = None

        st.success(f"{file_type} file uploaded successfully!")

        if sheet_name:
            st.info(f"Selected Excel sheet: {sheet_name}")

        st.subheader("1. Dataset Preview")
        st.dataframe(df.head())

        st.subheader("2. Dataset Shape")
        st.write(f"Rows: {df.shape[0]}")
        st.write(f"Columns: {df.shape[1]}")

        st.subheader("3. Column Information")
        column_info = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str),
            "Missing Values": df.isnull().sum().values,
            "Unique Values": df.nunique().values
        })
        st.dataframe(column_info)

        numeric_cols, categorical_cols, id_cols = detect_column_types(df)

        st.subheader("4. Duplicate Rows")
        duplicate_count = df.duplicated().sum()
        st.write(f"Number of duplicate rows: {duplicate_count}")

        st.subheader("5. Data Quality Assessment")
        data_quality_score, data_quality_suggestions = calculate_data_quality_score(df)
        st.metric("Data Quality Score", f"{data_quality_score}/100")

        for suggestion in data_quality_suggestions:
            if data_quality_score >= 80:
                st.info(suggestion)
            else:
                st.warning(suggestion)

        st.subheader("6. Descriptive Statistics")

        if len(numeric_cols) == 0:
            st.info("No suitable numerical columns found.")
        else:
            st.dataframe(df[numeric_cols].describe())

        st.subheader("7. Machine Learning Task Suggestion")

        target_options = [
            col for col in df.columns
            if col not in id_cols
        ]
        selected_target = None
        task_info = None

        if len(target_options) == 0:
            st.info("No suitable target column found.")
        else:
            default_index = 0

            if "Churn" in target_options:
                default_index = target_options.index("Churn")

            selected_target = st.selectbox(
                "Choose a target column",
                target_options,
                index=default_index
            )

            task_info = suggest_ml_task(
                df,
                selected_target,
                numeric_cols,
                categorical_cols,
                id_cols
            )

            st.success(f"Suggested Machine Learning Task: {task_info['task_type']}")

            st.write("**Reason:**")
            st.write(task_info["reason"])

            st.write("**Target Column Summary:**")
            st.write(f"- Target column: {selected_target}")
            st.write(f"- Unique values: {task_info['target_unique']}")
            st.write(f"- Missing values: {task_info['target_missing']}")

            st.write("**Available Numerical Features:**")
            st.write(task_info["numeric_features"])

            st.write("**Available Categorical Features:**")
            st.write(task_info["categorical_features"])

            st.write("**Suggested Models:**")
            for model in task_info["suggested_models"]:
                st.write(f"- {model}")

            st.write("**Suggested Evaluation Metrics:**")
            for metric in task_info["suggested_metrics"]:
                st.write(f"- {metric}")

        st.subheader("8. Correlation Analysis")
        correlation_summary = analyze_correlations(df, numeric_cols)

        if correlation_summary is None:
            st.info("Correlation analysis requires at least two suitable numerical columns.")
        else:
            corr_matrix = correlation_summary["corr_matrix"]
            st.write("**Correlation Matrix:**")
            st.dataframe(corr_matrix)

            fig, ax = plt.subplots()
            heatmap = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr_matrix.columns)))
            ax.set_yticks(range(len(corr_matrix.columns)))
            ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
            ax.set_yticklabels(corr_matrix.columns)
            ax.set_title("Correlation Heatmap")
            fig.colorbar(heatmap, ax=ax)
            fig.tight_layout()
            st.pyplot(fig)

            st.write("**Top 3 Strongest Correlations:**")
            if len(correlation_summary["top_correlations"]) == 0:
                st.info("No valid pairwise correlations were found.")
            else:
                for item in correlation_summary["top_correlations"]:
                    st.write(
                        f"- {item['column_a']} and {item['column_b']}: "
                        f"{item['correlation']:.3f}"
                    )

        st.subheader("9. Numerical Column Visualization")

        if len(numeric_cols) == 0:
            st.info("No numerical columns found.")
        else:
            selected_num_col = st.selectbox(
                "Choose a numerical column",
                numeric_cols
            )

            fig, ax = plt.subplots()
            ax.hist(df[selected_num_col].dropna(), bins=20)
            ax.set_title(f"Distribution of {selected_num_col}")
            ax.set_xlabel(selected_num_col)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

        st.subheader("10. Categorical Column Visualization")

        if len(categorical_cols) == 0:
            st.info("No categorical columns found.")
        else:
            selected_cat_col = st.selectbox(
                "Choose a categorical column",
                categorical_cols
            )

            value_counts = df[selected_cat_col].value_counts().head(10)

            fig, ax = plt.subplots()
            ax.bar(value_counts.index.astype(str), value_counts.values)
            ax.set_title(f"Top Categories in {selected_cat_col}")
            ax.set_xlabel(selected_cat_col)
            ax.set_ylabel("Count")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)

        st.subheader("11. AI Generated Insights")
        openai_key = get_openai_api_key()
        gemini_key = get_gemini_api_key()
        api_key_available = bool(openai_key or gemini_key)

        data_quality_summary = {
            "score": data_quality_score,
            "suggestions": data_quality_suggestions,
            "duplicate_rows": int(duplicate_count),
            "missing_values_by_column": {
                col: int(value)
                for col, value in df.isnull().sum().items()
                if value > 0
            }
        }

        ml_task_summary = None
        if task_info is not None and selected_target is not None:
            ml_task_summary = {
                "selected_target": selected_target,
                "task_type": task_info["task_type"],
                "reason": task_info["reason"],
                "suggested_models": task_info["suggested_models"],
                "suggested_metrics": task_info["suggested_metrics"]
            }

        if not api_key_available:
            st.info(
                "AI insights are optional. Add OPENAI_API_KEY or GEMINI_API_KEY "
                "to your environment to enable the AI Insight Generator."
            )

        st.info(
            f"OpenAI configured: {'Yes' if openai_key else 'No'}\n\n"
            f"Gemini configured: {'Yes' if gemini_key else 'No'}"
        )

        if st.button("Generate AI Insights", disabled=not api_key_available):
            with st.spinner("Generating AI insights from the dataset summary..."):
                ai_insights, ai_error, provider_name = generate_ai_insights(
                    df,
                    column_info,
                    data_quality_summary,
                    ml_task_summary
                )

            if ai_error:
                st.warning(ai_error)
            else:
                st.session_state["ai_insights"] = ai_insights
                st.success(f"AI insights generated successfully using {provider_name}.")

        if st.session_state.get("ai_insights"):
            st.markdown(st.session_state["ai_insights"])

        st.subheader("12. Baseline Model Training")
        st.write(
            "This trains simple baseline models so you can compare a first modeling result. "
            "It is not a final production model."
        )

        can_train_model = task_info is not None and selected_target is not None

        if not can_train_model:
            st.info("Choose a suitable target column before training a baseline model.")

        if st.button("Train Baseline Model", disabled=not can_train_model):
            with st.spinner("Training baseline models..."):
                try:
                    model_results = train_baseline_model(
                        df,
                        selected_target,
                        task_info,
                        numeric_cols,
                        categorical_cols,
                        id_cols
                    )
                    st.session_state["model_results"] = model_results
                    st.success("Baseline model training completed.")
                except Exception as training_error:
                    st.session_state["model_results"] = None
                    st.warning(f"Baseline model training could not be completed: {training_error}")

        model_results = st.session_state.get("model_results")
        current_model_results = None

        if model_results and model_results["target_column"] == selected_target:
            current_model_results = model_results
            st.write("**Model Comparison:**")
            st.dataframe(model_results["comparison_table"])

            st.write(f"**Best model:** {model_results['best_model_name']}")
            st.write(model_results["interpretation"])

            if model_results["task_type"] in ["Binary Classification", "Multi-class Classification"]:
                confusion_info = model_results["confusion_matrix"]
                confusion_df = pd.DataFrame(
                    confusion_info["matrix"],
                    index=confusion_info["labels"],
                    columns=confusion_info["labels"]
                )
                st.write("**Confusion Matrix for Best Model:**")
                st.dataframe(confusion_df)
            elif model_results["task_type"] == "Regression":
                st.info(
                    f"The best regression baseline was selected using "
                    f"{model_results['best_metric_name']}."
                )

            if current_model_results and "best_pipeline" in current_model_results:
                st.write(
                    "The downloaded file contains the best baseline sklearn pipeline, "
                    "including preprocessing steps and the trained model."
                )
                st.download_button(
                    label="Download Best Baseline Model (.pkl)",
                    data=create_model_download_bytes(current_model_results),
                    file_name=current_model_results["model_file_name"],
                    mime="application/octet-stream"
                )
        elif model_results:
            st.info("Stored model results are for a different target column. Train again to update them.")

        st.subheader("13. Prediction Demo")

        if current_model_results and "best_pipeline" in current_model_results:
            st.write(
                "This is a simple demo using the best baseline model trained above."
            )

            prediction_input_df = create_prediction_input(df, current_model_results)

            if st.button("Predict with Best Baseline Model"):
                try:
                    prediction, probabilities = make_single_prediction(
                        current_model_results,
                        prediction_input_df
                    )
                    st.success(f"Predicted value: {prediction}")

                    if probabilities is not None:
                        st.write("**Class Probabilities:**")
                        st.dataframe(probabilities)

                    st.info(
                        "This is a baseline prediction and should not be used as a "
                        "final production decision."
                    )
                except Exception as prediction_error:
                    st.warning(f"Prediction could not be completed: {prediction_error}")
        else:
            st.info("Train a baseline model first to use the prediction demo.")

        st.subheader("14. Generated Report")

        report = generate_markdown_report(
            df,
            data_quality_score=data_quality_score,
            data_quality_suggestions=data_quality_suggestions,
            selected_target=selected_target,
            task_info=task_info,
            correlation_summary=correlation_summary,
            ai_insights=st.session_state.get("ai_insights"),
            model_results=current_model_results
        )
        st.markdown(report)

        st.download_button(
            label="Download Report as Markdown",
            data=report,
            file_name="data_analysis_report.md",
            mime="text/markdown"
        )

        st.download_button(
            label="Download Report as Word (.docx)",
            data=create_docx_report_bytes(report),
            file_name="ai_data_analysis_ml_report.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

        st.download_button(
            label="Download Report as PDF",
            data=create_pdf_report_bytes(report),
            file_name="ai_data_analysis_ml_report.pdf",
            mime="application/pdf"
        )

    except Exception as e:
        st.error("Something went wrong while reading the file.")
        st.write(e)
else:
    st.info("Please upload a CSV or Excel file to start the analysis.")
