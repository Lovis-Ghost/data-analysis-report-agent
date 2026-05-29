import pandas as pd
import streamlit as st


def load_dataset(uploaded_file):
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        return df, "CSV", None

    if file_name.endswith(".xlsx"):
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_name = st.selectbox(
            "Choose an Excel sheet",
            excel_file.sheet_names
        )
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        return df, "Excel", sheet_name

    raise ValueError("Unsupported file format. Please upload a CSV or XLSX file.")


def prepare_data(df):
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":
            converted_col = pd.to_numeric(df[col], errors="coerce")
            valid_ratio = converted_col.notna().sum() / len(df)

            if valid_ratio > 0.8:
                df[col] = converted_col

    return df
