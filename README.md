# Data Analysis Report Agent

A Streamlit-based data analysis assistant that automatically analyzes uploaded CSV datasets, detects data quality issues, generates visualizations, suggests suitable machine learning tasks, and creates a downloadable Markdown report.

## Project Overview

Data Analysis Report Agent is designed to help users quickly understand a dataset without manually writing repeated exploratory data analysis code.
Users can upload a CSV file, and the application will automatically generate dataset summaries, missing value checks, duplicate row checks, descriptive statistics, visualizations, and machine learning task suggestions.

This project demonstrates practical skills in Python, data analysis automation, Streamlit web development, and basic AI-agent-style workflow design.

## Demo Screenshots

### Dataset Overview

![Dataset Overview](assets/screenshots/app_overview.png)

### Machine Learning Task Suggestion

![Machine Learning Task Suggestion](assets/screenshots/ml_task_suggestion.png)

### Generated Report

![Generated Report](assets/screenshots/generated_report.png)

## Features

* Upload CSV files through a web interface
* Preview dataset records
* Display dataset shape and column information
* Check missing values and duplicate rows
* Automatically detect numerical, categorical, and ID-like columns
* Exclude ID-like columns from unsuitable visualizations
* Treat binary numerical columns as categorical features
* Generate numerical column visualizations
* Generate categorical column visualizations
* Suggest suitable machine learning task types:

  * Regression
  * Binary classification
  * Multi-class classification
* Recommend suitable machine learning models and evaluation metrics
* Generate and download a Markdown data analysis report

## Tech Stack

* Python
* Streamlit
* pandas
* matplotlib
* tabulate

## Project Structure

```text
data-analysis-report-agent/
├── app.py
├── requirements.txt
├── CHANGELOG.md
├── README.md
├── .gitignore
└── venv/
```

## How to Run the Project

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd data-analysis-report-agent
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

## Example Use Case

For a customer churn dataset, the agent can automatically detect that the target column `Churn` is suitable for a binary classification task.

It then suggests possible models such as:

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost Classifier

It also recommends suitable evaluation metrics such as:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

## Current Version

### V1.2 - Machine Learning Task Suggestion

The current version supports automatic dataset analysis, smart column detection, visualizations, Markdown report generation, and machine learning task suggestion.

## Future Improvements

* Add automatic data cleaning suggestions
* Add correlation analysis
* Add model training for baseline classification and regression models
* Add PDF or Word report export
* Add LLM-generated natural language insights
* Add an agent workflow page to explain the reasoning process

## Author

Chen Hongyu
Master of Artificial Intelligence
Universiti Kebangsaan Malaysia
