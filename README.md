# emp-salary-prediction

# Income Classification on Adult Census Dataset

This project builds, evaluates, and compares several machine learning models to predict income levels using the Adult Census dataset. The workflow includes data cleaning, preprocessing, model training, evaluation, hyperparameter tuning, feature importance analysis, and model saving.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [How to Run](#how-to-run)
- [Code Structure](#code-structure)
- [Model Evaluation & Results](#model-evaluation--results)
- [Feature Importance](#feature-importance)
- [Saving the Model](#saving-the-model)
- [Notes](#notes)

---

## Overview

The script performs the following steps:

1. *Loads the Adult Census dataset* (CSV format).
2. *Cleans and preprocesses* the data (handles missing values, encodes categorical variables, scales numerical features).
3. *Splits* the data into training and testing sets.
4. *Builds pipelines* for three classifiers:
   - Logistic Regression
   - Random Forest
   - XGBoost
5. *Evaluates* each model using cross-validation and test set metrics.
6. *Compares* model performance visually.
7. *Tunes hyperparameters* for the best model.
8. *Analyzes feature importance* for tree-based models.
9. *Saves* the best model to disk.

---

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- joblib

Install dependencies with:

bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib


---

## Dataset

- The script expects the dataset at:  
  C:\Users\karth\Downloads\adult 3.csv
- The dataset should be in CSV format, similar to the UCI Adult dataset.

---

## How to Run

1. *Place the dataset* at the specified path or update the path in the script.
2. *Run the script*:

   bash
   python your_script_name.py
   

3. *Outputs*:
   - Console logs for each step
   - Model performance plots
   - The best model saved as a .joblib file in the working directory

---

## Code Structure

### 1. Data Loading

- Loads the CSV file into a pandas DataFrame.
- Prints the first few rows and dataset info.

### 2. Data Cleaning & Preprocessing

- Strips whitespace from column names and string values.
- Replaces '?' with NaN.
- Imputes missing values (mode for categorical, median for numerical).

### 3. Feature/Target Split & Train-Test Split

- Separates features (X) and target (y).
- Encodes the target variable.
- Splits data into training and testing sets (80/20 split, stratified).

### 4. Preprocessing Pipeline

- Uses ColumnTransformer to:
  - Scale numerical features
  - One-hot encode categorical features

### 5. Model Pipelines & Evaluation

- Defines pipelines for:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Trains and evaluates each model using cross-validation and test set metrics:
  - Accuracy, Precision, Recall, F1-Score, ROC AUC, Confusion Matrix

### 6. Model Comparison

- Summarizes and plots model performance for easy comparison.

### 7. Hyperparameter Tuning

- Performs grid search for the best model (Random Forest or XGBoost).
- Prints best parameters and improved metrics.

### 8. Feature Importance

- Displays and plots the top features for tree-based models.

### 9. Model Saving

- Saves the best-performing model as a .joblib file.

---

## Model Evaluation & Results

- The script prints and plots:
  - Accuracy, F1-Score, ROC AUC for each model
  - Cross-validation accuracy with error bars
- The best model is selected based on test set accuracy.

---

## Feature Importance

- For Random Forest and XGBoost, the script displays the top 10 most important features.
- For Logistic Regression, coefficients can be interpreted as feature importance.

---

## Saving the Model

- The best model is saved as:
  
  <model_name>_income_classifier_model.joblib
  
  (e.g., random_forest_classifier_income_classifier_model.joblib)

---

## Notes

- *Warnings*: The script suppresses some sklearn and pandas warnings for cleaner output.
- *Custom Path*: Update the dataset path in the script if your file is elsewhere.
- *Extensibility*: You can easily add more models or tune additional hyperparameters.

---

## License

This project is for educational and demonstration purposes.

---
