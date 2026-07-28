# MediRisk - Heart Disease Prediction Using Machine Learning

### Overview
MediRisk is an educational Data Science project designed to demonstrate the complete machine learning workflow. It leverages the UCI Cleveland Heart Disease dataset to predict the probability of the positive heart disease class. The project includes data cleaning, exploratory data analysis, preprocessing, training three separate ML algorithms, and deploying the best-performing model via a Streamlit web application.

### Problem Statement
Clinical decision-making can be supported by statistical models. This project aims to determine the statistical probability of heart disease presence using historical health measurements.

### Dataset
**Source:** UCI Machine Learning Repository (Cleveland Dataset).
**Target:** Binary classification. `0` represents the Negative class (no disease). Values > 0 are mapped to `1`, representing the Positive class.

### Tech Stack
* Python, Pandas, NumPy, Scikit-learn, Plotly, Streamlit

### ML Workflow
Data ➔ Cleaning ➔ EDA ➔ Train/Test Split (Stratified) ➔ Preprocessing ➔ Train Models ➔ Evaluate ➔ Select Model ➔ Streamlit Prediction

### Installation
1. `python -m venv venv`
2. Activate environment: `venv\Scripts\activate` (Windows)
3. `pip install -r requirements.txt`
4. `python train_model.py`
5. `streamlit run app.py`

### Limitations & Disclaimer
**Disclaimer:** Educational machine-learning project only. Predictions from this application are not medical diagnoses.
**Limitations:** Small/public dataset, potential population bias, limited variables, no clinical validation.
