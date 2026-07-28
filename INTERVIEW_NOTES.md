# Interview Preparation Notes

**Explain your MediRisk project.**
MediRisk is an end-to-end Machine Learning web app built with Python, Scikit-learn, and Streamlit that predicts the likelihood of heart disease using clinical data.

**What dataset did you use?**
The UCI Cleveland Heart Disease dataset, containing 14 attributes.

**What preprocessing did you perform?**
I used Scikit-learn ColumnTransformer. Numerical features were imputed with the median and scaled using StandardScaler. Categorical features were imputed with the mode and encoded using OneHotEncoder.

**What is data leakage?**
Data leakage occurs when information from the test dataset is accidentally used to train the model. I avoided this by using Scikit-learn Pipelines.

**Why use train/test split?**
To evaluate how well the model generalizes to unseen data, preventing overfitting.

**Why use stratify?**
To ensure the ratio of positive to negative classes is identical in both the training and testing sets.

**Why did you select your final model?**
Random Forest typically provided the best F1 Score and Recall, which is crucial for medical-context datasets where False Negatives are risky.
