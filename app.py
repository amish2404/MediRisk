import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import json
import os

# 1. Page Config (Must be first)
st.set_page_config(page_title="MediRisk AI", layout="wide", page_icon="🧬", initial_sidebar_state="expanded")

# 2. Inject Custom CSS for that sleek, modern UI
custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Neon Accent Colors for Headers */
    h1, h2, h3 {
        color: #BB86FC !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    
    /* Style the sidebar */
    [data-testid="stSidebar"] {
        background-color: #121212;
        border-right: 1px solid #333;
    }
    
    /* Sleek buttons */
    .stButton>button {
        background: linear-gradient(90deg, #BB86FC 0%, #3700B3 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(187, 134, 252, 0.4);
    }
    
    /* Custom Hero Title */
    .hero-title {
        font-size: 4rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #BB86FC, #03DAC6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        padding-bottom: 0;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        color: #A0A0A0;
        margin-top: -10px;
        margin-bottom: 2rem;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# 3. Load Assets
@st.cache_data
def load_data():
    if os.path.exists('data/heart.csv'):
        return pd.read_csv('data/heart.csv')
    return None

@st.cache_resource
def load_model():
    if os.path.exists('models/model.pkl'):
        return joblib.load('models/model.pkl')
    return None

df = load_data()
model = load_model()

# 4. Sidebar Navigation
st.sidebar.markdown("## 🧬 MediRisk AI")
page = st.sidebar.radio("Navigation", ["🏠 Home", "📊 Data Explorer", "📈 Model Performance", "🔮 Predict", "ℹ️ About"])

st.sidebar.markdown("---")
st.sidebar.caption("⚠️ **DISCLAIMER:** Educational ML project only. Not for clinical or diagnostic use.")

# ==========================================
# PAGE: HOME
# ==========================================
if page == "🏠 Home":
    st.markdown('<p class="hero-title"><h1>Medirisk.</h1></p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Predictive cardiovascular analytics powered by Machine Learning.</p>', unsafe_allow_html=True)
    
    if df is not None and model is not None:
        # Load metrics to show best model stats dynamically
        with open('models/metrics.json', 'r') as f:
            metrics = json.load(f)
        best = max(metrics, key=lambda x: x['F1 Score'])
        
        st.markdown("### ⚡ System Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Dataset Records", f"{df.shape[0]:,}", "UCI Cleveland")
        col2.metric("Features Analyzed", f"{df.shape[1] - 1}", "Clinical points")
        col3.metric("Selected Algorithm", best['Model'], "Optimized")
        col4.metric("Test ROC-AUC", f"{best['ROC-AUC'] * 100:.1f}%", "Confidence")
        
    st.markdown("---")
    st.markdown("### ⚙️ The ML Pipeline")
    st.info("Patient Data Input ➔ Scikit-learn Preprocessing Pipeline ➔ Trained Classifier ➔ Probability Score")

# ==========================================
# PAGE: DATA EXPLORER
# ==========================================
elif page == "📊 Data Explorer":
    st.markdown("## 📊 Dataset Insights")
    if df is not None:
        st.write("Explore the underlying data used to train the machine learning models.")
        with st.expander("🔎 View Raw Dataset", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.histogram(df, x='target', color='target', 
                                title="Target Class Balance", 
                                color_discrete_sequence=['#03DAC6', '#CF6679'],
                                template="plotly_dark")
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            fig2 = px.histogram(df, x='age', nbins=20, 
                                title="Patient Age Distribution",
                                color_discrete_sequence=['#BB86FC'],
                                template="plotly_dark")
            st.plotly_chart(fig2, use_container_width=True)
            
        st.markdown("### 🔗 Feature Correlation Map")
        fig3 = px.imshow(df.corr(), aspect="auto", 
                         color_continuous_scale='Purpor', 
                         template="plotly_dark")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.error("Data not found. Please run `train_model.py` first.")

# ==========================================
# PAGE: MODEL PERFORMANCE
# ==========================================
elif page == "📈 Model Performance":
    st.markdown("## 📈 Algorithm Evaluation")
    if os.path.exists('models/metrics.json'):
        with open('models/metrics.json', 'r') as f:
            metrics = json.load(f)
            
        metrics_df = pd.DataFrame(metrics)
        
        st.markdown("### Model Comparison Matrix")
        st.dataframe(metrics_df.style.highlight_max(axis=0, color='#3700B3'), use_container_width=True)
        
        metrics_melted = metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
        fig = px.bar(metrics_melted, x='Metric', y='Score', color='Model', barmode='group',
                     color_discrete_sequence=['#BB86FC', '#03DAC6', '#CF6679'],
                     template="plotly_dark", title="Cross-Metric Performance")
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.error("Metrics not found. Please run `train_model.py` first.")

# ==========================================
# PAGE: PREDICT
# ==========================================
elif page == "🔮 Predict":
    st.markdown("## 🔮 Run Inference")
    st.write("Enter patient vitals to generate a real-time statistical prediction using the trained model.")
    
    with st.form("predict_form"):
        st.markdown("#### Patient Vitals")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", 1, 120, 50)
            sex = st.selectbox("Sex", ["Female", "Male"])
            cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 50, 250, 120)
            chol = st.number_input("Serum Cholestoral (mg/dl)", 100, 600, 200)
            
        with col2:
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["False", "True"])
            restecg = st.selectbox("Resting ECG", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
            thalach = st.number_input("Max Heart Rate Achieved", 60, 250, 150)
            exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
            
        with col3:
            oldpeak = st.number_input("ST depression induced by exercise", 0.0, 10.0, 1.0)
            slope = st.selectbox("Slope of peak ST segment", ["Upsloping", "Flat", "Downsloping"])
            ca = st.selectbox("Number of major vessels (0-3)", [0.0, 1.0, 2.0, 3.0])
            thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversable Defect"])
            
        st.markdown("<br>", unsafe_allow_html=True)
        submit = st.form_submit_button("Generate Prediction", use_container_width=True)
        
    if submit:
        # Map inputs to dataset format
        input_data = pd.DataFrame([{
            'age': age, 'sex': 1 if sex == "Male" else 0,
            'cp': {"Typical Angina": 1, "Atypical Angina": 2, "Non-anginal Pain": 3, "Asymptomatic": 4}[cp],
            'trestbps': trestbps, 'chol': chol, 'fbs': 1 if fbs == "True" else 0,
            'restecg': {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}[restecg],
            'thalach': thalach, 'exang': 1 if exang == "Yes" else 0, 'oldpeak': oldpeak,
            'slope': {"Upsloping": 1, "Flat": 2, "Downsloping": 3}[slope],
            'ca': ca, 'thal': {"Normal": 3.0, "Fixed Defect": 6.0, "Reversable Defect": 7.0}[thal]
        }])
        
        if model:
            prob = model.predict_proba(input_data)[0][1]
            pred_class = "High Risk (Positive)" if prob > 0.5 else "Low Risk (Negative)"
            
            st.markdown("---")
            st.markdown("### 🤖 Model Output")
            
            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                if prob > 0.5:
                    st.error(f"**{pred_class}**")
                else:
                    st.success(f"**{pred_class}**")
            with res_col2:
                st.write(f"**Estimated Probability:** {prob * 100:.1f}%")
                st.progress(float(prob))
                
        else:
            st.error("Model not loaded. Train the model first.")

# ==========================================
# PAGE: ABOUT
# ==========================================
elif page == "ℹ️ About":
    st.markdown("## ℹ️ About Project")
    st.write("This project demonstrates a complete, end-to-end Machine Learning pipeline built for portfolio demonstration.")
    
    st.markdown("""
    ### 🛠 Tech Stack
    * **Data Manipulation:** `pandas`, `numpy`
    * **Machine Learning:** `scikit-learn` (Pipelines, ColumnTransformers)
    * **Visualizations:** `plotly`
    * **Web Framework:** `streamlit`
    
    ### ⚠️ Limitations & Real-World Context
    Machine learning models in healthcare require rigorous clinical trials, massive datasets (millions of rows), and FDA/regulatory approval. This project is trained on the standard UCI benchmark dataset (~300 records) to demonstrate **software engineering and data science fundamentals**, not to provide actual medical advice.
    """)