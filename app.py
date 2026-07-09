import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# --- 1. PAGE CONFIGURATION (Must be first) ---
st.set_page_config(page_title="AI Adoption Predictor", page_icon="📊", layout="wide")

# --- 2. CUSTOM CSS FOR PROFESSIONAL UI/UX ---
# This changes colors and fonts to look like custom software
st.markdown("""
<style>
    .main-header {
        background-color: #1a3c5e;
        padding: 30px;
        border-radius: 10px;
        margin-bottom: 30px;
        color: white;
    }
    .thesis-title {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 5px;
        color: #f0f2f6;
    }
    .author-info {
        font-size: 16px;
        color: #a8b9cc;
        font-style: italic;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-left: 5px solid #1a3c5e;
        padding: 20px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. ACADEMIC HEADER (CHANGE THESE TO YOUR DETAILS) ---
st.markdown("""
<div class="main-header">
    <div class="thesis-title">📊 AI in Business: Predicting Investment Adoption</div>
    <div class="author-info">MBA Thesis Tool | Thesis Title: Determinants and Application of Artifical Intelligence in Financial Decision Making Among Investors in Pakistan <br>Developed by: Ammar Abdul Jabbar | Student ID: [01-322251033]</div>
</div>
""", unsafe_allow_html=True)

# --- 4. TRAIN THE AI MODEL (Hidden in background) ---
@st.cache_resource
def train_ai():
    df = pd.read_csv('MBA_AI_Investment_Dataset.csv')
    X = df[['TR', 'PU', 'PEOU', 'PR', 'FL', 'TA']]
    y = df['AD']
    model = LinearRegression()
    model.fit(X, y)
    return model

model = train_ai()

# --- 5. APP LAYOUT (Two Columns) ---
col_left, col_right = st.columns([1, 1])

# LEFT COLUMN: Context & Inputs
with col_left:
    st.subheader("🧠 Model Variables (Extended TAM Framework)")
    st.markdown("""
    Adjust the sliders below to simulate a business leader's perceptions. 
    This model is trained on empirical data (N=250) evaluating Trust, TAM constructs, Risk, Literacy, and Awareness.
    """)
    
    st.write("**Input the Variable Scores (1 to 5):**")
    
    # Using st.form groups the sliders nicely and adds a submit button
    with st.form("prediction_form"):
        trust = st.slider("Trust (TR)", 1.0, 5.0, 3.5, 0.1)
        usefulness = st.slider("Perceived Usefulness (PU)", 1.0, 5.0, 3.5, 0.1)
        ease = st.slider("Perceived Ease of Use (PEOU)", 1.0, 5.0, 3.5, 0.1)
        risk = st.slider("Perceived Risk (PR)", 1.0, 5.0, 3.0, 0.1)
        fin_lit = st.slider("Financial Literacy (FL)", 1.0, 5.0, 3.5, 0.1)
        tech_aware = st.slider("Technological Awareness (TA)", 1.0, 5.0, 3.5, 0.1)
        
        submitted = st.form_submit_button("🎯 Predict Adoption Intent")

# RIGHT COLUMN: Results & Insights
with col_right:
    st.subheader("📈 Prediction Results")
    
    if submitted:
        user_inputs = [[trust, usefulness, ease, risk, fin_lit, tech_aware]]
        prediction = model.predict(user_inputs)[0]
        
        # Display the main score in a nice card
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(label="Predicted Adoption (AD) Score", value=f"{prediction:.2f} / 5.00")
        st.progress(min(prediction / 5.0, 1.0))
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Academic Interpretation
        st.write("**Academic Interpretation:**")
        if prediction >= 4.0:
            st.success("**HIGH ADOPTION INTENT.** The subject's high PU and TR outweigh perceived risks. Represents an 'Innovator' or 'Early Adopter' profile in the Diffusion of Innovations theory.")
        elif prediction >= 3.0:
            st.warning("**MODERATE ADOPTION INTENT.** The subject is in the 'Early Majority' fence. Targeted interventions to increase PEOU or reduce PR could convert them.")
        elif prediction >= 2.0:
            st.error("**LOW ADOPTION INTENT.** Significant barriers exist. High PR or low FL/Literacy is likely inhibiting the TAM acceptance process.")
        else:
            st.error("**VERY LOW ADOPTION INTENT.** Strong resistance. The subject falls into the 'Laggard' category for this specific technology.")
    else:
        st.info("👈 Adjust the variables on the left and click 'Predict Adoption Intent' to run the machine learning model.")

# --- 6. FOOTER (Optional but looks great) ---
st.markdown("---")
st.caption("Note: This predictive model is developed strictly for academic research purposes as part of an thesis dissertation. The algorithm utilizes Multiple Linear Regression based on primary survey data.")




























