import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# --- 1. TRAIN THE AI MODEL (This happens instantly in the background) ---
@st.cache_resource
def train_ai():
    # Load your dataset
    df = pd.read_csv('MBA_AI_Investment_Dataset.csv')
    X = df[['TR', 'PU', 'PEOU', 'PR', 'FL', 'TA']]
    y = df['AD']
    model = LinearRegression()
    model.fit(X, y)
    return model

model = train_ai()

# --- 2. DESIGN THE WEBPAGE ---
st.set_page_config(page_title="AI Adoption Predictor", page_icon="🤖")
st.title("🤖 AI in Business: Adoption Predictor")
st.markdown("Adjust the sliders below to represent a business leader's perceptions. The AI will predict their likelihood of adopting AI for investment purposes.")

st.sidebar.header("Input Variables (Scale 1-5)")

# Create interactive sliders for the user
trust = st.sidebar.slider("Trust (TR)", 1.0, 5.0, 3.0, 0.1)
usefulness = st.sidebar.slider("Perceived Usefulness (PU)", 1.0, 5.0, 3.0, 0.1)
ease = st.sidebar.slider("Perceived Ease of Use (PEOU)", 1.0, 5.0, 3.0, 0.1)
risk = st.sidebar.slider("Perceived Risk (PR)", 1.0, 5.0, 3.0, 0.1)
fin_lit = st.sidebar.slider("Financial Literacy (FL)", 1.0, 5.0, 3.0, 0.1)
tech_aware = st.sidebar.slider("Technological Awareness (TA)", 1.0, 5.0, 3.0, 0.1)

# --- 3. MAKE THE PREDICTION ---
# Create a button to trigger the prediction
if st.sidebar.button("Predict Adoption Score"):
    # Put user inputs into a format the AI understands
    user_inputs = [[trust, usefulness, ease, risk, fin_lit, tech_aware]]
    prediction = model.predict(user_inputs)[0]
    
    # Display the results beautifully
    st.subheader("🎯 Prediction Result")
    st.metric(label="Predicted Adoption (AD) Score", value=f"{prediction:.2f} / 5.00")
    
    # Add a visual progress bar
    st.progress(min(prediction / 5.0, 1.0))
    
    # Give a business interpretation
    if prediction >= 4.0:
        st.success("Insight: **HIGHLY LIKELY** to adopt AI. This profile represents an innovator or early adopter.")
    elif prediction >= 3.0:
        st.warning("Insight: **MODERATE LIKELIHOOD**. They are on the fence; increasing Usefulness or Ease of Use could push them to adopt.")
    elif prediction >= 2.0:
        st.error("Insight: **LOW LIKELIHOOD**. Significant barriers exist (likely driven by high Risk or low Trust).")
    else:
        st.error("Insight: **VERY LOW LIKELIHOOD**. Strong resistance to AI adoption in their business.")
else:
    st.info("👈 Move the sliders on the left and click 'Predict Adoption Score' to start.")
