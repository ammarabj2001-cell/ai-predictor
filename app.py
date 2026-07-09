import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="AI Adoption Predictor | MBA Thesis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# 2. DESIGN SYSTEM — CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Lora:ital@1&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hide default Streamlit chrome */
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }

    :root {
        --navy: #0f2942;
        --navy-light: #1e4266;
        --accent: #2f6fed;
        --accent-soft: #eaf1fe;
        --ink: #10192b;
        --muted: #64748b;
        --border: #e6e9f0;
        --surface: #ffffff;
        --bg: #f6f8fb;
        --success: #16a34a;
        --warning: #d97706;
        --danger: #dc2626;
    }

    .stApp { background-color: var(--bg); }

    /* ---------- Header ---------- */
    .hero {
        background: linear-gradient(135deg, var(--navy) 0%, var(--navy-light) 100%);
        border-radius: 16px;
        padding: 40px 44px;
        margin-bottom: 32px;
        box-shadow: 0 10px 30px -12px rgba(15, 41, 66, 0.45);
        position: relative;
        overflow: hidden;
    }
    .hero::after {
        content: "";
        position: absolute;
        top: -60px; right: -60px;
        width: 220px; height: 220px;
        background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-eyebrow {
        display: inline-block;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #9fc0ff;
        background: rgba(255,255,255,0.08);
        padding: 5px 12px;
        border-radius: 20px;
        margin-bottom: 14px;
    }
    .hero-title {
        font-size: 30px;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.3;
        margin-bottom: 10px;
        max-width: 760px;
    }
    .hero-sub {
        font-size: 14.5px;
        color: #c3d3e8;
        line-height: 1.6;
        max-width: 700px;
    }
    .hero-meta {
        margin-top: 18px;
        font-size: 13px;
        color: #8fa7c4;
        border-top: 1px solid rgba(255,255,255,0.12);
        padding-top: 14px;
    }

    /* ---------- Section labels ---------- */
    .section-label {
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 4px;
    }
    .section-desc {
        font-size: 13.5px;
        color: var(--muted);
        margin-bottom: 18px;
        line-height: 1.55;
    }

    /* ---------- Panels ---------- */
    .panel {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 28px 30px;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        height: 100%;
    }

    /* ---------- Sliders ---------- */
    div[data-testid="stSlider"] label p {
        font-size: 13.5px !important;
        font-weight: 600 !important;
        color: var(--ink) !important;
    }
    div[data-testid="stSlider"] {
        margin-bottom: 6px;
    }
    .stSlider [data-baseweb="slider"] > div > div {
        background: var(--accent) !important;
    }

    /* ---------- Submit button ---------- */
    div[data-testid="stFormSubmitButton"] button {
        background: var(--accent);
        color: white;
        border: none;
        border-radius: 9px;
        padding: 12px 0;
        font-weight: 700;
        font-size: 14.5px;
        width: 100%;
        margin-top: 14px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        box-shadow: 0 4px 12px -2px rgba(47, 111, 237, 0.45);
    }
    div[data-testid="stFormSubmitButton"] button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px -2px rgba(47, 111, 237, 0.55);
    }

    /* ---------- Result cards ---------- */
    .score-card {
        background: linear-gradient(180deg, var(--accent-soft) 0%, #ffffff 60%);
        border: 1px solid #d7e6fd;
        border-radius: 14px;
        padding: 10px 8px 4px 8px;
        text-align: center;
        margin-bottom: 18px;
    }
    .score-number {
        font-size: 44px;
        font-weight: 800;
        color: var(--navy);
        line-height: 1;
    }
    .score-label {
        font-size: 12.5px;
        font-weight: 600;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 6px;
    }

    .verdict-card {
        border-radius: 12px;
        padding: 18px 20px;
        border-left: 5px solid var(--accent);
        background: var(--accent-soft);
        margin-top: 6px;
    }
    .verdict-card.high { border-color: var(--success); background: #f0fdf4; }
    .verdict-card.mid  { border-color: var(--warning); background: #fffbeb; }
    .verdict-card.low  { border-color: var(--danger);  background: #fef2f2; }
    .verdict-title {
        font-weight: 700;
        font-size: 14.5px;
        margin-bottom: 5px;
        color: var(--ink);
    }
    .verdict-body {
        font-size: 13.5px;
        color: #3a4657;
        line-height: 1.55;
    }

    .empty-state {
        text-align: center;
        padding: 60px 20px;
        color: var(--muted);
    }
    .empty-state .icon { font-size: 34px; margin-bottom: 10px; }
    .empty-state .msg { font-size: 14px; font-weight: 500; }

    .footer-note {
        font-size: 12px;
        color: var(--muted);
        text-align: center;
        margin-top: 36px;
        padding-top: 18px;
        border-top: 1px solid var(--border);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 3. HERO HEADER
# ============================================================
st.markdown("""
<div class="hero">
    <span class="hero-eyebrow">MBA Thesis · Research Tool</span>
    <div class="hero-title">AI in Business: Predicting Investment Adoption</div>
    <div class="hero-sub">
        Determinants and Application of Artificial Intelligence in Financial Decision Making
        Among Investors in Pakistan — an interactive model built on the Extended
        Technology Acceptance Model (TAM), trained on primary survey data (N = 250).
    </div>
    <div class="hero-meta">Developed by Ammar Abdul Jabbar &nbsp;·&nbsp; Student ID 01-322251033</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 4. TRAIN THE MODEL
# ============================================================
@st.cache_resource
def train_ai():
    df = pd.read_csv('MBA_AI_Investment_Dataset.csv')
    X = df[['TR', 'PU', 'PEOU', 'PR', 'FL', 'TA']]
    y = df['AD']
    model = LinearRegression()
    model.fit(X, y)
    return model

model = train_ai()

# ============================================================
# 5. GAUGE CHART HELPER
# ============================================================
def make_gauge(value, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'suffix': " / 5", 'font': {'size': 34, 'color': '#10192b'}},
        gauge={
            'axis': {'range': [0, 5], 'tickwidth': 1, 'tickcolor': "#c8d2e0"},
            'bar': {'color': color, 'thickness': 0.28},
            'bgcolor': "white",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 2], 'color': '#fef2f2'},
                {'range': [2, 3], 'color': '#fef9ec'},
                {'range': [3, 4], 'color': '#fffbeb'},
                {'range': [4, 5], 'color': '#f0fdf4'},
            ],
        },
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'family': "Inter"},
    )
    return fig

# ============================================================
# 6. LAYOUT
# ============================================================
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Model Variables</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Adjust the sliders to simulate a business leader\'s '
        'perceptions across the Extended TAM constructs, then run the model.</div>',
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):
        trust = st.slider("Trust (TR)", 1.0, 5.0, 3.5, 0.1)
        usefulness = st.slider("Perceived Usefulness (PU)", 1.0, 5.0, 3.5, 0.1)
        ease = st.slider("Perceived Ease of Use (PEOU)", 1.0, 5.0, 3.5, 0.1)
        risk = st.slider("Perceived Risk (PR)", 1.0, 5.0, 3.0, 0.1)
        fin_lit = st.slider("Financial Literacy (FL)", 1.0, 5.0, 3.5, 0.1)
        tech_aware = st.slider("Technological Awareness (TA)", 1.0, 5.0, 3.5, 0.1)
        submitted = st.form_submit_button("🎯  Predict Adoption Intent")
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Prediction Results</div>', unsafe_allow_html=True)

    if submitted:
        user_inputs = [[trust, usefulness, ease, risk, fin_lit, tech_aware]]
        prediction = float(model.predict(user_inputs)[0])
        prediction = max(0.0, min(prediction, 5.0))

        if prediction >= 4.0:
            tone, color, cls = "success", "#16a34a", "high"
            title = "High Adoption Intent"
            body = ("Strong Trust and Perceived Usefulness outweigh perceived risk. "
                    "This profile aligns with an <b>Innovator / Early Adopter</b> "
                    "segment in Diffusion of Innovations theory.")
        elif prediction >= 3.0:
            tone, color, cls = "warning", "#d97706", "mid"
            title = "Moderate Adoption Intent"
            body = ("This profile sits on the <b>Early Majority</b> fence. "
                    "Targeted interventions that raise Perceived Ease of Use or "
                    "lower Perceived Risk could shift it toward adoption.")
        elif prediction >= 2.0:
            tone, color, cls = "error", "#dc2626", "low"
            title = "Low Adoption Intent"
            body = ("Meaningful barriers exist. Elevated Perceived Risk or limited "
                    "Financial Literacy is likely inhibiting the TAM acceptance process.")
        else:
            tone, color, cls = "error", "#dc2626", "low"
            title = "Very Low Adoption Intent"
            body = ("Strong resistance is indicated. This profile falls into the "
                    "<b>Laggard</b> category for this specific technology.")

        st.markdown('<div class="score-card">', unsafe_allow_html=True)
        st.plotly_chart(make_gauge(prediction, color), use_container_width=True, config={'displayModeBar': False})
        st.markdown(
            '<div class="score-label">Predicted Adoption (AD) Score</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="verdict-card {cls}">
            <div class="verdict-title">{title}</div>
            <div class="verdict-body">{body}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">👈</div>
            <div class="msg">Set the variables and click <b>Predict Adoption Intent</b><br>to run the model.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# 7. FOOTER
# ============================================================
st.markdown("""
<div class="footer-note">
    This predictive tool is developed strictly for academic research purposes as part of an MBA thesis dissertation.
    The underlying algorithm uses Multiple Linear Regression trained on primary survey data.
</div>
""", unsafe_allow_html=True)
