import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="AI Adoption Predictor | MBA Thesis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

VAR_LABELS = {
    "TR": "Trust",
    "PU": "Perceived Usefulness",
    "PEOU": "Perceived Ease of Use",
    "PR": "Perceived Risk",
    "FL": "Financial Literacy",
    "TA": "Technological Awareness",
}
VAR_ICONS = {
    "TR": "🤝", "PU": "⚡", "PEOU": "🧭", "PR": "⚠️", "FL": "📚", "TA": "📡",
}
VAR_ORDER = ["TR", "PU", "PEOU", "PR", "FL", "TA"]

ARTIFACT_PATH = "model_artifacts.pkl"

# ============================================================
# 2. DESIGN SYSTEM — CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    #MainMenu, footer, header {visibility: hidden;}
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1220px; }

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

    /* ---------- Hero ---------- */
    .hero {
        background: linear-gradient(135deg, var(--navy) 0%, var(--navy-light) 100%);
        border-radius: 16px 16px 0 0;
        padding: 38px 44px 30px 44px;
        box-shadow: 0 10px 30px -12px rgba(15, 41, 66, 0.45);
        position: relative;
        overflow: hidden;
    }
    .hero::after {
        content: "";
        position: absolute; top: -60px; right: -60px;
        width: 220px; height: 220px;
        background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-eyebrow {
        display: inline-block; font-size: 12px; font-weight: 700;
        letter-spacing: 0.12em; text-transform: uppercase; color: #9fc0ff;
        background: rgba(255,255,255,0.08); padding: 5px 12px; border-radius: 20px;
        margin-bottom: 14px;
    }
    .hero-title { font-size: 29px; font-weight: 800; color: #fff; line-height: 1.3; margin-bottom: 9px; max-width: 780px; }
    .hero-sub { font-size: 14px; color: #c3d3e8; line-height: 1.6; max-width: 720px; }
    .hero-meta { margin-top: 16px; font-size: 12.5px; color: #8fa7c4; border-top: 1px solid rgba(255,255,255,0.12); padding-top: 12px; }

    /* ---------- Model status strip ---------- */
    .status-strip {
        background: #0c2036;
        border-radius: 0 0 16px 16px;
        padding: 10px 44px;
        margin-bottom: 26px;
        display: flex; align-items: center; gap: 8px;
        font-size: 12px; color: #a9bcd6; font-weight: 500;
    }
    .status-dot {
        width: 8px; height: 8px; border-radius: 50%;
        background: #22c55e; box-shadow: 0 0 0 3px rgba(34,197,94,0.2);
        display: inline-block; flex-shrink: 0;
    }
    .status-strip b { color: #d7e3f5; font-weight: 700; }

    /* ---------- Tabs (pill style) ---------- */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; background: #eef1f6; padding: 5px; border-radius: 12px; }
    .stTabs [data-baseweb="tab"] {
        height: 40px; border-radius: 9px; padding: 0 18px; background: transparent;
        font-weight: 600; font-size: 13.5px; color: var(--muted);
    }
    .stTabs [data-baseweb="tab"] p { font-size: 13.5px !important; }
    .stTabs [aria-selected="true"] { background: var(--surface) !important; color: var(--navy) !important;
        box-shadow: 0 1px 3px rgba(16,24,40,0.12); }

    /* ---------- Section labels ---------- */
    .section-label { font-size: 13px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: var(--accent); margin-bottom: 4px; }
    .section-desc { font-size: 13.5px; color: var(--muted); margin-bottom: 16px; line-height: 1.55; }

    /* ---------- Real Streamlit bordered containers, styled as cards ---------- */
    .st-key-left_panel, .st-key-right_panel,
    .st-key-diag_panel_1, .st-key-diag_panel_2, .st-key-diag_panel_3 {
        border-radius: 14px !important;
        border: 1px solid var(--border) !important;
        box-shadow: 0 1px 2px rgba(16,24,40,0.04);
        padding: 8px 6px !important;
        background: var(--surface) !important;
    }
    .st-key-score_card {
        background: linear-gradient(180deg, var(--accent-soft) 0%, #ffffff 65%) !important;
        border: 1px solid #d7e6fd !important;
        border-radius: 14px !important;
        text-align: center;
        padding: 6px 6px 2px 6px !important;
    }

    /* ---------- Sliders ---------- */
    div[data-testid="stSlider"] label p { font-size: 13.5px !important; font-weight: 600 !important; color: var(--ink) !important; }
    .stSlider [data-baseweb="slider"] > div > div { background: var(--accent) !important; }

    /* ---------- Submit button ---------- */
    div[data-testid="stFormSubmitButton"] button {
        background: var(--accent); color: white; border: none; border-radius: 9px;
        padding: 12px 0; font-weight: 700; font-size: 14.5px; width: 100%; margin-top: 14px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        box-shadow: 0 4px 12px -2px rgba(47, 111, 237, 0.45);
    }
    div[data-testid="stFormSubmitButton"] button:hover { transform: translateY(-1px); box-shadow: 0 6px 16px -2px rgba(47, 111, 237, 0.55); }

    /* ---------- Result elements ---------- */
    .score-label { font-size: 12px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; margin-top: -8px; padding-bottom: 10px; }

    .verdict-card { border-radius: 12px; padding: 16px 18px; border-left: 5px solid var(--accent); background: var(--accent-soft); margin-top: 4px; margin-bottom: 16px; }
    .verdict-card.high { border-color: var(--success); background: #f0fdf4; }
    .verdict-card.mid  { border-color: var(--warning); background: #fffbeb; }
    .verdict-card.low  { border-color: var(--danger);  background: #fef2f2; }
    .verdict-title { font-weight: 700; font-size: 14px; margin-bottom: 4px; color: var(--ink); }
    .verdict-body { font-size: 13px; color: #3a4657; line-height: 1.55; }

    .badge {
        background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
        padding: 12px 14px; text-align: center; margin-bottom: 10px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .badge:hover { transform: translateY(-2px); box-shadow: 0 4px 10px rgba(16,24,40,0.08); }
    .badge .val { font-size: 20px; font-weight: 800; color: var(--navy); }
    .badge .lab { font-size: 11px; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; margin-top: 2px; }

    .insight-box {
        background: #f8faff; border: 1px dashed #c7d7fb; border-radius: 12px;
        padding: 14px 18px; font-size: 13px; color: #33415c; line-height: 1.6; margin-top: 4px;
    }
    .insight-box b { color: var(--navy); }

    .kpi-grid { display: flex; gap: 14px; margin-bottom: 24px; }
    .kpi {
        flex: 1; background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
        padding: 18px 20px; transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi:hover { transform: translateY(-2px); box-shadow: 0 6px 14px rgba(16,24,40,0.08); }
    .kpi .kpi-val { font-size: 26px; font-weight: 800; color: var(--navy); }
    .kpi .kpi-lab { font-size: 12px; color: var(--muted); font-weight: 600; margin-top: 4px; }

    .empty-state { text-align: center; padding: 50px 20px; color: var(--muted); }
    .empty-state .icon { font-size: 32px; margin-bottom: 8px; }
    .empty-state .msg { font-size: 13.5px; font-weight: 500; }

    .footer-note { font-size: 12px; color: var(--muted); text-align: center; margin-top: 30px; padding-top: 16px; border-top: 1px solid var(--border); }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 3. LOAD TRAINED ARTIFACTS (no training happens here)
# ============================================================
@st.cache_resource
def load_artifacts():
    if not os.path.exists(ARTIFACT_PATH):
        st.error(
            f"Model file '{ARTIFACT_PATH}' not found. Run `python train_model.py` "
            "first to train and save the model, then restart this app."
        )
        st.stop()
    return joblib.load(ARTIFACT_PATH)

DATA = load_artifacts()
model = DATA["model"]

# ============================================================
# 4. HERO HEADER + MODEL STATUS STRIP
# ============================================================
st.markdown("""
<div class="hero">
    <span class="hero-eyebrow">MBA Thesis · Research Tool</span>
    <div class="hero-title">AI in Business: Predicting Investment Adoption</div>
    <div class="hero-sub">
        Determinants and Application of Artificial Intelligence in Financial Decision Making
        Among Investors in Pakistan — an interactive, explainable model built on the Extended
        Technology Acceptance Model (TAM), trained on primary survey data (N = 250).
    </div>
    <div class="hero-meta">Developed by Ammar Abdul Jabbar &nbsp;·&nbsp; Student ID 01-322251033</div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="status-strip">
    <span class="status-dot"></span>
    <span><b>Model loaded from trained artifact</b> &nbsp;·&nbsp;
    trained {DATA['trained_at']} &nbsp;·&nbsp;
    N={DATA['n']} &nbsp;·&nbsp;
    R²={DATA['r2']:.3f} &nbsp;·&nbsp;
    inference only, no retraining at runtime</span>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 5. CHART HELPERS
# ============================================================
def make_gauge(value, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'suffix': " / 5", 'font': {'size': 32, 'color': '#10192b'}},
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
    fig.update_layout(height=190, margin=dict(l=24, r=24, t=20, b=6),
                       paper_bgcolor="rgba(0,0,0,0)", font={'family': "Inter"})
    return fig


def make_waterfall(contributions, baseline, final):
    labels = [VAR_LABELS[k] for k in contributions.index]
    all_vals = [baseline] + list(contributions.values) + [final]
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute"] + ["relative"] * len(contributions) + ["total"],
        x=["Sample<br>baseline"] + labels + ["Your<br>score"],
        y=[baseline] + list(contributions.values) + [final],
        connector={"line": {"color": "#cbd5e1"}},
        decreasing={"marker": {"color": "#dc2626"}},
        increasing={"marker": {"color": "#16a34a"}},
        totals={"marker": {"color": "#2f6fed"}},
        text=[f"{v:+.2f}" if i not in (0, len(contributions) + 1) else f"{v:.2f}"
              for i, v in enumerate(all_vals)],
        textposition="outside",
        textfont={'size': 11},
    ))
    y_pad = max(0.4, (max(all_vals) - min(all_vals)) * 0.25)
    fig.update_layout(
        height=340, margin=dict(l=50, r=20, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={'family': "Inter", 'size': 12}, showlegend=False,
        yaxis=dict(title="Adoption Score", gridcolor="#eef1f6",
                   range=[min(all_vals) - y_pad, max(all_vals) + y_pad]),
        xaxis=dict(tickfont={'size': 11}),
    )
    return fig


def make_importance_chart(std_coefs):
    ordered = std_coefs.reindex(std_coefs.abs().sort_values(ascending=True).index)
    colors = ["#16a34a" if v > 0 else "#dc2626" for v in ordered.values]
    max_abs = max(0.05, ordered.abs().max())
    fig = go.Figure(go.Bar(
        x=ordered.values, y=[VAR_LABELS[k] for k in ordered.index],
        orientation="h", marker_color=colors,
        text=[f"{v:+.3f}" for v in ordered.values], textposition="outside",
        textfont={'size': 11.5}, cliponaxis=False,
    ))
    fig.update_layout(
        height=300, margin=dict(l=10, r=70, t=15, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={'family': "Inter", 'size': 12},
        xaxis=dict(
            title="Standardized Coefficient (Impact on Adoption)",
            gridcolor="#eef1f6", zeroline=True, zerolinecolor="#cbd5e1",
            range=[-max_abs * 1.45, max_abs * 1.45],
        ),
    )
    return fig


def make_corr_heatmap(corr):
    labels = [VAR_LABELS[k] for k in corr.columns]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=labels, y=labels,
        colorscale=[[0, "#dc2626"], [0.5, "#f8fafc"], [1, "#2f6fed"]],
        zmin=-1, zmax=1, text=np.round(corr.values, 2), texttemplate="%{text}",
        textfont={'size': 11},
        colorbar=dict(title="r", thickness=14),
    ))
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=15, b=80),
        paper_bgcolor="rgba(0,0,0,0)", font={'family': "Inter", 'size': 11},
        xaxis=dict(tickangle=-40),
    )
    return fig

# ============================================================
# 6. TABS
# ============================================================
tab_predict, tab_diagnostics = st.tabs(["🎯Predictor", "📊 Model Diagnostics"])

# ------------------------------------------------------------
# TAB 1: PREDICTOR
# ------------------------------------------------------------
with tab_predict:
    col_left, col_right = st.columns([1, 1.15], gap="large")

    with col_left:
        with st.container(border=True, key="left_panel"):
            st.markdown('<div class="section-label">Model Variables</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-desc">Adjust the sliders to simulate a business leader\'s '
                'perceptions across the Extended TAM constructs, then run the model.</div>',
                unsafe_allow_html=True,
            )
            with st.form("prediction_form"):
                trust = st.slider(f"{VAR_ICONS['TR']}  Trust (TR)", 1.0, 5.0, 3.5, 0.1)
                usefulness = st.slider(f"{VAR_ICONS['PU']}  Perceived Usefulness (PU)", 1.0, 5.0, 3.5, 0.1)
                ease = st.slider(f"{VAR_ICONS['PEOU']}  Perceived Ease of Use (PEOU)", 1.0, 5.0, 3.5, 0.1)
                risk = st.slider(f"{VAR_ICONS['PR']}  Perceived Risk (PR)", 1.0, 5.0, 3.0, 0.1)
                fin_lit = st.slider(f"{VAR_ICONS['FL']}  Financial Literacy (FL)", 1.0, 5.0, 3.5, 0.1)
                tech_aware = st.slider(f"{VAR_ICONS['TA']}  Technological Awareness (TA)", 1.0, 5.0, 3.5, 0.1)
                submitted = st.form_submit_button("🎯 Predict AI Adoption")

    with col_right:
        with st.container(border=True, key="right_panel"):
            st.markdown('<div class="section-label">Prediction Results</div>', unsafe_allow_html=True)

            if submitted:
                x_vals = pd.Series(
                    [trust, usefulness, ease, risk, fin_lit, tech_aware], index=VAR_ORDER
                )
                prediction = float(model.predict([x_vals.values])[0])
                prediction_clamped = max(0.0, min(prediction, 5.0))

                baseline = float(model.predict([DATA["means"].values])[0])
                contributions = pd.Series(model.coef_, index=VAR_ORDER) * (x_vals - DATA["means"])
                percentile = float((DATA["y"] < prediction).mean() * 100)

                if prediction >= 4.0:
                    color, cls = "#16a34a", "high"
                    title = "High Adoption Intent"
                    body = ("Strong Trust and Perceived Usefulness outweigh perceived risk. "
                            "This profile aligns with an <b>Innovator / Early Adopter</b> "
                            "segment in Diffusion of Innovations theory.")
                elif prediction >= 3.0:
                    color, cls = "#d97706", "mid"
                    title = "Moderate Adoption Intent"
                    body = ("This profile sits on the <b>Early Majority</b> fence. "
                            "Targeted interventions that raise Perceived Ease of Use or "
                            "lower Perceived Risk could shift it toward adoption.")
                elif prediction >= 2.0:
                    color, cls = "#dc2626", "low"
                    title = "Low Adoption Intent"
                    body = ("Meaningful barriers exist. Elevated Perceived Risk or limited "
                            "Financial Literacy is likely inhibiting the TAM acceptance process.")
                else:
                    color, cls = "#dc2626", "low"
                    title = "Very Low Adoption Intent"
                    body = ("Strong resistance is indicated. This profile falls into the "
                            "<b>Laggard</b> category for this specific technology.")

                gcol, bcol = st.columns([1, 1])
                with gcol:
                    with st.container(border=False, key="score_card"):
                        st.plotly_chart(make_gauge(prediction_clamped, color), use_container_width=True, config={'displayModeBar': False})
                        st.markdown('<div class="score-label">Predicted Adoption Score</div>', unsafe_allow_html=True)
                with bcol:
                    st.markdown(f"""
                    <div class="badge"><div class="val">{percentile:.0f}<span style="font-size:13px;">th</span></div><div class="lab">Percentile vs. N=250 Sample</div></div>
                    <div class="badge"><div class="val">{baseline:.2f}</div><div class="lab">Sample Baseline Score</div></div>
                    """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="verdict-card {cls}">
                    <div class="verdict-title">{title}</div>
                    <div class="verdict-body">{body}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<div class="section-label" style="margin-top:6px;">Why This Score? — Variable Contribution</div>', unsafe_allow_html=True)
                st.markdown(
                    '<div class="section-desc">Each bar shows how far this profile\'s inputs pushed the '
                    'prediction above or below the sample baseline.</div>',
                    unsafe_allow_html=True,
                )
                st.plotly_chart(make_waterfall(contributions, baseline, prediction), use_container_width=True, config={'displayModeBar': False})

                top_driver = contributions.abs().idxmax()
                direction = "increased" if contributions[top_driver] > 0 else "decreased"
                st.markdown(f"""
                <div class="insight-box">
                    💡 <b>Key insight:</b> <b>{VAR_LABELS[top_driver]}</b> was the strongest driver for this
                    profile, which {direction} the predicted adoption score by
                    <b>{abs(contributions[top_driver]):.2f} points</b> relative to the sample average.
                    This profile ranks higher than <b>{percentile:.0f}%</b> of the surveyed investors (N={DATA['n']}).
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="empty-state">
                    <div class="icon">👈</div>
                    <div class="msg">Set the variables and click <b>Predict Adoption Intent</b><br>to run the model.</div>
                </div>
                """, unsafe_allow_html=True)

# ------------------------------------------------------------
# TAB 2: MODEL DIAGNOSTICS
# ------------------------------------------------------------
with tab_diagnostics:
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi"><div class="kpi-val">{DATA['r2']:.3f}</div><div class="kpi-lab">R² (Model Fit)</div></div>
        <div class="kpi"><div class="kpi-val">{DATA['adj_r2']:.3f}</div><div class="kpi-lab">Adjusted R²</div></div>
        <div class="kpi"><div class="kpi-val">{DATA['n']}</div><div class="kpi-lab">Sample Size (N)</div></div>
        <div class="kpi"><div class="kpi-val">6</div><div class="kpi-lab">TAM Predictor Variables</div></div>
    </div>
    """, unsafe_allow_html=True)

    dcol1, dcol2 = st.columns([1, 1], gap="large")

    with dcol1:
        with st.container(border=True, key="diag_panel_1"):
            st.markdown('<div class="section-label">Standardized Coefficient Importance</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-desc">Which TAM construct moves the needle most on adoption intent, '
                'holding the others constant.</div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(make_importance_chart(DATA["std_coefs"]), use_container_width=True, config={'displayModeBar': False})

    with dcol2:
        with st.container(border=True, key="diag_panel_2"):
            st.markdown('<div class="section-label">Construct Correlation Matrix</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-desc">Pairwise correlation across the six survey constructs, '
                'useful for spotting multicollinearity.</div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(make_corr_heatmap(DATA["corr"]), use_container_width=True, config={'displayModeBar': False})

    with st.container(border=True, key="diag_panel_3"):
        st.markdown('<div class="section-label">Sample Descriptive Statistics</div>', unsafe_allow_html=True)
        desc = DATA["desc"].copy()
        desc.index = [VAR_LABELS.get(i, i) for i in desc.index]
        st.dataframe(desc, use_container_width=True)

# ============================================================
# 7. FOOTER
# ============================================================
st.markdown("""
<div class="footer-note">
    This predictive tool is developed strictly for academic research purposes as part of our thesis dissertation.
</div>
""", unsafe_allow_html=True)
