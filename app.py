import textwrap
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Placement Predictor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg: #08090b;
    --panel: #101113;
    --panel-2: #141518;
    --border: #292b30;
    --border-soft: #202226;
    --text: #f1f3f5;
    --muted: #858a94;
    --blue: #1769ff;
    --blue-bright: #2878ff;
    --green: #19d37a;
    --yellow: #f5c542;
    --red: #ff4d5a;
}

/* ---------------- BASE ---------------- */

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: var(--bg);
    color: var(--text);
}

.main {
    background: var(--bg);
}

.block-container {
    padding: 38px 48px 60px 48px;
    max-width: 1500px;
}

/* ---------------- HIDE STREAMLIT ---------------- */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}

/* ---------------- SIDEBAR ---------------- */

section[data-testid="stSidebar"] {
    background: #0a0b0d;
    border-right: 1px solid #24262a;
}

section[data-testid="stSidebar"] > div {
    padding-top: 0;
}

.sidebar-brand {
    padding: 24px 20px 22px 20px;
    border-bottom: 1px solid #24262a;
}

.brand-row {
    display: flex;
    align-items: center;
    gap: 12px;
}

.brand-logo {
    width: 34px;
    height: 34px;
    background: var(--blue);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 700;
    font-size: 15px;
}

.brand-title {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.8px;
    color: #f5f5f5;
}

.brand-version {
    font-family: 'DM Mono', monospace;
    font-size: 8px;
    color: #777b84;
    letter-spacing: 1.5px;
    margin-top: 2px;
}

.sidebar-status {
    position: fixed;
    bottom: 18px;
    left: 20px;
    width: 170px;
    border-top: 1px solid #24262a;
    padding-top: 12px;
    font-family: 'DM Mono', monospace;
    font-size: 8px;
    color: #70747c;
    letter-spacing: 0.6px;
}

.status-dot {
    display: inline-block;
    width: 6px;
    height: 6px;
    background: var(--green);
    border-radius: 50%;
    margin-right: 5px;
}

/* ---------------- SIDEBAR RADIO ---------------- */

div[data-testid="stRadio"] > label {
    display: none;
}

div[data-testid="stRadio"] > div {
    gap: 2px;
    padding: 12px 8px;
}

div[data-testid="stRadio"] label {
    background: transparent;
    border-left: 2px solid transparent;
    padding: 11px 14px !important;
    color: #8d9199 !important;
    transition: 0.15s;
    cursor: pointer;
    font-size: 13px !important;
}

div[data-testid="stRadio"] label:hover {
    background: #111317;
    color: #e8e9eb !important;
}

div[data-testid="stRadio"] label:has(input:checked) {
    background: #111317;
    border-left: 2px solid var(--blue);
    color: white !important;
}

div[data-testid="stRadio"] label p {
    font-size: 13px !important;
}

/* ---------------- HEADINGS ---------------- */

.page-kicker {
    font-family: 'DM Mono', monospace;
    color: #367cff;
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.page-title {
    font-size: 32px;
    line-height: 1.1;
    font-weight: 700;
    letter-spacing: -1.3px;
    color: #f4f5f7;
    margin-bottom: 9px;
}

.page-subtitle {
    color: #81858e;
    font-size: 13px;
    max-width: 720px;
    line-height: 1.7;
    margin-bottom: 28px;
}

/* ---------------- SECTION ---------------- */

.section-title {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 1.7px;
    color: #a0a4ac;
    text-transform: uppercase;
    margin-bottom: 15px;
}

.section-card {
    background: var(--panel);
    border: 1px solid var(--border);
    padding: 20px;
    margin-bottom: 16px;
}

/* ---------------- INPUTS ---------------- */

label {
    color: #9ca0a8 !important;
    font-size: 10px !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

input, textarea {
    background: #151619 !important;
    color: #f3f3f3 !important;
    border: 1px solid #303238 !important;
    border-radius: 2px !important;
}

div[data-baseweb="select"] > div {
    background: #151619 !important;
    border: 1px solid #303238 !important;
    border-radius: 2px !important;
    color: white !important;
}

div[data-testid="stNumberInput"] button {
    background: #1a1c20 !important;
    color: #8c919a !important;
    border: none !important;
}

/* ---------------- BUTTON ---------------- */

.stButton > button {
    background: var(--blue) !important;
    color: white !important;
    border: 0 !important;
    border-radius: 2px !important;
    height: 42px !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    padding: 0 24px !important;
    transition: 0.2s;
}

.stButton > button:hover {
    background: var(--blue-bright) !important;
    box-shadow: 0 0 20px rgba(23,105,255,0.18);
}

/* ---------------- METRIC CARDS ---------------- */

.metric-card {
    background: var(--panel);
    border: 1px solid var(--border);
    padding: 19px;
    min-height: 110px;
}

.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 9px;
    color: #777c85;
    text-transform: uppercase;
    letter-spacing: 1.3px;
    margin-bottom: 14px;
}

.metric-value {
    font-size: 25px;
    font-weight: 600;
    letter-spacing: -0.8px;
    color: #f3f4f6;
}

.metric-value.blue {
    color: #3b82ff;
}

.metric-value.green {
    color: var(--green);
}

.metric-value.yellow {
    color: var(--yellow);
}

.metric-small {
    color: #6d7179;
    font-size: 9px;
    margin-top: 8px;
}

/* ---------------- RESULT ---------------- */

.result-card {
    background: #0e1110;
    border: 1px solid #26352d;
    padding: 24px;
    margin-top: 22px;
}

.result-label {
    font-family: 'DM Mono', monospace;
    color: #777d84;
    font-size: 9px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

.result-value {
    font-size: 43px;
    font-weight: 600;
    letter-spacing: -2px;
    margin-top: 5px;
}

.status-pill {
    display: inline-block;
    padding: 5px 9px;
    font-family: 'DM Mono', monospace;
    font-size: 8px;
    letter-spacing: 0.8px;
    border: 1px solid;
    margin-top: 8px;
}

.status-success {
    color: var(--green);
    border-color: #1d6b46;
    background: #0c2118;
}

.status-warning {
    color: var(--yellow);
    border-color: #685a25;
    background: #211e0c;
}

.status-danger {
    color: var(--red);
    border-color: #6c292f;
    background: #210d10;
}

/* ---------------- PROGRESS ---------------- */

.progress-container {
    background: #24272d;
    height: 5px;
    width: 100%;
    margin-top: 17px;
}

.progress-fill {
    height: 5px;
    background: var(--blue);
}

/* ---------------- INFO BOX ---------------- */

.info-box {
    border: 1px solid #2b2e34;
    background: #111215;
    padding: 16px;
    color: #858991;
    font-size: 11px;
    line-height: 1.7;
}

/* ---------------- TABLE ---------------- */

.dataframe {
    background: #101113 !important;
    color: #ddd !important;
}

/* ---------------- DIVIDER ---------------- */

hr {
    border: none !important;
    border-top: 1px solid #25272b !important;
    margin: 28px 0 !important;
}
/* ============================================================
   HOME PAGE
   ============================================================ */

.home-hero {
    padding: 35px 0 25px 0;
}

.hero-kicker,
.section-kicker {
    color: #247BFF;
    font-family: monospace;
    font-size: 12px;
    letter-spacing: 3px;
    margin-bottom: 14px;
}

.home-hero h1 {
    font-size: 48px;
    line-height: 1.08;
    margin: 0;
    font-weight: 700;
    letter-spacing: -2px;
}

.home-hero h1 span {
    color: #247BFF;
    display: block;
}

.hero-description {
    max-width: 850px;
    color: #8b94a7;
    font-size: 16px;
    line-height: 1.8;
    margin-top: 20px;
}


/* STATS */

.home-stats {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin: 30px 0 50px 0;
}

.home-stat {
    background: #111318;
    border: 1px solid #252932;
    padding: 22px;
    min-height: 120px;
    transition: 0.2s ease;
}

.home-stat:hover {
    border-color: #247BFF;
    transform: translateY(-2px);
}

.stat-label {
    font-family: monospace;
    font-size: 10px;
    letter-spacing: 2px;
    color: #657083;
}

.stat-value {
    font-size: 20px;
    font-weight: 700;
    margin-top: 18px;
    color: #f2f4f8;
}

.stat-value.online {
    color: #16e98a;
}

.stat-sub {
    color: #697386;
    font-size: 12px;
    margin-top: 7px;
}


/* CAPABILITIES */

.capability-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 18px;
    margin-top: 18px;
}

.capability-card {
    background: #111318;
    border: 1px solid #252932;
    padding: 28px;
    min-height: 180px;
    transition: 0.2s ease;
}

.capability-card:hover {
    border-color: #247BFF;
    transform: translateY(-3px);
}

.cap-number {
    color: #247BFF;
    font-family: monospace;
    font-size: 12px;
    margin-bottom: 35px;
}

.capability-card h3 {
    color: #f3f5f8;
    font-size: 20px;
    margin-bottom: 12px;
}

.capability-card p {
    color: #7d8798;
    line-height: 1.6;
    font-size: 13px;
}


/* WORKFLOW */

.workflow-kicker {
    margin-top: 55px;
}

.workflow {
    display: flex;
    align-items: center;
    background: #111318;
    border: 1px solid #252932;
    padding: 25px 30px;
    margin-top: 18px;
}

.workflow-step {
    display: flex;
    align-items: flex-start;
    gap: 15px;
    flex: 1;
}

.workflow-step span {
    color: #247BFF;
    font-family: monospace;
    font-size: 12px;
}

.workflow-step strong {
    color: #e9edf3;
    font-size: 14px;
}

.workflow-step p {
    color: #697386;
    font-size: 11px;
    margin: 5px 0 0 0;
    line-height: 1.5;
}

.workflow-line {
    width: 45px;
    height: 1px;
    background: #303641;
    margin: 0 20px;
}


/* MOBILE */

@media (max-width: 900px) {

    .home-stats {
        grid-template-columns: repeat(2, 1fr);
    }

    .capability-grid {
        grid-template-columns: 1fr;
    }

    .workflow {
        flex-direction: column;
        align-items: flex-start;
        gap: 20px;
    }

    .workflow-line {
        display: none;
    }

    .home-hero h1 {
        font-size: 38px;
    }
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL LOADING
# ============================================================

MODEL_DIR = "models"

CLASSIFIER_PATH = os.path.join(MODEL_DIR, "logistic_regression.joblib")
RF_PATH = os.path.join(MODEL_DIR, "random_forest.joblib")
XGB_PATH = os.path.join(MODEL_DIR, "xgboost.joblib")

@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None


logistic_model = load_model(CLASSIFIER_PATH)
rf_model = load_model(RF_PATH)
xgb_model = load_model(XGB_PATH)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def unwrap_model(model):
    """
    Supports both:
    1. Direct sklearn Pipeline
    2. {'pipeline': pipeline, 'feature_columns': [...]}
    """
    if isinstance(model, dict):
        return model.get("pipeline", model), model.get("feature_columns")
    return model, None


def make_prediction(model, data):
    model, feature_columns = unwrap_model(model)

    df = pd.DataFrame([data])

    if feature_columns:
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0

        df = df[feature_columns]

    prediction = model.predict(df)[0]

    probability = None

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(df)[0][1])

    return int(prediction), probability


def calculate_readiness(data):
    """
    Simple transparent readiness score.
    This is NOT the ML prediction.
    It is a dashboard score for student guidance.
    """

    score = 0

    score += min(data["cgpa"] / 10 * 25, 25)
    score += min(data["coding"] / 10 * 20, 20)
    score += min(data["communication"] / 10 * 15, 15)
    score += min(data["projects"] / 5 * 15, 15)
    score += min(data["internships"] / 2 * 10, 10)
    score += min(data["certifications"] / 3 * 5, 5)
    score += min(data["aptitude"] / 100 * 10, 10)

    return round(min(score, 100))


def recommendation(data):
    gaps = []

    if data["cgpa"] < 7.5:
        gaps.append(("Academic performance", "Raise CGPA above 7.5"))

    if data["coding"] < 7:
        gaps.append(("Coding", "Strengthen DSA and problem solving"))

    if data["communication"] < 7:
        gaps.append(("Communication", "Practice interviews and speaking"))

    if data["projects"] < 2:
        gaps.append(("Projects", "Build 2–3 strong projects"))

    if data["internships"] == 0:
        gaps.append(("Experience", "Target an internship or research project"))

    if data["aptitude"] < 65:
        gaps.append(("Aptitude", "Practice quantitative and logical reasoning"))

    if not gaps:
        gaps.append(("Profile", "Profile is well balanced. Focus on interview preparation"))

    return gaps


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("""
    <div class="sidebar-brand">
        <div class="brand-row">
            <div class="brand-logo">P</div>
            <div>
                <div class="brand-title">PLACEMENT</div>
                <div class="brand-version">PREDICTOR · V1.0</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# NAVIGATION
# ============================================================

pages = [
    "Home",
    "Prediction",
    "EDA Dashboard",
    "Batch Upload"
]

# Current page
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

with st.sidebar:

    selected_page = st.radio(
        "",
        pages,
        index=pages.index(st.session_state["page"]),
        label_visibility="collapsed"
    )

    if selected_page != st.session_state["page"]:
        st.session_state["page"] = selected_page
        st.rerun()

    # Sidebar status
st.markdown(
    """
    <div class="sidebar-status">
        <div class="status-line">
            <span class="status-dot"></span>
            SYSTEM ONLINE
        </div>

        <div class="sidebar-meta">
            MODEL: LOGISTIC / RF / XGB
        </div>

        <div class="sidebar-meta">
            DATA: TRAINING CORPUS
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PAGE ROUTING
# ============================================================

page = st.session_state["page"]

if page == "Home":
    # HOME PAGE CODE HERE
    pass

elif page == "Prediction":
    # PREDICTION PAGE CODE HERE
    pass

elif page == "EDA Dashboard":
    # EDA DASHBOARD CODE HERE
    pass

elif page == "Batch Upload":
    # BATCH UPLOAD PAGE CODE HERE
    pass
# ============================================================
# HOME
# ============================================================

if page == "Home":

    st.markdown(
        '<div class="page-kicker">// placement intelligence platform</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-title">Predict placement outcomes<br>'
        '<span style="color:#1769ff;">before recruitment day.</span></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'An end-to-end machine learning system that evaluates academic '
        'performance, technical skills and experience to estimate placement outcomes.'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">MODEL TYPE</div>
            <div class="metric-value blue">ML</div>
            <div class="metric-small">Classification engine</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">PREDICTION</div>
            <div class="metric-value">PLACEMENT</div>
            <div class="metric-small">Binary outcome</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">ANALYSIS</div>
            <div class="metric-value">PROFILE</div>
            <div class="metric-small">Skills + academics</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">ENGINE</div>
            <div class="metric-value green">ONLINE</div>
            <div class="metric-small">Local inference</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">// system capabilities</div>',
        unsafe_allow_html=True
    )

    cols = st.columns(3)

    capabilities = [
        ("01", "Placement Probability",
         "Estimate the likelihood of a successful placement from the student profile."),
        ("02", "Profile Analysis",
         "Identify the academic and technical factors influencing placement."),
        ("03", "Career Readiness",
         "Generate a transparent readiness score and improvement areas.")
    ]

    for col, item in zip(cols, capabilities):

        with col:

            st.markdown(f"""
            <div class="section-card" style="min-height:170px;">
                <div style="font-family:'DM Mono';font-size:10px;color:#367cff;">
                    {item[0]}
                </div>
                <div style="font-size:17px;font-weight:600;margin-top:28px;">
                    {item[1]}
                </div>
                <div style="font-size:11px;color:#777c85;line-height:1.7;margin-top:10px;">
                    {item[2]}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Run Prediction →"):
        st.session_state["page"] = "Prediction"
        st.rerun()


# ============================================================
# PREDICTION
# ============================================================

elif page == "Prediction":

    st.markdown(
        '<div class="page-kicker">// module 01</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-title">Prediction Engine</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Enter a candidate profile. The trained classifier evaluates the '
        'profile and returns a placement assessment with actionable insights.'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # Candidate
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">// candidate</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        name = st.text_input(
            "Student name",
            placeholder="e.g. Meghashree Mohapatra"
        )

    with col2:
        gender = st.selectbox(
            "Gender",
            ["Female", "Male", "Other"]
        )

    with col3:
        branch = st.selectbox(
            "Department",
            ["CSE", "IT", "ECE", "EEE", "MECH", "CIVIL"]
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # --------------------------------------------------------
    # Academic + Skills
    # --------------------------------------------------------

    left, right = st.columns(2)

    with left:

        st.markdown(
            '<div class="section-title">// academic profile</div>',
            unsafe_allow_html=True
        )

        st.markdown('<div class="section-card">', unsafe_allow_html=True)

        a1, a2 = st.columns(2)

        with a1:
            cgpa = st.number_input(
                "CGPA",
                min_value=0.0,
                max_value=10.0,
                value=7.5,
                step=0.1
            )

        with a2:
            age = st.number_input(
                "Age",
                min_value=17,
                max_value=35,
                value=21,
                step=1
            )

        a3, a4 = st.columns(2)

        with a3:
            backlogs = st.number_input(
                "Backlogs",
                min_value=0,
                max_value=20,
                value=0,
                step=1
            )

        with a4:
            aptitude = st.number_input(
                "Aptitude score",
                min_value=0,
                max_value=100,
                value=60,
                step=1
            )

        st.markdown('</div>', unsafe_allow_html=True)

    with right:

        st.markdown(
            '<div class="section-title">// technical & soft skills</div>',
            unsafe_allow_html=True
        )

        st.markdown('<div class="section-card">', unsafe_allow_html=True)

        b1, b2 = st.columns(2)

        with b1:
            coding = st.number_input(
                "Coding score",
                min_value=0.0,
                max_value=10.0,
                value=6.0,
                step=0.1
            )

        with b2:
            communication = st.number_input(
                "Communication",
                min_value=0.0,
                max_value=10.0,
                value=6.0,
                step=0.1
            )

        b3, b4 = st.columns(2)

        with b3:
            projects = st.number_input(
                "Projects",
                min_value=0,
                max_value=20,
                value=2,
                step=1
            )

        with b4:
            internships = st.number_input(
                "Internships",
                min_value=0,
                max_value=10,
                value=0,
                step=1
            )

        b5, b6 = st.columns(2)

        with b5:
            certifications = st.number_input(
                "Certifications",
                min_value=0,
                max_value=20,
                value=1,
                step=1
            )

        with b6:
            soft_skills = st.number_input(
                "Soft skills",
                min_value=0.0,
                max_value=10.0,
                value=6.0,
                step=0.1
            )

        st.markdown('</div>', unsafe_allow_html=True)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    st.markdown("<br>", unsafe_allow_html=True)

    predict = st.button("Predict Placement →")

    if predict:

        data = {
            "Gender": gender,
            "Degree": "B.Tech",
            "Branch": branch,
            "Age": age,
            "CGPA": cgpa,
            "Backlogs": backlogs,
            "Internships": internships,
            "Projects": projects,
            "Certifications": certifications,
            "Coding_Skills": coding,
            "Communication_Skills": communication,
            "Soft_Skills_Rating": soft_skills,
            "Aptitude_Test_Score": aptitude
        }

        try:

            if rf_model is None:
                st.error("No trained classification model found.")
                st.stop()

            prediction, probability = make_prediction(
                rf_model,
                data
            )

            if probability is None:
                probability = 1.0 if prediction == 1 else 0.0

            probability_pct = probability * 100

            readiness = calculate_readiness({
                "cgpa": cgpa,
                "coding": coding,
                "communication": communication,
                "projects": projects,
                "internships": internships,
                "certifications": certifications,
                "aptitude": aptitude
            })

            # ------------------------------------------------
            # Result
            # ------------------------------------------------

            st.markdown(
                '<div class="section-title">// prediction output</div>',
                unsafe_allow_html=True
            )

            if prediction == 1:

                status = "LIKELY PLACED"
                status_class = "status-success"

            else:

                status = "PLACEMENT AT RISK"
                status_class = "status-danger"

            c1, c2, c3 = st.columns(3)

            with c1:

                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">PLACEMENT PROBABILITY</div>
                    <div class="metric-value blue">
                        {probability_pct:.1f}%
                    </div>
                    <div class="status-pill {status_class}">
                        {status}
                    </div>
                    <div class="progress-container">
                        <div class="progress-fill"
                             style="width:{probability_pct}%;">
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with c2:

                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">CAREER READINESS</div>
                    <div class="metric-value green">
                        {readiness}/100
                    </div>
                    <div class="metric-small">
                        Profile strength index
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with c3:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">PLACEMENT STATUS</div>
                    <div class="metric-value green">
                """ + ("LIKELY" if prediction == 1 else "AT RISK") + """
                    </div>
                    <div class="metric-small">
                        Based on trained Random Forest classifier
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ------------------------------------------------
            # Recommendations
            # ------------------------------------------------

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(
                '<div class="section-title">// improvement analysis</div>',
                unsafe_allow_html=True
            )

            recommendations = recommendation({
                "cgpa": cgpa,
                "coding": coding,
                "communication": communication,
                "projects": projects,
                "internships": internships,
                "aptitude": aptitude
            })

            for title, action in recommendations:

                st.markdown(f"""
                <div class="info-box" style="margin-bottom:8px;">
                    <span style="color:#3b82ff;font-weight:600;">
                        {title}
                    </span>
                    <span style="margin-left:15px;">
                        {action}
                    </span>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:

            st.error(
                "Prediction could not be completed. "
                "Make sure the input feature names match the trained model."
            )

            with st.expander("Technical details"):
                st.code(str(e))


# ============================================================
# EDA DASHBOARD
# ============================================================

elif page == "EDA Dashboard":

    st.markdown(
        '<div class="page-kicker">// module 02</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-title">Exploratory Data Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Dataset distributions, placement patterns and model-oriented '
        'statistics from the training corpus.'
        '</div>',
        unsafe_allow_html=True
    )

    data_path = "data/raw/train.csv"

    if os.path.exists(data_path):

        df = pd.read_csv(data_path)

        placed = (
            df["Placement_Status"].astype(str).str.lower() == "placed"
        ).sum()

        placement_rate = (
            placed / len(df) * 100
            if len(df) > 0 else 0
        )

        avg_cgpa = (
            df["CGPA"].mean()
            if "CGPA" in df.columns else 0
        )

        salary = (
            df.loc[
                df["Placement_Status"].astype(str).str.lower() == "placed",
                "Salary"
            ].mean()
            if "Salary" in df.columns else None
        )

        c1, c2, c3, c4 = st.columns(4)

        metrics = [
            ("DATASET ROWS", f"{len(df):,}", ""),
            ("PLACEMENT RATE", f"{placement_rate:.1f}%", ""),
            ("AVERAGE CGPA", f"{avg_cgpa:.2f}", ""),
            ("AVG SALARY", f"₹{salary:.2f} LPA" if salary else "—", "")
        ]

        for col, (label, value, small) in zip(
            [c1, c2, c3, c4], metrics
        ):

            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-small">{small}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        left, right = st.columns(2)

        with left:

            st.markdown(
                '<div class="section-title">// placement split</div>',
                unsafe_allow_html=True
            )

            if "Placement_Status" in df.columns:

                counts = df["Placement_Status"].value_counts()

                st.bar_chart(counts)

        with right:

            st.markdown(
                '<div class="section-title">// cgpa distribution</div>',
                unsafe_allow_html=True
            )

            if "CGPA" in df.columns:

                hist = pd.DataFrame({
                    "CGPA": df["CGPA"]
                })

                st.bar_chart(
                    hist["CGPA"].value_counts().sort_index()
                )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            '<div class="section-title">// dataset preview</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            df.head(20),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.markdown("""
        <div class="info-box">
            Training dataset not found at
            <b>data/raw/train.csv</b>.
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# BATCH UPLOAD
# ============================================================

elif page == "Batch Upload":

    st.markdown(
        '<div class="page-kicker">// MODULE 03</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-title">Batch Upload</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Upload a CSV file containing multiple student profiles '
        'and generate placement predictions in one go.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload student CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            batch_df = pd.read_csv(uploaded_file)

            st.success(
                f"File uploaded successfully — {len(batch_df)} students found."
            )

            st.markdown("### Preview")

            st.dataframe(
                batch_df.head(10),
                use_container_width=True
            )

            if st.button("Generate Batch Predictions →"):

                required_columns = [
                    "Age",
                    "Gender",
                    "Degree",
                    "Branch",
                    "CGPA",
                    "Internships",
                    "Projects",
                    "Coding_Skills",
                    "Communication_Skills",
                    "Aptitude_Test_Score",
                    "Soft_Skills_Rating",
                    "Certifications",
                    "Backlogs"
                ]

                missing_columns = [
                    col for col in required_columns
                    if col not in batch_df.columns
                ]

                if missing_columns:

                    st.error(
                        "Missing columns: "
                        + ", ".join(missing_columns)
                    )

                else:

                    prediction_data = batch_df[required_columns].copy()

                    predictions = rf_model.predict(prediction_data)

                    if hasattr(rf_model, "predict_proba"):
                        probabilities = rf_model.predict_proba(
                            prediction_data
                        )[:, 1]
                    else:
                        probabilities = predictions.astype(float)

                    results = batch_df.copy()

                    results["Placement_Probability"] = (
                        probabilities * 100
                    ).round(2)

                    results["Predicted_Placement"] = predictions

                    results["Predicted_Status"] = results[
                        "Predicted_Placement"
                    ].map({
                        1: "Likely Placed",
                        0: "Unlikely Placed"
                    })

                    st.markdown("### Prediction Results")

                    st.dataframe(
                        results,
                        use_container_width=True
                    )

                    csv = results.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        label="Download Results CSV ↓",
                        data=csv,
                        file_name="placement_predictions.csv",
                        mime="text/csv"
                    )

        except Exception as e:

            st.error(f"Error processing file: {e}")

