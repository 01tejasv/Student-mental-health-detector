from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Student Mental Health Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# FILE PATHS
# Keep app.py, the model, and CSV in the same folder.
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Mental_Health_Model.pkl"
DATASET_PATH = BASE_DIR / "Student Social Media And Mental Health Impact.csv"


# =========================================================
# MODERN BLUE + PURPLE UI
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #1c2140;
        --muted: #737b98;
        --purple: #7257e8;
        --blue: #4285f4;
        --line: #e7e9f5;
        --panel: #ffffff;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 5% 0%, rgba(114, 87, 232, 0.10), transparent 32%),
            radial-gradient(ellipse at 95% 5%, rgba(66, 133, 244, 0.10), transparent 30%),
            #f6f7fc;
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: rgba(246, 247, 252, 0.85);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #211d50 0%, #30266d 55%, #3c3488 100%);
    }

    [data-testid="stSidebar"] * {
        color: #f5f4ff !important;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #d8d6f7 !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.18);
    }

    .hero {
        padding: 1.65rem 1.8rem;
        border-radius: 24px;
        color: white;
        background: linear-gradient(120deg, #33266f 0%, #6046cb 48%, #4285f4 100%);
        box-shadow: 0 16px 38px rgba(72, 67, 161, 0.20);
        margin: 0.25rem 0 1.35rem 0;
    }

    .hero-kicker {
        color: #e3e1ff;
        text-transform: uppercase;
        letter-spacing: 0.13em;
        font-size: 0.75rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .hero h1 {
        font-family: 'Manrope', sans-serif;
        font-size: clamp(1.8rem, 3.2vw, 2.7rem);
        line-height: 1.15;
        font-weight: 800;
        color: white;
        margin: 0 0 0.6rem 0;
    }

    .hero p {
        color: #eeedff;
        font-size: 1rem;
        max-width: 760px;
        margin: 0;
    }

    .section-heading {
        font-family: 'Manrope', sans-serif;
        color: #292650;
        font-weight: 800;
        font-size: 1.25rem;
        margin: 0.45rem 0 0.2rem 0;
    }

    .section-subtitle {
        color: var(--muted);
        margin-bottom: 1rem;
        font-size: 0.92rem;
    }

    div[data-testid="stForm"] {
        background: rgba(255,255,255,0.94);
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 1.35rem 1.4rem 1rem 1.4rem;
        box-shadow: 0 8px 26px rgba(38, 42, 94, 0.055);
    }

    div[data-testid="stForm"] h3 {
        color: #44358e;
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
        font-size: 1rem;
        padding-top: 0.25rem;
    }

    div[data-testid="stFormSubmitButton"] > button {
        color: white;
        border: 0;
        border-radius: 12px;
        padding: 0.8rem 1.2rem;
        font-weight: 700;
        background: linear-gradient(100deg, #7257e8 0%, #4285f4 100%);
        box-shadow: 0 7px 18px rgba(93, 87, 218, 0.23);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }

    div[data-testid="stFormSubmitButton"] > button:hover {
        color: white;
        transform: translateY(-1px);
        box-shadow: 0 10px 22px rgba(93, 87, 218, 0.30);
    }

    div[data-testid="stSelectbox"] label,
    div[data-testid="stNumberInput"] label,
    div[data-testid="stSlider"] label,
    div[data-testid="stSelectSlider"] label {
        color: #454b6b;
        font-weight: 600;
    }

    [data-testid="stMetric"] {
        background: white;
        padding: 1.1rem 1.2rem;
        border: 1px solid var(--line);
        border-radius: 17px;
        box-shadow: 0 6px 20px rgba(38, 42, 94, 0.045);
    }

    [data-testid="stMetricLabel"] {
        color: #737b98 !important;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: #30266d !important;
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
    }

    div[data-testid="stAlert"] {
        border-radius: 13px;
    }

    .mini-card {
        background: white;
        border: 1px solid var(--line);
        border-radius: 17px;
        padding: 1rem 1.1rem;
        height: 100%;
    }

    .mini-card .label {
        color: #737b98;
        font-size: 0.82rem;
        margin-bottom: 0.3rem;
    }

    .mini-card .value {
        color: #30266d;
        font-family: 'Manrope', sans-serif;
        font-size: 1.3rem;
        font-weight: 800;
    }

    .pill {
        display: inline-block;
        border-radius: 999px;
        padding: 0.32rem 0.7rem;
        color: #4e3cb0;
        background: #efedff;
        font-size: 0.78rem;
        font-weight: 700;
    }

    .footer-note {
        color: #7a819a;
        font-size: 0.83rem;
        text-align: center;
        padding: 1.1rem 0 0.5rem 0;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD MODEL AND DATASET
# =========================================================
@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


@st.cache_data
def load_dataset(dataset_path: str):
    return pd.read_csv(dataset_path)


try:
    if not MODEL_PATH.is_file():
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()

    if not DATASET_PATH.is_file():
        st.error(f"Dataset file not found: {DATASET_PATH}")
        st.stop()

    model = load_model(str(MODEL_PATH))
    df = load_dataset(str(DATASET_PATH))

except Exception as error:
    st.error("Unable to load the model or dataset.")
    st.exception(error)
    st.stop()


# =========================================================
# VALIDATE REQUIRED DATASET COLUMNS
# =========================================================
required_dataset_columns = [
    "Country",
    "Gender",
    "Academic_Level",
    "Most_Used_Platform",
    "Purpose_Of_Use",
    "Age",
    "Mental_Health_Score",
    "Study_Hours",
    "Avg_Daily_Usage_Hours",
    "Daily_Unlocks",
    "Physical_Activity_Hours",
    "Sleep_Hours_Per_Night",
    "Stress_Level",
]

missing_dataset_columns = [
    column for column in required_dataset_columns if column not in df.columns
]

if missing_dataset_columns:
    st.error(
        "The dataset is missing required columns: "
        + ", ".join(missing_dataset_columns)
    )
    st.stop()


# =========================================================
# COUNTRY GROUPING — PRESERVED FROM TRAINING NOTEBOOK
# =========================================================
top_countries = df["Country"].value_counts().index[:10].tolist()


def group_countries(country):
    """Keep a top-10 country unchanged; map all others to Other."""
    if pd.isna(country):
        return "Other"
    if country in top_countries:
        return country
    return "Other"


# Create Grouped_country even though it is not present in the original CSV.
df["Grouped_country"] = df["Country"].apply(group_countries)


# =========================================================
# INPUT OPTIONS AND VALIDATION
# =========================================================
def unique_values(column):
    return sorted(df[column].dropna().astype(str).unique().tolist())


genders = unique_values("Gender")
academic_levels = unique_values("Academic_Level")
platforms = unique_values("Most_Used_Platform")
purposes = unique_values("Purpose_Of_Use")
countries = unique_values("Country")
stress_levels = ["Low", "Medium", "High", "Very High"]

if not all([genders, academic_levels, platforms, purposes, countries]):
    st.error("One or more dropdown fields have no valid options in the dataset.")
    st.stop()

age_min = max(1, int(df["Age"].min()))
age_max = max(age_min, int(df["Age"].max()))
default_age = min(max(21, age_min), age_max)


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 🧠 Mindful")
    st.markdown("### Student Wellbeing")
    st.write(
        "A machine-learning project exploring how daily habits and social "
        "media use relate to a predicted mental-health score."
    )
    st.divider()
    st.markdown("**WHAT THIS APP USES**")
    st.markdown("📚 Academic routine")
    st.markdown("📱 Social media habits")
    st.markdown("🌙 Sleep and physical activity")
    st.markdown("🧩 Self-reported stress")
    st.divider()
    st.markdown("**PLEASE NOTE**")
    st.caption(
        "This is an educational prediction tool, not a medical device. "
        "It does not diagnose a mental-health condition."
    )


# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Student wellbeing • ML insights</div>
        <h1>Understand your everyday wellbeing</h1>
        <p>
            Enter a few details about your routine to explore the score predicted
            by the trained model. Your habits are only one part of your wellbeing
            story.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Compact dataset context metrics.
metric1, metric2, metric3 = st.columns(3)
with metric1:
    st.metric("Student records", f"{len(df):,}")
with metric2:
    st.metric("Platforms in dataset", f"{df['Most_Used_Platform'].nunique():,}")
with metric3:
    st.metric(
        "Observed score range",
        f"{df['Mental_Health_Score'].min():g}–{df['Mental_Health_Score'].max():g}",
    )

st.markdown('<div class="section-heading">Build your student profile</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Complete the sections below, then generate your model prediction.</div>',
    unsafe_allow_html=True,
)


# =========================================================
# INPUT FORM
# =========================================================
with st.form("mental_health_form"):
    st.markdown("### 👤  Personal & academic details")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age",
            min_value=age_min,
            max_value=age_max,
            value=default_age,
            step=1,
        )

    with col2:
        gender = st.selectbox("Gender", genders)

    with col3:
        academic_level = st.selectbox("Academic level", academic_levels)

    st.markdown("### 🌍  Location")
    country = st.selectbox(
        "Country",
        countries,
        index=countries.index("India") if "India" in countries else 0,
    )
    selected_grouped_country = group_countries(country)
    st.caption(f"Model country category: **{selected_grouped_country}**")

    st.markdown("### 📱  Social media habits")
    col4, col5, col6 = st.columns(3)

    with col4:
        platform = st.selectbox("Most-used platform", platforms)

    with col5:
        purpose = st.selectbox("Primary purpose", purposes)

    with col6:
        usage_hours = st.number_input(
            "Average daily usage (hours)",
            min_value=0.0,
            max_value=24.0,
            value=4.0,
            step=0.5,
        )

    daily_unlocks = st.number_input(
        "Daily phone unlocks",
        min_value=0,
        max_value=2000,
        value=100,
        step=5,
    )

    st.markdown("### 🌙  Lifestyle & wellbeing")
    col7, col8, col9 = st.columns(3)

    with col7:
        study_hours = st.number_input(
            "Study hours per day",
            min_value=0.0,
            max_value=24.0,
            value=4.0,
            step=0.5,
        )

    with col8:
        physical_activity = st.number_input(
            "Physical activity (hours/day)",
            min_value=0.0,
            max_value=24.0,
            value=1.0,
            step=0.5,
        )

    with col9:
        sleep_hours = st.number_input(
            "Sleep per night (hours)",
            min_value=0.0,
            max_value=24.0,
            value=7.0,
            step=0.5,
        )

    stress_level = st.select_slider(
        "Current stress level",
        options=stress_levels,
        value="Medium",
    )

    submitted = st.form_submit_button(
        "✨  Generate my prediction",
        use_container_width=True,
    )


# =========================================================
# PREDICTION + VISUAL ANALYTICS
# =========================================================
if submitted:
    grouped_country = group_countries(country)

    # Feature names and order exactly match the saved sklearn pipeline.
    input_data = pd.DataFrame(
        [
            {
                "Study_Hours": float(study_hours),
                "Age": int(age),
                "Avg_Daily_Usage_Hours": float(usage_hours),
                "Daily_Unlocks": int(daily_unlocks),
                "Physical_Activity_Hours": float(physical_activity),
                "Sleep_Hours_Per_Night": float(sleep_hours),
                "Stress_Level": stress_level,
                "Gender": gender,
                "Academic_Level": academic_level,
                "Most_Used_Platform": platform,
                "Purpose_Of_Use": purpose,
                "Grouped_country": grouped_country,
            }
        ],
        columns=[
            "Study_Hours",
            "Age",
            "Avg_Daily_Usage_Hours",
            "Daily_Unlocks",
            "Physical_Activity_Hours",
            "Sleep_Hours_Per_Night",
            "Stress_Level",
            "Gender",
            "Academic_Level",
            "Most_Used_Platform",
            "Purpose_Of_Use",
            "Grouped_country",
        ],
    )

    st.divider()
    st.markdown('<div class="section-heading">Your prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">A model estimate based on the information entered above.</div>',
        unsafe_allow_html=True,
    )

    try:
        prediction = float(model.predict(input_data)[0])

        if not pd.notna(prediction):
            st.error("The model returned an invalid prediction.")
            st.stop()

        target_min = float(df["Mental_Health_Score"].min())
        target_max = float(df["Mental_Health_Score"].max())
        target_mean = float(df["Mental_Health_Score"].mean())
        target_median = float(df["Mental_Health_Score"].median())

        score_col, range_col = st.columns([1, 1.5], gap="large")

        with score_col:
            st.markdown(
                f"""
                <div class="mini-card">
                    <div class="label">PREDICTED MENTAL HEALTH SCORE</div>
                    <div class="value" style="font-size:2.25rem;">{prediction:.2f}</div>
                    <div style="margin-top:0.55rem;">
                        <span class="pill">Model estimate</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if target_max > target_min:
                normalized_score = (prediction - target_min) / (target_max - target_min)
                normalized_score = max(0.0, min(1.0, normalized_score))
                st.progress(normalized_score, text="Position within observed dataset range")

        with range_col:
            st.markdown("#### Score context")
            st.write(
                f"The dataset's observed scores range from **{target_min:.2f}** "
                f"to **{target_max:.2f}**."
            )
            st.write(
                f"Dataset average: **{target_mean:.2f}** · "
                f"Dataset median: **{target_median:.2f}**"
            )
            st.caption(
                "The score is interpreted only in relation to this dataset. "
                "It is not a clinical assessment, diagnosis, or validated personal risk level."
            )

        st.markdown("### 📈 Score distribution in the dataset")
        st.caption("The chart shows how the dataset's recorded scores are distributed; the prediction is shown above.")

        # Histogram-style chart with robust handling for a constant target.
        if target_max > target_min:
            bin_edges = pd.cut(
                df["Mental_Health_Score"],
                bins=10,
                include_lowest=True,
                duplicates="drop",
            )
            distribution = (
                bin_edges.value_counts(sort=False)
                .rename_axis("Score interval")
                .reset_index(name="Student records")
            )
            distribution["Score interval"] = distribution["Score interval"].astype(str)
            st.bar_chart(
                distribution.set_index("Score interval")["Student records"],
                use_container_width=True,
            )
        else:
            st.info("The dataset contains only one unique mental-health score, so a distribution chart is not available.")

        st.markdown("### 🧭 Your routine at a glance")
        st.caption(
            "These values are compared with dataset averages for context only; "
            "they are not targets or judgments."
        )

        comparison = pd.DataFrame(
            {
                "Measure": [
                    "Study hours/day",
                    "Social media hours/day",
                    "Physical activity hours/day",
                    "Sleep hours/night",
                ],
                "Your input": [
                    float(study_hours),
                    float(usage_hours),
                    float(physical_activity),
                    float(sleep_hours),
                ],
                "Dataset average": [
                    float(df["Study_Hours"].mean()),
                    float(df["Avg_Daily_Usage_Hours"].mean()),
                    float(df["Physical_Activity_Hours"].mean()),
                    float(df["Sleep_Hours_Per_Night"].mean()),
                ],
            }
        ).set_index("Measure")
        st.bar_chart(comparison, use_container_width=True)

        st.markdown("### 🪪 Profile summary")
        summary1, summary2, summary3 = st.columns(3)
        with summary1:
            st.metric("Age", f"{int(age)}")
            st.metric("Academic level", academic_level)
        with summary2:
            st.metric("Daily social media", f"{usage_hours:.1f} hrs")
            st.metric("Daily unlocks", f"{int(daily_unlocks):,}")
        with summary3:
            st.metric("Sleep per night", f"{sleep_hours:.1f} hrs")
            st.metric("Stress input", stress_level)

        st.caption(
            f"Platform: {platform} · Purpose: {purpose} · "
            f"Country selected: {country} · Country group sent to model: {grouped_country}"
        )

    except Exception as error:
        st.error("Prediction failed. Check the model and input features.")
        st.exception(error)


# =========================================================
# FOOTER
# =========================================================
st.divider()
st.markdown(
    """
    <div class="footer-note">
        Student Mental Health Detector · Machine Learning Project<br>
        For educational use only — not a substitute for professional mental-health support.
    </div>
    """,
    unsafe_allow_html=True,
)
