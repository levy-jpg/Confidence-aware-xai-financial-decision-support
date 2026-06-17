"""
Confidence-Aware XAI Streamlit Prototype
Author: Levy Thiga Kariuki
Student Number: G20893080
"""

import time
import base64
from datetime import datetime
from html import escape
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RAW_DATA_DIR = DATA_DIR / "raw"
RESPONSE_FILE = DATA_DIR / "responses" / "user_study_responses.csv"
CONFIDENCE_TRAINING_FILE = DATA_DIR / "simulated_behavioural_confidence_data.csv"
GERMAN_CREDIT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
LOCAL_GERMAN_CREDIT_FILE = RAW_DATA_DIR / "german.data"
CONFIDENCE_TARGET_COLUMN = "confidence_level"
FALLBACK_CONFIDENCE_FEATURES = [
    "decision_time",
    "click_count",
    "scroll_depth",
    "hover_count",
    "explanation_view_time",
]
RESPONSE_COLUMNS = [
    "timestamp",
    "session_id",
    "participant_id",
    "profile_id",
    "dataset_index",
    "profile_type",
    "condition",
    "user_prediction",
    "user_confidence",
    "ai_prediction",
    "probability_bad_credit",
    "user_agreed_with_ai",
    "decision_time",
    "interaction_count",
    "ai_prediction_review_time",
    "explanation_reading_time",
    "explanation_view_time",
    "confidence_scroll_depth_proxy",
    "confidence_hover_count_proxy",
    "predicted_confidence",
    "reported_confidence_level",
    "adaptation_signal",
    "explanation_depth",
    "top_features_shown",
    "trust",
    "understanding",
    "usefulness",
    "reliance",
    "comments",
]


# ---------------------------------------------------------------------------
# Page setup and visual styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Confidence-Aware XAI Prototype",
    page_icon="CA",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        :root {
            --bg-soft: color-mix(in srgb, var(--secondary-background-color, #f6f8fb) 86%, var(--primary-color, #7c3aed));
            --border: color-mix(in srgb, var(--text-color, #172033) 18%, transparent);
            --ink: var(--text-color, #172033);
            --muted: color-mix(in srgb, var(--text-color, #172033) 68%, transparent);
            --panel: var(--secondary-background-color, #ffffff);
            --hero-start: color-mix(in srgb, var(--background-color, #ffffff) 92%, var(--accent) 8%);
            --hero-end: color-mix(in srgb, var(--secondary-background-color, #eef4ff) 82%, var(--accent-3) 18%);
            --step-bg: #e8eef8;
            --step-ink: #22304a;
            --good: #0f7a55;
            --bad: #b42318;
            --accent: #7c3aed;
            --accent-2: #f97316;
            --accent-3: #06b6d4;
            --glow: rgba(124, 58, 237, .24);
            --shadow: 0 10px 28px rgba(15, 23, 42, .08);
        }

        @media (prefers-color-scheme: dark) {
            :root {
                --bg-soft: #1f2937;
                --border: #3b4758;
                --ink: #f3f7fb;
                --muted: #c7d0dd;
                --panel: #111827;
                --hero-start: #111827;
                --hero-end: #1f2937;
                --step-bg: #263244;
                --step-ink: #edf4ff;
                --good: #34d399;
                --bad: #f87171;
                --accent: #a78bfa;
                --accent-2: #fb923c;
                --accent-3: #22d3ee;
                --glow: rgba(167, 139, 250, .22);
                --shadow: 0 12px 32px rgba(0, 0, 0, .22);
            }
        }

        .main .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .section-gap {
            height: 1.15rem;
        }

        .mode-badge-wrap {
            margin-top: .9rem;
            margin-bottom: 1.05rem;
        }

        h1, h2, h3 {
            color: var(--ink);
            letter-spacing: 0;
        }

        .hero {
            position: relative;
            overflow: hidden;
            padding: 1.35rem 1.5rem 1.25rem 1.5rem;
            border: 1px solid var(--border);
            border-radius: 8px;
            background:
                radial-gradient(circle at 88% 20%, color-mix(in srgb, var(--accent-2) 18%, transparent), transparent 28%),
                linear-gradient(135deg, color-mix(in srgb, var(--panel) 86%, transparent), color-mix(in srgb, var(--hero-end) 72%, transparent));
            backdrop-filter: blur(10px);
            margin-bottom: 1.25rem;
            box-shadow: var(--shadow);
            transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
        }

        .hero:hover {
            transform: translateY(-1px);
            border-color: color-mix(in srgb, var(--accent) 42%, var(--border));
        }

        .hero::before {
            content: "";
            position: absolute;
            inset: 0 0 auto 0;
            height: 4px;
            background: linear-gradient(90deg, var(--accent), var(--accent-2), var(--accent-3), var(--accent));
        }

        .hero::after {
            content: "";
            position: absolute;
            top: -38px;
            right: -42px;
            width: 210px;
            height: 120px;
            background:
                radial-gradient(circle at 30% 45%, color-mix(in srgb, var(--accent-2) 24%, transparent), transparent 36%),
                radial-gradient(circle at 68% 38%, var(--glow), transparent 42%),
                radial-gradient(circle at 52% 72%, color-mix(in srgb, var(--accent-3) 20%, transparent), transparent 38%);
            filter: blur(2px);
            pointer-events: none;
        }

        .hero-content {
            position: relative;
            z-index: 1;
        }

        .ai-tag {
            display: inline-flex;
            align-items: center;
            gap: .4rem;
            padding: .2rem .55rem;
            border: 1px solid color-mix(in srgb, var(--accent) 50%, var(--border));
            border-radius: 999px;
            color: var(--ink);
            background: color-mix(in srgb, var(--panel) 82%, transparent);
            font-size: .76rem;
            font-weight: 750;
            margin-bottom: .55rem;
        }

        .ai-tag-dot {
            width: .52rem;
            height: .52rem;
            border-radius: 999px;
            background: linear-gradient(135deg, var(--accent), var(--accent-2), var(--accent-3));
            box-shadow: 0 0 18px var(--glow);
        }

        .hero h1 {
            margin: 0 0 .35rem 0;
            font-size: 2rem;
        }

        .hero p {
            margin: 0;
            color: var(--muted);
            font-size: 1rem;
        }

        .metric-tile {
            padding: .8rem .9rem;
            border: 1px solid var(--border);
            border-radius: 8px;
            background: color-mix(in srgb, var(--panel) 72%, transparent);
            backdrop-filter: blur(8px);
            min-height: 92px;
            box-shadow: 0 1px 0 rgba(15, 23, 42, .04);
            transition: transform .16s ease, border-color .16s ease, box-shadow .16s ease;
        }

        .metric-tile:hover {
            transform: translateY(-1px);
            border-color: color-mix(in srgb, var(--accent) 36%, var(--border));
            box-shadow: var(--shadow);
        }

        .metric-label {
            color: var(--muted);
            font-size: .78rem;
            text-transform: uppercase;
            font-weight: 700;
            margin-bottom: .25rem;
        }

        .metric-value {
            color: var(--ink);
            font-size: 1.2rem;
            font-weight: 750;
            line-height: 1.25;
            overflow-wrap: anywhere;
        }

        .comparison-panel {
            margin-top: 1rem;
            margin-bottom: 1rem;
            padding: .95rem 1rem;
            border: 1px solid var(--border);
            border-radius: 8px;
            background: color-mix(in srgb, var(--panel) 78%, transparent);
            backdrop-filter: blur(8px);
        }

        .comparison-title {
            color: var(--ink);
            font-weight: 800;
            margin-bottom: .7rem;
        }

        .comparison-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: .7rem;
        }

        .comparison-item {
            min-height: 72px;
            padding: .65rem .7rem;
            border: 1px solid var(--border);
            border-radius: 8px;
            background: color-mix(in srgb, var(--bg-soft) 86%, transparent);
            overflow: hidden;
        }

        .comparison-label {
            color: var(--muted);
            font-size: .72rem;
            line-height: 1.15;
            text-transform: uppercase;
            font-weight: 750;
            margin-bottom: .28rem;
        }

        .comparison-value {
            color: var(--ink);
            font-size: .95rem;
            line-height: 1.2;
            font-weight: 800;
            overflow-wrap: anywhere;
        }

        .good-risk {
            color: var(--good);
            font-weight: 800;
        }

        .bad-risk {
            color: var(--bad);
            font-weight: 800;
        }

        .step-pill {
            display: inline-block;
            padding: .18rem .55rem;
            border-radius: 999px;
            background: var(--step-bg);
            color: var(--step-ink);
            font-size: .78rem;
            font-weight: 750;
            margin-bottom: .45rem;
        }

        .risk-meter {
            padding: 1rem 1.05rem;
            border: 1px solid var(--border);
            border-radius: 8px;
            background: color-mix(in srgb, var(--panel) 78%, transparent);
            backdrop-filter: blur(8px);
            margin: .65rem 0 1rem 0;
            box-shadow: var(--shadow);
        }

        .risk-meter-head {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: .55rem;
            color: var(--ink);
            font-weight: 750;
        }

        .risk-track {
            height: 14px;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--good) 0%, var(--accent-2) 55%, var(--bad) 100%);
            overflow: hidden;
            position: relative;
        }

        .risk-marker {
            position: absolute;
            top: -4px;
            width: 4px;
            height: 22px;
            border-radius: 999px;
            background: var(--ink);
            box-shadow: 0 0 0 3px color-mix(in srgb, var(--panel) 80%, transparent);
        }

        .risk-scale {
            display: flex;
            justify-content: space-between;
            margin-top: .35rem;
            color: var(--muted);
            font-size: .76rem;
        }

        .impact-up {
            color: var(--bad);
            font-weight: 750;
        }

        .impact-down {
            color: var(--good);
            font-weight: 750;
        }

        div[data-testid="stProgress"] > div > div {
            background-color: var(--accent);
        }

        div.stButton > button {
            min-height: 2.75rem;
            border-radius: 8px;
            font-weight: 750;
        }

        div[data-testid="stHorizontalBlock"] {
            gap: 1rem;
        }

        .helper-text {
            color: var(--muted);
            font-size: .9rem;
            margin: -.2rem 0 .8rem 0;
        }

        .action-spacer {
            height: 1.72rem;
        }

        .sidebar-action {
            display: block;
            width: 100%;
            padding: .54rem .72rem;
            border-radius: 8px;
            color: var(--ink) !important;
            text-align: center;
            text-decoration: none !important;
            font-weight: 750;
            font-size: .9rem;
            margin: .35rem 0 .45rem 0;
            border: 1px solid var(--border);
            transition: transform .16s ease, filter .16s ease, box-shadow .16s ease;
        }

        .sidebar-action:hover {
            transform: translateY(-1px);
            filter: brightness(1.04);
            box-shadow: var(--shadow);
        }

        .sidebar-blue {
            background: color-mix(in srgb, #60a5fa 46%, var(--panel));
            border-color: color-mix(in srgb, #2563eb 46%, var(--border));
        }

        .sidebar-red {
            background: color-mix(in srgb, #f87171 42%, var(--panel));
            border-color: color-mix(in srgb, #dc2626 42%, var(--border));
        }

        .sidebar-green {
            background: color-mix(in srgb, #4ade80 44%, var(--panel));
            border-color: color-mix(in srgb, #16a34a 42%, var(--border));
        }

        @media (max-width: 760px) {
            .hero h1 {
                font-size: 1.55rem;
                line-height: 1.2;
            }

            .metric-tile {
                min-height: auto;
            }

            .comparison-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }

            .comparison-value {
                font-size: .88rem;
            }

            .risk-meter-head {
                flex-direction: column;
                gap: .2rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data and model loading
# ---------------------------------------------------------------------------

@st.cache_resource
def load_models():
    """Load trained models and preprocessing objects once per Streamlit session."""
    return (
        joblib.load(MODELS_DIR / "xgboost_credit_model.pkl"),
        joblib.load(MODELS_DIR / "label_encoders.pkl"),
        joblib.load(MODELS_DIR / "confidence_model.pkl"),
        joblib.load(MODELS_DIR / "confidence_scaler.pkl"),
        joblib.load(MODELS_DIR / "confidence_label_encoder.pkl"),
    )


@st.cache_data
def load_profiles():
    """Load the curated applicant profiles used in the user study."""
    return pd.read_csv(DATA_DIR / "selected_applicant_profiles.csv")


@st.cache_data
def load_dataset():
    """Load and encode the German Credit dataset to match the trained model."""
    columns = [
        "checking_account_status", "duration_months", "credit_history", "purpose",
        "credit_amount", "savings_account", "employment_since", "installment_rate",
        "personal_status_sex", "other_debtors", "present_residence_since", "property",
        "age", "other_installment_plans", "housing", "existing_credits", "job",
        "people_liable", "telephone", "foreign_worker", "target",
    ]

    try:
        df = pd.read_csv(GERMAN_CREDIT_URL, sep=" ", names=columns)
    except Exception:
        df = pd.read_csv(LOCAL_GERMAN_CREDIT_FILE, sep=" ", names=columns)

    df["target"] = df["target"].map({1: 0, 2: 1})

    model_df = df.copy()
    for col, encoder in label_encoders.items():
        model_df[col] = encoder.transform(model_df[col])

    X_data = model_df.drop("target", axis=1)
    y_data = model_df["target"]
    return df, X_data, y_data


final_model, label_encoders, confidence_model, confidence_scaler, confidence_label_encoder = load_models()
profiles = load_profiles()
df_raw, X, y = load_dataset()


def get_confidence_feature_names():
    """Return the full ordered feature schema expected by the confidence model."""
    for fitted_object in (confidence_scaler, confidence_model):
        feature_names = getattr(fitted_object, "feature_names_in_", None)
        if feature_names is not None:
            return list(feature_names)

    if CONFIDENCE_TRAINING_FILE.exists():
        training_columns = pd.read_csv(CONFIDENCE_TRAINING_FILE, nrows=0).columns.tolist()
        feature_columns = [
            column for column in training_columns
            if column != CONFIDENCE_TARGET_COLUMN
        ]
        if feature_columns:
            return feature_columns

    return FALLBACK_CONFIDENCE_FEATURES.copy()


confidence_feature_names = get_confidence_feature_names()


# ---------------------------------------------------------------------------
# Explanation engine
# ---------------------------------------------------------------------------

@st.cache_resource
def create_shap_values(_model, X_data):
    """
    Create local contribution values for each prediction.

    XGBoost's built-in pred_contribs path is used for the deployed model because
    some SHAP/XGBoost version combinations fail when parsing XGBoost base_score.
    """
    if hasattr(_model, "get_booster"):
        import xgboost as xgb

        booster = _model.get_booster()
        if booster.feature_names:
            dmatrix = xgb.DMatrix(X_data, feature_names=list(X_data.columns))
        else:
            dmatrix = xgb.DMatrix(X_data)

        contributions = booster.predict(dmatrix, pred_contribs=True)
        return None, contributions[:, :-1]

    explainer = shap.TreeExplainer(_model)
    return explainer, explainer.shap_values(X_data)


explainer, shap_values = create_shap_values(final_model, X)


# ---------------------------------------------------------------------------
# Human-readable labels and decoding helpers
# ---------------------------------------------------------------------------

DM_TO_GBP_APPROX = 0.75


def dm_to_gbp_label(amount):
    return f"approx. GBP {amount * DM_TO_GBP_APPROX:,.0f}"


feature_name_map = {
    "checking_account_status": "Checking account status",
    "duration_months": "Loan duration",
    "credit_history": "Credit history",
    "purpose": "Loan purpose",
    "credit_amount": "Credit amount",
    "savings_account": "Savings account status",
    "employment_since": "Employment duration",
    "installment_rate": "Instalment rate",
    "personal_status_sex": "Personal status / sex",
    "other_debtors": "Other debtors or guarantors",
    "present_residence_since": "Years at current residence",
    "property": "Property status",
    "age": "Age",
    "other_installment_plans": "Other instalment plans",
    "housing": "Housing status",
    "existing_credits": "Number of existing credits",
    "job": "Job type",
    "people_liable": "Number of dependants",
    "telephone": "Telephone registered",
    "foreign_worker": "Foreign worker status",
}

value_maps = {
    "checking_account_status": {
        "A11": "Negative balance (less than approx. GBP 0)",
        "A12": "Low positive balance (approx. GBP 0 to GBP 150)",
        "A13": "Strong positive balance (greater than or equal to approx. GBP 150)",
        "A14": "No checking account",
    },
    "credit_history": {
        "A30": "no credits taken / all credits paid back duly",
        "A31": "all credits at this bank paid back duly",
        "A32": "existing credits paid back duly until now",
        "A33": "delay in paying off in the past",
        "A34": "critical account / other credits existing",
    },
    "purpose": {
        "A40": "car (new)", "A41": "car (used)", "A42": "furniture/equipment",
        "A43": "radio/television", "A44": "domestic appliances", "A45": "repairs",
        "A46": "education", "A47": "vacation", "A48": "retraining",
        "A49": "business", "A410": "others",
    },
    "savings_account": {
        "A61": "Low savings (less than approx. GBP 75)",
        "A62": "Moderate savings (approx. GBP 75 to GBP 375)",
        "A63": "High savings (approx. GBP 375 to GBP 750)",
        "A64": "Very high savings (greater than or equal to approx. GBP 750)",
        "A65": "No savings information",
    },
    "employment_since": {
        "A71": "unemployed", "A72": "less than 1 year", "A73": "1 to 4 years",
        "A74": "4 to 7 years", "A75": "greater than 7 years",
    },
    "property": {
        "A121": "real estate",
        "A122": "building society savings agreement / life insurance",
        "A123": "car or other property",
        "A124": "unknown / no property",
    },
    "housing": {"A151": "rent", "A152": "own", "A153": "for free"},
    "job": {
        "A171": "unemployed / unskilled non-resident",
        "A172": "unskilled resident",
        "A173": "skilled employee / official",
        "A174": "management / self-employed / highly qualified",
    },
    "other_installment_plans": {"A141": "bank", "A142": "stores", "A143": "none"},
    "other_debtors": {"A101": "none", "A102": "co-applicant", "A103": "guarantor"},
    "telephone": {"A191": "none", "A192": "yes, registered"},
    "foreign_worker": {"A201": "yes", "A202": "no"},
}

reason_templates = {
    "checking_account_status": "Checking account status can indicate the applicant's short-term liquidity and ability to absorb repayment shocks.",
    "duration_months": "Longer loan durations can increase uncertainty because the repayment period extends over more future events.",
    "credit_history": "Credit history provides evidence about previous repayment behaviour and reliability.",
    "purpose": "Loan purpose can affect risk because some purposes are associated with different repayment patterns.",
    "credit_amount": "Larger credit amounts can increase exposure because more money must be repaid.",
    "savings_account": "Savings information can indicate the applicant's financial buffer if income or expenses change.",
    "employment_since": "Employment duration can indicate income stability and continuity.",
    "installment_rate": "A higher instalment burden can leave less disposable income for unexpected expenses.",
    "personal_status_sex": "This feature reflects demographic groupings in the original dataset and should be interpreted cautiously.",
    "other_debtors": "A co-applicant or guarantor can change risk because repayment responsibility may be shared or supported.",
    "present_residence_since": "Residence duration can indicate stability in the applicant's living situation.",
    "property": "Property status can indicate available assets or financial security.",
    "age": "Age can influence risk patterns in the historical dataset, although it should be interpreted cautiously.",
    "other_installment_plans": "Other instalment plans can indicate existing repayment commitments outside this credit application.",
    "housing": "Housing status can affect financial stability and recurring living costs.",
    "existing_credits": "Existing credits can indicate current debt load and repayment obligations.",
    "job": "Job type can indicate income regularity and employment security.",
    "people_liable": "More dependants can increase financial pressure on the applicant's disposable income.",
    "telephone": "Telephone registration is a historical dataset signal and should be treated as a weak contextual feature.",
    "foreign_worker": "Foreign worker status is part of the original dataset and should be interpreted cautiously due to fairness considerations.",
}


def decode_feature_value(feature, encoded_value):
    """Convert encoded model inputs back into participant-readable values."""
    if feature == "credit_amount":
        return dm_to_gbp_label(float(encoded_value))
    if feature == "duration_months":
        return f"{int(encoded_value)} months"
    if feature == "age":
        return f"{int(encoded_value)} years"
    if feature == "installment_rate":
        return f"{int(encoded_value)}% of disposable income"
    if feature == "present_residence_since":
        return f"{int(encoded_value)} year(s)"
    if feature == "existing_credits":
        return f"{int(encoded_value)} existing credit(s)"
    if feature == "people_liable":
        return f"{int(encoded_value)} dependant(s)"

    if feature in label_encoders:
        original_code = label_encoders[feature].inverse_transform([int(encoded_value)])[0]
        return value_maps.get(feature, {}).get(original_code, original_code)

    return encoded_value


# ---------------------------------------------------------------------------
# Study logic and adaptive explanation helpers
# ---------------------------------------------------------------------------

def initialise_session_state():
    """Create all Streamlit session fields used across the study flow."""
    defaults = {
        "session_id": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "participant_id": None,
        "consent_accepted": False,
        "start_time": time.time(),
        "interaction_count": 0,
        "submitted_initial": False,
        "ai_reveal_time": None,
        "explanation_display_time": None,
        "ai_prediction_review_time": None,
        "explanation_reading_time": None,
        "explanation_generated": False,
        "evaluation_submitted": False,
        "selected_condition": None,
        "adaptation_trace": "",
        "confidence_feature_inputs": {},
        "confidence_features_used": confidence_feature_names,
        "reviewed_profile_before_explanation": False,
        "compared_ai_prediction_before_explanation": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if st.session_state.participant_id is None or not str(st.session_state.participant_id).strip():
        st.session_state.participant_id = get_current_participant_id()


def reset_case_state():
    """Reset downstream state when the participant switches applicant profile."""
    st.session_state.start_time = time.time()
    st.session_state.interaction_count = 0
    st.session_state.submitted_initial = False
    st.session_state.ai_reveal_time = None
    st.session_state.explanation_display_time = None
    st.session_state.ai_prediction_review_time = None
    st.session_state.explanation_reading_time = None
    st.session_state.explanation_generated = False
    st.session_state.evaluation_submitted = False
    st.session_state.selected_condition = None
    st.session_state.adaptation_trace = ""
    st.session_state.confidence_feature_inputs = {}
    st.session_state.confidence_features_used = confidence_feature_names
    st.session_state.reviewed_profile_before_explanation = False
    st.session_state.compared_ai_prediction_before_explanation = False


def start_new_participant():
    """Assign a new participant ID and restart the study flow."""
    current_participant_id = st.session_state.get("participant_id", "")
    reset_case_state()
    st.session_state.session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    st.session_state.participant_id = get_next_participant_id(current_participant_id=current_participant_id)
    st.session_state.consent_accepted = False
    if "consent_checkbox" in st.session_state:
        st.session_state.consent_checkbox = False


def repair_incomplete_explanation_state():
    """Clear stale explanation state left over from older app versions."""
    if not st.session_state.explanation_generated:
        return

    required_keys = [
        "predicted_confidence",
        "applied_confidence_signal",
        "explanation_depth",
        "explanation_rationale",
        "explanation_style",
        "explanation_summary",
        "show_probability_context",
        "show_extra_guidance",
        "explanation_df",
        "top_n",
        "ai_prediction_review_time",
        "explanation_display_time",
        "confidence_feature_inputs",
        "confidence_features_used",
    ]

    if not all(key in st.session_state for key in required_keys):
        st.session_state.explanation_generated = False
        st.session_state.evaluation_submitted = False
        st.session_state.adaptation_trace = ""


def get_applicant_profile(sample_index):
    """Return the key applicant fields shown before the AI prediction."""
    row = X.iloc[sample_index]
    fields = [
        "checking_account_status", "savings_account", "credit_history", "purpose",
        "credit_amount", "duration_months", "employment_since", "age",
        "housing", "property", "job",
    ]
    return {
        feature_name_map[field]: decode_feature_value(field, row[field])
        for field in fields
    }


def build_confidence_input(feature_values):
    """Build a complete ordered confidence-model input row.

    The trained confidence model expects a feature named explanation_view_time.
    In the live app this value is the pre-explanation AI prediction review time,
    because confidence must be inferred before the explanation is selected.
    """
    ordered_values = {
        feature_name: [float(feature_values.get(feature_name, 0.0))]
        for feature_name in confidence_feature_names
    }
    return pd.DataFrame(ordered_values, columns=confidence_feature_names)


def estimate_confidence_behaviour_signals(click_count, profile_reviewed, ai_compared):
    """Estimate Streamlit-compatible behavioural proxies for the confidence model.

    Native Streamlit does not expose browser hover or scroll telemetry without a
    custom component. These values therefore use observable workflow progress
    and participant-confirmed review actions instead of fixed constants.
    """
    review_actions = int(profile_reviewed) + int(ai_compared)
    scroll_depth_proxy = min(1.0, 0.5 + (0.2 * review_actions))
    hover_count_proxy = click_count + review_actions
    return round(scroll_depth_proxy, 2), int(hover_count_proxy)


def predict_user_confidence(input_df):
    """Infer behavioural confidence level using all trained confidence features."""
    input_scaled = confidence_scaler.transform(input_df)
    pred_encoded = confidence_model.predict(input_scaled)[0]
    return confidence_label_encoder.inverse_transform([pred_encoded])[0]


def self_report_to_confidence(confidence_score):
    """Convert the participant's 1-5 confidence rating into low/medium/high."""
    if confidence_score <= 2:
        return "low"
    if confidence_score == 3:
        return "medium"
    return "high"


def combine_confidence_signals(model_confidence, reported_confidence):
    """Use the less confident signal so uncertain users receive enough detail."""
    levels = {"low": 0, "medium": 1, "high": 2}
    reverse_levels = {0: "low", 1: "medium", 2: "high"}
    model_level = levels.get(model_confidence, 1)
    reported_level = levels.get(reported_confidence, 1)
    return reverse_levels[min(model_level, reported_level)]


def get_applicant_case_letter(profile_id):
    """Extract the A-F case letter from labels such as Applicant A."""
    profile_id = str(profile_id).strip()
    if not profile_id:
        return ""
    return profile_id.split()[-1].upper()


def get_study_pattern(participant_id):
    """Return the counterbalanced condition pattern for this participant."""
    participant_number = get_participant_number(participant_id)
    if participant_number is not None and participant_number % 2 == 0:
        return "Pattern B"
    return "Pattern A"


def get_assigned_condition(participant_id, profile_id):
    """Assign Static/Adaptive using the study's A/B counterbalancing plan."""
    case_letter = get_applicant_case_letter(profile_id)
    pattern_a = {
        "A": "Static",
        "B": "Adaptive",
        "C": "Static",
        "D": "Adaptive",
        "E": "Static",
        "F": "Adaptive",
    }
    pattern_b = {
        "A": "Adaptive",
        "B": "Static",
        "C": "Adaptive",
        "D": "Static",
        "E": "Adaptive",
        "F": "Static",
    }
    pattern = pattern_b if get_study_pattern(participant_id) == "Pattern B" else pattern_a
    return pattern.get(case_letter, "Static")


def get_next_applicant_profile_id(current_profile_id):
    """Return the next applicant profile in the A-F study order."""
    profile_ids = profiles["Profile ID"].tolist()
    if current_profile_id not in profile_ids:
        return profile_ids[0], False

    current_index = profile_ids.index(current_profile_id)
    next_index = current_index + 1
    if next_index >= len(profile_ids):
        return None, True
    return profile_ids[next_index], False


def apply_pending_profile_selection():
    """Apply a queued profile change before the selectbox widget is rendered."""
    pending_profile_id = st.session_state.pop("pending_profile_id", None)
    if pending_profile_id:
        st.session_state.profile_selector = pending_profile_id


def get_adaptive_depth(confidence_level):
    """Map confidence level to explanation depth and presentation style."""
    if confidence_level == "low":
        return {
            "top_n": 6,
            "depth": "Detailed",
            "style": "expanded",
            "summary": "Detailed support mode",
            "show_probability_context": True,
            "show_extra_guidance": True,
            "rationale": "Adaptive mode detected lower confidence, so it shows more factors and extra context.",
        }
    if confidence_level == "medium":
        return {
            "top_n": 4,
            "depth": "Moderate",
            "style": "balanced",
            "summary": "Standard support mode",
            "show_probability_context": False,
            "show_extra_guidance": False,
            "rationale": "Adaptive mode detected moderate confidence, so it shows a focused explanation.",
        }
    return {
        "top_n": 2,
        "depth": "Concise",
        "style": "concise",
        "summary": "Quick summary mode",
        "show_probability_context": False,
        "show_extra_guidance": False,
        "rationale": "Adaptive mode detected higher confidence, so it keeps the explanation brief.",
    }


def get_static_depth():
    """Return a fixed explanation configuration for the static condition."""
    return {
        "top_n": 4,
        "depth": "Standard",
        "style": "balanced",
        "summary": "Static standard mode",
        "show_probability_context": False,
        "show_extra_guidance": False,
        "rationale": "Static mode uses the same explanation depth for every participant.",
    }


def build_explanation(sample_index, top_n):
    """Create a ranked explanation table from local SHAP-style contributions."""
    shap_row = shap_values[sample_index]
    explanation_df = pd.DataFrame({
        "feature": X.columns,
        "raw_value": X.iloc[sample_index].values,
        "shap_value": shap_row,
        "abs_impact": np.abs(shap_row),
    }).sort_values("abs_impact", ascending=False).head(top_n)

    explanation_df["readable_feature"] = explanation_df["feature"].map(
        lambda feature: feature_name_map.get(feature, feature)
    )
    explanation_df["readable_value"] = explanation_df.apply(
        lambda row: decode_feature_value(row["feature"], row["raw_value"]),
        axis=1,
    )
    explanation_df["reason_template"] = explanation_df["feature"].map(
        lambda feature: reason_templates.get(feature, "This feature is part of the model's learned risk pattern for this applicant.")
    )
    explanation_df["effect"] = np.where(
        explanation_df["shap_value"] > 0,
        "increased",
        "reduced",
    )
    return explanation_df.reset_index(drop=True)


def get_prediction(sample_index):
    """Return the model prediction and probability for one applicant."""
    sample = X.iloc[[sample_index]]
    prediction = int(final_model.predict(sample)[0])
    probability_bad = float(final_model.predict_proba(sample)[0][1])
    return prediction, probability_bad


def get_participant_number(participant_id):
    """Extract the numeric part from IDs like P001."""
    participant_id = str(participant_id).strip().upper()
    if participant_id.startswith("P") and participant_id[1:].isdigit():
        return int(participant_id[1:])
    return None


def get_next_participant_id(file_path=RESPONSE_FILE, current_participant_id=None):
    """Assign the next anonymised participant ID from existing response data."""
    numeric_ids = []
    current_number = get_participant_number(current_participant_id)
    if current_number is not None:
        numeric_ids.append(current_number)

    file_path = Path(file_path)
    if not file_path.exists():
        next_number = max(numeric_ids) + 1 if numeric_ids else 1
        return f"P{next_number:03d}"

    try:
        responses = pd.read_csv(file_path)
    except (pd.errors.EmptyDataError, ValueError):
        next_number = max(numeric_ids) + 1 if numeric_ids else 1
        return f"P{next_number:03d}"

    if "participant_id" not in responses.columns:
        next_number = max(numeric_ids) + 1 if numeric_ids else 1
        return f"P{next_number:03d}"

    participant_ids = responses["participant_id"].dropna().astype(str)
    for participant_id in participant_ids:
        participant_number = get_participant_number(participant_id)
        if participant_number is not None:
            numeric_ids.append(participant_number)

    if not numeric_ids:
        return "P001"

    return f"P{max(numeric_ids) + 1:03d}"


def get_current_participant_id(file_path=RESPONSE_FILE):
    """Return the latest existing participant ID without incrementing it."""
    file_path = Path(file_path)
    if not file_path.exists():
        return "P001"

    try:
        responses = pd.read_csv(file_path)
    except (pd.errors.EmptyDataError, ValueError):
        return "P001"

    if "participant_id" not in responses.columns:
        return "P001"

    numeric_ids = [
        participant_number
        for participant_number in (
            get_participant_number(participant_id)
            for participant_id in responses["participant_id"].dropna().astype(str)
        )
        if participant_number is not None
    ]
    if not numeric_ids:
        return "P001"

    return f"P{max(numeric_ids):03d}"


def get_model_confidence_badge(probability_bad):
    """Summarise model certainty from distance away from the decision boundary."""
    distance_from_boundary = abs(probability_bad - 0.5)
    if distance_from_boundary < 0.15:
        return "Borderline case"
    if distance_from_boundary < 0.35:
        return "Moderate model confidence"
    return "High model confidence"


def get_response_counts(file_path=RESPONSE_FILE):
    """Return lightweight response counts for the sidebar."""
    file_path = Path(file_path)
    if not file_path.exists():
        return {"total": 0, "static": 0, "adaptive": 0}

    try:
        responses = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        return {"total": 0, "static": 0, "adaptive": 0}

    if "condition" not in responses.columns:
        return {"total": len(responses), "static": 0, "adaptive": 0}

    condition_counts = responses["condition"].value_counts()
    return {
        "total": len(responses),
        "static": int(condition_counts.get("Static", 0)),
        "adaptive": int(condition_counts.get("Adaptive", 0)),
    }


def append_study_response(response, file_path=RESPONSE_FILE):
    """Append one response while preserving older rows if the schema expands."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_columns = list(dict.fromkeys(RESPONSE_COLUMNS + list(response.keys())))
    new_response = pd.DataFrame([response]).reindex(columns=ordered_columns)

    if file_path.exists():
        existing = pd.read_csv(file_path)
        combined_columns = list(dict.fromkeys(RESPONSE_COLUMNS + list(existing.columns) + list(new_response.columns)))
        existing = existing.reindex(columns=combined_columns)
        new_response = new_response.reindex(columns=combined_columns)
        pd.concat([existing, new_response], ignore_index=True).to_csv(file_path, index=False)
    else:
        new_response.to_csv(file_path, index=False)


def get_response_file_bytes(file_path=RESPONSE_FILE):
    """Return saved response data for the sidebar download button."""
    file_path = Path(file_path)
    if not file_path.exists():
        return None

    with file_path.open("rb") as response_file:
        return response_file.read()


def get_query_param(name):
    """Read a Streamlit query parameter across old/new Streamlit versions."""
    value = st.query_params.get(name)
    if isinstance(value, list):
        return value[0] if value else None
    return value


def handle_sidebar_actions():
    """Handle coloured sidebar action links."""
    action = get_query_param("sidebar_action")
    if not action:
        return

    if action == "new_participant":
        start_new_participant()
    elif action == "reset_case":
        reset_case_state()

    st.query_params.clear()
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


# ---------------------------------------------------------------------------
# UI rendering helpers
# ---------------------------------------------------------------------------

def render_metric(label, value, css_class=""):
    st.markdown(
        f"""
        <div class="metric-tile">
            <div class="metric-label">{escape(str(label))}</div>
            <div class="metric-value {css_class}">{escape(str(value))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_action(label, action, colour_class):
    """Render a coloured sidebar action link."""
    st.sidebar.markdown(
        f"""
        <a class="sidebar-action {colour_class}" href="?sidebar_action={action}" target="_self">
            {escape(label)}
        </a>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_download(label, data, file_name, colour_class):
    """Render a coloured sidebar download link."""
    encoded_data = base64.b64encode(data).decode("utf-8")
    st.sidebar.markdown(
        f"""
        <a class="sidebar-action {colour_class}" href="data:text/csv;base64,{encoded_data}" download="{escape(file_name)}">
            {escape(label)}
        </a>
        """,
        unsafe_allow_html=True,
    )


def render_condition_badge(condition):
    """Display the current explanation mode as a compact badge."""
    badge_label = f"{condition.upper()} MODE"
    st.markdown(
        f"""
        <div class="mode-badge-wrap">
            <div class="ai-tag">
                <span class="ai-tag-dot"></span>
                <span>{escape(badge_label)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_step_header(step, title):
    st.markdown(f'<span class="step-pill">{step}</span>', unsafe_allow_html=True)
    st.subheader(title)


def render_helper_text(text):
    st.markdown(f'<div class="helper-text">{escape(str(text))}</div>', unsafe_allow_html=True)


def render_action_spacer():
    st.markdown('<div class="action-spacer"></div>', unsafe_allow_html=True)


def render_consent_intro():
    """Show the study information and consent gate before the task starts."""
    st.markdown(
        """
        <div class="hero">
            <div class="hero-content">
                <div class="ai-tag"><span class="ai-tag-dot"></span><span>Study information</span></div>
                <h1>Confidence-Aware Explainable AI Prototype</h1>
                <p>This simulated study evaluates how static and adaptive explanations affect trust, understanding, and reliance in AI-assisted credit-risk decisions.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown("**Before You Begin**")
        st.write("This is a simulated financial decision-support task using predefined dataset profiles.")
        st.write("The task does not involve real lending decisions and will not affect any real person, account, or financial outcome.")
        st.write("No real personal financial data is collected or processed.")
        st.write("Responses are stored for dissertation analysis and are anonymised using a participant ID rather than a real name.")
        st.write("A participant ID will be assigned to you at the start of the study.")
        st.write("Participation is voluntary. You may stop before submitting your evaluation response if you do not wish to continue.")
        st.write("The AI prediction is for research purposes only and must not be treated as real lending or financial advice.")
        st.write("Static and adaptive explanation modes are assigned automatically from the study pattern.")

        consent_checked = st.checkbox(
            "I understand this is a simulated, voluntary study and consent to my anonymised responses being recorded for dissertation analysis.",
            key="consent_checkbox",
        )

        if st.button("Start study", type="primary", disabled=not consent_checked):
            st.session_state.consent_accepted = True
            if hasattr(st, "rerun"):
                st.rerun()
            else:
                st.experimental_rerun()


def render_profile(profile):
    profile_items = list(profile.items())
    left_items = profile_items[:6]
    right_items = profile_items[6:]
    left_col, right_col = st.columns(2)

    with left_col:
        for key, value in left_items:
            st.markdown(f"**{key}:** {value}")

    with right_col:
        for key, value in right_items:
            st.markdown(f"**{key}:** {value}")


def render_risk_meter(probability_bad):
    """Render a compact finance-style risk meter for bad-credit probability."""
    probability_pct = probability_bad * 100
    if probability_bad < 0.35:
        risk_band = "Low risk"
        risk_class = "good-risk"
    elif probability_bad < 0.65:
        risk_band = "Borderline"
        risk_class = ""
    else:
        risk_band = "High risk"
        risk_class = "bad-risk"

    st.markdown(
        f"""
        <div class="risk-meter">
            <div class="risk-meter-head">
                <span>Bad-credit risk meter</span>
                <span class="{risk_class}">{risk_band} | {probability_pct:.1f}%</span>
            </div>
            <div class="risk-track">
                <div class="risk-marker" style="left: calc({probability_pct:.1f}% - 2px);"></div>
            </div>
            <div class="risk-scale">
                <span>0%</span>
                <span>50%</span>
                <span>100%</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_decision_comparison(user_prediction, ai_label, probability_bad):
    """Show a compact user-versus-AI decision comparison panel."""
    confidence_badge = get_model_confidence_badge(probability_bad)
    agreement = user_prediction == ai_label
    items = [
        ("Your judgement", user_prediction),
        ("AI judgement", ai_label),
        ("Agreement", "Yes" if agreement else "No"),
        ("Model signal", confidence_badge),
    ]
    item_html = "".join(
        f"""
        <div class="comparison-item">
            <div class="comparison-label">{escape(label)}</div>
            <div class="comparison-value">{escape(value)}</div>
        </div>
        """
        for label, value in items
    )

    st.markdown(
        f"""
        <div class="comparison-panel">
            <div class="comparison-title">Decision comparison</div>
            <div class="comparison-grid">{item_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_depth_indicator(depth):
    """Highlight the active explanation depth."""
    levels = ["Concise", "Moderate", "Detailed"]
    cols = st.columns(3)

    for col, level in zip(cols, levels):
        with col:
            if depth == level:
                st.markdown(f"**{level}**")
                st.progress(1.0)
            elif depth == "Standard" and level == "Moderate":
                st.markdown("**Standard**")
                st.progress(0.66)
            else:
                st.caption(level)
                st.progress(0.15)


def render_feature_impact_chart(explanation_df, style):
    """Render signed local impacts and adaptive support visuals."""
    st.markdown("**Local feature impact**")
    chart_df = explanation_df[["readable_feature", "shap_value"]].copy()
    chart_df = chart_df.rename(
        columns={
            "readable_feature": "Feature",
            "shap_value": "Impact",
        }
    )
    chart_df["Feature"] = chart_df["Feature"].astype(str)
    chart_df["Impact share"] = explanation_df["abs_impact"].astype(float)
    chart_df["Direction"] = np.where(chart_df["Impact"] >= 0, "Raises risk", "Lowers risk")
    chart_df["Rank"] = np.arange(1, len(chart_df) + 1)
    chart_df["Cumulative impact"] = chart_df["Impact"].cumsum()

    try:
        import altair as alt
    except ImportError:
        st.bar_chart(chart_df.set_index("Feature")["Impact"], height=280)
        st.caption("Install Altair for the orange bar chart and multicolour impact-share donut chart.")
        return

    axis_color = "#94a3b8"
    palette = [
        "#f97316", "#7c3aed", "#06b6d4", "#22c55e", "#ef4444",
        "#eab308", "#ec4899", "#14b8a6", "#6366f1", "#84cc16",
    ]

    bar_chart = (
        alt.Chart(chart_df)
        .mark_bar(color="#f97316", cornerRadius=4)
        .encode(
            x=alt.X("Impact:Q", title="Signed impact"),
            y=alt.Y("Feature:N", sort="-x", title=None),
            tooltip=[
                alt.Tooltip("Feature:N"),
                alt.Tooltip("Impact:Q", format=".3f"),
            ],
        )
        .properties(height=260)
        .configure_view(strokeOpacity=0)
        .configure_axis(labelColor=axis_color, titleColor=axis_color, gridColor="#33415533")
        .configure(background="transparent")
    )

    st.altair_chart(bar_chart, use_container_width=True)
    st.caption("Positive values raise predicted bad-credit risk; negative values lower it.")

    if style in ["balanced", "expanded"]:
        st.markdown("**Risk direction balance**")
        direction_df = (
            chart_df.groupby("Direction", as_index=False)["Impact share"]
            .sum()
            .rename(columns={"Impact share": "Total absolute impact"})
        )
        direction_chart = (
            alt.Chart(direction_df)
            .mark_bar(cornerRadius=4)
            .encode(
                x=alt.X("Total absolute impact:Q", title="Total local impact"),
                y=alt.Y("Direction:N", title=None),
                color=alt.Color(
                    "Direction:N",
                    scale=alt.Scale(
                        domain=["Raises risk", "Lowers risk"],
                        range=["#e80c0c", "#22c55e"],
                    ),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("Direction:N"),
                    alt.Tooltip("Total absolute impact:Q", format=".3f"),
                ],
            )
            .properties(height=120)
            .configure_view(strokeOpacity=0)
            .configure_axis(labelColor=axis_color, titleColor=axis_color, gridColor="#33415533")
            .configure(background="transparent")
        )
        st.altair_chart(direction_chart, use_container_width=True)
        st.caption("This separates evidence that pushes the prediction toward higher risk from evidence that pushes it toward lower risk.")

    if style == "expanded":
        st.markdown("**Cumulative explanation path**")
        path_chart = (
            alt.Chart(chart_df)
            .mark_line(point=True, color="#7c3aed")
            .encode(
                x=alt.X("Rank:O", title="Reason rank"),
                y=alt.Y("Cumulative impact:Q", title="Cumulative signed impact"),
                tooltip=[
                    alt.Tooltip("Rank:O"),
                    alt.Tooltip("Feature:N"),
                    alt.Tooltip("Impact:Q", format=".3f"),
                    alt.Tooltip("Cumulative impact:Q", format=".3f"),
                ],
            )
            .properties(height=190)
        )
        zero_rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
            color="#94a3b8",
            strokeDash=[4, 4],
        ).encode(y="y:Q")
        layered_path_chart = (
            alt.layer(path_chart, zero_rule)
            .properties(height=190)
            .configure_view(strokeOpacity=0)
            .configure_axis(labelColor=axis_color, titleColor=axis_color, gridColor="#33415533")
            .configure(background="transparent")
        )
        st.altair_chart(layered_path_chart, use_container_width=True)
        st.caption("For detailed mode, this shows how the ranked reasons accumulate as supporting evidence.")

    pie_df = chart_df[["Feature", "Impact share"]].copy()
    pie_chart = (
        alt.Chart(pie_df)
        .mark_arc(innerRadius=48, outerRadius=100, stroke="#ffffff", strokeWidth=1)
        .encode(
            theta=alt.Theta("Impact share:Q", title="Impact share"),
            color=alt.Color(
                "Feature:N",
                scale=alt.Scale(range=palette),
                legend=alt.Legend(title="Feature", orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip("Feature:N"),
                alt.Tooltip("Impact share:Q", format=".3f"),
            ],
        )
        .properties(height=330)
        .configure_view(strokeOpacity=0)
        .configure_legend(labelColor=axis_color, titleColor=axis_color)
        .configure(background="transparent")
    )

    st.markdown("**Impact share**")
    st.altair_chart(pie_chart, use_container_width=True)


def render_explanation(explanation_df, style):
    """Display ranked feature explanations with detail based on condition depth."""
    max_impact = max(float(explanation_df["abs_impact"].max()), 0.0001)

    for index, row in explanation_df.iterrows():
        direction_class = "impact-up" if row["shap_value"] > 0 else "impact-down"
        direction_label = "Raises risk" if row["shap_value"] > 0 else "Lowers risk"
        impact_pct = min(float(row["abs_impact"]) / max_impact, 1.0)
        effect_sentence = f"{row['readable_feature']} ({row['readable_value']}) {row['effect']} the predicted credit risk."
        reason_sentence = (
            f"{effect_sentence} This is relevant because {str(row['reason_template'])[0].lower()}"
            f"{str(row['reason_template'])[1:]}"
        )

        with st.container(border=True):
            cols = st.columns([0.6, 3.0, 1.2])
            with cols[0]:
                st.markdown(f"**#{index + 1}**")
            with cols[1]:
                st.markdown(f"**{row['readable_feature']}**")
                st.caption(str(row["readable_value"]))
            with cols[2]:
                st.markdown(f'<span class="{direction_class}">{direction_label}</span>', unsafe_allow_html=True)

            if style == "concise":
                st.caption(effect_sentence)
            elif style == "balanced":
                st.caption(reason_sentence)
            else:
                st.write(reason_sentence)
                st.caption(
                    "This is one contributing factor in the model's prediction and should not be interpreted as a standalone lending rule."
                )

            st.progress(impact_pct)
            if style == "expanded":
                st.caption(
                    f"Relative impact in this explanation: {impact_pct:.0%}."
                )


def render_sidebar(condition):
    st.sidebar.title("Study Control")
    selected_profile_for_pattern = st.session_state.get("profile_selector", profiles["Profile ID"].iloc[0])
    study_pattern = get_study_pattern(st.session_state.participant_id)

    st.sidebar.subheader("Participant")
    id_cols = st.sidebar.columns([1, 1])
    with id_cols[0]:
        st.caption("Participant ID")
        st.markdown(f"**{st.session_state.participant_id}**")
    with id_cols[1]:
        st.caption("Session")
        st.markdown(f"**{st.session_state.session_id[-6:]}**")

    st.sidebar.caption("Use one participant ID for all cases completed by the same person.")

    render_sidebar_action("Reset current case", "reset_case", "sidebar-red")

    st.sidebar.divider()
    st.sidebar.subheader("Explanation Mode")
    condition_disabled = True
    st.sidebar.radio(
        "Assigned explanation condition",
        ["Static", "Adaptive"],
        index=0 if condition == "Static" else 1,
        key="condition_radio",
        disabled=condition_disabled,
    )
    st.sidebar.caption(
        f"{study_pattern}: {selected_profile_for_pattern} is assigned to {condition}. "
        "This is locked by the study design."
    )

    with st.sidebar.expander("Mode guide", expanded=False):
        st.write("Static: fixed standard explanation depth.")
        st.write("Adaptive: explanation depth changes using confidence signals.")
        st.caption("All profiles are simulated dataset cases. No real financial data is collected.")

    with st.sidebar.expander("Task order", expanded=False):
        st.write("Complete the six applicant cases in order from Applicant A to Applicant F.")
        st.write("The app assigns Static or Adaptive automatically from the participant ID pattern.")
        st.write("After submitting each evaluation, use the continue button to move to the next applicant.")

    st.sidebar.divider()
    st.sidebar.subheader("Progress")
    progress_steps = [
        ("Profile reviewed", True),
        ("Initial judgement", st.session_state.submitted_initial),
        ("AI prediction shown", st.session_state.submitted_initial),
        ("Explanation generated", st.session_state.explanation_generated),
        ("Evaluation submitted", st.session_state.evaluation_submitted),
    ]

    for label, complete in progress_steps:
        marker = "[x]" if complete else "[ ]"
        st.sidebar.write(f"{marker} {label}")

    render_sidebar_action("Start new participant", "new_participant", "sidebar-blue")

    st.sidebar.divider()
    st.sidebar.subheader("Responses")
    response_counts = get_response_counts()
    count_cols = st.sidebar.columns(3)
    with count_cols[0]:
        st.metric("Total", response_counts["total"])
    with count_cols[1]:
        st.metric("Static", response_counts["static"])
    with count_cols[2]:
        st.metric("Adaptive", response_counts["adaptive"])

    response_data = get_response_file_bytes()
    if response_data is not None:
        render_sidebar_download(
            "Download responses",
            response_data,
            "data-responses-user_study_responses.csv",
            "sidebar-green",
        )

    st.sidebar.divider()
    with st.sidebar.expander("System Status", expanded=False):
        st.write("[x] Models loaded")
        st.write(f"[x] Profiles loaded: {len(profiles)}")
        st.write(f"[x] Dataset rows: {len(X)}")
        st.write("[x] Explanations ready")
        response_status = "available" if RESPONSE_FILE.exists() else "not created yet"
        st.write(f"Responses file: {response_status}")

    with st.sidebar.expander("Analysis Tools", expanded=False):
        st.code("streamlit run scripts/analysis_dashboard.py")
        st.code("py scripts/analysis_summary.py")
        st.caption("Use scripts/reset_responses.py only when you deliberately want to archive and clear the CSV.")


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

initialise_session_state()
repair_incomplete_explanation_state()
handle_sidebar_actions()
apply_pending_profile_selection()

selected_profile_for_condition = st.session_state.get("profile_selector", profiles["Profile ID"].iloc[0])
condition = get_assigned_condition(st.session_state.participant_id, selected_profile_for_condition)
st.session_state.condition_radio = condition
render_sidebar(condition)

if not st.session_state.consent_accepted:
    render_consent_intro()
    st.stop()

st.markdown(
    """
    <div class="hero">
        <div class="hero-content">
            <div class="ai-tag"><span class="ai-tag-dot"></span><span>AI-assisted credit risk review</span></div>
            <h1>Confidence-Aware Explainable AI Prototype</h1>
            <p>Financial decision support with XGBoost predictions, SHAP-style explanations, and confidence-aware explanation depth.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

summary_cols = st.columns(4)
with summary_cols[0]:
    render_metric("Dataset", "German Credit")
with summary_cols[1]:
    render_metric("Profiles", str(len(profiles)))
with summary_cols[2]:
    render_metric("Model", "XGBoost")
with summary_cols[3]:
    render_metric("Condition", condition)

render_condition_badge(condition)

st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Step 1: Applicant profile
# ---------------------------------------------------------------------------

with st.container():
    render_step_header("Step 1", "Review Applicant Profile")
    render_helper_text(
        "Complete the applicant cases in order from A to F. The explanation mode is assigned automatically "
        "from your participant ID, so review the applicant information first and make your own judgement before seeing the AI prediction."
    )

    selected_profile_id = st.selectbox(
        "Select applicant profile",
        profiles["Profile ID"].tolist(),
        key="profile_selector",
        on_change=reset_case_state,
    )

    selected_profile = profiles[profiles["Profile ID"] == selected_profile_id].iloc[0]
    sample_index = int(selected_profile["Dataset Index"])
    profile = get_applicant_profile(sample_index)

    profile_col, case_col = st.columns([2.2, 1])
    with profile_col:
        with st.container(border=True):
            render_profile(profile)

    with case_col:
        render_metric("Applicant case", selected_profile_id)
        render_action_spacer()
        render_metric("Assigned mode", condition)
        render_action_spacer()
        render_metric("Decision stage", "Human review")


# ---------------------------------------------------------------------------
# Step 2: Initial human judgement
# ---------------------------------------------------------------------------

with st.container():
    render_step_header("Step 2", "Your Initial Judgement")
    render_helper_text("Record your own decision before seeing the model output. The assigned explanation condition is fixed by the study design.")
    judgement_col, confidence_col, action_col = st.columns([1.2, 1.5, 1])
    judgement_locked = st.session_state.submitted_initial
    prediction_options = ["Good Credit", "Bad Credit"]
    prediction_index = prediction_options.index(
        st.session_state.get("user_prediction", "Good Credit")
    )
    confidence_value = st.session_state.get("user_confidence", 3)

    with judgement_col:
        user_prediction = st.radio(
            "What do you think the outcome should be?",
            prediction_options,
            index=prediction_index,
            horizontal=True,
            disabled=judgement_locked,
        )

    with confidence_col:
        user_confidence = st.slider(
            "How confident are you in your judgement?",
            1,
            5,
            confidence_value,
            help="1 = not confident, 5 = very confident",
            disabled=judgement_locked,
        )

    with action_col:
        render_action_spacer()
        submit_initial = st.button(
            "Submit initial judgement",
            use_container_width=True,
            disabled=judgement_locked,
        )

    if submit_initial:
        st.session_state.interaction_count += 1
        st.session_state.decision_time = round(time.time() - st.session_state.start_time, 2)
        st.session_state.user_prediction = user_prediction
        st.session_state.user_confidence = user_confidence
        st.session_state.sample_index = sample_index
        st.session_state.selected_profile_id = selected_profile_id
        st.session_state.selected_condition = condition
        st.session_state.submitted_initial = True
        st.session_state.ai_reveal_time = time.time()
        st.session_state.explanation_generated = False
        st.session_state.evaluation_submitted = False
        st.session_state.adaptation_trace = ""
        st.success(f"Initial judgement submitted in {st.session_state.decision_time} seconds.")


# ---------------------------------------------------------------------------
# Step 3: AI prediction
# ---------------------------------------------------------------------------

if st.session_state.submitted_initial:
    render_step_header("Step 3", "AI Prediction")
    render_helper_text("The model prediction is now visible. Generate the explanation to continue to the evaluation stage.")

    ai_prediction, probability_bad = get_prediction(st.session_state.sample_index)
    ai_label = "Bad Credit" if ai_prediction == 1 else "Good Credit"
    ai_class = "bad-risk" if ai_prediction == 1 else "good-risk"
    user_agrees = st.session_state.user_prediction == ai_label

    pred_cols = st.columns(4)
    with pred_cols[0]:
        render_metric("AI prediction", ai_label, ai_class)
    with pred_cols[1]:
        render_metric("Bad-credit probability", f"{probability_bad:.1%}")
    with pred_cols[2]:
        render_metric("Your judgement", st.session_state.user_prediction)
    with pred_cols[3]:
        render_metric("Agreement", "Yes" if user_agrees else "No")

    render_decision_comparison(st.session_state.user_prediction, ai_label, probability_bad)
    render_risk_meter(probability_bad)

    with st.container(border=True):
        st.markdown("**Review checks**")
        st.caption("These checks help the prototype estimate review behaviour before selecting the explanation depth.")
        review_col, compare_col = st.columns(2)
        with review_col:
            profile_reviewed = st.checkbox(
                "I reviewed the applicant details before requesting the explanation.",
                key="reviewed_profile_before_explanation",
                disabled=st.session_state.explanation_generated,
            )
        with compare_col:
            ai_compared = st.checkbox(
                "I compared my judgement with the AI prediction.",
                key="compared_ai_prediction_before_explanation",
                disabled=st.session_state.explanation_generated,
            )

    generate_explanation = st.button(
        "Generate explanation",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.explanation_generated,
    )

    if generate_explanation:
        st.session_state.interaction_count += 1
        st.session_state.ai_prediction_review_time = round(time.time() - st.session_state.ai_reveal_time, 2)
        st.session_state.explanation_display_time = time.time()
        st.session_state.explanation_reading_time = None

        decision_time = st.session_state.decision_time
        click_count = st.session_state.interaction_count
        scroll_depth, hover_count = estimate_confidence_behaviour_signals(
            click_count,
            st.session_state.reviewed_profile_before_explanation,
            st.session_state.compared_ai_prediction_before_explanation,
        )
        confidence_model_review_time = st.session_state.ai_prediction_review_time

        confidence_feature_values = {
            "decision_time": decision_time,
            "click_count": click_count,
            "interaction_count": click_count,
            "scroll_depth": scroll_depth,
            "hover_count": hover_count,
            "explanation_view_time": confidence_model_review_time,
            "ai_prediction_review_time": confidence_model_review_time,
        }
        confidence_input_df = build_confidence_input(confidence_feature_values)
        predicted_confidence = predict_user_confidence(confidence_input_df)
        reported_confidence = self_report_to_confidence(st.session_state.user_confidence)
        adaptive_confidence = combine_confidence_signals(predicted_confidence, reported_confidence)

        if st.session_state.selected_condition == "Static":
            explanation_config = get_static_depth()
            applied_confidence_signal = "Not applied"
            adaptation_trace = "Static condition -> Standard explanation"
        else:
            explanation_config = get_adaptive_depth(adaptive_confidence)
            applied_confidence_signal = adaptive_confidence.title()
            adaptation_trace = f"Explanation adapted from confidence signal: {adaptive_confidence.title()} -> {explanation_config['depth']}"

        explanation_df = build_explanation(
            st.session_state.sample_index,
            explanation_config["top_n"],
        )

        st.session_state.predicted_confidence = predicted_confidence
        st.session_state.reported_confidence_level = reported_confidence
        st.session_state.adaptive_confidence_level = adaptive_confidence
        st.session_state.applied_confidence_signal = applied_confidence_signal
        st.session_state.adaptation_trace = adaptation_trace
        st.session_state.confidence_feature_inputs = confidence_input_df.iloc[0].to_dict()
        st.session_state.confidence_features_used = list(confidence_input_df.columns)
        st.session_state.explanation_depth = explanation_config["depth"]
        st.session_state.explanation_style = explanation_config["style"]
        st.session_state.explanation_summary = explanation_config["summary"]
        st.session_state.show_probability_context = explanation_config["show_probability_context"]
        st.session_state.show_extra_guidance = explanation_config["show_extra_guidance"]
        st.session_state.explanation_rationale = explanation_config["rationale"]
        st.session_state.top_n = explanation_config["top_n"]
        st.session_state.explanation_df = explanation_df
        st.session_state.ai_prediction = ai_label
        st.session_state.probability_bad = probability_bad
        st.session_state.explanation_generated = True

    if st.session_state.explanation_generated:
        st.info("Explanation generated for this case.")


# ---------------------------------------------------------------------------
# Step 4: Explanation
# ---------------------------------------------------------------------------

if st.session_state.explanation_generated:
    render_step_header("Step 4", "Explanation")
    render_helper_text("The chart shows signed local feature impact. Positive values raise predicted bad-credit risk; negative values lower it.")

    info_cols = st.columns(4)
    with info_cols[0]:
        render_metric("Condition used", st.session_state.selected_condition)
    with info_cols[1]:
        render_metric("Behavioural confidence", st.session_state.predicted_confidence.title())
    with info_cols[2]:
        render_metric("Adaptation signal", st.session_state.applied_confidence_signal)
    with info_cols[3]:
        render_metric("Depth", st.session_state.explanation_depth)

    render_depth_indicator(st.session_state.explanation_depth)
    st.markdown(f"**Presentation style:** {st.session_state.explanation_summary}")
    st.markdown(f"**Explanation policy:** {st.session_state.explanation_rationale}")
    adaptation_trace = st.session_state.get(
        "adaptation_trace",
        f"{st.session_state.selected_condition} condition -> {st.session_state.explanation_depth} explanation",
    )
    st.caption(adaptation_trace)

    if st.session_state.show_probability_context:
        st.info(
            f"Detailed context: the model estimated a bad-credit probability of "
            f"{st.session_state.probability_bad:.1%}. Review the reasons below as supporting evidence, "
            "not as independent lending rules."
        )

    if st.session_state.show_extra_guidance:
        with st.expander("How to read this detailed explanation", expanded=True):
            st.write("Each reason shows a feature value, whether it raised or lowered predicted risk, and a deterministic context note.")
            st.write("The explanation is selective: it highlights the strongest local contributors for this applicant.")
            st.write("For appropriate reliance, compare the AI reasons against your own judgement before completing the evaluation.")
    elif st.session_state.explanation_style == "concise" and st.session_state.selected_condition == "Adaptive":
        st.info(
            "You reported high confidence, so a concise explanation is shown. "
            "You may still review the main factors carefully, especially if the AI prediction differs from your initial judgement."
        )

    with st.expander("Confidence model transparency", expanded=False):
        st.write("Because native Streamlit does not expose browser hover or scroll telemetry, scroll depth and hover count are represented as transparent workflow/review proxies rather than hidden browser tracking.")
        st.info(
            "Methodology note: behavioural confidence is estimated using prototype interaction signals and "
            "review-confirmation proxies. This demonstrates adaptive explanation logic; it is not intended "
            "to represent full browser-based behavioural tracking."
        )
        confidence_inputs = st.session_state.get("confidence_feature_inputs", {})
        confidence_feature_table = pd.DataFrame({
            "Feature used by confidence model": st.session_state.get("confidence_features_used", confidence_feature_names),
            "Value for this case": [
                confidence_inputs.get(feature_name, 0.0)
                for feature_name in st.session_state.get("confidence_features_used", confidence_feature_names)
            ],
        })
        st.dataframe(
            confidence_feature_table,
            hide_index=True,
            use_container_width=True,
        )
        st.caption("The confidence model uses the time spent reviewing the AI prediction before requesting the explanation. Actual explanation-reading time is recorded when the evaluation is submitted.")

    render_action_spacer()
    chart_col, ranked_col = st.columns([1, 1.35])
    with chart_col:
        render_feature_impact_chart(
            st.session_state.explanation_df,
            st.session_state.explanation_style,
        )
    with ranked_col:
        render_explanation(st.session_state.explanation_df, st.session_state.explanation_style)


# ---------------------------------------------------------------------------
# Step 5: Evaluation and response saving
# ---------------------------------------------------------------------------

if st.session_state.explanation_generated:
    render_step_header("Step 5", "Evaluation")
    render_helper_text("Submit once for this case. After submission, use the continue button to move to the next assigned applicant case.")

    eval_cols = st.columns(2)
    with eval_cols[0]:
        trust = st.slider("I trust the AI prediction.", 1, 5, 3)
        understanding = st.slider("The explanation helped me understand the prediction.", 1, 5, 3)
    with eval_cols[1]:
        usefulness = st.slider("The explanation was useful.", 1, 5, 3)
        reliance = st.slider("I would rely on this AI recommendation.", 1, 5, 3)

    comments = st.text_area("Please provide any comments about the explanation you received. For example, you may comment on its clarity, usefulness, level of detail, trustworthiness, or anything you found confusing.")

    submit_evaluation = st.button(
        "Submit evaluation",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.evaluation_submitted,
    )

    if submit_evaluation:
        user_agreed_with_ai = st.session_state.user_prediction == st.session_state.ai_prediction
        explanation_reading_time = round(time.time() - st.session_state.explanation_display_time, 2)
        st.session_state.explanation_reading_time = explanation_reading_time
        response = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "session_id": st.session_state.session_id,
            "participant_id": st.session_state.participant_id.strip(),
            "profile_id": st.session_state.selected_profile_id,
            "dataset_index": st.session_state.sample_index,
            "profile_type": selected_profile["Profile Type"],
            "condition": st.session_state.selected_condition,
            "user_prediction": st.session_state.user_prediction,
            "user_confidence": st.session_state.user_confidence,
            "ai_prediction": st.session_state.ai_prediction,
            "probability_bad_credit": st.session_state.probability_bad,
            "user_agreed_with_ai": user_agreed_with_ai,
            "decision_time": st.session_state.decision_time,
            "interaction_count": st.session_state.interaction_count,
            "ai_prediction_review_time": st.session_state.ai_prediction_review_time,
            "explanation_reading_time": explanation_reading_time,
            "explanation_view_time": explanation_reading_time,
            "confidence_scroll_depth_proxy": st.session_state.confidence_feature_inputs.get("scroll_depth", 0.0),
            "confidence_hover_count_proxy": st.session_state.confidence_feature_inputs.get("hover_count", 0.0),
            "predicted_confidence": st.session_state.predicted_confidence,
            "reported_confidence_level": st.session_state.reported_confidence_level,
            "adaptation_signal": st.session_state.applied_confidence_signal,
            "explanation_depth": st.session_state.explanation_depth,
            "top_features_shown": st.session_state.top_n,
            "trust": trust,
            "understanding": understanding,
            "usefulness": usefulness,
            "reliance": reliance,
            "comments": comments,
        }

        append_study_response(response)
        st.session_state.evaluation_submitted = True
        st.success("Evaluation submitted and saved.")

    if st.session_state.evaluation_submitted:
        with st.container(border=True):
            st.markdown("**Thank you. Response saved.**")
            st.write("This case is complete and has been added to the study response file.")
            next_profile_id, study_complete = get_next_applicant_profile_id(
                st.session_state.selected_profile_id
            )

            if study_complete:
                st.success("All six applicant cases are complete for this participant.")
                st.caption("Use Start new participant in the sidebar when the next participant begins.")
            else:
                next_condition = get_assigned_condition(
                    st.session_state.participant_id,
                    next_profile_id,
                )
                st.caption(
                    f"Next case: {next_profile_id}, assigned to {next_condition}."
                )
                if st.button(
                    f"Continue to {next_profile_id}",
                    type="primary",
                    use_container_width=True,
                ):
                    st.session_state.pending_profile_id = next_profile_id
                    reset_case_state()
                    if hasattr(st, "rerun"):
                        st.rerun()
                    else:
                        st.experimental_rerun()
