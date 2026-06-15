from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1]
RESPONSE_FILE = BASE_DIR / "data" / "responses" / "user_study_responses.csv"
RATING_METRICS = ["trust", "understanding", "usefulness", "reliance"]


def load_responses():
    if not RESPONSE_FILE.exists():
        return None

    responses = pd.read_csv(RESPONSE_FILE)
    if responses.empty:
        return responses

    for column in RATING_METRICS + [
        "top_features_shown",
        "probability_bad_credit",
        "decision_time",
        "ai_prediction_review_time",
        "explanation_reading_time",
        "explanation_view_time",
        "interaction_count",
        "user_confidence",
    ]:
        if column in responses.columns:
            responses[column] = pd.to_numeric(responses[column], errors="coerce")

    if "user_agreed_with_ai" in responses.columns:
        responses["user_agreed_with_ai"] = responses["user_agreed_with_ai"].map(
            lambda value: str(value).strip().lower() in ["true", "1", "1.0", "yes"]
        )

    if "participant_id" in responses.columns:
        responses["participant_id"] = responses["participant_id"].fillna("").astype(str)

    return responses


st.set_page_config(
    page_title="Confidence-Aware XAI Study Dashboard",
    page_icon="DA",
    layout="wide",
)

st.title("Confidence-Aware XAI Study Dashboard")
st.caption("Visual summary of static versus adaptive explanation responses.")

responses = load_responses()
if responses is None:
    st.warning(f"No response file found: {RESPONSE_FILE}")
    st.stop()
if responses.empty:
    st.warning("The response file exists, but it does not contain any rows yet.")
    st.stop()

st.sidebar.header("Filters")

conditions = sorted(responses["condition"].dropna().unique().tolist())
selected_conditions = st.sidebar.multiselect("Condition", conditions, default=conditions)

profiles = sorted(responses["profile_id"].dropna().unique().tolist()) if "profile_id" in responses.columns else []
selected_profiles = st.sidebar.multiselect("Profile", profiles, default=profiles)

filtered = responses[responses["condition"].isin(selected_conditions)].copy()
if selected_profiles:
    filtered = filtered[filtered["profile_id"].isin(selected_profiles)]

if filtered.empty:
    st.warning("No responses match the selected filters.")
    st.stop()

summary_cols = st.columns(5)
with summary_cols[0]:
    st.metric("Responses", len(filtered))
with summary_cols[1]:
    participant_count = 0
    if "participant_id" in filtered.columns:
        participant_count = filtered["participant_id"].replace("", pd.NA).dropna().nunique()
    st.metric("Participants", participant_count)
with summary_cols[2]:
    st.metric("Profiles", filtered["profile_id"].nunique() if "profile_id" in filtered.columns else 0)
with summary_cols[3]:
    st.metric("Static", int((filtered["condition"] == "Static").sum()))
with summary_cols[4]:
    st.metric("Adaptive", int((filtered["condition"] == "Adaptive").sum()))

st.divider()

tab_overview, tab_ratings, tab_behaviour, tab_raw = st.tabs(
    ["Overview", "Ratings", "Behaviour & Depth", "Raw Data"]
)

with tab_overview:
    st.subheader("Responses by Condition")
    condition_counts = filtered["condition"].value_counts().rename_axis("Condition").reset_index(name="Responses")
    st.bar_chart(condition_counts.set_index("Condition"))

    if "user_agreed_with_ai" in filtered.columns:
        st.subheader("Agreement with AI")
        agreement_summary = filtered.groupby("condition")["user_agreed_with_ai"].mean().mul(100).round(1)
        st.bar_chart(agreement_summary)
        st.caption("Percentage of responses where the initial user judgement matched the AI prediction.")

with tab_ratings:
    available_metrics = [metric for metric in RATING_METRICS if metric in filtered.columns]
    if available_metrics:
        st.subheader("Mean Ratings by Condition")
        metric_summary = filtered.groupby("condition")[available_metrics].mean().round(2)
        st.dataframe(metric_summary, use_container_width=True)
        st.bar_chart(metric_summary)

        st.subheader("Overall Rating Distribution")
        long_ratings = filtered.melt(
            id_vars=["condition"],
            value_vars=available_metrics,
            var_name="Metric",
            value_name="Rating",
        ).dropna()
        st.bar_chart(long_ratings.groupby(["Metric", "Rating"]).size().unstack(fill_value=0))
    else:
        st.info("No rating columns are available in the response file.")

with tab_behaviour:
    if "top_features_shown" in filtered.columns:
        st.subheader("Explanation Depth by Condition")
        depth_summary = filtered.groupby("condition")["top_features_shown"].mean().round(2)
        st.bar_chart(depth_summary)

    timing_cols = [
        col for col in [
            "decision_time",
            "ai_prediction_review_time",
            "explanation_reading_time",
            "explanation_view_time",
            "interaction_count",
        ]
        if col in filtered.columns
    ]
    if timing_cols:
        st.subheader("Interaction Signals by Condition")
        timing_summary = filtered.groupby("condition")[timing_cols].mean().round(2)
        st.dataframe(timing_summary, use_container_width=True)
        st.bar_chart(timing_summary)

    if "adaptation_signal" in filtered.columns:
        st.subheader("Adaptation Signals")
        signal_counts = filtered["adaptation_signal"].fillna("Missing").value_counts()
        st.bar_chart(signal_counts)

with tab_raw:
    st.subheader("Raw Responses")
    st.dataframe(filtered, use_container_width=True)

    csv_data = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered responses",
        data=csv_data,
        file_name="filtered_user_study_responses.csv",
        mime="text/csv",
    )
