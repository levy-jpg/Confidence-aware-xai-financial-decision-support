"""
Visual analysis dashboard for the Confidence-Aware XAI user study.
Author: Levy Thiga Kariuki
Student Number: G20893080
"""

import math
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    from scipy import stats
except ImportError:
    stats = None


BASE_DIR = Path(__file__).resolve().parents[1]
RESPONSE_FILE = BASE_DIR / "data" / "responses" / "user_study_responses.csv"
BACKUP_DIR = BASE_DIR / "data" / "response_backups"
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
        "confidence_scroll_depth_proxy",
        "confidence_hover_count_proxy",
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


def cohen_d(group_a, group_b):
    """Return Cohen's d for two independent groups."""
    group_a = pd.Series(group_a).dropna().astype(float)
    group_b = pd.Series(group_b).dropna().astype(float)
    if len(group_a) < 2 or len(group_b) < 2:
        return np.nan

    pooled_variance = (
        ((len(group_a) - 1) * group_a.var(ddof=1))
        + ((len(group_b) - 1) * group_b.var(ddof=1))
    ) / (len(group_a) + len(group_b) - 2)

    if pooled_variance <= 0:
        return 0.0
    return (group_b.mean() - group_a.mean()) / math.sqrt(pooled_variance)


def bootstrap_ci(group_a, group_b, iterations=2000, seed=42):
    """Estimate a 95% bootstrap CI for the Adaptive minus Static mean difference."""
    group_a = pd.Series(group_a).dropna().astype(float).to_numpy()
    group_b = pd.Series(group_b).dropna().astype(float).to_numpy()
    if len(group_a) == 0 or len(group_b) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(iterations):
        sample_a = rng.choice(group_a, size=len(group_a), replace=True)
        sample_b = rng.choice(group_b, size=len(group_b), replace=True)
        differences.append(sample_b.mean() - sample_a.mean())

    lower, upper = np.percentile(differences, [2.5, 97.5])
    return lower, upper


def compare_static_adaptive(filtered, metrics):
    """Build a Static vs Adaptive comparison table for results reporting."""
    rows = []
    static_rows = filtered[filtered["condition"] == "Static"]
    adaptive_rows = filtered[filtered["condition"] == "Adaptive"]

    for metric in metrics:
        static_values = static_rows[metric].dropna().astype(float)
        adaptive_values = adaptive_rows[metric].dropna().astype(float)
        if static_values.empty or adaptive_values.empty:
            continue

        mean_difference = adaptive_values.mean() - static_values.mean()
        ci_low, ci_high = bootstrap_ci(static_values, adaptive_values)
        p_value = np.nan
        if stats is not None and len(static_values) >= 2 and len(adaptive_values) >= 2:
            p_value = stats.mannwhitneyu(
                static_values,
                adaptive_values,
                alternative="two-sided",
            ).pvalue

        rows.append({
            "Metric": metric,
            "Static n": len(static_values),
            "Adaptive n": len(adaptive_values),
            "Static mean": round(static_values.mean(), 2),
            "Adaptive mean": round(adaptive_values.mean(), 2),
            "Mean difference": round(mean_difference, 2),
            "95% CI lower": round(ci_low, 2) if not np.isnan(ci_low) else np.nan,
            "95% CI upper": round(ci_high, 2) if not np.isnan(ci_high) else np.nan,
            "Cohen d": round(cohen_d(static_values, adaptive_values), 2),
            "Mann-Whitney p": round(p_value, 4) if not np.isnan(p_value) else "Install scipy",
        })

    return pd.DataFrame(rows)


def backup_response_file(reason="manual_row_delete"):
    """Back up the full response CSV before any admin deletion."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = BACKUP_DIR / f"user_study_responses_{reason}_{timestamp}.csv"
    shutil.copy2(RESPONSE_FILE, backup_path)
    return backup_path


def delete_response_rows(row_indices):
    """Delete selected zero-based response rows from the stored CSV."""
    stored_responses = pd.read_csv(RESPONSE_FILE)
    valid_indices = [index for index in row_indices if index in stored_responses.index]
    if not valid_indices:
        return None, 0

    backup_path = backup_response_file()
    stored_responses = stored_responses.drop(index=valid_indices).reset_index(drop=True)
    stored_responses.to_csv(RESPONSE_FILE, index=False)
    return backup_path, len(valid_indices)


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

tab_overview, tab_ratings, tab_comparison, tab_behaviour, tab_raw = st.tabs(
    ["Overview", "Ratings", "Static vs Adaptive", "Behaviour & Depth", "Raw Data"]
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

    if "participant_id" in filtered.columns:
        st.subheader("Participant Completion Check")
        participant_rows = (
            filtered["participant_id"]
            .replace("", pd.NA)
            .dropna()
            .value_counts()
            .rename_axis("Participant ID")
            .reset_index(name="Saved rows")
            .sort_values("Participant ID")
        )
        if not participant_rows.empty:
            participant_rows["Expected rows"] = 6
            participant_rows["Complete"] = participant_rows["Saved rows"] == 6
            st.dataframe(participant_rows, use_container_width=True)
            st.caption("Each participant should have exactly six saved rows: Applicant A to Applicant F.")

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

with tab_comparison:
    available_metrics = [metric for metric in RATING_METRICS if metric in filtered.columns]
    if {"Static", "Adaptive"}.issubset(set(filtered["condition"].dropna())) and available_metrics:
        st.subheader("Static vs Adaptive Rating Comparison")
        comparison_table = compare_static_adaptive(filtered, available_metrics)
        if comparison_table.empty:
            st.info("There is not enough rating data to compare the two conditions yet.")
        else:
            st.dataframe(comparison_table, use_container_width=True)
            st.caption(
                "Mean difference is Adaptive minus Static. Confidence intervals are bootstrapped. "
                "Mann-Whitney p-values are suitable for small ordinal rating samples, but should be interpreted cautiously."
            )
    else:
        st.info("Collect responses in both Static and Adaptive conditions before running this comparison.")

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
            "confidence_scroll_depth_proxy",
            "confidence_hover_count_proxy",
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
    raw_display = filtered.copy()
    raw_display.insert(0, "csv_row_number", raw_display.index + 2)
    st.dataframe(raw_display, use_container_width=True)

    with st.expander("Remove invalid responses", expanded=False):
        st.warning(
            "Use this only for invalid test rows or responses where the participant did not follow the study procedure. "
            "The full CSV is backed up before deletion."
        )
        deletion_options = [
            f"CSV row {index + 2} | {row.get('participant_id', '')} | {row.get('profile_id', '')} | {row.get('condition', '')}"
            for index, row in filtered.iterrows()
        ]
        option_to_index = {
            option: index
            for option, index in zip(deletion_options, filtered.index)
        }

        with st.form("delete_invalid_response_rows"):
            selected_rows = st.multiselect(
                "Select response rows to remove",
                deletion_options,
                help="CSV row numbers include the header row, matching spreadsheet view.",
            )
            confirmation = st.text_input(
                "Type DELETE to confirm",
                help="This prevents accidental deletion while reviewing data.",
            )
            delete_rows = st.form_submit_button("Delete selected rows")

        if delete_rows:
            if confirmation != "DELETE":
                st.error("Rows were not deleted because the confirmation text did not match DELETE.")
            elif not selected_rows:
                st.error("No rows selected for deletion.")
            else:
                selected_indices = [option_to_index[option] for option in selected_rows]
                backup_path, deleted_count = delete_response_rows(selected_indices)
                if deleted_count:
                    st.success(f"Deleted {deleted_count} row(s). Backup saved to: {backup_path}")
                    if hasattr(st, "rerun"):
                        st.rerun()
                    else:
                        st.experimental_rerun()
                else:
                    st.error("No matching stored rows were found to delete.")

    csv_data = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered responses",
        data=csv_data,
        file_name="filtered_user_study_responses.csv",
        mime="text/csv",
    )
