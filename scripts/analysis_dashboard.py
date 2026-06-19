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
import altair as alt
import streamlit as st

try:
    from scipy import stats
except ImportError:
    stats = None


BASE_DIR = Path(__file__).resolve().parents[1]
RESPONSE_FILE = BASE_DIR / "data" / "responses" / "user_study_responses.csv"
BACKUP_DIR = BASE_DIR / "data" / "response_backups"
RATING_METRICS = ["trust", "understanding", "usefulness", "reliance"]
TIMING_METRICS = [
    "decision_time",
    "ai_prediction_review_time",
    "explanation_reading_time",
    "interaction_count",
]
DISPLAY_LABELS = {
    "trust": "Trust",
    "understanding": "Understanding",
    "usefulness": "Usefulness",
    "reliance": "Reliance",
    "decision_time": "Initial decision time (seconds)",
    "ai_prediction_review_time": "AI prediction review time (seconds)",
    "explanation_reading_time": "Explanation reading time (seconds)",
    "explanation_view_time": "Explanation reading time (seconds)",
    "interaction_count": "Interaction count",
    "confidence_scroll_depth_proxy": "Review depth proxy",
    "confidence_hover_count_proxy": "Review confirmation proxy",
    "top_features_shown": "Top features shown",
    "probability_bad_credit": "Bad-credit probability",
    "user_confidence": "Self-reported confidence",
}
SECOND_METRICS = {
    "decision_time",
    "ai_prediction_review_time",
    "explanation_reading_time",
    "explanation_view_time",
}
CONFIDENCE_ORDER = ["low", "medium", "high"]
STOPWORDS = {
    "the", "and", "was", "were", "that", "this", "with", "for", "from",
    "about", "because", "very", "really", "good", "clear", "case", "they",
    "their", "explanation", "prediction", "credit", "risk", "more", "less",
    "useful", "helped", "understand", "understanding", "would", "could",
    "should", "applicant", "details", "feature", "features",
}


def display_label(column):
    """Return a reader-friendly label for dashboard columns."""
    return DISPLAY_LABELS.get(column, column.replace("_", " ").title())


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
            "Metric": display_label(metric),
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


def get_available_columns(dataframe, columns):
    """Return columns that exist in the current response file."""
    return [column for column in columns if column in dataframe.columns]


def make_long_ratings(dataframe):
    """Convert trust/understanding/usefulness/reliance ratings to long format."""
    available_metrics = get_available_columns(dataframe, RATING_METRICS)
    if not available_metrics:
        return pd.DataFrame()

    id_vars = [column for column in ["participant_id", "profile_id", "condition"] if column in dataframe.columns]
    return dataframe.melt(
        id_vars=id_vars,
        value_vars=available_metrics,
        var_name="Metric",
        value_name="Rating",
    ).dropna(subset=["Rating"]).assign(
        Metric=lambda frame: frame["Metric"].map(display_label)
    )


def summarise_by_condition(dataframe, metrics):
    """Create count, mean, median, and spread summaries by condition."""
    rows = []
    for condition, condition_rows in dataframe.groupby("condition"):
        for metric in metrics:
            values = condition_rows[metric].dropna().astype(float)
            if values.empty:
                continue
            rows.append({
                "Condition": condition,
                "Metric": display_label(metric),
                "n": len(values),
                "Mean": round(values.mean(), 2),
                "Median": round(values.median(), 2),
                "SD": round(values.std(ddof=1), 2) if len(values) > 1 else 0.0,
                "Min": round(values.min(), 2),
                "Max": round(values.max(), 2),
            })
    return pd.DataFrame(rows)


def summarise_selected_signal(dataframe, signal):
    """Summarise one interaction signal with clearer columns for dashboard display."""
    rows = []
    for condition, condition_rows in dataframe.groupby("condition"):
        values = condition_rows[signal].dropna().astype(float)
        if values.empty:
            continue
        rows.append({
            "Condition": condition,
            "n": len(values),
            "Mean": round(values.mean(), 2),
            "Median": round(values.median(), 2),
            "SD": round(values.std(ddof=1), 2) if len(values) > 1 else 0.0,
            "Min": round(values.min(), 2),
            "Max": round(values.max(), 2),
        })
    return pd.DataFrame(rows)


def participant_completion_table(dataframe):
    """Summarise whether each participant completed the six-case procedure."""
    if "participant_id" not in dataframe.columns:
        return pd.DataFrame()

    table = (
        dataframe["participant_id"]
        .replace("", pd.NA)
        .dropna()
        .value_counts()
        .rename_axis("Participant ID")
        .reset_index(name="Saved rows")
        .sort_values("Participant ID")
    )
    if table.empty:
        return table

    table["Expected rows"] = 6
    table["Complete"] = table["Saved rows"] == 6
    return table


def extract_comment_terms(dataframe, top_n=20):
    """Return simple word-frequency counts from optional participant comments."""
    if "comments" not in dataframe.columns:
        return pd.DataFrame()

    comments = dataframe["comments"].dropna().astype(str)
    terms = []
    for comment in comments:
        cleaned = "".join(character.lower() if character.isalnum() else " " for character in comment)
        terms.extend(
            term for term in cleaned.split()
            if len(term) > 2 and term not in STOPWORDS and not term.isdigit()
        )

    if not terms:
        return pd.DataFrame()

    return (
        pd.Series(terms)
        .value_counts()
        .head(top_n)
        .rename_axis("Term")
        .reset_index(name="Mentions")
    )


def render_bar_chart(dataframe, x, y, color=None, title=None):
    """Render a compact Altair bar chart."""
    chart = alt.Chart(dataframe).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X(x),
        y=alt.Y(y),
        tooltip=list(dataframe.columns),
    )
    if color:
        chart = chart.encode(color=alt.Color(color))
    if title:
        chart = chart.properties(title=title)
    st.altair_chart(chart, use_container_width=True)


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
st.caption("Results dashboard for the static versus adaptive explanation user study.")

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

if "condition" in filtered.columns:
    condition_counts = filtered["condition"].value_counts()
    static_count = int(condition_counts.get("Static", 0))
    adaptive_count = int(condition_counts.get("Adaptive", 0))
    balance_gap = abs(static_count - adaptive_count)
    st.caption(
        f"Condition balance gap: {balance_gap} response(s). "
        "A balanced full study should have 30 Static and 30 Adaptive responses."
    )

st.divider()

tab_overview, tab_ratings, tab_comparison, tab_behaviour, tab_comments, tab_raw = st.tabs(
    ["Overview", "Ratings", "Static vs Adaptive", "Behaviour & Confidence", "Comments", "Raw Data"]
)

with tab_overview:
    st.subheader("Responses by Condition")
    condition_counts = filtered["condition"].value_counts().rename_axis("Condition").reset_index(name="Responses")
    render_bar_chart(condition_counts, "Condition:N", "Responses:Q", color="Condition:N")

    if "user_agreed_with_ai" in filtered.columns:
        st.subheader("Agreement with AI")
        agreement_summary = (
            filtered.groupby("condition")["user_agreed_with_ai"]
            .mean()
            .mul(100)
            .round(1)
            .rename_axis("Condition")
            .reset_index(name="Agreement rate (%)")
        )
        render_bar_chart(agreement_summary, "Condition:N", "Agreement rate (%):Q", color="Condition:N")
        st.caption("Percentage of responses where the initial user judgement matched the AI prediction.")

    if "participant_id" in filtered.columns:
        st.subheader("Participant Completion Check")
        participant_rows = participant_completion_table(filtered)
        if not participant_rows.empty:
            st.dataframe(participant_rows, use_container_width=True)
            st.caption("Each participant should have exactly six saved rows: Applicant A to Applicant F.")

    if {"profile_id", "condition"}.issubset(filtered.columns):
        st.subheader("Applicant Coverage by Condition")
        profile_condition = (
            filtered.groupby(["profile_id", "condition"])
            .size()
            .reset_index(name="Responses")
        )
        heatmap = alt.Chart(profile_condition).mark_rect().encode(
            x=alt.X("profile_id:N", title="Applicant case"),
            y=alt.Y("condition:N", title="Condition"),
            color=alt.Color("Responses:Q", scale=alt.Scale(scheme="blues")),
            tooltip=["profile_id", "condition", "Responses"],
        )
        text = alt.Chart(profile_condition).mark_text(color="white", fontWeight="bold").encode(
            x="profile_id:N",
            y="condition:N",
            text="Responses:Q",
        )
        st.altair_chart((heatmap + text), use_container_width=True)
        st.caption("This checks whether each applicant case appears under both explanation conditions across the study.")

with tab_ratings:
    available_metrics = [metric for metric in RATING_METRICS if metric in filtered.columns]
    if available_metrics:
        st.subheader("Mean Ratings by Condition")
        metric_summary = filtered.groupby("condition")[available_metrics].mean().round(2)
        st.dataframe(metric_summary, use_container_width=True)
        metric_summary_long = (
            metric_summary
            .reset_index()
            .melt(id_vars="condition", var_name="Metric", value_name="Mean rating")
        )
        metric_summary_long["Metric"] = metric_summary_long["Metric"].map(display_label)
        mean_chart = alt.Chart(metric_summary_long).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("Metric:N", title="Evaluation metric"),
            y=alt.Y("Mean rating:Q", scale=alt.Scale(domain=[1, 5])),
            color=alt.Color("condition:N", title="Condition"),
            xOffset="condition:N",
            tooltip=["condition", "Metric", "Mean rating"],
        )
        st.altair_chart(mean_chart, use_container_width=True)

        st.subheader("Likert Rating Distribution")
        long_ratings = make_long_ratings(filtered)
        likert_counts = (
            long_ratings.groupby(["Metric", "condition", "Rating"])
            .size()
            .reset_index(name="Responses")
        )
        likert_chart = alt.Chart(likert_counts).mark_bar().encode(
            x=alt.X("Rating:O", title="Rating (1-5)"),
            y=alt.Y("Responses:Q"),
            color=alt.Color("condition:N", title="Condition"),
            column=alt.Column("Metric:N", title=None),
            tooltip=["Metric", "condition", "Rating", "Responses"],
        ).properties(width=150)
        st.altair_chart(likert_chart, use_container_width=True)

        st.subheader("Rating Spread by Condition")
        box_chart = alt.Chart(long_ratings).mark_boxplot(extent="min-max").encode(
            x=alt.X("condition:N", title="Condition"),
            y=alt.Y("Rating:Q", scale=alt.Scale(domain=[1, 5])),
            color=alt.Color("condition:N", legend=None),
            column=alt.Column("Metric:N", title=None),
            tooltip=["Metric", "condition", "Rating"],
        ).properties(width=150)
        st.altair_chart(box_chart, use_container_width=True)

        st.subheader("Descriptive Rating Table")
        st.dataframe(summarise_by_condition(filtered, available_metrics), use_container_width=True)
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
            diff_chart = alt.Chart(comparison_table).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                x=alt.X("Metric:N", title="Evaluation metric"),
                y=alt.Y("Mean difference:Q", title="Adaptive minus Static"),
                color=alt.condition(
                    alt.datum["Mean difference"] >= 0,
                    alt.value("#22c55e"),
                    alt.value("#ef4444"),
                ),
                tooltip=[
                    "Metric",
                    "Static mean",
                    "Adaptive mean",
                    "Mean difference",
                    "95% CI lower",
                    "95% CI upper",
                    "Cohen d",
                    "Mann-Whitney p",
                ],
            )
            zero_rule = alt.Chart(pd.DataFrame({"Mean difference": [0]})).mark_rule(color="#64748b").encode(
                y="Mean difference:Q"
            )
            error_bars = alt.Chart(comparison_table).mark_errorbar(ticks=True).encode(
                x=alt.X("Metric:N", title="Evaluation metric"),
                y=alt.Y("95% CI lower:Q", title="Adaptive minus Static"),
                y2="95% CI upper:Q",
                tooltip=["Metric", "95% CI lower", "95% CI upper"],
            )
            st.altair_chart((diff_chart + error_bars + zero_rule), use_container_width=True)
            st.caption(
                "Mean difference is Adaptive minus Static. Confidence intervals are bootstrapped. "
                "Mann-Whitney p-values are suitable for small ordinal rating samples, but should be interpreted cautiously."
            )

            st.subheader("Interpretation Notes")
            strongest_metric = comparison_table.iloc[
                comparison_table["Mean difference"].abs().idxmax()
            ]
            st.write(
                f"Largest observed mean difference: **{strongest_metric['Metric']}** "
                f"({strongest_metric['Mean difference']} points, Adaptive minus Static)."
            )
            st.write(
                "Use this table as descriptive support for the results chapter, then discuss sample size and "
                "the exploratory nature of the inferential tests."
            )
            st.download_button(
                "Download comparison table",
                data=comparison_table.to_csv(index=False).encode("utf-8"),
                file_name="static_vs_adaptive_comparison.csv",
                mime="text/csv",
            )
    else:
        st.info("Collect responses in both Static and Adaptive conditions before running this comparison.")

with tab_behaviour:
    if "top_features_shown" in filtered.columns:
        st.subheader("Explanation Depth by Condition")
        depth_summary = (
            filtered.groupby("condition")["top_features_shown"]
            .mean()
            .round(2)
            .rename_axis("Condition")
            .reset_index(name="Mean top features shown")
        )
        render_bar_chart(depth_summary, "Condition:N", "Mean top features shown:Q", color="Condition:N")

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
        selected_signal = st.selectbox(
            "Choose interaction signal",
            timing_cols,
            index=timing_cols.index("ai_prediction_review_time")
            if "ai_prediction_review_time" in timing_cols else 0,
            format_func=display_label,
        )

        signal_df = filtered[["condition", selected_signal]].dropna().copy()
        signal_df[selected_signal] = pd.to_numeric(signal_df[selected_signal], errors="coerce")
        signal_df = signal_df.dropna(subset=[selected_signal])
        signal_df["Display value"] = signal_df[selected_signal]

        cap_value = np.nan
        if selected_signal in SECOND_METRICS and not signal_df.empty:
            cap_value = signal_df[selected_signal].quantile(0.95)
            if pd.notna(cap_value) and cap_value > 0:
                signal_df["Display value"] = signal_df[selected_signal].clip(upper=cap_value)

        signal_cols = st.columns([1, 2])
        with signal_cols[0]:
            st.dataframe(
                summarise_selected_signal(filtered, selected_signal),
                use_container_width=True,
                hide_index=True,
            )
        with signal_cols[1]:
            box_chart = alt.Chart(signal_df).mark_boxplot(extent="min-max", size=55).encode(
                x=alt.X("condition:N", title="Condition"),
                y=alt.Y("Display value:Q", title=display_label(selected_signal)),
                color=alt.Color("condition:N", legend=None),
                tooltip=[
                    alt.Tooltip("condition:N", title="Condition"),
                    alt.Tooltip(f"{selected_signal}:Q", title=display_label(selected_signal), format=".2f"),
                ],
            )
            st.altair_chart(box_chart, use_container_width=True)
            histogram_chart = alt.Chart(signal_df).mark_bar(
                opacity=0.85,
                cornerRadiusTopLeft=2,
                cornerRadiusTopRight=2,
            ).encode(
                x=alt.X(
                    "Display value:Q",
                    bin=alt.Bin(maxbins=16),
                    title=display_label(selected_signal),
                ),
                y=alt.Y("count():Q", title="Responses"),
                color=alt.Color("condition:N", title="Condition"),
                row=alt.Row("condition:N", title=None),
                tooltip=[
                    alt.Tooltip("condition:N", title="Condition"),
                    alt.Tooltip("count():Q", title="Responses"),
                ],
            ).properties(height=90)
            st.altair_chart(histogram_chart, use_container_width=True)

        if selected_signal in SECOND_METRICS and pd.notna(cap_value):
            max_value = signal_df[selected_signal].max()
            if max_value > cap_value:
                st.caption(
                    f"Visual capped at the 95th percentile ({cap_value:.2f} seconds) so a long pause "
                    "does not flatten the chart. The summary table keeps the raw values."
                )

        timing_summary = summarise_by_condition(filtered, timing_cols)
        timing_summary["Signal"] = timing_summary["Metric"]
        max_mean_by_signal = timing_summary.groupby("Signal")["Mean"].transform("max")
        timing_summary["Relative mean"] = np.where(
            max_mean_by_signal > 0,
            timing_summary["Mean"] / max_mean_by_signal * 100,
            0,
        )
        timing_summary_chart = alt.Chart(timing_summary).mark_bar(
            cornerRadiusTopLeft=3,
            cornerRadiusTopRight=3,
        ).encode(
            x=alt.X(
                "Relative mean:Q",
                title="Relative mean within each signal",
                scale=alt.Scale(domain=[0, 100]),
            ),
            y=alt.Y("Signal:N", title=None, sort="-x"),
            color=alt.Color("Condition:N"),
            row=alt.Row("Condition:N", title=None),
            tooltip=["Condition", "Signal", "Mean", "Median", "SD", "Relative mean"],
        ).properties(height=120)
        st.altair_chart(timing_summary_chart, use_container_width=True)
        st.caption(
            "The selected signal chart uses its own scale. The summary bars give a quick overview of all "
            "available behavioural signals by comparing each condition against the largest mean for that signal."
        )

    if "adaptation_signal" in filtered.columns:
        st.subheader("Adaptation Signals")
        signal_counts = (
            filtered["adaptation_signal"]
            .fillna("Missing")
            .value_counts()
            .rename_axis("Adaptation signal")
            .reset_index(name="Responses")
        )
        render_bar_chart(signal_counts, "Adaptation signal:N", "Responses:Q", color="Adaptation signal:N")

    if {"predicted_confidence", "reported_confidence_level"}.issubset(filtered.columns):
        st.subheader("Predicted vs Reported Confidence")
        confidence_table = (
            filtered.groupby(["reported_confidence_level", "predicted_confidence"])
            .size()
            .reset_index(name="Responses")
        )
        confidence_heatmap = alt.Chart(confidence_table).mark_rect().encode(
            x=alt.X("reported_confidence_level:N", sort=CONFIDENCE_ORDER, title="Self-reported confidence"),
            y=alt.Y("predicted_confidence:N", sort=CONFIDENCE_ORDER, title="Behavioural confidence"),
            color=alt.Color("Responses:Q", scale=alt.Scale(scheme="purples")),
            tooltip=["reported_confidence_level", "predicted_confidence", "Responses"],
        )
        confidence_text = alt.Chart(confidence_table).mark_text(color="white", fontWeight="bold").encode(
            x=alt.X("reported_confidence_level:N", sort=CONFIDENCE_ORDER),
            y=alt.Y("predicted_confidence:N", sort=CONFIDENCE_ORDER),
            text="Responses:Q",
        )
        st.altair_chart((confidence_heatmap + confidence_text), use_container_width=True)

    if {"user_agreed_with_ai", "reliance", "condition"}.issubset(filtered.columns):
        st.subheader("Reliance by AI Agreement")
        reliance_summary = (
            filtered.groupby(["condition", "user_agreed_with_ai"])["reliance"]
            .mean()
            .round(2)
            .reset_index(name="Mean reliance")
        )
        reliance_summary["Agreement with AI"] = reliance_summary["user_agreed_with_ai"].map({
            True: "Agreed",
            False: "Disagreed",
        })
        reliance_chart = alt.Chart(reliance_summary).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("Agreement with AI:N"),
            y=alt.Y("Mean reliance:Q", scale=alt.Scale(domain=[1, 5])),
            color=alt.Color("condition:N", title="Condition"),
            xOffset="condition:N",
            tooltip=["condition", "Agreement with AI", "Mean reliance"],
        )
        st.altair_chart(reliance_chart, use_container_width=True)

with tab_comments:
    st.subheader("Participant Comments")
    if "comments" not in filtered.columns:
        st.info("No comments column is available in the response file.")
    else:
        comments_df = filtered[
            filtered["comments"].notna()
            & (filtered["comments"].astype(str).str.strip() != "")
        ].copy()

        comment_cols = st.columns(3)
        with comment_cols[0]:
            st.metric("Commented rows", len(comments_df))
        with comment_cols[1]:
            comment_rate = (len(comments_df) / len(filtered) * 100) if len(filtered) else 0
            st.metric("Comment rate", f"{comment_rate:.1f}%")
        with comment_cols[2]:
            avg_length = comments_df["comments"].astype(str).str.len().mean() if not comments_df.empty else 0
            st.metric("Avg. comment length", f"{avg_length:.0f} chars")

        if comments_df.empty:
            st.info("No written comments are available for the selected filters.")
        else:
            st.subheader("Comment Count by Condition")
            comment_condition = (
                comments_df.groupby("condition")
                .size()
                .rename_axis("Condition")
                .reset_index(name="Comments")
            )
            render_bar_chart(comment_condition, "Condition:N", "Comments:Q", color="Condition:N")

            st.subheader("Frequent Comment Terms")
            term_counts = extract_comment_terms(comments_df)
            if term_counts.empty:
                st.info("There are too few comment terms to summarise.")
            else:
                render_bar_chart(term_counts, "Mentions:Q", "Term:N")

            st.subheader("Comment Review Table")
            review_columns = [
                column for column in [
                    "participant_id",
                    "profile_id",
                    "condition",
                    "explanation_depth",
                    "trust",
                    "understanding",
                    "usefulness",
                    "reliance",
                    "comments",
                ]
                if column in comments_df.columns
            ]
            st.dataframe(comments_df[review_columns], use_container_width=True)

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
