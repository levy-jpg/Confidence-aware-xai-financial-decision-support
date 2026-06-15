"""
Command-line response summary for the Confidence-Aware XAI user study.
Author: Levy Thiga Kariuki
Student Number: G20893080
"""

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RESPONSE_FILE = BASE_DIR / "data" / "responses" / "user_study_responses.csv"
METRICS = ["trust", "understanding", "usefulness", "reliance"]


def load_responses():
    responses = pd.read_csv(RESPONSE_FILE)

    for column in METRICS + [
        "top_features_shown",
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

    return responses


def main():
    if not RESPONSE_FILE.exists():
        print(f"No response file found: {RESPONSE_FILE}")
        return

    responses = load_responses()
    if responses.empty:
        print("Response file exists, but it has no rows yet.")
        return

    print("Study response summary")
    print("======================")
    print(f"Rows: {len(responses)}")

    if "participant_id" in responses.columns:
        participant_count = responses["participant_id"].dropna().astype(str).replace("", pd.NA).dropna().nunique()
        print(f"Participants with IDs: {participant_count}")

    print("\nResponses by condition")
    print(responses["condition"].value_counts(dropna=False).to_string())

    available_metrics = [metric for metric in METRICS if metric in responses.columns]
    if available_metrics:
        print("\nMean ratings by condition")
        print(responses.groupby("condition")[available_metrics].mean().round(2).to_string())

    if "top_features_shown" in responses.columns:
        print("\nAverage explanation depth by condition")
        print(responses.groupby("condition")["top_features_shown"].mean().round(2).to_string())

    if "user_agreed_with_ai" in responses.columns:
        print("\nAgreement with AI by condition")
        agreement = responses.groupby("condition")["user_agreed_with_ai"].mean().mul(100).round(1)
        print(agreement.to_string())


if __name__ == "__main__":
    main()
