"""
Repository smoke test for the Confidence-Aware XAI project.
Author: Levy Thiga Kariuki
Student Number: G20893080
"""

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "app.py",
    "README.md",
    "requirements.txt",
    "data/raw/german.data",
    "data/selected_applicant_profiles.csv",
    "data/simulated_behavioural_confidence_data.csv",
    "models/xgboost_credit_model.pkl",
    "models/label_encoders.pkl",
    "models/confidence_model.pkl",
    "models/confidence_scaler.pkl",
    "models/confidence_label_encoder.pkl",
    "notebooks/MSc_Project_Applicant_Profiles.ipynb",
    "notebooks/MSc_Project_Confidence_Aware_Adaptive_XAI_for_Financial_Decision_Support.ipynb",
    "docs/methodology_notes.md",
    "docs/study_design_summary.md",
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


def check_required_files():
    missing = [path for path in REQUIRED_FILES if not (BASE_DIR / path).exists()]
    if missing:
        print("Missing required files:")
        for path in missing:
            print(f" - {path}")
        return False

    print("Required project files: OK")
    return True


def check_data_files():
    profiles = pd.read_csv(BASE_DIR / "data" / "selected_applicant_profiles.csv")
    behaviour = pd.read_csv(BASE_DIR / "data" / "simulated_behavioural_confidence_data.csv")

    if profiles.empty:
        print("Applicant profile CSV is empty.")
        return False
    if behaviour.empty:
        print("Simulated behavioural confidence CSV is empty.")
        return False

    print(f"Applicant profiles: OK ({len(profiles)} rows)")
    print(f"Behavioural confidence data: OK ({len(behaviour)} rows)")
    return True


def check_response_schema():
    response_file = BASE_DIR / "data" / "responses" / "user_study_responses.csv"
    if not response_file.exists():
        print("Response CSV not present yet; this is OK before data collection.")
        return True

    responses = pd.read_csv(response_file, nrows=0)
    missing_columns = [column for column in RESPONSE_COLUMNS if column not in responses.columns]
    if missing_columns:
        print("Response CSV uses an older schema and is missing:")
        for column in missing_columns:
            print(f" - {column}")
        print("Run python scripts/reset_responses.py before formal data collection to recreate clean headers.")
        return True

    print("Response CSV schema: OK")
    return True


def main():
    checks = [
        check_required_files(),
        check_data_files(),
        check_response_schema(),
    ]

    if all(checks):
        print("Smoke test passed.")
    else:
        raise SystemExit("Smoke test failed.")


if __name__ == "__main__":
    main()
