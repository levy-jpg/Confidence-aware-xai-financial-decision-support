import os
import shutil
from datetime import datetime

import pandas as pd


RESPONSE_FILE = "user_study_responses.csv"
BACKUP_DIR = "response_backups"
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


def main():
    if os.path.exists(RESPONSE_FILE):
        os.makedirs(BACKUP_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_path = os.path.join(BACKUP_DIR, f"user_study_responses_{timestamp}.csv")
        shutil.copy2(RESPONSE_FILE, backup_path)
        print(f"Backed up existing responses to: {backup_path}")
    else:
        print("No existing response file found. Creating a clean response file.")

    pd.DataFrame(columns=RESPONSE_COLUMNS).to_csv(RESPONSE_FILE, index=False)
    print(f"Reset complete: {RESPONSE_FILE} now contains headers only.")


if __name__ == "__main__":
    main()
