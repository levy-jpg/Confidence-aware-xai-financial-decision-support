# Confidence-Aware XAI Prototype

Streamlit prototype for a confidence-aware explainable AI financial decision-support study.

The app uses a trained XGBoost credit-risk model, SHAP-style local feature contributions, and a lightweight behavioural confidence model to compare static and adaptive explanations.

## Run

```bash
streamlit run app.py
```

## Project Structure

```text
app.py                         # Main Streamlit participant-facing prototype
data/                          # Study data and response storage
  selected_applicant_profiles.csv
  simulated_behavioural_confidence_data.csv
  responses/                   # Local response CSVs are ignored by Git
  response_backups/            # Local reset backups are ignored by Git
models/                        # Trained model and preprocessing artefacts
scripts/                       # Analysis and maintenance utilities
```

## Study Flow

1. Review a predefined applicant profile.
2. Submit an initial good/bad credit judgement and confidence rating.
3. View the AI prediction and bad-credit probability.
4. Generate either a static or adaptive explanation.
5. Submit evaluation responses for trust, understanding, usefulness, reliance, and comments.

## Explanation Conditions

- Static: shows a fixed standard explanation with four features.
- Adaptive: adjusts explanation depth using the full trained behavioural confidence-feature row plus self-reported confidence.
  - Low confidence: detailed explanation.
  - Medium confidence: moderate explanation.
  - High confidence: concise explanation.

Responses are saved to `data/responses/user_study_responses.csv`.

Participant IDs are assigned automatically as anonymised codes such as `P001`, `P002`, and so on. The same ID should be used for all cases completed by one participant. Use the sidebar `Start new participant` button when a new person begins the study.

Timing fields distinguish between `ai_prediction_review_time`, measured before the explanation is requested, and `explanation_reading_time`, measured after the explanation is shown and before the evaluation is submitted. The legacy `explanation_view_time` column is kept as the actual explanation-reading time for compatibility.

## Study Tools

The sidebar includes an optional anonymised participant ID, explanation condition guidance, study progress, a system status panel, and a download button for saved responses.

The app also starts with a short consent/introduction screen explaining that the task is simulated, uses predefined profiles, and does not collect real financial data.

The interface includes a condition badge, model confidence signal, user-versus-AI comparison panel, and explanation-depth indicator to make the adaptive behaviour visible during testing.

## Quick Analysis

After collecting responses, run:

```bash
python scripts/analysis_summary.py
```

This prints response counts and mean trust, understanding, usefulness, and reliance scores by condition.

For a visual dashboard, run:

```bash
streamlit run scripts/analysis_dashboard.py
```

The dashboard plots condition-level ratings, explanation depth, AI agreement, and raw responses.

## Reset Responses

Do not reset responses from the participant-facing app. To clear the CSV deliberately, run:

```bash
python scripts/reset_responses.py
```

The script backs up the current CSV into `data/response_backups/` first, then recreates `data/responses/user_study_responses.csv` with clean headers.
