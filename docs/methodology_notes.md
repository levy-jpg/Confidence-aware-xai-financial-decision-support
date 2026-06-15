# Methodology Notes

Author: Levy Thiga Kariuki  
Student Number: G20893080

## Project Aim

This project investigates how confidence-aware adaptation can change the presentation of explainable AI outputs in a financial decision-support context. The system compares a static explanation condition with an adaptive condition that changes explanation depth based on confidence signals.

## Dataset

The financial prediction model uses the German Credit dataset from the UCI Machine Learning Repository.

- UCI dataset page: <https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data>
- Direct data file: <https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data>

The app attempts to load the dataset from UCI first. If the online source is unavailable, it falls back to the local copy stored at `data/raw/german.data`.

## Financial Prediction Model

The credit-risk prediction component uses an XGBoost classifier trained on the German Credit dataset. The saved model artefacts are stored in `models/` and loaded by the Streamlit app.

The app uses local feature-contribution values to explain individual predictions. In deployment, XGBoost's built-in contribution output is used because it avoids SHAP/XGBoost compatibility issues seen with some package versions.

## Static and Adaptive Explanation Conditions

The study compares two explanation conditions:

- Static: shows a fixed standard explanation with four local features.
- Adaptive: changes the explanation depth and presentation style using confidence signals.

Adaptive explanation depth:

- Low confidence: detailed explanation with more local factors and additional guidance.
- Medium confidence: moderate explanation with a focused set of factors.
- High confidence: concise explanation with the strongest local factors and a safety note.

## Confidence Model

The confidence model is a lightweight behavioural classifier trained on simulated interaction data. It supports the adaptive explanation controller by estimating whether the participant appears to have low, medium, or high confidence.

The deployed Streamlit prototype uses:

- decision time
- click count
- AI prediction review time
- workflow/review proxy for scroll depth
- workflow/review proxy for hover count
- self-reported confidence rating

Native Streamlit does not expose browser hover or scroll telemetry without a custom JavaScript component. For that reason, the app uses explicit review-confirmation checks as transparent proxies rather than hidden browser tracking. In this implementation, those values are used to demonstrate confidence-aware adaptive XAI logic rather than full behavioural monitoring.

## Response Data

Participant responses are saved locally to `data/responses/user_study_responses.csv`. The response CSV is ignored by Git so participant-study data is not accidentally pushed to the repository.

Key response fields include:

- participant ID
- applicant profile ID
- study condition
- initial user judgement
- self-reported confidence
- AI prediction and probability
- agreement with AI
- timing and interaction signals
- predicted behavioural confidence
- explanation depth
- trust, understanding, usefulness, and reliance ratings
- optional comments

## Analysis Approach

The analysis dashboard provides descriptive and comparative outputs for the results chapter.

Recommended reporting:

- response counts by condition
- mean trust, understanding, usefulness, and reliance by condition
- explanation depth by condition
- AI agreement rate by condition
- Static-vs-Adaptive mean differences
- bootstrapped confidence intervals
- Cohen's d effect sizes
- Mann-Whitney p-values for small ordinal rating samples

Statistical outputs should be interpreted cautiously if the study sample is small. The main academic value is the comparison of user experience and reliance patterns between static and adaptive explanation presentations.

## Ethics and Scope

The app is a simulated decision-support study. It does not make real financial decisions, does not process real personal financial data, and should not be interpreted as financial advice. Responses are anonymised using participant IDs and are intended for dissertation analysis only.
