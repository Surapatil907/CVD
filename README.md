# Streamlit UI — Multiclass CVD Dissertation Model

This folder contains a ready-to-run Streamlit interface for the final tuned
Gradient Boosting model used in the dissertation.

## Included files

- `app.py` — Streamlit user interface
- `final_gradient_boosting_pipeline.joblib` — fitted preprocessing + classifier pipeline
- `final_feature_importance.csv` — final global feature-importance table
- `requirements.txt` — Python dependencies
- `.streamlit/config.toml` — light UI theme configuration

## Run locally

Recommended: Python 3.11 or 3.12.

```bash
python -m venv .venv
```

Activate the environment:

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
streamlit run app.py
```

Streamlit will normally open the app in your browser at:

```text
http://localhost:8501
```

## Deploy with Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload all files in this folder.
3. Sign in to Streamlit Community Cloud.
4. Create a new app from the GitHub repository.
5. Select `app.py` as the entry-point file.
6. Deploy.

## Model inputs

The app expects these 10 predictors in this exact schema:

- `age`
- `sex`
- `bmi`
- `systolic_bp`
- `diastolic_bp`
- `ldl`
- `hdl`
- `fasting_glucose`
- `smoking_status`
- `resting_heart_rate`

Target classes:

- 0 — No established CVD / Control
- 1 — Coronary Artery Disease (CAD)
- 2 — Hypertensive Heart Disease (HHD)
- 3 — Heart Failure with Reduced Ejection Fraction (HFrEF)

## Final reported model performance

- Accuracy: 0.5592
- Macro F1: 0.5548
- Macro ROC-AUC (One-vs-Rest): 0.8048

## Important limitation

This is a research prototype trained on a synthetic dataset. It is not a
clinically validated diagnostic system and must not be used for medical
decision-making.
