from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "final_gradient_boosting_pipeline.joblib"
IMPORTANCE_PATH = APP_DIR / "final_feature_importance.csv"

FEATURES = [
    "age",
    "sex",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "ldl",
    "hdl",
    "fasting_glucose",
    "smoking_status",
    "resting_heart_rate",
]

CLASS_NAMES = {
    0: "No established CVD / Control",
    1: "Coronary Artery Disease (CAD)",
    2: "Hypertensive Heart Disease (HHD)",
    3: "Heart Failure with Reduced Ejection Fraction (HFrEF)",
}

CLASS_SHORT = {
    0: "Control",
    1: "CAD",
    2: "HHD",
    3: "HFrEF",
}

SEX_MAP = {"Female": 0, "Male": 1}
SMOKING_MAP = {"Never": 0, "Former": 1, "Current": 2}

TRAINING_RANGES = {
    "Age": "40–85 years",
    "BMI": "18–45 kg/m²",
    "Systolic BP": "90–200 mmHg",
    "Diastolic BP": "50–120 mmHg",
    "LDL": "40–250 mg/dL",
    "HDL": "20–100 mg/dL",
    "Fasting glucose": "65–250 mg/dL",
    "Resting heart rate": "45–120 bpm",
}


st.set_page_config(
    page_title="CVD Multiclass Classifier",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2rem;
            max-width: 1250px;
        }
        .hero {
            padding: 1.35rem 1.5rem;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(124,58,237,0.14), rgba(37,99,235,0.08));
            border: 1px solid rgba(124,58,237,0.18);
            margin-bottom: 1.2rem;
        }
        .hero h1 {
            margin: 0 0 .35rem 0;
            font-size: 2rem;
        }
        .hero p {
            margin: 0;
            opacity: 0.82;
        }
        .result-card {
            border: 1px solid rgba(124,58,237,0.25);
            border-radius: 16px;
            padding: 1.1rem 1.2rem;
            margin: .4rem 0 1rem 0;
        }
        .small-note {
            opacity: 0.75;
            font-size: .9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found."
        )
    return joblib.load(MODEL_PATH)


def make_input_frame(
    age,
    sex,
    bmi,
    systolic_bp,
    diastolic_bp,
    ldl,
    hdl,
    fasting_glucose,
    smoking_status,
    resting_heart_rate,
):
    return pd.DataFrame(
        [
            {
                "age": int(age),
                "sex": int(sex),
                "bmi": float(bmi),
                "systolic_bp": int(systolic_bp),
                "diastolic_bp": int(diastolic_bp),
                "ldl": float(ldl),
                "hdl": float(hdl),
                "fasting_glucose": float(fasting_glucose),
                "smoking_status": int(smoking_status),
                "resting_heart_rate": float(resting_heart_rate),
            }
        ],
        columns=FEATURES,
    )


def validate_profile(frame):
    warnings = []
    row = frame.iloc[0]

    if row["systolic_bp"] <= row["diastolic_bp"]:
        warnings.append(
            "Systolic blood pressure should be greater than diastolic blood pressure."
        )

    pulse_pressure = row["systolic_bp"] - row["diastolic_bp"]
    if pulse_pressure < 15 or pulse_pressure > 100:
        warnings.append(
            "Pulse pressure is outside the 15–100 mmHg range used by the synthetic generator."
        )

    return warnings


try:
    model = load_model()
except Exception as exc:
    st.error(f"Unable to load the trained model: {exc}")
    st.stop()


with st.sidebar:
    st.title("Model summary")
    st.metric("Final test accuracy", "55.92%")
    st.metric("Macro F1", "0.5548")
    st.metric("Macro ROC-AUC (OvR)", "0.8048")
    st.divider()
    st.caption(
        "Final tuned Gradient Boosting classifier trained on 12,000 "
        "synthetic records across 4 balanced classes."
    )
    st.warning(
        "Research prototype only. This application is not a medical device "
        "and must not be used for diagnosis or treatment decisions."
    )


st.markdown(
    """
    <div class="hero">
        <h1> Multiclass Cardiovascular Disease Classifier</h1>
        <p>
            Streamlit interface for testing the dissertation's final tuned
            Gradient Boosting model on the 10 clinical/risk-factor predictors.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

predict_tab, model_tab, guide_tab = st.tabs(
    ["Single prediction", "Model information", "Input guide"]
)


with predict_tab:
    st.subheader("Enter a clinical profile")
    st.caption(
        "Inputs are constrained to the synthetic-data ranges used during model development."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.slider("Age (years)", 40, 85, 60, 1)
        sex_label = st.selectbox("Sex", list(SEX_MAP.keys()), index=1)
        bmi = st.slider("BMI (kg/m²)", 18.0, 45.0, 27.0, 0.1)
        smoking_label = st.selectbox(
            "Smoking status", list(SMOKING_MAP.keys()), index=0
        )

    with c2:
        systolic_bp = st.slider("Systolic BP (mmHg)", 90, 200, 135, 1)
        diastolic_bp = st.slider("Diastolic BP (mmHg)", 50, 120, 80, 1)
        resting_heart_rate = st.slider(
            "Resting heart rate (bpm)", 45.0, 120.0, 70.0, 0.5
        )

    with c3:
        ldl = st.slider("LDL cholesterol (mg/dL)", 40.0, 250.0, 140.0, 1.0)
        hdl = st.slider("HDL cholesterol (mg/dL)", 20.0, 100.0, 55.0, 1.0)
        fasting_glucose = st.slider(
            "Fasting glucose (mg/dL)", 65.0, 250.0, 100.0, 1.0
        )

    profile = make_input_frame(
        age=age,
        sex=SEX_MAP[sex_label],
        bmi=bmi,
        systolic_bp=systolic_bp,
        diastolic_bp=diastolic_bp,
        ldl=ldl,
        hdl=hdl,
        fasting_glucose=fasting_glucose,
        smoking_status=SMOKING_MAP[smoking_label],
        resting_heart_rate=resting_heart_rate,
    )

    profile_warnings = validate_profile(profile)

    if st.button("Predict CVD class", type="primary", width="stretch"):
        if any(
            "Systolic blood pressure should be greater" in w
            for w in profile_warnings
        ):
            for message in profile_warnings:
                st.error(message)
        else:
            for message in profile_warnings:
                st.warning(message)

            predicted_id = int(model.predict(profile)[0])
            probabilities = model.predict_proba(profile)[0]
            confidence = float(np.max(probabilities))

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="small-note">Predicted class</div>
                    <h2>{CLASS_NAMES[predicted_id]}</h2>
                    <div class="small-note">
                        Highest model probability: {confidence:.1%}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            probability_df = pd.DataFrame(
                {
                    "Class": [CLASS_SHORT[i] for i in range(4)],
                    "Probability": probabilities,
                }
            ).set_index("Class")

            left, right = st.columns([1, 1.25])

            with left:
                st.dataframe(
                    probability_df.style.format({"Probability": "{:.2%}"}),
                    width='stretch',
                )

            with right:
                st.bar_chart(probability_df, y="Probability")

            with st.expander("Show model input row"):
                display_profile = profile.copy()
                display_profile["sex"] = sex_label
                display_profile["smoking_status"] = smoking_label
                st.dataframe(display_profile, width='stretch')

            st.info(
                "Interpret this as a model prediction on synthetic-data patterns, "
                "not as a clinical diagnosis."
            )


with model_tab:
    st.subheader("Final dissertation model")

    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", "0.5592")
    m2.metric("Macro F1", "0.5548")
    m3.metric("Macro ROC-AUC", "0.8048")

    st.markdown(
        """
        **Classifier:** Gradient Boosting

        **Best hyperparameters**
        - `n_estimators = 150`
        - `learning_rate = 0.05`
        - `max_depth = 3`
        - `min_samples_leaf = 20`
        - `subsample = 0.9`
        - `max_features = "sqrt"`

        **Target classes**
        - 0 — No established CVD / Control
        - 1 — Coronary Artery Disease (CAD)
        - 2 — Hypertensive Heart Disease (HHD)
        - 3 — Heart Failure with Reduced Ejection Fraction (HFrEF)
        """
    )

    st.subheader("Global feature importance")
    if IMPORTANCE_PATH.exists():
        importance_df = pd.read_csv(IMPORTANCE_PATH)
        if {"feature", "importance"}.issubset(importance_df.columns):
            chart_data = (
                importance_df[["feature", "importance"]]
                .sort_values("importance", ascending=False)
                .set_index("feature")
            )
            st.bar_chart(chart_data)
            st.dataframe(
                importance_df.sort_values("importance", ascending=False),
                width='stretch',
                hide_index=True,
            )
    else:
        st.caption("Feature-importance CSV was not bundled with this copy of the app.")

    st.info(
        "Feature importance describes how the trained model uses predictors. "
        "It does not establish clinical causality."
    )


with guide_tab:
    st.subheader("Input ranges used by the app")

    range_df = pd.DataFrame(
        [{"Feature": key, "Range": value} for key, value in TRAINING_RANGES.items()]
    )
    st.dataframe(range_df, width='stretch', hide_index=True)

    st.markdown(
        """
        **Categorical coding used by the trained pipeline**

        - `sex`: Female = 0, Male = 1
        - `smoking_status`: Never = 0, Former = 1, Current = 2

        The ranges above describe the synthetic generator's supported input space.
        They are **not diagnostic reference intervals**.
        """
    )

    st.warning(
        "Predictions for profiles that differ materially from the synthetic training "
        "distribution may be unreliable."
    )
