# app.py

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# =========================
# 1. LOAD DATA & TRAIN MODEL
# =========================

@st.cache_data
def load_data(path: str = r"C:\Users\pc\PycharmProjects\Data2010Project\student_lifestyle_dataset.csv") -> pd.DataFrame:
    """Load the student lifestyle dataset."""
    df = pd.read_csv(path)
    return df


@st.cache_resource
def train_model(df: pd.DataFrame):
    """Train a Random Forest model and return model, encoder, stats, feature cols."""

    # Encode Stress_Level
    le = LabelEncoder()
    df["Stress_Level_Encoded"] = le.fit_transform(df["Stress_Level"])

    feature_cols = [
        "Study_Hours_Per_Day",
        "Extracurricular_Hours_Per_Day",
        "Sleep_Hours_Per_Day",
        "Social_Hours_Per_Day",
        "Physical_Activity_Hours_Per_Day",
        "Stress_Level_Encoded",
    ]
    target_col = "GPA"

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # Ranges used for sliders & recommendation logic
    stats = {
        "min_study": float(df["Study_Hours_Per_Day"].min()),
        "max_study": float(df["Study_Hours_Per_Day"].max()),
        "min_sleep": float(df["Sleep_Hours_Per_Day"].min()),
        "max_sleep": float(df["Sleep_Hours_Per_Day"].max()),
        "min_phys": float(df["Physical_Activity_Hours_Per_Day"].min()),
        "max_phys": float(df["Physical_Activity_Hours_Per_Day"].max()),
        "min_extra": float(df["Extracurricular_Hours_Per_Day"].min()),
        "max_extra": float(df["Extracurricular_Hours_Per_Day"].max()),
        "min_social": float(df["Social_Hours_Per_Day"].min()),
        "max_social": float(df["Social_Hours_Per_Day"].max()),
        "min_stress_encoded": int(df["Stress_Level_Encoded"].min()),
        "max_stress_encoded": int(df["Stress_Level_Encoded"].max()),
    }

    return rf, le, stats, feature_cols


# =========================
# 2. RECOMMENDATION ENGINE
# =========================

def generate_recommendations(
    base_input: pd.DataFrame,
    base_pred: float,
    model: RandomForestRegressor,
    stats: dict,
) -> pd.DataFrame:
    """
    Try small, realistic changes to each habit and estimate
    how much each change could improve predicted GPA.
    Returns a DataFrame of recommendations sorted by delta_gpa.
    """

    candidates = []
    row = base_input.copy()

    # 1. Increase study hours (+1.5h, up to max)
    current_study = float(row["Study_Hours_Per_Day"].iloc[0])
    new_study = min(current_study + 1.5, stats["max_study"])
    if new_study > current_study:
        tmp = row.copy()
        tmp["Study_Hours_Per_Day"] = new_study
        new_pred = model.predict(tmp)[0]
        candidates.append({
            "habit": "Study_Hours_Per_Day",
            "description": f"Increase study time from {current_study:.1f}h to {new_study:.1f}h per day.",
            "new_pred": new_pred,
            "delta_gpa": new_pred - base_pred,
        })

    # 2. Improve sleep (aim at least 7h, up to max)
    current_sleep = float(row["Sleep_Hours_Per_Day"].iloc[0])
    target_sleep = max(current_sleep, 7.0)
    target_sleep = min(target_sleep, stats["max_sleep"])
    if target_sleep > current_sleep:
        tmp = row.copy()
        tmp["Sleep_Hours_Per_Day"] = target_sleep
        new_pred = model.predict(tmp)[0]
        candidates.append({
            "habit": "Sleep_Hours_Per_Day",
            "description": f"Increase sleep from {current_sleep:.1f}h to {target_sleep:.1f}h per day.",
            "new_pred": new_pred,
            "delta_gpa": new_pred - base_pred,
        })

    # 3. Increase physical activity (aim at least 1h)
    current_phys = float(row["Physical_Activity_Hours_Per_Day"].iloc[0])
    target_phys = max(current_phys, 1.0)
    target_phys = min(target_phys, stats["max_phys"])
    if target_phys > current_phys:
        tmp = row.copy()
        tmp["Physical_Activity_Hours_Per_Day"] = target_phys
        new_pred = model.predict(tmp)[0]
        candidates.append({
            "habit": "Physical_Activity_Hours_Per_Day",
            "description": f"Increase physical activity from {current_phys:.1f}h to {target_phys:.1f}h per day.",
            "new_pred": new_pred,
            "delta_gpa": new_pred - base_pred,
        })

    # 4. Reduce social hours if very high (> 4h → 3h)
    current_social = float(row["Social_Hours_Per_Day"].iloc[0])
    if current_social > 4.0:
        target_social = 3.0
        tmp = row.copy()
        tmp["Social_Hours_Per_Day"] = target_social
        new_pred = model.predict(tmp)[0]
        candidates.append({
            "habit": "Social_Hours_Per_Day",
            "description": f"Reduce social time from {current_social:.1f}h to {target_social:.1f}h per day.",
            "new_pred": new_pred,
            "delta_gpa": new_pred - base_pred,
        })

    # 5. Reduce extracurricular time if very high (> 3h → 2h)
    current_extra = float(row["Extracurricular_Hours_Per_Day"].iloc[0])
    if current_extra > 3.0:
        target_extra = 2.0
        tmp = row.copy()
        tmp["Extracurricular_Hours_Per_Day"] = target_extra
        new_pred = model.predict(tmp)[0]
        candidates.append({
            "habit": "Extracurricular_Hours_Per_Day",
            "description": f"Reduce extracurricular time from {current_extra:.1f}h to {target_extra:.1f}h per day.",
            "new_pred": new_pred,
            "delta_gpa": new_pred - base_pred,
        })

    # 6. Lower stress (one level down if possible)
    current_stress = int(row["Stress_Level_Encoded"].iloc[0])
    if current_stress > stats["min_stress_encoded"]:
        target_stress = current_stress - 1
        tmp = row.copy()
        tmp["Stress_Level_Encoded"] = target_stress
        new_pred = model.predict(tmp)[0]
        candidates.append({
            "habit": "Stress_Level",
            "description": "Work on reducing stress by one level (e.g. from High → Medium).",
            "new_pred": new_pred,
            "delta_gpa": new_pred - base_pred,
        })

    if not candidates:
        return pd.DataFrame(columns=["habit", "description", "new_pred", "delta_gpa"])

    rec_df = pd.DataFrame(candidates)
    rec_df = rec_df.sort_values("delta_gpa", ascending=False)

    # Keep only improvements larger than a small threshold
    rec_df = rec_df[rec_df["delta_gpa"] > 0.01]
    return rec_df


# =========================
# 3. STREAMLIT UI
# =========================

def main():
    st.title("🎓 Student Lifestyle → GPA Predictor & Habit Advisor")

    st.write(
        """
        This app uses a machine learning model trained on student lifestyle data
        to **predict GPA** and suggest **habit changes** that may improve academic performance.
        """
    )

    df = load_data()
    model, stress_encoder, stats, feature_cols = train_model(df)

    # Sidebar: user inputs
    st.sidebar.header("Input Your Daily Habits")

    study_hours = st.sidebar.slider(
        "Study Hours Per Day",
        stats["min_study"],
        stats["max_study"],
        float(df["Study_Hours_Per_Day"].mean()),
        step=0.5,
    )

    sleep_hours = st.sidebar.slider(
        "Sleep Hours Per Day",
        stats["min_sleep"],
        stats["max_sleep"],
        float(df["Sleep_Hours_Per_Day"].mean()),
        step=0.5,
    )

    extra_hours = st.sidebar.slider(
        "Extracurricular Hours Per Day",
        stats["min_extra"],
        stats["max_extra"],
        float(df["Extracurricular_Hours_Per_Day"].mean()),
        step=0.5,
    )

    social_hours = st.sidebar.slider(
        "Social Hours Per Day",
        stats["min_social"],
        stats["max_social"],
        float(df["Social_Hours_Per_Day"].mean()),
        step=0.5,
    )

    phys_hours = st.sidebar.slider(
        "Physical Activity Hours Per Day",
        stats["min_phys"],
        stats["max_phys"],
        float(df["Physical_Activity_Hours_Per_Day"].mean()),
        step=0.5,
    )

    stress_options = sorted(df["Stress_Level"].unique().tolist())
    stress_level = st.sidebar.selectbox("Stress Level", stress_options)
    stress_encoded = int(stress_encoder.transform([stress_level])[0])

    # Build single-row input for prediction
    input_dict = {
        "Study_Hours_Per_Day": [study_hours],
        "Extracurricular_Hours_Per_Day": [extra_hours],
        "Sleep_Hours_Per_Day": [sleep_hours],
        "Social_Hours_Per_Day": [social_hours],
        "Physical_Activity_Hours_Per_Day": [phys_hours],
        "Stress_Level_Encoded": [stress_encoded],
    }
    input_df = pd.DataFrame(input_dict)

    if st.button("🔮 Predict My GPA"):
        pred_gpa = model.predict(input_df)[0]
        st.subheader(f"Predicted GPA: **{pred_gpa:.2f}**")

        st.write("---")
        st.subheader("🧠 Study Habit Advisor")

        rec_df = generate_recommendations(
            base_input=input_df,
            base_pred=pred_gpa,
            model=model,
            stats=stats,
        )

        if rec_df.empty:
            st.info(
                "No strong recommendations found – your current habits already look close to optimal according to the model."
            )
        else:
            st.write(
                "Based on the model, the following changes are estimated to **increase your predicted GPA**:"
            )
            for _, row in rec_df.head(3).iterrows():
                st.markdown(
                    f"- {row['description']} "
                    f"(estimated GPA: **{row['new_pred']:.2f}**, "
                    f"change: **+{row['delta_gpa']:.2f}**)"
                )
            st.caption(
                "These are model-based estimates, not guarantees. Correlation does not imply causation."
            )

    st.write("---")
    st.subheader("ℹ️ Model Information")
    st.write("Features used:", ", ".join(feature_cols))
    st.write(
        "The model is a **Random Forest Regressor** trained on the uploaded student lifestyle dataset."
    )


if __name__ == "__main__":
    main()
