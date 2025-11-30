# app.py

import pandas as pd
import numpy as np
from pathlib import Path

import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
import plotly.graph_objects as go

# =========================
# 0. BASIC PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Student Lifestyle → GPA Advisor",
    page_icon="🎓",
    layout="wide",
)


# =========================
# 1. LOAD DATA & TRAIN MODEL
# =========================

@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Load the student lifestyle dataset from the same folder as this app.
    """
    csv_path = Path(__file__).parent / "student_lifestyle_dataset.csv"

    if not csv_path.exists():
        st.error(f"File not found: {csv_path}")
        st.stop()

    return pd.read_csv(csv_path)


@st.cache_resource
def train_model(df: pd.DataFrame):
    """
    Train a Random Forest model and return:
    - model: trained RandomForestRegressor
    - stress_encoder: LabelEncoder for Stress_Level
    - stats: min/max values for sliders and logic
    - feature_cols: columns used as model features
    """

    # Encode Stress_Level from text → numbers
    le = LabelEncoder()
    df["Stress_Level_Encoded"] = le.fit_transform(df["Stress_Level"])

    # Features and target
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

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Random Forest model
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
    Returns a DataFrame of recommendations sorted by delta_gpa (best first).
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
            "habit": "Study time",
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
            "habit": "Sleep",
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
            "habit": "Physical activity",
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
            "habit": "Social time",
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
            "habit": "Extracurriculars",
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
            "habit": "Stress level",
            "description": "Work on reducing stress by one level (e.g. from High → Medium).",
            "new_pred": new_pred,
            "delta_gpa": new_pred - base_pred,
        })

    # No candidates at all
    if not candidates:
        return pd.DataFrame(columns=["habit", "description", "new_pred", "delta_gpa"])

    # Convert to DataFrame, sort, and filter small changes
    rec_df = pd.DataFrame(candidates)
    rec_df = rec_df.sort_values("delta_gpa", ascending=False)

    # Keep only improvements larger than a small threshold
    rec_df = rec_df[rec_df["delta_gpa"] > 0.01]
    return rec_df


# =========================
# 3. VISUALIZATION HELPERS
# =========================

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)

    fig, ax = plt.subplots()
    ax.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx])
    ax.set_title("Feature importance")
    ax.set_xlabel("Importance score")
    st.pyplot(fig)


def radar_chart(user_df: pd.DataFrame, df: pd.DataFrame):
    """
    Radar chart: user profile vs dataset average (for main lifestyle habits).
    """
    categories = [
        "Study_Hours_Per_Day",
        "Sleep_Hours_Per_Day",
        "Extracurricular_Hours_Per_Day",
        "Social_Hours_Per_Day",
        "Physical_Activity_Hours_Per_Day",
    ]

    user_values = [float(user_df[c].iloc[0]) for c in categories]
    avg_values = [float(df[c].mean()) for c in categories]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=user_values,
        theta=categories,
        fill='toself',
        name='You',
        fillcolor = 'rgba(0, 102, 255, 0.4)',
        line = dict(color='rgba(0, 102, 255, 1)', width=3)
    ))

    fig.add_trace(go.Scatterpolar(
        r=avg_values,
        theta=categories,
        fill='toself',
        name='Dataset average',
        fillcolor = 'rgba(255, 153, 0, 0.3)',
        line = dict(color='rgba(255, 153, 0, 1)', width=3)
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        height=650,
        width=650,
        margin=dict(l=30, r=30, t=50, b=30),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"staticPlot": True}
    )


def gpa_distribution(df: pd.DataFrame, predicted_gpa: float):
    """
    Histogram of GPA with a vertical line at the user's predicted GPA.
    """
    fig, ax = plt.subplots()
    ax.hist(df["GPA"], bins=15, alpha=0.7)
    ax.axvline(predicted_gpa, linewidth=2)
    ax.set_title("GPA distribution")
    ax.set_xlabel("GPA")
    ax.set_ylabel("Number of students")
    st.pyplot(fig)


def correlation_heatmap(df: pd.DataFrame):
    """
    Simple correlation heatmap using matplotlib only.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots()
    im = ax.imshow(corr.values, aspect="auto", cmap = "coolwarm")

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)

    ax.set_title("Correlation heatmap (numerical features)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)


# =========================
# 4. STREAMLIT UI
# =========================

def main():
    st.title("GPA Predictor based on Student Lifestyle")

    st.markdown(
        """
        This tool uses a machine learning model trained on student lifestyle data to:

        - **Predict your GPA** based on your daily habits  
        - Suggest **small, realistic changes** that may improve your predicted GPA  

        Use the controls on the left to describe a typical day.
        """
    )

    # Load data and train model (cached)
    df = load_data()
    model, stress_encoder, stats, feature_cols = train_model(df)

    # ========== SIDEBAR: INPUTS ==========
    with st.sidebar:
        st.header("Your Daily Profile")

        st.caption("Adjust these sliders to match a *normal* day for you.")

        study_hours = st.slider(
            "Study hours per day",
            stats["min_study"],
            stats["max_study"],
            float(df["Study_Hours_Per_Day"].mean()),
            step=0.5,
        )

        sleep_hours = st.slider(
            "Sleep hours per day",
            stats["min_sleep"],
            stats["max_sleep"],
            float(df["Sleep_Hours_Per_Day"].mean()),
            step=0.5,
        )

        extra_hours = st.slider(
            "Extracurricular hours per day",
            stats["min_extra"],
            stats["max_extra"],
            float(df["Extracurricular_Hours_Per_Day"].mean()),
            step=0.5,
        )

        social_hours = st.slider(
            "Social hours per day",
            stats["min_social"],
            stats["max_social"],
            float(df["Social_Hours_Per_Day"].mean()),
            step=0.5,
        )

        phys_hours = st.slider(
            "Physical activity hours per day",
            stats["min_phys"],
            stats["max_phys"],
            float(df["Physical_Activity_Hours_Per_Day"].mean()),
            step=0.5,
        )

        stress_options = sorted(df["Stress_Level"].unique().tolist())
        default_stress_index = 0
        if "Medium" in stress_options:
            default_stress_index = stress_options.index("Medium")

        stress_level = st.selectbox(
            "Stress level",
            stress_options,
            index=default_stress_index,
        )
        stress_encoded = int(stress_encoder.transform([stress_level])[0])

        st.markdown("---")
        st.caption(
            "This is a **predictive model**, not a guarantee. "
            "Use the suggestions as guidance, not strict rules."
        )

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

    # ========== MAIN CONTENT: TABS ==========
    tab_pred, tab_data = st.tabs(
        ["My GPA & Advice", "Dataset & Model"]
    )

    # ----- TAB 1: MAIN PREDICTION + ADVICE -----
    with tab_pred:
        st.subheader("Predicted GPA")

        # Two-column layout (left: prediction, right: summary)
        col_left, col_right = st.columns([1.2, 1])

        with col_left:
            if st.button("Predict my GPA"):

                pred_gpa = model.predict(input_df)[0]

                st.metric(
                    label="Estimated GPA",
                    value=f"{pred_gpa:.2f}",
                    help="Prediction based on your current daily habits."
                )

                st.markdown("---")
                st.subheader("Habit suggestions")

                rec_df = generate_recommendations(
                    base_input=input_df,
                    base_pred=pred_gpa,
                    model=model,
                    stats=stats,
                )

                if rec_df.empty:
                    st.info(
                        "No strong recommendations found – your current habits already look close to optimal."
                    )
                else:
                    st.write("These changes could **increase** your predicted GPA:")
                    for _, row_rec in rec_df.head(3).iterrows():
                        st.markdown(
                            f"- **{row_rec['habit']}:** {row_rec['description']}  \n"
                            f"  ↳ New GPA: **{row_rec['new_pred']:.2f}** "
                            f"(change: **+{row_rec['delta_gpa']:.2f}**)"
                        )
                    st.caption("These are model-based estimates, not guarantees.")

            else:
                st.info("Click **'Predict my GPA'** to see results.")

        with col_right:
            st.markdown("#### Quick summary of your current day")

            st.write(
                pd.DataFrame(
                    {
                        "Habit": [
                            "Study hours",
                            "Sleep hours",
                            "Extracurricular",
                            "Social time",
                            "Physical activity",
                            "Stress level",
                        ],
                        "Value": [
                            f"{study_hours:.1f} h",
                            f"{sleep_hours:.1f} h",
                            f"{extra_hours:.1f} h",
                            f"{social_hours:.1f} h",
                            f"{phys_hours:.1f} h",
                            stress_level,
                        ],
                    }
                )
            )

        # ---------------------------------------
        # 📊 CENTERED GPA DISTRIBUTION CHART
        # ---------------------------------------
        if "pred_gpa" in locals():
            st.markdown("<br><h3 style='text-align:center;'> Where do you sit in the class?</h3>",
                        unsafe_allow_html=True)

            st.markdown("<div style='display:flex; justify-content:center;'>",
                        unsafe_allow_html=True)
            gpa_distribution(df, pred_gpa)
            st.markdown("</div>", unsafe_allow_html=True)

            st.caption(
                "<div style='text-align:center;'>"
                "The histogram shows how GPAs are distributed in the dataset.<br>"
                "The vertical line marks **your predicted GPA**."
                "</div>",
                unsafe_allow_html=True
            )

            st.markdown("<br><h3 style='text-align:center;'> How your habits compare to others</h3>",
                        unsafe_allow_html=True)

            st.markdown("<div style='display:flex; justify-content:center;'>",
                        unsafe_allow_html=True)
            radar_chart(input_df, df)
            st.markdown("</div>", unsafe_allow_html=True)

            st.caption(
                "<div style='text-align:center;'>"
                "The radar chart compares **your lifestyle** with the dataset average.<br>"
                "Larger distance from the center = higher value for that habit."
                "</div>",
                unsafe_allow_html=True
            )

    # ----- TAB 2: DATASET & MODEL -----
    with tab_data:
        st.subheader("Dataset snapshot")

        st.markdown(
            "Below is a small preview of the dataset used to train the model, "
            "along with basic summary statistics and model insights."
        )

        st.markdown("##### Sample rows")
        st.dataframe(df.head(), use_container_width=True)

        st.markdown("##### Summary statistics (numerical columns)")
        st.write(df.describe())
        st.caption(
            "These statistics show the overall range and central tendency of the numerical variables "
            "used in the model (e.g. average study hours, typical GPA, etc.)."
        )

        st.markdown("##### Feature importance (Random Forest)")
        plot_feature_importance(model, feature_cols)
        st.caption(
            "Feature importance indicates which habits matter most for the model’s GPA predictions. "
            "Bars further to the right contribute more to the model’s decisions."
        )

        st.markdown("##### Correlation heatmap")
        correlation_heatmap(df)
        st.caption(
            "The heatmap shows how strongly each pair of numerical variables move together. "
            "Brighter or darker colors (far from zero) mean a stronger positive or negative relationship."
        )

    # ----- FOOTER -----
    st.markdown("---")
    st.subheader("Model details")
    st.write("**Features used:**", ", ".join(feature_cols))
    st.write(
        "The model is a **Random Forest Regressor** trained on the uploaded student lifestyle dataset."
    )


if __name__ == "__main__":
    main()
