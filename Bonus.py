# app.py

import pandas as pd
import numpy as np
from pathlib import Path

import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Try to import XGBoost
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# =========================
# 0. BASIC PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Student Lifestyle → GPA Advisor",
    page_icon="🎓",
    layout="wide",
)

# =========================
# 1. DATA LOADING & CLEANING
# =========================

def remove_outliers_iqr(df: pd.DataFrame, cols) -> pd.DataFrame:
    """
    Remove outliers using the IQR rule for the given columns.
    This helps avoid crazy values like 15h of sleep, 12h of study, etc.
    """
    clean_df = df.copy()
    for col in cols:
        Q1 = clean_df[col].quantile(0.25)
        Q3 = clean_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        clean_df = clean_df[(clean_df[col] >= lower) & (clean_df[col] <= upper)]
    return clean_df


@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Load the student lifestyle dataset from the same folder as this app
    and remove extreme outliers in hour-based columns.
    """
    csv_path = Path(__file__).parent / "student_lifestyle_dataset.csv"

    if not csv_path.exists():
        st.error(f"File not found: {csv_path}")
        st.stop()

    df = pd.read_csv(csv_path)

    hour_cols = [
        "Study_Hours_Per_Day",
        "Extracurricular_Hours_Per_Day",
        "Sleep_Hours_Per_Day",
        "Social_Hours_Per_Day",
        "Physical_Activity_Hours_Per_Day",
    ]

    # Clean outliers
    df = remove_outliers_iqr(df, hour_cols)

    return df


# =========================
# 2. MODEL TRAINING
# =========================

@st.cache_resource
def train_models(df: pd.DataFrame):
    """
    Train multiple models (Linear Regression, Random Forest, XGBoost if available)
    and return:
    - models: dict of trained models
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

    # Train/test split (same for all models)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {}

    # 1. Random Forest
    rf = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf

    # 2. Linear Regression
    lin = LinearRegression()
    lin.fit(X_train, y_train)
    models["Linear Regression"] = lin

    # 3. XGBoost (if available)
    if XGBOOST_AVAILABLE:
        xgb = XGBRegressor(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective="reg:squarederror",
            n_jobs=-1,
        )
        xgb.fit(X_train, y_train)
        models["XGBoost"] = xgb

    # Stats for sliders and target GPA
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
        "min_gpa": float(df["GPA"].min()),
        "max_gpa": float(df["GPA"].max()),
        "mean_gpa": float(df["GPA"].mean()),
    }

    return models, le, stats, feature_cols


# =========================
# 3. RECOMMENDATION ENGINE (JOINT OPTIMIZATION)
# =========================

def generate_recommendations(
    base_input: pd.DataFrame,
    base_pred: float,
    model,
    stats: dict,
    target_gpa=None,
    n_candidates: int = 400,
) -> pd.DataFrame:
    """
    Jointly optimize ALL habits at once.

    Strategy:
    - Start from the user's current lifestyle.
    - Randomly sample candidate lifestyles around it (changing multiple habits).
    - Enforce realistic bounds for each habit.
    - Enforce a 24-hour total time constraint.
    - Predict GPA for all candidates.
    - Rank candidates based on:
        * if target_gpa is None: largest GPA increase
        * if target_gpa is set: closest to target while still improving
    - Return a few best combined lifestyle plans.
    """

    base_row = base_input.iloc[0].copy()

    # Reasonable "human" bounds (you can tune these)
    safe_ranges = {
        "Study_Hours_Per_Day": (0.5, 8.0),
        "Sleep_Hours_Per_Day": (5.0, 9.0),
        "Extracurricular_Hours_Per_Day": (0.0, 5.0),
        "Social_Hours_Per_Day": (0.0, 6.0),
        "Physical_Activity_Hours_Per_Day": (0.0, 4.0),
    }

    # Helper: sample a new continuous value around current, within bounds, 0.5h steps
    def sample_continuous(col_name, current):
        low, high = safe_ranges[col_name]
        # random shift between -2 and +2 hours
        delta = np.random.uniform(-2.0, 2.0)
        val = current + delta
        val = max(low, min(high, val))
        # snap to nearest 0.5 hours
        return round(val * 2) / 2.0

    # Stress: discrete levels
    current_stress = int(base_row["Stress_Level_Encoded"])
    min_stress = stats["min_stress_encoded"]
    max_stress = stats["max_stress_encoded"]

    candidates = []

    # --- 1. Generate candidate lifestyles ---
    for _ in range(n_candidates):
        cand = base_row.copy()

        # Continuous habits (sample around current)
        cand["Study_Hours_Per_Day"] = sample_continuous(
            "Study_Hours_Per_Day",
            base_row["Study_Hours_Per_Day"],
        )
        cand["Sleep_Hours_Per_Day"] = sample_continuous(
            "Sleep_Hours_Per_Day",
            base_row["Sleep_Hours_Per_Day"],
        )
        cand["Extracurricular_Hours_Per_Day"] = sample_continuous(
            "Extracurricular_Hours_Per_Day",
            base_row["Extracurricular_Hours_Per_Day"],
        )
        cand["Social_Hours_Per_Day"] = sample_continuous(
            "Social_Hours_Per_Day",
            base_row["Social_Hours_Per_Day"],
        )
        cand["Physical_Activity_Hours_Per_Day"] = sample_continuous(
            "Physical_Activity_Hours_Per_Day",
            base_row["Physical_Activity_Hours_Per_Day"],
        )

        # Stress: bias toward equal or lower stress
        possible_stress = list(range(min_stress, max_stress + 1))
        # Favor values <= current_stress
        if np.random.rand() < 0.7:
            possible_stress = [s for s in possible_stress if s <= current_stress]
        if not possible_stress:
            possible_stress = [current_stress]
        cand["Stress_Level_Encoded"] = int(np.random.choice(possible_stress))

        # 24h constraint
        total_hours = (
            cand["Study_Hours_Per_Day"]
            + cand["Sleep_Hours_Per_Day"]
            + cand["Extracurricular_Hours_Per_Day"]
            + cand["Social_Hours_Per_Day"]
            + cand["Physical_Activity_Hours_Per_Day"]
        )
        if total_hours > 24:
            continue

        candidates.append(cand)

    if not candidates:
        return pd.DataFrame(columns=["habit", "description", "new_pred", "delta_gpa"])

    cand_df = pd.DataFrame(candidates)

    # --- 2. Predict GPA for all candidates ---
    preds = model.predict(cand_df[base_input.columns])
    cand_df["pred_gpa"] = preds
    cand_df["delta_gpa"] = cand_df["pred_gpa"] - base_pred

    # Keep only lifestyle plans that improve GPA meaningfully
    cand_df = cand_df[cand_df["delta_gpa"] > 0.01]
    if cand_df.empty:
        return pd.DataFrame(columns=["habit", "description", "new_pred", "delta_gpa"])

    # --- 3. Rank candidates ---
    if target_gpa is not None:
        cand_df["dist_to_target"] = np.abs(cand_df["pred_gpa"] - target_gpa)
        cand_df = cand_df.sort_values(
            ["dist_to_target", "delta_gpa"],
            ascending=[True, False],
        )
    else:
        cand_df = cand_df.sort_values("delta_gpa", ascending=False)

    # --- 4. Convert candidates to human-readable plans ---
    recs = []
    for _, row in cand_df.head(5).iterrows():
        changes = []

        def add_change(name, label, unit="h"):
            old = float(base_row[name])
            new = float(row[name])
            if abs(new - old) >= 0.25:  # only mention real changes
                direction = "Increase" if new > old else "Reduce"
                changes.append(
                    f"{direction} {label} from {old:.1f}{unit} to {new:.1f}{unit} per day"
                )

        add_change("Study_Hours_Per_Day", "study time")
        add_change("Sleep_Hours_Per_Day", "sleep")
        add_change("Extracurricular_Hours_Per_Day", "extracurricular time")
        add_change("Social_Hours_Per_Day", "social time")
        add_change("Physical_Activity_Hours_Per_Day", "physical activity")

        new_stress = int(row["Stress_Level_Encoded"])
        if new_stress < current_stress:
            changes.append(
                "Work on lowering your stress level (e.g. better time management, relaxation)."
            )

        if not changes:
            continue

        description = "; ".join(changes)
        recs.append({
            "habit": "Combined lifestyle plan",
            "description": description,
            "new_pred": float(row["pred_gpa"]),
            "delta_gpa": float(row["delta_gpa"]),
        })

    if not recs:
        return pd.DataFrame(columns=["habit", "description", "new_pred", "delta_gpa"])

    rec_df = pd.DataFrame(recs)
    if target_gpa is None:
        rec_df = rec_df.sort_values("delta_gpa", ascending=False)

    return rec_df


# =========================
# 4. VISUALIZATION HELPERS
# =========================

def plot_feature_importance(model, feature_names, model_name: str):
    """
    Generic feature-importance / coefficient plot for the selected model.
    - Tree models (RF, XGBoost): use feature_importances_
    - Linear Regression: use |coefficients|
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        importances = np.abs(np.ravel(coef))
    else:
        st.warning("This model does not provide feature importance information.")
        return

    sorted_idx = np.argsort(importances)

    fig, ax = plt.subplots()
    ax.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx])
    ax.set_title(f"Feature importance – {model_name}")
    ax.set_xlabel("Importance / |Coefficient|")
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
        fillcolor='rgba(0, 102, 255, 0.4)',
        line=dict(color='rgba(0, 102, 255, 1)', width=3)
    ))

    fig.add_trace(go.Scatterpolar(
        r=avg_values,
        theta=categories,
        fill='toself',
        name='Dataset average',
        fillcolor='rgba(255, 153, 0, 0.3)',
        line=dict(color='rgba(255, 153, 0, 1)', width=3)
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
    im = ax.imshow(corr.values, aspect="auto", cmap="coolwarm")

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)

    ax.set_title("Correlation heatmap (numerical features)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)


# =========================
# 5. STREAMLIT UI
# =========================

def main():
    st.title("GPA Predictor based on Student Lifestyle")

    st.markdown(
        """
        This tool uses machine learning models trained on student lifestyle data to:

        - **Predict your GPA** based on your daily habits  
        - Suggest **combined lifestyle changes** (study, sleep, social, etc.)  
        - Let you set a **target GPA** and see which plans move you toward it  

        Use the controls on the left to describe a typical day.
        """
    )

    # Load data and train models (cached)
    df = load_data()
    models, stress_encoder, stats, feature_cols = train_models(df)

    # ========== SIDEBAR: INPUTS ==========
    with st.sidebar:
        st.header("Your Daily Profile")

        # Model choice
        st.markdown("### Model settings")
        model_options = list(models.keys())
        default_model = "Random Forest" if "Random Forest" in model_options else model_options[0]
        selected_model_name = st.selectbox(
            "Choose prediction model",
            model_options,
            index=model_options.index(default_model),
        )
        selected_model = models[selected_model_name]

        st.caption("Different models may give slightly different predictions.")

        st.markdown("---")
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

    # Init session state
    if "pred_gpa" not in st.session_state:
        st.session_state["pred_gpa"] = None
    if "target_gpa" not in st.session_state:
        st.session_state["target_gpa"] = stats["mean_gpa"]

    # ========== MAIN CONTENT: TABS ==========
    tab_pred, tab_target, tab_data = st.tabs(
        ["My GPA & Advice", "🎯 Target GPA Planner", "Dataset & Model"]
    )

    # ----- TAB 2 FIRST: TARGET GPA PLANNER -----
    with tab_target:
        st.subheader("🎯 Target GPA Planner")

        st.markdown(
            """
            Use this tab to choose a **goal GPA** you want to aim for.  
            The prediction tab will then tell you how far you are from this goal and
            how each recommended lifestyle plan moves you toward it.
            """
        )

        current_target = st.session_state.get("target_gpa", stats["mean_gpa"])

        new_target = st.slider(
            "Choose your goal GPA",
            stats["min_gpa"],
            stats["max_gpa"],
            float(current_target),
            step=0.1,
        )

        st.session_state["target_gpa"] = float(new_target)

        st.info(
            f"Your current target GPA is set to **{new_target:.2f}**. "
            "Go to **'My GPA & Advice'** and run the prediction to see "
            "how close you are and which lifestyle plans help you get there."
        )

    # ----- TAB 1: MAIN PREDICTION + ADVICE -----
    with tab_pred:
        st.subheader("Predicted GPA")

        target_gpa = st.session_state.get("target_gpa", stats["mean_gpa"])

        # Two-column layout (left: prediction, right: summary)
        col_left, col_right = st.columns([1.3, 1])

        with col_left:
            if st.button("Predict my GPA"):
                pred_gpa = float(selected_model.predict(input_df)[0])
                st.session_state["pred_gpa"] = pred_gpa

                st.metric(
                    label=f"Estimated GPA ({selected_model_name})",
                    value=f"{pred_gpa:.2f}",
                    help="Prediction based on your current daily habits and chosen model."
                )

                gap = target_gpa - pred_gpa
                st.write(f"🎯 **Your target GPA (from Planner tab):** {target_gpa:.2f}")
                if gap <= 0:
                    st.success(
                        f"Your predicted GPA already meets or exceeds your target "
                        f"(by {-gap:.2f} points). Focus on maintaining healthy habits!"
                    )
                else:
                    st.info(
                        f"You are currently **{gap:.2f} GPA points below** your target."
                    )

                st.markdown("---")
                st.subheader("Lifestyle plans to move toward your goal")

                rec_df = generate_recommendations(
                    base_input=input_df,
                    base_pred=pred_gpa,
                    model=selected_model,
                    stats=stats,
                    target_gpa=target_gpa,
                )

                if rec_df.empty:
                    st.info(
                        "No strong improvement plans found – your current habits already look close to optimal "
                        "under this model."
                    )
                else:
                    for _, row_rec in rec_df.head(3).iterrows():
                        new_pred = row_rec["new_pred"]
                        delta = row_rec["delta_gpa"]
                        gap_after = target_gpa - new_pred

                        if gap_after <= 0:
                            goal_msg = (
                                f"✅ This plan would **reach or exceed your goal** "
                                f"(new GPA {new_pred:.2f}, overshoot {abs(gap_after):.2f})."
                            )
                        else:
                            goal_msg = (
                                f"📉 After this plan, you would still be **{gap_after:.2f} points below** your goal "
                                f"(new GPA {new_pred:.2f})."
                            )

                        st.markdown(
                            f"- **Plan:** {row_rec['description']}  \n"
                            f"  ↳ Predicted GPA change: **+{delta:.2f}** → **{new_pred:.2f}**  \n"
                            f"  {goal_msg}"
                        )

                    st.caption("These are model-based estimates from observational data, not guarantees.")

            else:
                st.info("Click **'Predict my GPA'** to see results and lifestyle plans.")

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

        # GPA distribution + radar only if we already predicted
        pred_gpa = st.session_state.get("pred_gpa", None)
        if pred_gpa is not None:
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

    # ----- TAB 3: DATASET & MODEL -----
    with tab_data:
        st.subheader("Dataset snapshot")

        st.markdown(
            "Below is a small preview of the dataset used to train the models, "
            "along with basic summary statistics and model insights."
        )

        st.markdown("##### Sample rows")
        st.dataframe(df.head(), use_container_width=True)

        st.markdown("##### Summary statistics (numerical columns)")
        st.write(df.describe())
        st.caption(
            "These statistics show the overall range and central tendency of the numerical variables "
            "used in the models (e.g. average study hours, typical GPA, etc.)."
        )

        st.markdown("##### Feature importance (selected model)")
        plot_feature_importance(selected_model, feature_cols, selected_model_name)
        st.caption(
            "The plot indicates which habits matter most for the selected model’s GPA predictions. "
            "Larger bars contribute more to the model’s decisions."
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

    model_list_str = ", ".join(models.keys())
    st.write(
        f"Available models: **{model_list_str}**. "
        f"Currently selected: **{selected_model_name}**."
    )
    if not XGBOOST_AVAILABLE:
        st.caption("XGBoost is not installed in this environment, so only Linear Regression and Random Forest may be available.")


if __name__ == "__main__":
    main()
