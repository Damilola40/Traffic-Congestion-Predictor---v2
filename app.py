import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import datetime

# -----------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Traffic Congestion Predictor",
    page_icon="🚦",
    layout="centered",
)

# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
MODEL_FILES = {
    "Logistic Regression": "traffic_binary_lr_pipeline.pkl",
}

TARGET_LABELS = {0: "Normal", 1: "Congested"}
LABEL_COLOR = {
    "Normal": "🟢",
    "Congested": "🔴",
    "Partially Normal": "🟡",
    "Partially Congested": "🟡",
}

@st.cache_resource
def load_model(filename: str):
    path = Path(__file__).parent / filename
    if not path.exists():
        return None
    return joblib.load(path)


# -----------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------
st.title("🚦 Traffic Congestion Predictor")
st.caption(
    "3MTT x MIT AI Innovation Challenge"
)
st.write(
    "Predicts whether traffic will be **Normal** or **Congested** on major "
    "Lagos roads and expressways, based on road, time, and weather conditions."
)

def load_model(name):
    try:
        return joblib.load(MODEL_FILES[name])
    except Exception as e:
        st.warning(
            f"`{MODEL_FILES[name]}` not found in this folder yet. "
            "The interface still runs, but predictions are placeholders until "
            f"you drop the trained pipeline file in here. (Error: {e})"
        )
        return None

model = load_model("Logistic Regression")

if "history" not in st.session_state:
    st.session_state["history"] = []

st.divider()

if "day" not in st.session_state:
    st.session_state["day"] = "Monday"
if "obs_hour" not in st.session_state:
    st.session_state["obs_hour"] = 8

# -----------------------------------------------------------------------
# INPUT FORM — column names match X in train_model_gen.ipynb exactly
# -----------------------------------------------------------------------
st.subheader("Enter conditions")

col1, col2 = st.columns(2)

ROADS = [
    "apapa-oshodi expressway", "funsho williams avenue", "ikorodu road",
    "lagos-abeokuta expressway", "lagos-badagry expressway",
    "lagos-ibadan expressway", "lekki-epe expressway", "third mainland bridge",
]
ROAD_TO_CORRIDOR = {
    "apapa-oshodi expressway": "mile2-oshodi",
    "funsho williams avenue": "unknown",
    "ikorodu road": "unknown",
    "lagos-abeokuta expressway": "abule egba-iyana ipaja",
    "lagos-badagry expressway": "mile2-festac",
    "lagos-ibadan expressway": "ojota-berger",
    "lekki-epe expressway": "lekkiphase1-ajah",
    "third mainland bridge": "oworonshole-adekunle",
}
WEATHERS = ["cloudy", "heavy rain", "light rain", "mostly cloudy", "partly sunny"]

with col1:
    road = st.selectbox("road", ROADS)
    fixed_corridor = ROAD_TO_CORRIDOR[road]
    st.text_input("fixed_corridor (auto-set from road)", value=fixed_corridor, disabled=True)
    day = st.selectbox(
        "day",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        key = "day",
    )
    weather = st.selectbox("weather", WEATHERS)

with col2:
    obs_hour = st.slider("hour of day (0-23)", 0, 23, key="obs_hour")
    temperature = st.slider("temperature (°C)", 20.0, 31.0, 26.0, step=0.1)
    rain_chance = st.slider("Rain chance (%)", 0, 100, 0, step=5) / 100

def use_current_time():
    now = datetime.datetime.now()
    st.session_state["day"] = now.strftime("%A")
    st.session_state["obs_hour"] = now.hour

btn_col1, btn_col2 = st.columns(2)

with btn_col1:
    st.button("Use current day & time", on_click=use_current_time, use_container_width=True)

with btn_col2:
    predict_clicked = st.button("Predict congestion", type="primary", use_container_width=True)

def hourly_forecast(model, base_input: dict) -> pd.DataFrame:
    """Predict congestion probability across all 24 hours, holding
    every other input fixed at what's currently selected."""
    rows = []
    for h in range(24):
        row = base_input.copy()
        row["obs_hour"] = h
        rows.append(row)
    hours_df = pd.DataFrame(rows)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(hours_df)[:, 1] * 100 # P(Congested)
    else:
        proba = model.predict(hours_df) * 100

    return pd.DataFrame({"Hour": range(24), "P(Congested)": proba}).set_index("Hour")

# -----------------------------------------------------------------------
# PREDICTION
# -----------------------------------------------------------------------
if predict_clicked:
    input_df = pd.DataFrame([{
        "road": road,
        "fixed_corridor": fixed_corridor,
        "day": day,
        "weather": weather,
        "temperature": temperature,
        "rain_chance": rain_chance,
        "obs_hour": obs_hour,
    }])

    st.divider()
    st.subheader("Result")

    def get_display_label(proba, labels=("Normal", "Congested")) -> str:
        """
        proba = [P(Normal), P(Congested)] from model.predict_proba().
        If both are within 20 points of 50/50 (i.e. each roughly 40-60%),
        call it "Partially <higher one>" instead of stating it as certain.
        """
        p_normal, p_congested = proba[0], proba[1]
        higher = labels[1] if p_congested >= p_normal else labels[0]

        if abs(p_congested - p_normal) <= 0.20:
            return f"Partially {higher}"
        return higher

    if model is not None:
        try:
            pred = model.predict(input_df)[0]

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_df)[0]
                label = get_display_label(proba)
            else:
                proba = None
                label = TARGET_LABELS.get(int(pred), str(pred))

            st.markdown(f"### {LABEL_COLOR.get(label, '')} Predicted: **{label}**")

            if proba is not None:
                proba_df = pd.DataFrame(
                    {"Class": [TARGET_LABELS[0], TARGET_LABELS[1]], "Probability (%)": proba * 100}
                ).set_index("Class")
                st.bar_chart(proba_df)

            st.markdown("**24-hour forecast for this road & day**")
            forecast_df = hourly_forecast(model, input_df.iloc[0].to_dict())
            st.line_chart(forecast_df)

            st.session_state["history"].insert(0, {
                "Road": road, "Day": day, "Hour": obs_hour,
                "Weather": weather, "Prediction": label,
            })
            st.session_state["history"] = st.session_state["history"][:5]

        except Exception as e:
            st.error(
                f"Prediction failed — check that column names/values match "
                f"what the pipeline was trained on. Details: {e}"
            )
    else:
        st.info("Placeholder result (no model loaded): Congested 🔴")

    def get_prep_tips(label: str, weather: str, obs_hour: int) -> list[str]:
        """Actionable advice based on the prediction and current conditions."""
        tips = []

        if label == "Congested":
            tips.append("Leave earlier than usual.")
            tips.append("Check for an alternate route before you set off.")
            if weather in ("heavy rain", "light rain"):
                tips.append("Rain on top of congestion usually means even slower going.")
            if obs_hour in range(7, 10) or obs_hour in range(15, 20):
                tips.append("You're heading out in a typical rush-hour window — expect stop-and-go.")

        elif label == "Partially Congested":
            tips.append("Conditions are borderline — leaning toward congested but not certain.")
            tips.append("Leave a little earlier than usual, just in case.")
            if weather in ("heavy rain", "light rain"):
                tips.append("Rain could tip this toward slower traffic — plan for delay.")

        elif label == "Partially Normal":
            tips.append("Conditions are borderline — leaning toward normal but not certain.")
            tips.append("Probably fine to head out as planned, but keep an eye on the road.")

        else:  # "Normal"
            tips.append("Traffic looks normal for this time and route.")
            tips.append("No extra buffer needed, but conditions can shift — worth a quick recheck before leaving.")

        return tips


    # --- call it right after label is known ---
    tips = get_prep_tips(label, weather, obs_hour)
    st.markdown("**What to prepare for:**")
    for tip in tips:
        st.write(f"- {tip}")

    with st.expander("Show input sent to model"):
            st.dataframe(input_df)

if st.session_state["history"]:
    st.divider()
    st.markdown("**Recent predictions**")
    st.dataframe(pd.DataFrame(st.session_state["history"]))

st.divider()
st.caption("Built by Muhammad Misbahudeen · FUT Minna · 3MTT/MIT OpenLearning - Cohort 1")