import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="California Water Quality – WQI Model",
    page_icon="💧",
    layout="wide"
)

# =========================================================
# CONSTANTS & UTIL FUNCTIONS
# =========================================================

MODEL_PATH = "wqi_xgb_pipeline.pkl"

NUMERIC_FEATURES = [
    "DissolvedOxygen_mg/L",
    "pH_pH units",
    "Turbidity_NTU",
    "SpecificConductance_µS/cm",
    "WaterTemperature_°C",
    "sample_depth_meter",
    "DO_Temp_Ratio",
    "latitude",
    "longitude",
    "Month_sin",
    "Month_cos",
]

CATEGORICAL_FEATURES = ["station_type"]

CLASS_MAPPING = {
    0: "Good",
    1: "Moderate",
    2: "Poor",
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering logic from training pipeline."""
    df = df.copy()

    # Temporal features
    if "sample_date" in df.columns:
        df["sample_date"] = pd.to_datetime(df["sample_date"], errors="coerce")
        df["Month"] = df["sample_date"].dt.month
        df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
        df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    else:
        df["Month_sin"] = 0.0
        df["Month_cos"] = 0.0

    # DO/Temperature ratio
    df["DO_Temp_Ratio"] = df["DissolvedOxygen_mg/L"] / (df["WaterTemperature_°C"] + 1)

    # Depth fill
    df["sample_depth_meter"] = df["sample_depth_meter"].fillna(0)

    return df


@st.cache_resource(show_spinner=False)
def load_model():
    """Load trained XGBoost pipeline."""
    return joblib.load(MODEL_PATH)


def predict_single_sample(sample_date, station_type, do, ph, turb, sc, temp, depth, lat, lon):
    """Predict WQI class from a single input row."""
    df = pd.DataFrame({
        "sample_date": [pd.to_datetime(sample_date)],
        "station_type": [station_type],
        "DissolvedOxygen_mg/L": [do],
        "pH_pH units": [ph],
        "Turbidity_NTU": [turb],
        "SpecificConductance_µS/cm": [sc],
        "WaterTemperature_°C": [temp],
        "sample_depth_meter": [depth],
        "latitude": [lat],
        "longitude": [lon]
    })

    df_eng = engineer_features(df)
    X = df_eng[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    model = load_model()

    pred_code = int(model.predict(X)[0])
    pred_label = CLASS_MAPPING.get(pred_code, str(pred_code))

    # Probability scores (if classifier supports)
    try:
        prob = model.predict_proba(X)[0]
        proba_dict = {
            "Good": prob[0],
            "Moderate": prob[1],
            "Poor": prob[2]
        }
    except:
        proba_dict = None

    return pred_label, pred_code, proba_dict


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("💧 Navigation")
page = st.sidebar.radio(
    "Go to",
    ("Home", "ML Predictions", "Single Sample Prediction", "Forecasting Predictions")
)

# =========================================================
# HOME PAGE
# =========================================================
if page == "Home":
    st.markdown("<h1 style='text-align:center;'>💧 California Water Quality Index (WQI)</h1>",
                unsafe_allow_html=True)

    st.markdown("<h3 style='text-align:center;'>Machine Learning Based Classification Dashboard</h3>",
                unsafe_allow_html=True)

    st.markdown("---")

    st.write("""
    Welcome to the **Water Quality Index (WQI) Prediction App**.

    This dashboard provides:
    - 🤖 ML-based WQI classification  
    - 🧪 Single sample prediction  
    - 📈 Forecasting placeholder for future enhancements  
    - 🌱 Clean student-friendly UI  

    Use the sidebar to navigate through the application.
    """)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    col1.markdown("""
        <div style="padding:15px; border-radius:10px; background:#27C8F5; text-align:center;">
            <h4>Model</h4>
            <p style="font-size:18px;">XGBoost</p>
        </div>
    """, unsafe_allow_html=True)

    col2.markdown("""
        <div style="padding:15px; border-radius:10px; background:#27C8F5; text-align:center;">
            <h4>Inputs</h4>
            <p style="font-size:18px;">10 Features</p>
        </div>
    """, unsafe_allow_html=True)

    col3.markdown("""
        <div style="padding:15px; border-radius:10px; background:#27C8F5; text-align:center;">
            <h4>Classes</h4>
            <p style="font-size:18px;">Good • Moderate • Poor</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

# =========================================================
# ML PREDICTIONS PAGE
# =========================================================
elif page == "ML Predictions":
    st.title("🌊 Water Quality Classification")

    col1, col2 = st.columns(2)

    with col1:
        pH = st.number_input("pH", 0.0, 14.0, 7.0)
        DO = st.number_input("Dissolved Oxygen (mg/L)", 0.0, 20.0, 8.0)
        Temp = st.number_input("Temperature (°C)", -5.0, 50.0, 20.0)
        Turbidity = st.number_input("Turbidity (NTU)", 0.0, 1000.0, 5.0)

    with col2:
        EC = st.number_input("Conductivity (µS/cm)", 0.0, 30000.0, 400.0)
        Depth = st.number_input("Sample Depth (m)", 0.0, 100.0, 1.0)
        Latitude = st.number_input("Latitude", -90.0, 90.0, 37.5)
        Longitude = st.number_input("Longitude", -180.0, 180.0, -121.9)
        StationType = st.selectbox("Station Type", ["River", "Lake", "Groundwater", "Reservoir", "Other"])
        SampleDate = st.date_input("Sample Date")

    if st.button("Predict Water Quality Index (WQI) Class"):
        # Use your existing prediction function
        pred_label, pred_code, probabilities = predict_single_sample(
            SampleDate, StationType, DO, pH, Turbidity, EC, Temp, Depth, Latitude, Longitude
        )

        st.success(f"### 🌟 Predicted Water Quality Class: **{pred_label}**")

        st.info("""
        🎯 Interpretation:
        - **Good**: Water quality is healthy and suitable for most uses.
        - **Moderate**: Water quality is acceptable but may require treatment.
        - **Poor**: Water quality is unsafe for direct use and may indicate pollution.
        """)

# =========================================================
# SINGLE SAMPLE PAGE
# =========================================================

elif page == "Single Sample Prediction":
    st.markdown("<h2 style='text-align:center;'>🧪 Single Sample Prediction</h2>",
                unsafe_allow_html=True)

    st.write("A compact form to quickly classify water quality from a single sample.")

    st.markdown("---")

    with st.form("single_form"):
        sample_date = st.date_input("Sample Date")
        station_type = st.selectbox("Station Type",
                                    ["River", "Lake", "Groundwater", "Reservoir", "Other"])

        col1, col2 = st.columns(2)
        with col1:
            do = st.number_input("Dissolved Oxygen (mg/L)", 0.0, 20.0, 8.0, 0.1)
            ph = st.number_input("pH", 0.0, 14.0, 7.0, 0.1)
            temp = st.number_input("Temperature (°C)", -5.0, 50.0, 20.0, 0.5)

        with col2:
            sc = st.number_input("Conductivity (µS/cm)", 0.0, 20000.0, 400.0, 10.0)
            turb = st.number_input("Turbidity (NTU)", 0.0, 1000.0, 5.0, 0.1)
            depth = st.number_input("Depth (m)", 0.0, 100.0, 1.0, 0.1)

        lat = st.number_input("Latitude", -90.0, 90.0, 37.5, 0.01)
        lon = st.number_input("Longitude", -180.0, 180.0, -121.9, 0.01)

        submit = st.form_submit_button("Predict")

    if submit:
        label, code, proba = predict_single_sample(sample_date, station_type, do, ph, turb, sc, temp, depth, lat, lon)

        st.success(f"### Predicted WQI Class: **{label}**")

        if proba:
            st.write("### Probability Breakdown")
            prob_df = pd.DataFrame(proba, index=["Probability"])
            st.bar_chart(prob_df.T)


# =========================================================
# FORECASTING PLACEHOLDER
# =========================================================

elif page == "Forecasting Predictions":
    st.markdown("<h2 style='text-align:center;'>📈 Forecasting Predictions (Coming Soon)</h2>",
                unsafe_allow_html=True)

    st.info("""
    This section will include:
    - Time-series forecasting of DO, pH, Temperature  
    - Seasonal/Trend analysis  
    - Predicting future WQI trends  

    (Planned for future enhancement.)
    """)

