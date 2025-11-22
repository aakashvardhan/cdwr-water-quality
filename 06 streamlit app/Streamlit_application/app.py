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
LABEL_ENCODER_PATH = "label_encoder.pkl"


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
def load_model_and_encoder():
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return model, label_encoder

# =========================================================
# LOAD GLOBAL MODEL, ENCODER, DATASET, PREPROCESSOR
# =========================================================

processed_df = pd.read_pickle("processed_dataset_WQ.pkl")

model, label_encoder = load_model_and_encoder()

preprocessor = model.named_steps["preprocessor"]


def predict_single_sample(sample_date, station_type, do, ph, turb, sc, temp, depth, lat, lon):
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

    model, label_encoder = load_model_and_encoder()

    y_pred_encoded = model.predict(X)[0]
    y_pred_label = label_encoder.inverse_transform([y_pred_encoded])[0]

    # Probabilities
    try:
        prob = model.predict_proba(X)[0]
        proba_dict = {
            label_encoder.inverse_transform([0])[0]: prob[0],
            label_encoder.inverse_transform([1])[0]: prob[1],
            label_encoder.inverse_transform([2])[0]: prob[2],
        }
    except:
        proba_dict = None

    return y_pred_label, int(y_pred_encoded), proba_dict


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("💧 Navigation")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "ML Predictions",
        "Single Sample Prediction",
        "Model Comparison",
        "Map – All Stations",
        "Map – Poor Quality Hotspots",
        "Map – Parameter Trends",
        "Map – Time Trends",
        "Forecasting Predictions"
    ]
)

# =========================================================
# HOME PAGE
# =========================================================

if page == "Home":

    # ==============================
    # HEADER
    # ==============================
    st.markdown("""
        <div style='text-align:center; padding: 20px;'>
            <h1 style='color:#1A73E8; font-size: 48px;'>💧 California Water Quality Index</h1>
            <h3 style='color:#1A73E8;'>Machine Learning – Environmental Monitoring Dashboard</h3>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ==============================
    # INTRODUCTION SECTION
    # ==============================
    st.markdown("""
    <div style='font-size:18px; line-height:1.7;'>
        Welcome to the <b>California Water Quality Index (WQI) Dashboard</b>.<br><br>
        This application demonstrates a <b>Machine Learning–based</b> approach to classify 
        water quality across California using features such as:
        <ul>
            <li>Dissolved Oxygen</li>
            <li>pH</li>
            <li>Turbidity</li>
            <li>Conductivity</li>
            <li>Temperature</li>
            <li>Geolocation</li>
        </ul>
        The ML model predicts one of three conditions:
        <b>Good</b>, <b>Moderate</b>, or <b>Poor</b> water quality.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ==============================
    # KPI METRICS SECTION
    # ==============================
    st.subheader("📊 Dashboard Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("ML Model", "XGBoost Classifier", "v1.0")
    col2.metric("Features Used", "11 Inputs")
    col3.metric("Classes", "Good / Moderate / Poor")

    st.markdown("<br>", unsafe_allow_html=True)

    # ==============================
    # WATER-THEMED INFO CARDS
    # ==============================
    col4, col5 = st.columns(2)

    with col4:
        st.markdown("""
        <div style='background:#27BEF5; padding:20px; border-radius:12px;'>
            <h4>🌎 Goal of the Project</h4>
            <p style='font-size:16px;'>
                The purpose of this project is to support environmental agencies 
                by providing scalable and automated water quality classification.  
                Using historical water sampling data, this model can assist in 
                detecting early signs of water degradation.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown("""
        <div style='background:#27BEF5; padding:20px; border-radius:12px;'>
            <h4>🧠 Why Machine Learning?</h4>
            <p style='font-size:16px;'>
                ML models capture complex interactions between hydrological, chemical, 
                and geographical parameters.  
                XGBoost, in particular, provides excellent accuracy and interpretability 
                for environmental datasets.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ==============================
    # MODEL SUMMARY SECTION
    # ==============================
    st.subheader("🤖 Model Summary")

    st.markdown("""
    <div style='font-size:16px; line-height:1.7;'>
        • Algorithm: <b>XGBoost</b> (Multi-Class Classification)<br>
        • Oversampling: <b>SMOTE</b> (to balance Poor class)<br>
        • Preprocessing: Scaling + One-Hot Encoding<br>
        • Feature Engineering: DO/Temp Ratio, Seasonal Encoding, Depth Handling<br>
        • Output Categories: Good, Moderate, Poor<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")


# =========================================================
# ML PREDICTIONS PAGE
# =========================================================

elif page == "ML Predictions":
    st.markdown("<h2 style='text-align:center;'>🤖 ML Predictions (Full Input)</h2>",
                unsafe_allow_html=True)

    st.markdown("Enter full water-quality parameters below to classify WQI.")

    st.markdown("---")

    with st.form("ml_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            sample_date = st.date_input("Sample Date")
            station_type = st.selectbox("Station Type", ["River", "Lake", "Groundwater", "Reservoir", "Other"])
            do = st.number_input("Dissolved Oxygen (mg/L)", 0.0, 20.0, 7.5, 0.1)

        with col2:
            ph = st.number_input("pH", 0.0, 14.0, 7.2, 0.1)
            turb = st.number_input("Turbidity (NTU)", 0.0, 1000.0, 2.0, 0.1)
            sc = st.number_input("Conductivity (µS/cm)", 0.0, 20000.0, 500.0, 10.0)

        with col3:
            temp = st.number_input("Temperature (°C)", -5.0, 50.0, 18.0, 0.5)
            depth = st.number_input("Depth (m)", 0.0, 100.0, 1.0, 0.1)
            lat = st.number_input("Latitude", -90.0, 90.0, 37.5, 0.01)
            lon = st.number_input("Longitude", -180.0, 180.0, -121.9, 0.01)

        submit = st.form_submit_button("🔍 Predict WQI")

    if submit:
        label, code, proba = predict_single_sample(sample_date, station_type, do, ph, turb, sc, temp, depth, lat, lon)

        st.success(f"### 🎯 Predicted Water Quality Class: **{label}**")

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

# =========================================================
# MODEL COMPARISON PAGE
# =========================================================

elif page == "Model Comparison":

    st.markdown("<h2 style='text-align:center;'>📊 Model Comparison Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("Compare XGBoost with other machine learning models evaluated during training.")
    st.markdown("---")

    st.info("""
    This dashboard compares multiple machine learning models using the same preprocessing,
    SMOTE balancing, and test dataset.  
    It helps demonstrate **why XGBoost was chosen** as the final model.
    """)

    # Load precomputed results
    results_df = pd.read_csv("model_comparison_results.csv")

    # ----------------------------------------------
    # Display Table
    # ----------------------------------------------
    st.subheader("📌 Overall Model Comparison Table")
    st.dataframe(
        results_df.style.highlight_max(
            color='#3D6357',
            subset=["Accuracy", "F1 Weighted", "F1 (Poor Class)"]
        )
    )

    st.markdown("---")

    # ----------------------------------------------
    # Charts
    # ----------------------------------------------
    st.markdown("### 📈 Accuracy Comparison")
    st.bar_chart(results_df.set_index("Model")["Accuracy"])

    st.markdown("### 📈 F1 Score (Weighted)")
    st.bar_chart(results_df.set_index("Model")["F1 Weighted"])

    st.markdown("### ⚠️ F1 Score for POOR Class (Most Important)")
    st.bar_chart(results_df.set_index("Model")["F1 (Poor Class)"])

    st.markdown("---")

    # ----------------------------------------------
    # Final Conclusion
    # ----------------------------------------------
    st.success("""
    ## 🧠 Conclusion: Why XGBoost is the Best Model?

    - XGBoost achieves **highest accuracy** among all tested models  
    - It provides the **best F1 Score for the Poor class**, which is the most critical for environmental classification  
    - Handles nonlinear relationships effectively  
    - Works extremely well with engineered features  
    - Robust even with imbalanced datasets (aided by SMOTE)  

    This comparison scientifically justifies choosing **XGBoost** as the final WQI classifier.
    """)

# =========================================================
# MAP VISUALIZATIONS PAGE
# =========================================================

elif page == "Map – All Stations":

    st.markdown("<h2 style='text-align:center;'>🗺️ All California Water Sampling Stations</h2>",
                unsafe_allow_html=True)

    df = processed_df.copy().dropna(subset=["latitude", "longitude"])

    st.write(f"Total Samples Plotted: **{len(df)}**")

    import plotly.express as px

    fig = px.scatter_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        color="WQI_Class",
        color_discrete_map={
            "Good": "green",
            "Moderate": "orange",
            "Poor": "red"
        },
        hover_name="station_type",
        hover_data={"latitude": True, "longitude": True, "WQI_Class": True},
        zoom=5,
        mapbox_style="open-street-map",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    ✔ Green = Good water quality  
    ✔ Orange = Moderate water quality  
    ✔ Red = Poor water quality  
    """)


elif page == "Map – Poor Quality Hotspots":

    st.markdown("<h2 style='text-align:center;'>🔥 Hotspots of Poor Water Quality</h2>",
                unsafe_allow_html=True)

    df = processed_df.copy().dropna(subset=["latitude", "longitude"])
    poor_df = df[df["WQI_Class"] == "Poor"]

    if poor_df.empty:
        st.success("No poor-quality samples found in the dataset.")
    else:
        st.write(f"Total POOR samples: **{len(poor_df)}**")

        import plotly.express as px

        # Scatter-mapbox for points
        fig1 = px.scatter_mapbox(
            poor_df,
            lat="latitude",
            lon="longitude",
            color="WQI_Class",
            color_discrete_map={"Poor": "red"},
            zoom=5,
            mapbox_style="open-street-map",
            hover_data=["station_type", "DissolvedOxygen_mg/L", "pH_pH units"]
        )
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown("### 🔥 Heatmap of Poor Quality Concentrations")

        # Heatmap
        fig2 = px.density_mapbox(
            poor_df,
            lat="latitude",
            lon="longitude",
            z=None,
            radius=25,
            center=dict(lat=36.8, lon=-119.5),
            zoom=5,
            mapbox_style="stamen-terrain",
            color_continuous_scale="jet",
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.info("""
        🔍 **Interpretation:**  
        - Red clusters indicate high density of poor samples  
        - Often found near urban waterways, agricultural runoff points, or industrial regions  
        """)

elif page == "Map – Parameter Trends":

    st.markdown("<h2 style='text-align:center;'>🌡 Geographic Trends in Water Quality Parameters</h2>",
                unsafe_allow_html=True)

    df = processed_df.copy().dropna(subset=["latitude", "longitude"])

    parameter = st.selectbox(
        "Choose parameter to visualize:",
        ["DissolvedOxygen_mg/L", "pH_pH units", "Turbidity_NTU", "SpecificConductance_µS/cm"]
    )

    import plotly.express as px

    fig = px.scatter_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        color=parameter,
        color_continuous_scale="Viridis",
        zoom=5,
        mapbox_style="carto-positron",
        hover_data=["WQI_Class", parameter]
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(f"""
    🔍 **About {parameter}:**

    - **Dissolved Oxygen** → Low levels indicate pollution or biological degradation  
    - **pH** → Extremes suggest chemical contamination  
    - **Turbidity** → High values show sediments or organic pollution  
    - **Specific Conductance** → High values indicate salts/minerals  
    """)

elif page == "Map – Time Trends":

    st.markdown("<h2 style='text-align:center;'>🕒 Time-Based Geographic Trends</h2>",
                unsafe_allow_html=True)

    st.markdown("""
    Explore how water quality and key parameters change over **time** across California.
    
    Use the time slider to animate changes month-by-month or year-by-year.
    """)

    df = processed_df.copy().dropna(subset=["latitude", "longitude", "sample_date"])

    # ------------------------
    # Extract temporal features
    # ------------------------
    df["sample_date"] = pd.to_datetime(df["sample_date"], errors="coerce")
    df["Year"] = df["sample_date"].dt.year
    df["Month"] = df["sample_date"].dt.month
    df["YearMonth"] = df["sample_date"].dt.to_period("M").astype(str)

    # ------------------------
    # Time granularity selector
    # ------------------------
    mode = st.radio(
        "Select Time Granularity:",
        ["Yearly", "Monthly"],
        horizontal=True
    )

    if mode == "Yearly":
        time_values = sorted(df["Year"].unique())
        selected_time = st.slider(
            "Select Year:",
            min_value=min(time_values),
            max_value=max(time_values),
            value=min(time_values),
            step=1
        )
        df_time = df[df["Year"] == selected_time]
        st.write(f"📅 Showing data for **Year {selected_time}**")

    else:
        time_values = sorted(df["YearMonth"].unique())
        selected_time = st.select_slider(
            "Select Year-Month:",
            options=time_values,
            value=time_values[0]
        )
        df_time = df[df["YearMonth"] == selected_time]
        st.write(f"📅 Showing data for **{selected_time}**")

    st.markdown("---")

    # ------------------------
    # Parameter selection
    # ------------------------
    parameter = st.selectbox(
        "Select parameter to visualize:",
        ["DissolvedOxygen_mg/L", "pH_pH units", "Turbidity_NTU", "SpecificConductance_µS/cm"]
    )

    import plotly.express as px

    if df_time.empty:
        st.warning("⚠ No samples available for this time period.")
    else:
        # ------------------------
        # Time-dependent scatter map
        # ------------------------
        fig = px.scatter_mapbox(
            df_time,
            lat="latitude",
            lon="longitude",
            color=parameter,
            color_continuous_scale="Viridis",
            zoom=5,
            height=600,
            mapbox_style="carto-positron",
            hover_data=["WQI_Class", "station_type", parameter, "sample_date"]
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.info("""
        🔍 **Interpretation Tips:**
        - Seasonal variations (winter vs summer) may influence DO & pH  
        - Year-to-year patterns can reveal long-term environmental trends  
        - Monthly trends help detect pollution spikes or recovery periods  
    """)
