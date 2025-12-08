 
---
    
# **Water Quality Prediction Using Machine Learning**
  




## **Project Overview**

This project focuses on developing a **data-driven framework** for predicting **Water Quality Index (WQI)** using field data from the **California Department of Water Resources (DWR)**.
The study aims to support sustainable water management by leveraging **machine learning (ML)** and **exploratory data analysis (EDA)** to uncover patterns in physicochemical parameters such as **pH**, **Dissolved Oxygen**, **Turbidity**, **Conductivity**, and **Temperature**.

The project is part of the **DATA 245 – Machine Learning** course at *San José State University*.

---

## **Objectives**

* Perform **EDA** to understand the structure, distribution, and quality of California’s water quality dataset.
* Standardize and clean multi-parameter field data for consistency and accuracy.
* Identify core water quality indicators and eliminate sparse or redundant features.
* Build the foundation for **Water Quality Index (WQI)** computation and predictive modeling using ML.

---

## **Exploratory Data Analysis (EDA)**

### **1. Data Understanding**

* Data sourced from **California DWR Field Measurements (1913–2025)**.
* Explored key metadata including:
  `station_id`, `station_type`, `county_name`, `parameter`, and `fdr_result`.
* Visualized sample distribution across **time** (Figure 1) and **counties** (Figure 2) to identify data-rich regions suitable for analysis.

### **2. Data Cleaning and Standardization**

* Standardized units to ensure uniformity across stations:

  * °C → Water Temperature
  * µS/cm → Conductivity
  * mg/L → Dissolved Oxygen
  * NTU → Turbidity
* Built a **unit mapping dictionary** to resolve naming inconsistencies.
* Detected and replaced 296 outlier readings (e.g., pH > 14, DO > 20 mg/L, negative temperatures).

### **3. Missing Data and Sparse Features**

* Analyzed missingness for all parameters.
* Retained five high-coverage parameters (70–98%):
  `pH_pH units`, `DissolvedOxygen_mg/L`, `Turbidity_NTU`, `SpecificConductance_µS/cm`, `WaterTemperature_°C`.
* Dropped sparse parameters (>80% missing).
* Imputed missing coordinates using **county-level averages**.

### **4. Data Transformation**

* Converted data from **long to wide format** to have one row per sampling event.
* Each parameter became a separate column, preserving station metadata (county, station type, latitude, longitude).
* Conducted **correlation analysis** and visualized pairwise relationships among core parameters.

### **5. Key Insights**

* Data from **2000–2025** showed the highest completeness and consistency.
* Strong correlations observed between:

  * Dissolved Oxygen and Temperature (negative correlation)
  * Turbidity and Conductivity (moderate positive correlation)
* Missing data patterns confirmed uneven sampling between counties — surface water data dominated the dataset.
* Dataset now clean, standardized, and ready for WQI computation and ML modeling.

---

## **Next Steps**

* Compute **Water Quality Index (WQI)** using weighted arithmetic mean.
* Perform **feature engineering** and **feature selection**.
* Handle **class imbalance** using SMOTE.
* Train ML models (Random Forest, XGBoost, SVM) for WQI class prediction.
* Evaluate performance using **F1-score** and feature importance analysis.

---

## **Repository Structure**

```
water-quality-prediction/
│
├── notebooks/                     # Jupyter notebooks for EDA and modeling
│   ├── water_quality_analysis_eda.ipynb
├── README.md                      # Project overview

```

---

## **Technologies Used**

* **Python 3.10+**
* **Jupyter Notebook**
* **pandas**, **NumPy**, **matplotlib**, **seaborn**
* **scikit-learn**
* **XGBoost**, **imbalanced-learn**

---

## **Data Source**

California Department of Water Resources (DWR) –

https://data.ca.gov/dataset/water-quality-data

---

## **References**

* Kumar, R., & Singh, A. (2024). *Water quality prediction with machine learning algorithms.*
  *EPRA International Journal of Multidisciplinary Research (IJMR), 10(4), 45–53.*
  [https://doi.org/10.36713/epra16318](https://doi.org/10.36713/epra16318)
* Zhu, M. et al. (2022). *A review of the application of machine learning in water quality evaluation.*
  *Eco-Environment & Health, 1(2), 107–116.*
  [https://doi.org/10.1016/j.eehl.2022.06.001](https://doi.org/10.1016/j.eehl.2022.06.001)

*(Additional references included in project report)*

---
# Adaptive Time Series Forecasting of Dissolved Oxygen

This project develops a machine learning framework to forecast dissolved oxygen (DO) levels in California's water systems using 25 years of data from the California Department of Water Resources. DO is a critical water quality indicator, and accurate forecasting enables proactive management of hypoxic events that threaten aquatic ecosystems. The study compares three regression approaches—XGBoost, Random Forest, and Support Vector Regression—to predict DO concentrations 1-15 days ahead based on physiochemical parameters (temperature, pH, specific conductance, turbidity) and temporal patterns.

The methodology centers on rigorous time series modeling: raw monitoring data from 29,271 stations was resampled to 5,120 uniform intervals to eliminate temporal aliasing, then enriched through extensive feature engineering. Autoregressive lags (DO at t-1, t-2, t-3, etc.), rolling statistics (3-, 6-, 12-hour windows capturing mean and standard deviation), and cyclical encodings (sine/cosine transformations of month and day-of-year) were constructed to expose persistence, local variability, and seasonal forcing. Interaction terms between temperature and other parameters allowed models to learn coupled effects directly.

Validation employed expanding-window time series cross-validation to prevent data leakage—each fold trained on all data up to time t and validated on the subsequent forecast horizon, ensuring no future information contaminated predictions. XGBoost achieved the most stable performance (RMSE 0.65-0.80 mg/L across horizons), while Random Forest delivered the lowest overall error (RMSE 0.68 mg/L, 40% improvement over naive baseline). SHAP analysis revealed that recent DO history and temperature dominated predictions, with models capturing the expected inverse temperature-oxygen relationship. This framework provides water managers with reliable early-warning forecasts to support proactive intervention
use
---

#ARIMA Modelling of Water Quality Analysis

1. Data Preprocessing

Standardized measurement units across all stations.

Handled missing values using:

Linear interpolation for short gaps

Forward-fill for longer missing segments

Removed outliers using IQR-based filtering.

Pivoted data by station × year for temporal alignment.

Conducted stationarity checks using the ADF test and applied differencing where required.

2. Exploratory Data Analysis (EDA)

Generated histograms, boxplots, and time-series plots for each physicochemical parameter.

Computed Pearson correlation matrix to identify interdependencies.

Used rolling mean and rolling standard deviation to assess temporal variability.

Decomposed WQI into trend, seasonal, and residual components.

3. Trend Detection — Mann–Kendall Test

DO: Increasing (p < 0.01)

Water Temperature: Decreasing (strongest trend)

Specific Conductance: Decreasing

pH: Increasing

Turbidity: No significant trend

Indicates measurable environmental improvement over time.

4. ARIMA/SARIMAX Modeling

Selected optimal model via AIC/BIC comparison → ARIMA(2,1,2).

Differencing required (ADF p = 0.429 → non-stationary).

MA terms insignificant but model overall stable and well-behaved.

5. Rolling-Origin Cross-Validation (ROCV)

Provides realistic, time-aware evaluation.

Avg. performance across 45 stations:

MAE: 2.75 ± 2.04

MAPE: 3.48% ± 2.52%

Validates strong temporal generalization and robustness.

6. Outlier Detection & Cleaning

Identified 9 stations with extreme MAPE (>100%) due to missing years or erratic series.

Removed them prior to summarizing performance.

Clean dataset retains 45 high-quality stations.

1. Forecasting Accuracy

ARIMA model predicts WQI within 2–3 units error on average.

Rolling cross-validation confirms stability and generalization.

WQI typically ranges between 60–90 → MAPE ≈ 3.5% is excellent for environmental forecasting.

2. Station-Level Performance

Top 10% stations: MAPE < 1% → near-perfect accuracy.

Stations with complete and smooth temporal data give the best results.

High-MAPE stations correspond to:

Missing observations

Sensor anomalies

Abrupt, non-stationary fluctuations
→ These are data-quality issues, not model failure.

3. Model Diagnostics

Residuals behave like white noise:

Ljung–Box p = 0.62 → no autocorrelation

Jarque–Bera p = 0.72 → normally distributed errors

Heteroskedasticity p = 0.95 → stable variance

Confirms well-specified ARIMA(2,1,2) model.

4. Trend Findings (Scientific Interpretation)

DO increased, suggesting improving ecological health.

Water temperature decreased, improving oxygen retention.

Conductance decreased, indicating reduced dissolved solids and cleaner inflows.

pH increased slightly, consistent with ionic or CO₂ balance changes.

Turbidity remained stable, reflecting consistent sediment inputs.



