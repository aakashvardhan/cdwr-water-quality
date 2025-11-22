# ============================================================
# CLEAN XGBOOST TRAINING SCRIPT (Compatible with XGB 1.7.6)
# ============================================================

import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_pickle("processed_dataset_WQ.pkl")   # <-- update path if needed


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

def engineer_features(df):
    df = df.copy()

    # Date processing
    df['sample_date'] = pd.to_datetime(df['sample_date'])
    df['Month'] = df['sample_date'].dt.month
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

    # DO/Temperature ratio
    df['DO_Temp_Ratio'] = df['DissolvedOxygen_mg/L'] / (df['WaterTemperature_°C'] + 1)

    # Default depth
    if 'sample_depth_meter' in df.columns:
        df['sample_depth_meter'] = df['sample_depth_meter'].fillna(0)

    return df


df = engineer_features(df)


# ============================================================
# 3. FEATURE COLUMNS
# ============================================================

numeric_features = [
    'DissolvedOxygen_mg/L',
    'pH_pH units',
    'Turbidity_NTU',
    'SpecificConductance_µS/cm',
    'WaterTemperature_°C',
    'sample_depth_meter',
    'DO_Temp_Ratio',
    'latitude',
    'longitude',
    'Month_sin',
    'Month_cos'
]

categorical_features = ['station_type']
target_col = 'WQI_Class'


# ============================================================
# 4. SPLIT DATA
# ============================================================

X = df[numeric_features + categorical_features]
y = df[target_col]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)


# ============================================================
# 5. PREPROCESSING PIPELINE
# ============================================================

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# ============================================================
# 6. XGBOOST CLASSIFIER (CLEAN VERSION FOR STREAMLIT)
# ============================================================

xgb_model = XGBClassifier(
    objective="multi:softprob",
    num_class=len(label_encoder.classes_),
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

# ============================================================
# 7. FULL PIPELINE (Preprocess → SMOTE → XGB)
# ============================================================

pipeline = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42, k_neighbors=5)),
    ("clf", xgb_model)
])

print("Training XGBoost model...")
pipeline.fit(X_train, y_train)


# ============================================================
# 8. SAVE ARTIFACTS FOR STREAMLIT
# ============================================================

joblib.dump(pipeline, "wqi_xgb_pipeline.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("\n🎉 Training completed successfully!")
print("Saved:")
print("  ➤ wqi_xgb_pipeline.pkl")
print("  ➤ label_encoder.pkl")

