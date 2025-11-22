import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Load processed dataset
df = pd.read_pickle("processed_dataset_WQ.pkl")
model = joblib.load("wqi_xgb_pipeline.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Extract the preprocessor from the XGB pipeline
preprocessor = model.named_steps["preprocessor"]

# -------------------------------
# Feature Names
# -------------------------------
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

# -------------------------------
# Feature Engineering Function
# -------------------------------
def engineer_features(df):
    df = df.copy()

    df["sample_date"] = pd.to_datetime(df["sample_date"], errors="coerce")
    df["Month"] = df["sample_date"].dt.month

    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    df["DO_Temp_Ratio"] = df["DissolvedOxygen_mg/L"] / (df["WaterTemperature_°C"] + 1)
    df["sample_depth_meter"] = df["sample_depth_meter"].fillna(0)

    return df

# -------------------------------
# APPLY FEATURE ENGINEERING
# -------------------------------
df_eng = engineer_features(df)

# Now all engineered columns exist
X = df_eng[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = label_encoder.transform(df_eng["WQI_Class"])

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# Models to compare
# -------------------------------
compare_models = {
    "XGBoost": model,
    "Random Forest": RandomForestClassifier(n_estimators=400, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=3000, class_weight='balanced'),
    "SVM (RBF Kernel)": SVC(probability=True, class_weight='balanced'),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
}

results = []

# Preprocessing steps
base_steps = [
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42, k_neighbors=5))
]

# -------------------------------
# Train & Evaluate Models
# -------------------------------
for name, clf in compare_models.items():

    if name == "XGBoost":
        y_pred = clf.predict(X_test)
    else:
        pipe = ImbPipeline(steps=base_steps + [('clf', clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    poor_class_index = list(label_encoder.classes_).index("Poor")
    f1_poor = f1_score(y_test, y_pred, labels=[poor_class_index], average="macro")

    results.append([name, acc, f1_weighted, f1_poor])

# Save results
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1 Weighted", "F1 (Poor Class)"])
results_df.to_csv("model_comparison_results.csv", index=False)

print("Saved model_comparison_results.csv ✔️")
