import pandas as pd
import numpy as np
import joblib
from config import DATA_PATH, MODEL_PATH  # Import paths from config.py
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ==============================
# 1) LOAD & FEATURE ENGINEERING
# ==============================
df = pd.read_csv(DATA_PATH)
print(f"Data loaded from: {DATA_PATH}")

# Convert date columns to datetime
df["Purchase_Date"] = pd.to_datetime(df["Purchase_Date"])
df["Decommission_Date"] = pd.to_datetime(df["Decommission_Date"])

# Add server lifespan in years
df["Server_Lifespan_Years"] = (df["Decommission_Date"] - df["Purchase_Date"]).dt.days / 365.25

# Drop unnecessary columns
drop_cols = ["Server_ID", "Purchase_Date", "Decommission_Date"]
df.drop(columns=drop_cols, inplace=True)

# ============================
# 2) SELECT FEATURES & TARGET
# ============================
FEATURES = [
    "Data_Center",
    "OS",
    "RAM_GB",
    "CPU_Cores",
    "Storage_GB",
    "Availability",
    "Server_Lifespan_Years"
]
TARGET = "Failure_Rate"

X = df[FEATURES]
y = df[TARGET]

# ================================
# 3) PREPROCESSING TRANSFORMATION
# ================================
# Categorical & numeric features
cat_features = ["Data_Center", "OS"]
num_features = ["RAM_GB", "CPU_Cores", "Storage_GB", "Availability", "Server_Lifespan_Years"]

preprocessor = ColumnTransformer([
    ("onehot", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ("scale", StandardScaler(), num_features)
])

# ===========================
# 4) TRAIN-TEST SPLIT (80-20)
# ===========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================
# 5) MODEL DEFINITIONS
# ======================
models = {
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42),
    "LightGBM": LGBMRegressor(random_state=42),
    "Linear Regression": LinearRegression()
}

# ============================================================
# 6) (Optional) HYPERPARAM TUNING DEMO (Example: Random Forest)
# ============================================================
# If you want to skip tuning, just comment out this block
param_dist = {
    "regressor__n_estimators": [50, 100, 200],
    "regressor__max_depth": [5, 10, 20, None],
    "regressor__min_samples_split": [2, 5, 10],
    "regressor__min_samples_leaf": [1, 2, 4],
}

rf_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("regressor", RandomForestRegressor(random_state=42))
])

print("Tuning Random Forest hyperparameters...")
search = RandomizedSearchCV(
    rf_pipeline,
    param_dist,
    n_iter=10,
    scoring="r2",
    cv=5,
    random_state=42,
    n_jobs=-1
)
search.fit(X_train, y_train)

print(f"Best Random Forest Params: {search.best_params_}")
best_rf = search.best_estimator_.named_steps["regressor"]  # Extract tuned model
models["Random Forest"] = best_rf

# =========================
# 7) TRAIN & EVALUATE ALL
# =========================
results = []
best_model_name = None
best_r2 = -999
best_pipeline = None

for model_name, model_obj in models.items():
    print(f"Training {model_name}...")

    # Build pipeline
    model_pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("regressor", model_obj)
    ])

    # Fit model
    model_pipeline.fit(X_train, y_train)

    # Predict & evaluate
    y_pred = model_pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"{model_name} Evaluation:")
    print(f"   MAE: {mae:.4f}")
    print(f"   R²:  {r2:.4f}\n")

    # Track best model
    results.append((model_name, mae, r2))
    if r2 > best_r2:
        best_r2 = r2
        best_model_name = model_name
        best_pipeline = model_pipeline

# ==========================
# 8) PICK & SAVE BEST MODEL
# ==========================
print("\n**Model Comparison**")
for r in results:
    print(f"{r[0]}: MAE = {r[1]:.4f}, R² = {r[2]:.4f}")

print(f"\nBest Model is: {best_model_name} with R² = {best_r2:.4f}")

# Save best
joblib.dump(best_pipeline, MODEL_PATH)
print(f"Saved best model ({best_model_name}) to {MODEL_PATH}")
