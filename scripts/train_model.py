import pandas as pd
import numpy as np
import joblib
from config import DATA_PATH, MODEL_PATH  # Import paths from config.py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv(DATA_PATH)
print(f"Data loaded from: {DATA_PATH}")

# Select features & target variable
FEATURES = ["Data_Center", "OS", "RAM_GB", "CPU_Cores", "Storage_GB", "Availability"]
TARGET = "Failure_Rate"

X = df[FEATURES]
y = df[TARGET]

# Preprocessing: One-hot encode categorical features, scale numerical features
preprocessor = ColumnTransformer([
    ("onehot", OneHotEncoder(handle_unknown="ignore"), ["Data_Center", "OS"]),
    ("scale", StandardScaler(), ["RAM_GB", "CPU_Cores", "Storage_GB", "Availability"])
])

# Define model pipeline (Random Forest)
model_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
print("Training model...")
model_pipeline.fit(X_train, y_train)

# Make predictions
y_pred = model_pipeline.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Save trained model
joblib.dump(model_pipeline, MODEL_PATH)
print(f"Model saved at: {MODEL_PATH}")
