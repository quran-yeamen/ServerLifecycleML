import pandas as pd
import joblib
from config import DATA_PATH, MODEL_PATH  # Import paths from config.py
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv(DATA_PATH)
print(f"Data loaded from: {DATA_PATH}")

# Select features & target variable
FEATURES = ["Data_Center", "OS", "RAM_GB", "CPU_Cores", "Storage_GB", "Availability"]
TARGET = "Failure_Rate"

X = df[FEATURES]
y = df[TARGET]

# Load trained model
model = joblib.load(MODEL_PATH)
print(f"Model loaded from: {MODEL_PATH}")

# Make predictions
y_pred = model.predict(X)

# Evaluate performance
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("\n Model Evaluation on Full Dataset:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")
