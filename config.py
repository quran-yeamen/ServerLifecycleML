import os

# Get the absolute path of the project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define file paths dynamically
DATA_PATH = os.path.join(BASE_DIR, "data", "synthetic_server_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "server_failure_model.pkl")
NOTEBOOKS_PATH = os.path.join(BASE_DIR, "notebooks")
SCRIPTS_PATH = os.path.join(BASE_DIR, "scripts")
DASHBOARD_PATH = os.path.join(BASE_DIR, "dashboard")

print(f"Project Base: {BASE_DIR}")
print(f"Dataset Path: {DATA_PATH}")
print(f"Model Path: {MODEL_PATH}")
print(f"Notebooks Path: {NOTEBOOKS_PATH}")
print(f"Scripts Path: {SCRIPTS_PATH}")
print(f"Dashboard Path: {DASHBOARD_PATH}")