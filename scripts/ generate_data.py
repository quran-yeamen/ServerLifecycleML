import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Function to generate a random date
def random_date(start, end):
    return start + timedelta(days=random.randint(0, (end - start).days))

# Define parameters for synthetic data
NUM_SERVERS = 1000
DATA_CENTERS = ["New York", "San Francisco", "Chicago", "Dallas", "Atlanta"]
OS_LIST = ["Linux", "Windows Server", "Ubuntu", "Red Hat", "CentOS"]

# Generate synthetic dataset
start_date = datetime(2015, 1, 1)
end_date = datetime(2025, 1, 1)

data = []
for i in range(NUM_SERVERS):
    purchase_date = random_date(start_date, end_date - timedelta(days=365*3))
    decommission_date = purchase_date + timedelta(days=random.randint(365*3, 365*7))  # Lifespan of 3-7 years

    data.append([
        f"SRV-{i+1}",
        random.choice(DATA_CENTERS),
        purchase_date.strftime("%Y-%m-%d"),
        decommission_date.strftime("%Y-%m-%d"),
        random.choice(OS_LIST),
        random.randint(16, 256),  # RAM in GB
        random.randint(4, 64),  # CPU Cores
        random.randint(500, 5000),  # Storage in GB
        round(random.uniform(50, 99.9), 2),  # Availability %
        round(random.uniform(0.01, 5.0), 2)  # Failure Rate %
    ])

# Create DataFrame
columns = ["Server_ID", "Data_Center", "Purchase_Date", "Decommission_Date", "OS", "RAM_GB", "CPU_Cores", "Storage_GB", "Availability", "Failure_Rate"]
df = pd.DataFrame(data, columns=columns)

# Save dataset
df.to_csv("../data/synthetic_server_data.csv", index=False)
print("Synthetic data generated and saved!")
