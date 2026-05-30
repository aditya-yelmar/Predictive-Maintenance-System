import pandas as pd
import joblib
from pathlib import Path

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

sys.path.append(str(BASE))

from maintenance.scheduler import maintenance_recommendation
BASE = Path(__file__).resolve().parent.parent

# Load anomaly model
anomaly_model = joblib.load(
    BASE / "models" / "anomaly_model.pkl"
)

# Load sample data
data = pd.read_csv(
    BASE / "data" / "processed_train.csv"
)

row = data.iloc[100]

features = data.drop(
    columns=["engine_id", "cycle"]
)

prediction = anomaly_model.predict(
    [features.iloc[100]]
)

is_anomaly = prediction[0] == -1

example_rul = 25

result = maintenance_recommendation(
    rul=example_rul,
    anomaly=is_anomaly
)

print("\nENGINE HEALTH REPORT\n")

print("Engine:", row["engine_id"])
print("Cycle:", row["cycle"])

print(
    "Anomaly:",
    "YES" if is_anomaly else "NO"
)

print(
    "Predicted RUL:",
    example_rul
)

print(
    "Priority:",
    result["priority"]
)

print(
    "Action:",
    result["action"]
)