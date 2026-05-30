import pandas as pd
from pathlib import Path

from sklearn.ensemble import IsolationForest
import joblib

BASE = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    BASE / "data" / "processed_train.csv"
)

features = [
    col
    for col in df.columns
    if col not in ["engine_id", "cycle"]
]

X = df[features]

model = IsolationForest(
    contamination=0.05,
    random_state=42
)

model.fit(X)

joblib.dump(
    model,
    BASE / "models" / "anomaly_model.pkl"
)

print("Anomaly Model Saved!")