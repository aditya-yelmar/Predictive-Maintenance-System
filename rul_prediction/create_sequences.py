import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    BASE / "data" / "train_with_rul.csv"
)

feature_cols = [
    col
    for col in df.columns
    if col not in ["engine_id", "cycle", "RUL"]
]

WINDOW_SIZE = 30

X = []
y = []

for engine_id in df["engine_id"].unique():

    engine_data = df[
        df["engine_id"] == engine_id
    ]

    features = engine_data[
        feature_cols
    ].values

    rul_values = engine_data[
        "RUL"
    ].values

    for i in range(
        len(engine_data) - WINDOW_SIZE
    ):

        X.append(
            features[i:i+WINDOW_SIZE]
        )

        y.append(
            rul_values[i+WINDOW_SIZE]
        )

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

np.save(
    BASE / "data" / "X_sequences.npy",
    X
)

np.save(
    BASE / "data" / "y_sequences.npy",
    y
)

print("Sequences Saved!")