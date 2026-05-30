import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

train = pd.read_csv(
    BASE / "data" / "processed_train.csv"
)

max_cycles = (
    train
    .groupby("engine_id")
    ["cycle"]
    .max()
)

train["RUL"] = (
    train["engine_id"]
    .map(max_cycles)
    -
    train["cycle"]
)

output = BASE / "data" / "train_with_rul.csv"

train.to_csv(
    output,
    index=False
)

print("\nDone!\n")

print(
    train[
        [
            "engine_id",
            "cycle",
            "RUL"
        ]
    ].head(10)
)