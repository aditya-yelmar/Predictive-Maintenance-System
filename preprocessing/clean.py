import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

BASE = Path(__file__).resolve().parent.parent

train_path = BASE / "data" / "train_FD001.txt"
test_path = BASE / "data" / "test_FD001.txt"

columns = (
    ["engine_id", "cycle"] +
    [f"setting_{i}" for i in range(1, 4)] +
    [f"sensor_{i}" for i in range(1, 22)]
)

train = pd.read_csv(
    train_path,
    sep=r"\s+",
    header=None
)

test = pd.read_csv(
    test_path,
    sep=r"\s+",
    header=None
)

train = train.iloc[:, :26]
test = test.iloc[:, :26]

train.columns = columns
test.columns = columns

drop_cols = []

for col in train.columns:
    if train[col].nunique() <= 1:
        drop_cols.append(col)

train = train.drop(columns=drop_cols)
test = test.drop(columns=drop_cols)

features = train.columns.drop(["engine_id", "cycle"])

scaler = MinMaxScaler()

train[features] = scaler.fit_transform(train[features])
test[features] = scaler.transform(test[features])

train.to_csv(
    BASE / "data" / "processed_train.csv",
    index=False
)

test.to_csv(
    BASE / "data" / "processed_test.csv",
    index=False
)

print("Done!")
print(train.head())python preprocessing/clean.py