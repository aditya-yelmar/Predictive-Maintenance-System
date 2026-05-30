import pandas as pd
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

BASE = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    BASE / "data" / "train_with_rul.csv"
)

features = [
    col for col in df.columns
    if col not in ["engine_id", "cycle", "RUL"]
]

X = df[features].values
y = df["RUL"].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
).unsqueeze(1)

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
).unsqueeze(1)

y_train = torch.tensor(
    y_train,
    dtype=torch.float32
).view(-1, 1)

y_test = torch.tensor(
    y_test,
    dtype=torch.float32
).view(-1, 1)


class RULModel(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            batch_first=True
        )

        self.fc = nn.Linear(
            64,
            1
        )

    def forward(self, x):

        out, _ = self.lstm(x)

        out = out[:, -1, :]

        out = self.fc(out)

        return out


model = RULModel(
    input_size=X_train.shape[2]
)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 10

for epoch in range(epochs):

    model.train()

    predictions = model(X_train)

    loss = criterion(
        predictions,
        y_train
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    print(
        f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}"
    )

torch.save(
    model.state_dict(),
    BASE / "models" / "rul_model.pt"
)

print("\nModel Saved!")