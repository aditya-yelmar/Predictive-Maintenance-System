import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parent.parent

X = np.load(BASE / "data" / "X_sequences.npy")
y = np.load(BASE / "data" / "y_sequences.npy")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1,1)

class SequenceLSTM(nn.Module):

    def __init__(self,input_size):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.2)

        self.fc = nn.Linear(64,1)

    def forward(self,x):

        out,_ = self.lstm(x)

        out = out[:,-1,:]

        out = self.dropout(out)

        out = self.fc(out)

        return out

model = SequenceLSTM(
    input_size=X_train.shape[2]
)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 10

for epoch in range(epochs):

    predictions = model(X_train)

    loss = criterion(
        predictions,
        y_train
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    print(
        f"Epoch {epoch+1}/{epochs} Loss: {loss.item():.4f}"
    )

torch.save(
    model.state_dict(),
    BASE / "models" / "sequence_rul_model.pt"
)

print("\nSequence LSTM Saved!")