import pandas as pd
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

data = pd.read_csv(
    BASE / "data" / "processed_train.csv"
)

print("Starting Sensor Stream...\n")

for index, row in data.head(50).iterrows():

    print(
        f"Engine: {row['engine_id']} | "
        f"Cycle: {row['cycle']}"
    )

    time.sleep(1)

print("\nStreaming Finished!")