import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

# Title
st.title("Predictive Maintenance Dashboard")

# Load data
train_df = pd.read_csv(
    BASE / "data" / "processed_train.csv"
)

# Remove extra columns
# train_df = train_df.iloc[:, :26]

# Column names
# columns = ['engine_id', 'cycle']

# for i in range(1, 4):
#     columns.append(f'op_setting_{i}')

# for i in range(1, 22):
#     columns.append(f'sensor_{i}')

# train_df.columns = columns

# Sidebar
st.sidebar.header("Engine Selection")

selected_engine = st.sidebar.slider(
    "Select Engine Number",
    min_value=int(train_df["engine_id"].min()),
    max_value=int(train_df["engine_id"].max()),
    value=1
)
# Filter engine
engine_data = train_df[
    train_df['engine_id'] == selected_engine
]

# Show dataframe
st.subheader("Engine Sensor Data")
st.dataframe(engine_data.head())

# Plot sensor
st.subheader("Sensor 2 Trend")

fig, ax = plt.subplots(figsize=(10,5))

ax.plot(
    engine_data['cycle'],
    engine_data['sensor_2']
)

ax.set_xlabel("Cycles")
ax.set_ylabel("Sensor 2")

st.pyplot(fig)

# RUL estimation
current_cycle = engine_data["cycle"].iloc[-1]

estimated_rul = max(1, 250 - current_cycle)

# Display metrics
st.subheader("Remaining Useful Life")

st.metric(
    label="Estimated RUL",
    value=f"{estimated_rul} cycles"
)

# Maintenance recommendation
st.subheader("Maintenance Recommendation")
# Health status logic
if estimated_rul < 20:
    health_status = "Critical"
elif estimated_rul < 50:
    health_status = "Warning"
else:
    health_status = "Healthy"

# Dashboard metric cards
col1, col2, col3 = st.columns(3)

col1.metric("Engine ID", selected_engine)
col2.metric("Estimated RUL", f"{estimated_rul} cycles")
col3.metric("Health Status", health_status)

if estimated_rul < 20:
    st.error(
        "Critical maintenance required!"
    )

elif estimated_rul < 50:
    st.warning(
        "Schedule maintenance soon."
    )

else:
    st.success(
        "Engine operating normally."
    )