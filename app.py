import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Page configuration
st.set_page_config(
    page_title="Stock Price Trend Predictor",
    page_icon="📈",
    layout="wide"
)

# Application title
st.title("📈 Stock Price Trend Predictor")

st.write(
    "A machine-learning application for analyzing "
    "historical stock data and estimating future closing prices."
)

st.divider()

# Load prepared data
@st.cache_data
def load_data():
    df = pd.read_csv("data/prepared_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)

try:
    df = load_data()

    model = joblib.load("models/stock_model.pkl")

except FileNotFoundError:
    st.error(
        "Required files were not found. Run download_data.py, "
        "prepare_data.py, and train_model.py first."
    )
    st.stop()

# Feature columns
features = [
    "Open",
    "High",
    "Low",
    "Volume",
    "Daily_Return",
    "Price_Range",
    "SMA_10",
    "SMA_20",
    "Volatility"
]

# Stock selection
st.subheader("Stock Analysis")

st.write("Currently supported stock: Apple (AAPL)")

# Select date range
min_date = df["Date"].min().date()

max_date = df["Date"].max().date()

start_date, end_date = st.date_input(
    "Select historical period",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

filtered = df[
    (df["Date"].dt.date >= start_date) &
    (df["Date"].dt.date <= end_date)
].copy()

if filtered.empty:
    st.warning("No data available for this date range.")
    st.stop()

# Latest price information
latest = df.iloc[-1]

latest_price = float(latest["Close"])

st.divider()

st.subheader("Stock Summary")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Latest Closing Price",
    f"${latest_price:.2f}"
)

col2.metric(
    "Historical Records",
    f"{len(filtered)}"
)

col3.metric(
    "Latest Data Date",
    latest["Date"].strftime("%Y-%m-%d")
)

st.divider()

# Historical stock graph
st.subheader("Historical Closing Price")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    filtered["Date"],
    filtered["Close"],
    label="Closing Price"
)

ax.set_xlabel("Date")

ax.set_ylabel("Price (USD)")

ax.set_title("Historical Stock Price")

ax.legend()

ax.grid(True)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)

st.divider()

# Predict next trading day's closing price
st.subheader("Next-Day Price Estimate")

latest_features = latest[features].to_frame().T

predicted_price = float(
    model.predict(latest_features)[0]
)

if predicted_price > latest_price:
    trend = "UP ↑"
elif predicted_price < latest_price:
    trend = "DOWN ↓"
else:
    trend = "UNCHANGED"

col1, col2 = st.columns(2)

col1.metric(
    "Estimated Next-Day Closing Price",
    f"${predicted_price:.2f}"
)

col2.metric(
    "Estimated Price Trend",
    trend
)

st.caption(
    "The predicted value is an estimate based on historical data, "
    "not a guarantee of future market performance."
)

st.divider()

# Model evaluation
st.subheader("Model Performance")

split = int(len(df) * 0.8)

test_data = df.iloc[split:].copy()

X_test = test_data[features]

y_test = test_data["Target_Close"]

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, predictions)

col1, col2, col3, col4 = st.columns(4)

col1.metric("MAE", f"{mae:.4f}")

col2.metric("MSE", f"{mse:.4f}")

col3.metric("RMSE", f"{rmse:.4f}")

col4.metric("R²", f"{r2:.4f}")

st.divider()

# Actual vs predicted graph
st.subheader("Actual vs Predicted Prices")

test_dates = test_data["Date"]

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    test_dates,
    y_test,
    label="Actual Next-Day Close"
)

ax.plot(
    test_dates,
    predictions,
    label="Predicted Next-Day Close"
)

ax.set_xlabel("Date")

ax.set_ylabel("Closing Price (USD)")

ax.set_title("Actual vs Predicted Stock Prices")

ax.legend()

ax.grid(True)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)

# Display prediction table
st.subheader("Prediction Results")

results = pd.DataFrame({
    "Date": test_dates.values,
    "Actual Closing Price": y_test.values,
    "Predicted Closing Price": predictions
})

st.dataframe(
    results.tail(20),
    use_container_width=True
)

# Footer
st.divider()

st.caption(
    "Stock Price Trend Predictor | Machine Learning Capstone Project"
)

st.caption(
    "Educational use only. This application does not guarantee "
    "future stock prices or investment returns."
)