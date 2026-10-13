import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load prepared data
df = pd.read_csv("data/prepared_data.csv")

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date").reset_index(drop=True)

# Features
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

X = df[features]

# Target: next trading day's closing price
y = df["Target_Close"]

# Chronological 80/20 split
split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

test_dates = df["Date"].iloc[split:]

# Train model
model = LinearRegression()

model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, predictions)

print("\nLINEAR REGRESSION RESULTS")
print("------------------------------")

print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")

# Save model
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/stock_model.pkl")

print("\nModel saved successfully!")

# Save test predictions
results = pd.DataFrame({
    "Date": test_dates.values,
    "Actual_Close": y_test.values,
    "Predicted_Close": predictions
})

results.to_csv("data/predictions.csv", index=False)

print("Predictions saved to data/predictions.csv")

# Plot actual vs predicted prices
plt.figure(figsize=(12, 6))

plt.plot(
    results["Date"],
    results["Actual_Close"],
    label="Actual Next-Day Close"
)

plt.plot(
    results["Date"],
    results["Predicted_Close"],
    label="Predicted Next-Day Close"
)

plt.title("Stock Price Prediction - Linear Regression")

plt.xlabel("Date")

plt.ylabel("Closing Price")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig("data/prediction_graph.png")

plt.show()