import pandas as pd
import numpy as np
import os

# Load data
df = pd.read_csv("data/stock_data.csv")

# Convert Date
df["Date"] = pd.to_datetime(df["Date"])

# Sort by date
df = df.sort_values("Date")

# Remove duplicate dates
df = df.drop_duplicates(subset=["Date"])

# Convert numerical columns
columns = ["Open", "High", "Low", "Close", "Volume"]

for col in columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove invalid values
df = df.replace([np.inf, -np.inf], np.nan)

df = df.dropna(subset=columns)

# Feature engineering
df["Daily_Return"] = df["Close"].pct_change()

df["Price_Range"] = df["High"] - df["Low"]

df["SMA_10"] = df["Close"].rolling(10).mean()

df["SMA_20"] = df["Close"].rolling(20).mean()

df["Volatility"] = df["Daily_Return"].rolling(10).std()

# Target: next trading day's closing price
df["Target_Close"] = df["Close"].shift(-1)

# Remove rows with missing feature or target values
df = df.replace([np.inf, -np.inf], np.nan)

df = df.dropna()

# Save prepared data
os.makedirs("data", exist_ok=True)

df.to_csv("data/prepared_data.csv", index=False)

print("Data preparation completed successfully!")

print("\nPrepared data:")
print(df.head())

print("\nDataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())