import yfinance as yf
import pandas as pd
import os

# Create data folder
os.makedirs("data", exist_ok=True)

# Download stock data
df = yf.download(
    "AAPL",
    period="5y",
    auto_adjust=False,
    progress=False
)

if df.empty:
    raise ValueError("No stock data downloaded. Check your internet connection.")

# Handle MultiIndex columns
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Convert index to Date column
df = df.reset_index()

# Save data
df.to_csv("data/stock_data.csv", index=False)

print("Stock data downloaded successfully!")
print(df.head())
print("\nTotal records:", len(df))