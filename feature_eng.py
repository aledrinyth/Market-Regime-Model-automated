import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from ta.trend import MACD
from sklearn.preprocessing import StandardScaler


# Read the historical data from the CSV file and compute additional features
df = pd.read_csv("data/VAS_historical_data.csv")

print(df.head())

df['Date'] = pd.to_datetime(df['Date'], utc=True)
df['Daily Log Return'] = df['Close'].pct_change().apply(lambda x: np.log(1 + x))
df['20-Day Rolling Volatility'] = df['Daily Log Return'].rolling(window=20).std() * np.sqrt(252)  # Annualized volatility
df['50-Day Moving Average'] = df['Close'].rolling(window=50).mean()
df['200-Day Moving Average'] = df['Close'].rolling(window=200).mean()
df['Distance from 50-Day MA'] = df['Close'] - df['50-Day Moving Average']
df['Distance from 200-Day MA'] = df['Close'] - df['200-Day Moving Average']
rolling_peak = df['Close'].rolling(window=20, min_periods=1).max()
df['Max Drawdown'] = (df['Close'] - rolling_peak) / rolling_peak
df['Volume Change'] = df['Volume'].pct_change()


df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()

# Calculate RSI
df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
macd_indicator = MACD(close=df['Close'], window_fast=12, window_slow=26, window_sign=9)
df['MACD'] = macd_indicator.macd()
df['MACD Signal'] = macd_indicator.macd_signal()

df_VIX = pd.read_csv("data/VIX_historical_data.csv")
df_VIX['Date'] = pd.to_datetime(df_VIX['Date'], utc=True)


# Merge the VIX data with the main DataFrame on the 'Date' column
df = df.merge(df_VIX[['Date', 'Close']], on='Date', how='left', suffixes=('', '_VIX'))

print(df)


# Drop rows with any NaN values
df.dropna(inplace=True)

print(df)
df = df[['Date', 'Close', 'Volume', 'Daily Log Return', '20-Day Rolling Volatility', 
         '50-Day Moving Average', '200-Day Moving Average', 'Distance from 50-Day MA', 
         'Distance from 200-Day MA', 'Max Drawdown', 'Volume Change', 'ATR', 'RSI',
          'MACD', 'MACD Signal', 'Close_VIX']]

# Scale the features using StandardScaler
scaler = StandardScaler()

df_scaled = df.copy()

