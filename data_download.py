import pandas as pd
import requests
import yfinance as yf

# ETF Proxy = ASX:VAS
ticker = "VAS.AX"

ticker_data = yf.Ticker(ticker)

historical_data = ticker_data.history(period="max")

# Save the historical data to a CSV file
historical_data.to_csv("data/VAS_historical_data.csv")

vol_ticker = "^AXVI"
volatility_data = yf.Ticker(vol_ticker).history(period="max")
volatility_data.to_csv("data/VIX_historical_data.csv")
