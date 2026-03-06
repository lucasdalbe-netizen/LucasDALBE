import yfinance as yf
import os

tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "JPM", "GS", "BAC",
    "XOM", "CVX",
    "^GSPC", "^VIX"
]

os.makedirs("data", exist_ok=True)

for t in tickers:
    print(f"Downloading {t}...")
    df = yf.download(
        t,
        period="5y",
        interval="1d",
        auto_adjust=True,
        progress=False
    )
    df.to_csv(f"data/{t}_prices_5y_1d.csv")
