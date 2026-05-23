import yfinance as yf
from datetime import date

tickers = ["^GSPC", "^DJI", "AAPL", "MSFT"]
for t in tickers:
    print(f"Checking {t}...")
    df = yf.download(t, start="1920-01-01", end="1930-01-01", progress=False)
    if not df.empty:
        print(f"  {t} has data from {df.index[0].date()}")
    else:
        print(f"  {t} has no data in 1920s")
