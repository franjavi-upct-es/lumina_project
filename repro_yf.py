import yfinance as yf
import pandas as pd
from datetime import date

ticker = "AAPL"
start = date(2024, 5, 20)
end = date(2024, 5, 21)

raw = yf.download(
    ticker,
    start=start,
    end=end,
    interval="1d",
    auto_adjust=False,
    progress=False,
    threads=False,
)
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)

print("Columns:", raw.columns)
print("Index name:", raw.index.name)
print("Head:\n", raw.head())

df = raw.reset_index()
print("Columns after reset_index:", df.columns)
