import yfinance as yf

tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]

for ticker in tickers:
    data = yf.download(ticker, start="2022-07-11", end="2026-07-11")
    data.to_csv(f"{ticker}.csv")
    print(f"Saved {ticker}.csv")