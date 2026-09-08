"""
Step 2 & 3 — Get and load stock data.

Real usage:
    1. Fetch data directly from Yahoo finance CSV to get most accurate data

If no CSVs are found, this module falls back to generating realistic synthetic
price data so the rest of the pipeline can be tested immediately.
"""

import os
import glob
import numpy as np
import pandas as pd


def load_prices_from_csvs(data_dir: str) -> pd.DataFrame:
    
    '''Read every CSV in data_dir, extract the Date/Close columns, and merge
    them into one DataFrame of closing prices (columns = tickers, index = Date).
    Dates that don't exist for all tickers are dropped (inner join) so the
    matrix is clean and rectangular which is required for covariance calculations.
    '''
    csv_paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in '{data_dir}'.")

    series_list = []
    for path in csv_paths:
        ticker = os.path.splitext(os.path.basename(path))[0].upper()
        df = pd.read_csv(path)


        date_col = next(
            (c for c in df.columns if c.lower() in ("date", "price")), None
        )
        close_col = next(
            (c for c in df.columns if c.lower() in ("close", "adj close")),
            None,
        )
        if date_col is None:
            raise ValueError(f"No 'Date' column found in {path}")
        if close_col is None:
            raise ValueError(f"No 'Close' column found in {path}")

        '''Yahoo's newer export also inserts junk rows under the header
        (e.g. a 'Ticker' row and a 'Date' label row). Coerce to datetime
         and drop any row that isn't a real date.'''
        
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", format="mixed")
        df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
        df = df.dropna(subset=[date_col, close_col])

        s = df.set_index(date_col)[close_col].rename(ticker)
        series_list.append(s)

    prices = pd.concat(series_list, axis=1, join="inner")  # align dates
    prices = prices.sort_index()

    if prices.isnull().values.any():
        prices = prices.dropna()

    return prices


def generate_synthetic_data(
    tickers=("AAPL", "MSFT", "AMZN", "NVDA", "GOOG"),
    days: int = 756,          # ~3 trading years
    seed: int = 42,
) -> pd.DataFrame:
    
    '''Generates plausible daily closing prices via geometric Brownian motion, with a rough correlation structure so the demo Sharpe ratios look sane. 
    This is only used as test values before we can use the real values'''
    
    rng = np.random.default_rng(seed)
    n = len(tickers)

    # Rough annualised drift/vol per ticker (illustrative, not real)
    annual_drift = rng.uniform(0.08, 0.25, n)
    annual_vol = rng.uniform(0.20, 0.45, n)

    # Build a correlated random shock matrix
    base_corr = 0.35  # baseline correlation between tech stocks
    corr = np.full((n, n), base_corr)
    np.fill_diagonal(corr, 1.0)
    chol = np.linalg.cholesky(corr)

    dt = 1 / 252
    daily_drift = annual_drift * dt
    daily_vol = annual_vol * np.sqrt(dt)

    z = rng.standard_normal((days, n))
    correlated_z = z @ chol.T

    log_returns = daily_drift + daily_vol * correlated_z
    log_prices = np.cumsum(log_returns, axis=0)
    prices = 100 * np.exp(log_prices)  # start every stock at $100

    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=days + 5)[-len(prices):]
    return pd.DataFrame(prices, index=dates, columns=list(tickers))