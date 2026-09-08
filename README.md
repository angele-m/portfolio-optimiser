# Quantitative Portfolio Optimiser

A Python tool that finds the allocation of money across a set of stocks that
maximises risk-adjusted return (the Sharpe Ratio), using historical price
data and Modern Portfolio Theory.

## What it does

Given historical prices for a handful of stocks, this project:

1. Computes each stock's expected return, risk (volatility), and how they
   move together (covariance)
2. Computes a naive equal-weight baseline for comparison
3. Searches thousands of random portfolio allocations via Monte Carlo
   simulation, scoring each by its Sharpe Ratio
4. Runs an exact numerical optimiser (`scipy.optimize`, SLSQP) to find the
   true maximum-Sharpe allocation, rather than just the best of N random guesses
5. Supports an optional cap on how much any single stock can hold, to avoid
   unrealistic, over-concentrated allocations
6. Plots the "efficient frontier" — every portfolio tried, colour-coded by
   Sharpe Ratio, with the best one highlighted

## Project structure

```
portfolio_optimiser/
├── data/                  # historical price CSVs, one per ticker
├── output/                # generated efficient frontier chart
├── src/
│   ├── download_data.py   # fetches live historical data via yfinance
│   ├── data_loader.py     # loads & aligns price CSVs (or generates synthetic demo data)
│   ├── stats_calc.py      # daily returns, mean, std dev, covariance
│   ├── optimizer.py       # Monte Carlo search + exact SLSQP optimiser
│   ├── visualize.py       # plots the efficient frontier
│   └── main.py             # orchestrates everything, CLI entry point
├── .gitignore
├── requirements.txt
└── README.md
```

## The maths

| Term | Meaning |
|---|---|
| **Daily return** | `(P_t - P_{t-1}) / P_{t-1}` — the % price change from one day to the next |
| **Expected return** | Average daily return, annualised (× 252 trading days/year) |
| **Risk (standard deviation)** | How much daily returns vary from their average, annualised (× √252) |
| **Covariance matrix** | How each pair of stocks' returns move together — the reason diversification reduces risk |
| **Portfolio weights** | Fraction of total money in each stock; must sum to 1 (100%) |
| **Sharpe Ratio** | `(portfolio return − risk-free rate) / portfolio risk` — return earned per unit of risk. Higher is better |

Portfolio return is the weighted average of each stock's expected return.
Portfolio risk is **not** just a weighted average of individual risks — it
depends on the covariance matrix, which is why combining stocks that don't
move in lockstep reduces overall risk without giving up return.

## Setup

```bash
pip install -r requirements.txt
```

## Step 1 — Get stock data

**Option A — fetch live data (recommended)**

```bash
cd src
python download_data.py
```

This uses [`yfinance`](https://pypi.org/project/yfinance/) to pull historical
prices directly and save one CSV per ticker into `data/`. Edit the `tickers`
list and date range inside `download_data.py` to customise.

**Option B — download manually from Yahoo Finance**

Visit a ticker's history page (e.g. `finance.yahoo.com/quote/AAPL/history`),
set a date range, and download the CSV into `data/`.

## Step 2 — Run the optimiser

```bash
python main.py --data-dir ../data
```

Optional flags:

```bash
python main.py --data-dir ../data --num-portfolios 20000 --risk-free-rate 0.03 --max-weight 0.4
```

- `--num-portfolios` — how many random portfolios to try (default 10,000)
- `--risk-free-rate` — used in the Sharpe Ratio calc (default 2%)
- `--max-weight` — caps any single stock's weight, e.g. `0.4` for max 40%
  (prevents the optimiser from piling everything into one or two stocks)

If `--data-dir` is omitted or empty, the script falls back to synthetic
demo data so the pipeline can be tested without any real data.

## Output

The script prints:
- Each stock's annualised expected return
- The best portfolio found by Monte Carlo search (max Sharpe)
- The lowest-risk portfolio found
- A naive equal-weight baseline, for comparison
- The exact optimiser's result (true max Sharpe, via SLSQP)

...and saves an efficient frontier chart to `output/efficient_frontier.png`.

## Example results

Run on 5 years of real data (AAPL, AMZN, GOOGL, MSFT, NVDA; 2019–2024):

| Portfolio | Sharpe Ratio | Expected Return | Risk |
|---|---|---|---|
| Naive baseline (equal weight) | 1.204 | 37.99% | 29.90% |
| Monte Carlo best (10,000 tries) | 1.424 | 59.46% | 40.34% |
| **Exact optimiser (SLSQP)** | **1.445** | 60.31% | 40.35% |

The optimised allocation improved the Sharpe Ratio by roughly **20% over naive
equal-weighting**, mainly by shifting weight toward the stocks with the
strongest risk-adjusted returns over that window (AAPL and NVDA). Worth
noting: this came from taking on *more* risk, not less — the gain is in
efficiency (return per unit of risk), not safety. The unconstrained optimiser
also put 0% into 3 of the 5 stocks, which is why the `--max-weight` flag
exists — to keep the result realistically diversified.

## A real bug I hit, and what it taught me

Yahoo Finance's CSV export format isn't fixed — I originally wrote the data
loader assuming the classic `Date, Open, High, Low, Close, Volume` layout,
which is what most documentation and guides describe. When I downloaded real
data directly from Yahoo Finance, the actual export used a different
structure: the date column was labelled `Price` instead of `Date`, and there
were extra junk rows under the header before the real data started.

The fix was to make the loader defensive rather than assuming one exact
format: it accepts either `Date` or `Price` as the date column, coerces
values to real dates, and drops any row that fails to parse. Switching to
fetching data via `yfinance` (`download_data.py`) sidesteps this problem
going forward, since it returns structured data directly rather than
depending on a webpage's CSV export format.

## Limitations

- Assumes historical returns/volatility/correlations predict the future,
  which real markets don't guarantee — correlations especially tend to
  spike during crises
- Assumes roughly normal return distributions; real markets have fatter
  tails (more extreme moves than a normal distribution predicts)
- No transaction costs, taxes, or liquidity constraints
- Highly sensitive to the chosen date range — a different historical window
  can produce a very different "optimal" allocation
- The unconstrained optimiser can concentrate heavily into 1–2 stocks;
  use `--max-weight` to enforce diversification

## Possible extensions

- Trace the full efficient frontier curve (min risk at each target return
  level) instead of just the single max-Sharpe point
- Allow short-selling (negative weights) and compare results
- Add sector-level constraints (e.g. "no more than 50% in tech")
- Sweep the risk-free rate and see how the optimal allocation shifts
