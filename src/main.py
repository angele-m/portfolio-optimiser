"""
Quantitative portfolio optimiser's main entry point.


Put one CSV per ticker in the data directory (e.g. AAPL.csv, MSFT.csv),
downloaded from Yahoo Finance's historical data page. See data_loader.py
for the expected format.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_prices_from_csvs, generate_synthetic_data
from stats_calc import calculate_daily_returns, annualised_mean_returns, annualised_cov_matrix
from optimizer import monte_carlo_simulation, best_portfolio_from_results, optimise_max_sharpe
from visualize import plot_efficient_frontier


def format_weights(weights, prices_cols):
    return "\n".join(f"    {t:<8s} {w:6.2%}" for t, w in weights.items())


def main():
    parser = argparse.ArgumentParser(description="Quantitative Portfolio Optimiser")
    parser.add_argument("--data-dir", default=None, help="Folder containing per-ticker CSVs")
    parser.add_argument("--num-portfolios", type=int, default=10000)
    parser.add_argument("--risk-free-rate", type=float, default=0.02)
    parser.add_argument("--out", default="../output/efficient_frontier.png")
    args = parser.parse_args()

    # Step 2/3: load data 
    if args.data_dir and os.path.isdir(args.data_dir) and os.listdir(args.data_dir):
        print(f"Loading real price data from '{args.data_dir}' ...")
        prices = load_prices_from_csvs(args.data_dir)
    else:
        print("No data directory given (or it's empty) — using synthetic demo data.")
        print("To use real data: download CSVs from Yahoo Finance and pass --data-dir\n")
        prices = generate_synthetic_data()

    tickers = list(prices.columns)
    print(f"Tickers: {tickers}")
    print(f"Date range: {prices.index.min().date()} to {prices.index.max().date()} "
          f"({len(prices)} trading days)\n")

    # Step 4: daily returns 
    returns = calculate_daily_returns(prices)

    #  Step 5: statistics 
    mean_returns = annualised_mean_returns(returns)
    cov_matrix = annualised_cov_matrix(returns)

    print("Annualised expected return per stock:")
    for t in tickers:
        print(f"    {t:<8s} {mean_returns[t]:6.2%}")
    print()

    #  Step 6 & 7: Monte Carlo search for best Sharpe 
    print(f"Running Monte Carlo simulation with {args.num_portfolios} random portfolios...")
    results, weights_record = monte_carlo_simulation(
        mean_returns, cov_matrix, num_portfolios=args.num_portfolios,
        risk_free_rate=args.risk_free_rate,
    )
    max_sharpe_pf, min_risk_pf = best_portfolio_from_results(results, weights_record, tickers)

    print("\n=== BEST PORTFOLIO FOUND (Monte Carlo, Max Sharpe) ===")
    print(format_weights(max_sharpe_pf["weights"], tickers))
    print(f"    Expected Return : {max_sharpe_pf['return']:.2%}")
    print(f"    Risk (Std Dev)  : {max_sharpe_pf['risk']:.2%}")
    print(f"    Sharpe Ratio    : {max_sharpe_pf['sharpe']:.3f}")

    print("\n=== LOWEST-RISK PORTFOLIO FOUND (Monte Carlo) ===")
    print(format_weights(min_risk_pf["weights"], tickers))
    print(f"    Expected Return : {min_risk_pf['return']:.2%}")
    print(f"    Risk (Std Dev)  : {min_risk_pf['risk']:.2%}")
    print(f"    Sharpe Ratio    : {min_risk_pf['sharpe']:.3f}")

    #  Step 8: exact optimiser (better than random search) 
    exact = optimise_max_sharpe(mean_returns, cov_matrix, risk_free_rate=args.risk_free_rate)
    print("\n=== EXACT OPTIMISER RESULT (SLSQP, true max Sharpe) ===")
    print(format_weights(exact["weights"], tickers))
    print(f"    Expected Return : {exact['return']:.2%}")
    print(f"    Risk (Std Dev)  : {exact['risk']:.2%}")
    print(f"    Sharpe Ratio    : {exact['sharpe']:.3f}")
    print(f"    Converged       : {exact['success']}")

    # Plot 
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plot_efficient_frontier(results, max_sharpe_pf, min_risk_pf, args.out)
    print(f"\nEfficient frontier chart saved to: {args.out}")


if __name__ == "__main__":
    main()
