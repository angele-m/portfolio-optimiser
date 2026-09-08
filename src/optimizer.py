"""
Step 6 & 7: Generate random portfolios, evaluate them, keep the best one.
Step 8: a proper numerical optimiser (SLSQP) that finds the exact
max-Sharpe portfolio instead of just the best of N random guesses, plus
optional per-stock weight constraints.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize 

from stats_calc import portfolio_performance, sharpe_ratio


def monte_carlo_simulation(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    num_portfolios: int = 10000,
    risk_free_rate: float = 0.02,
    seed: int = 7,
):
    """
    Generates `num_portfolios` random weight combinations (each summing to 1),
    computes return/risk/Sharpe for each, and returns:
        results: ndarray shape (3, num_portfolios) -> [returns, risks, sharpes]
        weights_record: list of weight arrays, same order as results columns
    """
    rng = np.random.default_rng(seed)
    n_assets = len(mean_returns)
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = rng.random(n_assets)
        weights /= np.sum(weights)  # ensure weights sum to 1
        weights_record.append(weights)

        port_return, port_risk = portfolio_performance(weights, mean_returns, cov_matrix)
        results[0, i] = port_return
        results[1, i] = port_risk
        results[2, i] = sharpe_ratio(port_return, port_risk, risk_free_rate)

    return results, weights_record


def best_portfolio_from_results(results, weights_record, tickers):
    """Pick out the max-Sharpe and min-risk portfolios from Monte Carlo results."""
    max_sharpe_idx = np.argmax(results[2])
    min_risk_idx = np.argmin(results[1])

    def _package(idx):
        return {
            "weights": pd.Series(weights_record[idx], index=tickers),
            "return": results[0, idx],
            "risk": results[1, idx],
            "sharpe": results[2, idx],
        }

    return _package(max_sharpe_idx), _package(min_risk_idx)



# Step 8 improvement: exact optimisation instead of random search


def optimise_max_sharpe(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.02,
    bounds=(0.0, 1.0),
    max_weight_per_asset: float = 1.0,
):
    """
    Uses SLSQP to find the weight vector that truly maximises the Sharpe Ratio,
    subject to: weights sum to 1, and each weight in [bounds[0], min(bounds[1], max_weight_per_asset)].
    This removes the randomness/approximation error of the Monte Carlo approach.
    """
    n_assets = len(mean_returns)

    def neg_sharpe(weights):
        port_return, port_risk = portfolio_performance(weights, mean_returns, cov_matrix)
        return -sharpe_ratio(port_return, port_risk, risk_free_rate)

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    asset_bounds = tuple((bounds[0], min(bounds[1], max_weight_per_asset)) for _ in range(n_assets))
    init_guess = np.repeat(1 / n_assets, n_assets)

    result = minimize(
        neg_sharpe,
        init_guess,
        method="SLSQP",
        bounds=asset_bounds,
        constraints=constraints,
    )

    weights = pd.Series(result.x, index=mean_returns.index)
    port_return, port_risk = portfolio_performance(result.x, mean_returns, cov_matrix)
    return {
        "weights": weights,
        "return": port_return,
        "risk": port_risk,
        "sharpe": sharpe_ratio(port_return, port_risk, risk_free_rate),
        "success": result.success,
    }
