"""
Step 4 & 5 — Daily returns and portfolio statistics.

Key maths recap:
    daily return_t   = (P_t - P_{t-1}) / P_{t-1}
    expected return   = mean of daily returns, annualised (x 252 trading days)
    risk (std dev)     = standard deviation of returns, annualised (x sqrt(252))
    covariance matrix  = how returns of each pair of stocks move together,
                          annualised the same way as variance
"""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def calculate_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple percentage returns: (P_t - P_{t-1}) / P_{t-1}"""
    return prices.pct_change().dropna()


def annualised_mean_returns(returns: pd.DataFrame) -> pd.Series:
    return returns.mean() * TRADING_DAYS_PER_YEAR


def annualised_cov_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.cov() * TRADING_DAYS_PER_YEAR


def portfolio_performance(weights: np.ndarray, mean_returns: pd.Series, cov_matrix: pd.DataFrame):
    """
    Given a weight vector, return (expected_return, risk, sharpe_ratio).
    Risk = portfolio standard deviation = sqrt(w^T . Cov . w)
    """
    port_return = np.dot(weights, mean_returns)
    port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    port_risk = np.sqrt(port_variance)
    return port_return, port_risk


def sharpe_ratio(port_return: float, port_risk: float, risk_free_rate: float = 0.02) -> float:
    """Sharpe Ratio = (return - risk_free_rate) / risk"""
    if port_risk == 0:
        return 0.0
    return (port_return - risk_free_rate) / port_risk
