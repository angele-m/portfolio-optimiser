"""Plots the Monte Carlo cloud of portfolios (the 'efficient frontier' picture)."""

import matplotlib
matplotlib.use("Agg")  # no display needed, just save to file
import matplotlib.pyplot as plt


def plot_efficient_frontier(results, max_sharpe_pf, min_risk_pf, save_path: str):
    fig, ax = plt.subplots(figsize=(10, 7))

    scatter = ax.scatter(
        results[1], results[0], c=results[2], cmap="viridis", s=8, alpha=0.6
    )
    fig.colorbar(scatter, label="Sharpe Ratio")

    ax.scatter(
        max_sharpe_pf["risk"], max_sharpe_pf["return"],
        marker="*", color="red", s=400, label="Max Sharpe Ratio", edgecolors="black",
    )
    ax.scatter(
        min_risk_pf["risk"], min_risk_pf["return"],
        marker="*", color="blue", s=400, label="Min Risk", edgecolors="black",
    )

    ax.set_xlabel("Risk (Annualised Standard Deviation)")
    ax.set_ylabel("Expected Annual Return")
    ax.set_title("Portfolio Optimisation — Monte Carlo Simulation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
