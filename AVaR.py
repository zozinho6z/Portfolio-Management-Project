import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import yfinance as yf


def calculate_risks(returns, capital, alpha=0.05):
    """Calculates Parametric & Historical VaR and AVaR (Expected Shortfall)."""
    # 1. Historical Approach
    var_hist_pct = -float(np.percentile(returns, alpha * 100))
    worst_losses = returns[returns <= -var_hist_pct]
    avar_hist_pct = -float(worst_losses.mean())

    # 2. Parametric Approach (Gaussian)
    mu = float(returns.mean())
    sigma = float(returns.std())
    z_score = norm.ppf(1 - alpha)
    var_param_pct = -(mu - sigma * z_score)
    phi_z = norm.pdf(z_score)
    avar_param_pct = -mu + sigma * (phi_z / alpha)

    return {
        "mu": mu,
        "sigma": sigma,
        "var_hist": (var_hist_pct, capital * var_hist_pct),
        "avar_hist": (avar_hist_pct, capital * avar_hist_pct),
        "var_param": (var_param_pct, capital * var_param_pct),
        "avar_param": (avar_param_pct, capital * avar_param_pct)
    }


def display_plot(returns, var_hist_pct, avar_hist_pct, ticker):
    """Generates return distribution and tail risk visualization."""
    plt.figure(figsize=(10, 6))
    plt.hist(returns, bins=50, density=True, alpha=0.6, color="skyblue", edgecolor="black", label="Observed Returns")
    plt.axvline(-var_hist_pct, color="orange", linestyle="--", linewidth=2, label=f"Historical VaR 95% (-{var_hist_pct:.2%})")
    plt.axvline(-avar_hist_pct, color="red", linestyle="-", linewidth=2, label=f"Historical AVaR 95% (-{avar_hist_pct:.2%})")

    x_tail = np.linspace(returns.min(), -var_hist_pct, 100)
    plt.fill_between(x_tail, 0, 10, color="red", alpha=0.3, label="AVaR Tail Risk Zone (Extreme Losses)")

    plt.title(f"Return Distribution & Tail Risk Analytics — {ticker}", fontsize=12)
    plt.xlabel("Daily Return")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def analyze_risk():
    """Main execution function called by main.py."""
    print("=== MODULE 1: MARKET RISK ANALYTICS (VaR & AVaR) ===")
    capital = float(input("Enter your capital in € [50000]: ") or 50000)
    ticker = input("Stock ticker [AAPL]: ").strip().upper() or "AAPL"

    df = yf.download(ticker, start="2023-01-01", end="2026-01-01")
    data = df["Close"].squeeze()
    returns = data.pct_change().dropna()

    res = calculate_risks(returns, capital)

    print("\n" + "=" * 65)
    print(f" RISK ANALYSIS RESULTS — CAPITAL: {capital:,.2f} € ({ticker})")
    print("=" * 65)
    print(f"1. HISTORICAL   -> VaR (95%) : {res['var_hist'][0]:.2%} ({res['var_hist'][1]:,.2f} €) | AVaR (95%) : {res['avar_hist'][0]:.2%} ({res['avar_hist'][1]:,.2f} €)")
    print(f"2. PARAMETRIC -> VaR (95%) : {res['var_param'][0]:.2%} ({res['var_param'][1]:,.2f} €) | AVaR (95%) : {res['avar_param'][0]:.2%} ({res['avar_param'][1]:,.2f} €)")
    print("=" * 65)

    display_plot(returns, res["var_hist"][0], res["avar_hist"][0], ticker)


# Standalone script runner
if __name__ == "__main__":
    analyze_risk()