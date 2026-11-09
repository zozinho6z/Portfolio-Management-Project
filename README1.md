# Module 1: Market Risk Analytics (VaR & AVaR)

## Overview
This module provides a quantitative risk analysis framework for equity portfolios. It computes both Value at Risk (VaR) and Average Value at Risk (AVaR / Expected Shortfall) at a 95% confidence level (alpha = 0.05) using two complementary approaches:
1. Non-Parametric (Historical): Evaluates risk directly from empirical return distributions without assuming any underlying probabilistic model.
2. Parametric (Gaussian): Uses closed-form analytical solutions based on the normal distribution N(mu, sigma^2).

Additionally, it generates a visualization of the return distribution, highlighting the extreme loss tail zone.

---

## Mathematical Formulation

### 1. Historical Approach
* Historical VaR: Empirical quantile of daily returns:
  VaR_hist = - quantile_alpha(R)

* Historical AVaR: Truncated expectation tail losses (where R <= -VaR_hist):
  AVaR_hist = - E[ R | R <= -VaR_hist ]

### 2. Parametric Approach (Gaussian)
Assuming returns R follow N(mu, sigma^2), with z_(1-alpha) = Phi^(-1)(1 - alpha):
* Parametric VaR:
  VaR_param = - (mu - sigma * z_(1-alpha))

* Parametric AVaR: Closed-form solution using the standard normal density function phi(z):
  AVaR_param = - mu + sigma * (phi(z_(1-alpha)) / alpha)

---

## File Structure & Usage

### Execution
Run the script directly to analyze any ticker via Yahoo Finance:
```bash
python risk_analysis.py