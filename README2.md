---

### 2. `README.md` pour le Module 2 (`module2_merton/README.md`)

```markdown
# Module 2: Discrete-Time Merton Portfolio Optimization

## Overview
This module implements the discrete-time Merton portfolio selection model under logarithmic utility U(x) = ln(x) in a single-period complete market Omega = {w_1, ..., w_K}.

The solver determines:
1. The unique Equivalent Martingale Measure (Q) ensuring No Arbitrage (NAFL).
2. The Radon-Nikodym Density (xi = dQ / dP) defining state prices.
3. The Optimal Terminal Wealth (X_hat) maximizing expected utility under the budget constraint.
4. The Replicating Portfolio Allocation (theta_0, H) at t = 0 across risk-free and risky assets.

---

## Mathematical Workflow

1. Risk-Neutral Measure (Q):
   Solve the linear system for vector Q = [q_1, ..., q_K]^T:
   (S_bar_T)^T * Q = S_0
   where S_bar_T = S_T / (1 + r)

2. Radon-Nikodym Derivative (xi):
   xi(w_i) = q_i / p_i

3. Optimal Terminal Wealth (X_hat):
   Derived from first-order conditions U'(x) = 1/x and budget constraint E_Q[ X_hat / (1 + r) ] = K:
   X_hat(w_i) = K * (1 + r) / xi(w_i)

4. Replicating Portfolio (H & theta_0):
   Solve delta_S_bar * H = (X_hat / (1 + r)) - K over linearly independent states to determine risky share holdings H = [H_1, ..., H_N]^T, then allocate residual capital to Cash:
   theta_0 = K - sum( H_j * S_0^j )

---

## Features & Usage

* Robust Parsing: Supports fraction strings ('1/8', '42/31', '21/124') and decimal inputs natively.
* Fallback Defaults: Includes benchmark 3-state / 2-risky-asset market configuration from coursework.

### Execution
Run directly to execute the interactive solver:
```bash
python merton_discrete.py