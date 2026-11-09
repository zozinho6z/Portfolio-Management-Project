from fractions import Fraction
import numpy as np

# ==============================================================================
# 1. CORE QUANTITATIVE ENGINE (Unchanged)
# ==============================================================================
def solve_merton_discrete(capital, r, P, S0, S_T):
    """
    Solves Merton's discrete-time portfolio optimization problem for U(x) = ln(x).
    
    Parameters:
        capital (float) : Initial capital K invested at t=0.
        r (float)       : Risk-free interest rate.
        P (np.array)    : Physical probability vector (size K).
        S0 (np.array)   : Asset price vector at t=0 (size N+1).
        S_T (np.array)  : Terminal asset price matrix (shape: K states x N+1 assets).
        
    Returns:
        dict: Optimization output containing Q, xi, X_hat, theta_cash, and H.
    """
    # 1. Equivalent Martingale Measure (Q) under No-Arbitrage (NAFL)
    S_T_discounted = S_T / (1.0 + r)
    Q = np.linalg.solve(S_T_discounted.T, S0)

    # 2. Radon-Nikodym Density: xi = dQ / dP
    xi = Q / P

    # 3. Optimal Terminal Wealth: X_hat for logarithmic utility U(x) = ln(x)
    X_hat = capital * (1.0 + r) / xi

    # 4. Replicating Portfolio Allocation: theta = (theta_cash, H)
    n_risky = len(S0) - 1
    delta_S = S_T_discounted[:, 1:] - S0[1:]
    rhs = (X_hat / (1.0 + r)) - capital

    # Solve linear system over the first N states (complete market)
    H = np.linalg.solve(delta_S[:n_risky, :], rhs[:n_risky])

    # Residual capital allocated to Cash
    theta_cash = capital - np.sum(H * S0[1:])

    return {
        "Q": Q,
        "xi": xi,
        "X_hat": X_hat,
        "theta_cash": theta_cash,
        "H": H
    }


# ==============================================================================
# 2. INPUT PARSERS (Supports Decimals & Fractions)
# ==============================================================================
def parse_value(expr_str):
    """Safely converts a string (e.g., '0.05', '1/8', '42/31') into a float."""
    try:
        return float(Fraction(expr_str.strip()))
    except (ValueError, ZeroDivisionError):
        return None


def prompt_float(prompt_text, default_val):
    """Prompts for a numeric value with a fallback default."""
    raw = input(f"{prompt_text} [{default_val}]: ").strip()
    if not raw:
        return float(default_val)
    val = parse_value(raw)
    return val if (val is not None and val >= 0) else float(default_val)


def prompt_vector(prompt_text, default_arr):
    """Prompts for a space-separated vector of floats/fractions."""
    default_str = " ".join([str(x) for x in default_arr])
    raw = input(f"{prompt_text} [{default_str}]: ").strip()
    if not raw:
        return np.array(default_arr)
    
    tokens = raw.split()
    parsed = [parse_value(t) for t in tokens]
    if len(parsed) != len(default_arr) or any(v is None for v in parsed):
        print(f"   [Warning] Invalid input. Using default: {default_str}")
        return np.array(default_arr)
    return np.array(parsed)


# ==============================================================================
# 3. INTERACTIVE RUNNER & USER INTERFACE
# ==============================================================================
def run_merton_solver():
    """Interactive execution function called by main.py."""
    print("=== MODULE 2: MERTON PORTFOLIO OPTIMIZATION (DISCRETE MARKET) ===\n")
    
    # 1. User inputs for Capital and Risk-Free Rate
    capital = prompt_float("Enter initial capital K (€)", 10000.0)
    r = prompt_float("Enter risk-free rate r (e.g., 0.05)", 0.05)
    
    print("\n--- Market Data Configuration (Press Enter for Defaults) ---")
    
    # 2. Vector of Physical Probabilities P
    default_P = [1/8, 3/8, 4/8]
    P = prompt_vector("Enter probabilities P (e.g., 1/8 3/8 4/8)", default_P)
    
    # Check probability sum
    if not np.isclose(np.sum(P), 1.0, atol=1e-3):
        print(f"   [Warning] Sum of P is {np.sum(P):.4f} != 1.0. Normalizing probabilities...")
        P = P / np.sum(P)

    # 3. Initial Asset Prices S0
    default_S0 = [1.0, 1.0, 1.0]
    S0 = prompt_vector("Enter initial prices S0 (Cash, Asset1, Asset2)", default_S0)

    # 4. Terminal Matrix Prices S_T per state
    print("Enter terminal prices S_T per state (Cash_T, Asset1_T, Asset2_T):")
    default_ST_w1 = [1.0 + r, 42/31, 21/124]
    default_ST_w2 = [1.0 + r, 21/31, 42/31]
    default_ST_w3 = [1.0 + r, 21/62, 168/31]

    st_w1 = prompt_vector("  • State w1 prices", default_ST_w1)
    st_w2 = prompt_vector("  • State w2 prices", default_ST_w2)
    st_w3 = prompt_vector("  • State w3 prices", default_ST_w3)
    
    S_T = np.array([st_w1, st_w2, st_w3])

    # 5. Run solver engine
    try:
        res = solve_merton_discrete(capital, r, P, S0, S_T)
    except np.linalg.LinAlgError:
        print("\n[Error] System matrix is singular/non-invertible. Check market completeness.")
        return

    # 6. Format and display results
    print("\n" + "="*65)
    print(f" MERTON OPTIMIZATION RESULTS (INVESTED CAPITAL: {capital:,.2f} €)")
    print("="*65)
    
    print("1. PROBABILISTIC MEASURES & DENSITIES:")
    for i in range(len(P)):
        print(f"   • State ω{i+1}: P = {P[i]:.4f} | Q = {res['Q'][i]:.4f} | ξ (dQ/dP) = {res['xi'][i]:.4f}")

    print("-" * 65)
    print("2. OPTIMAL TERMINAL WEALTH (X_hat):")
    for i in range(len(P)):
        print(f"   • Target wealth in state ω{i+1}: {res['X_hat'][i]:,.2f} €")

    print("-" * 65)
    print("3. OPTIMAL PORTFOLIO ALLOCATION AT t=0:")
    print(f"   • Risk-Free Asset (Cash) : {res['theta_cash']:,.2f} € ({res['theta_cash']/capital:.2%})")
    for idx, qty in enumerate(res['H'], start=1):
        value = qty * S0[idx]
        print(f"   • Risky Asset {idx} (H{idx})     : {qty:.4f} shares => {value:,.2f} € ({value/capital:.2%})")
    print("="*65)


if __name__ == "__main__":
    run_merton_solver()