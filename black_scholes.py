import numpy as np
from scipy.stats import norm
from math import log, sqrt, exp
from scipy.optimize import brentq

def _d1d2(S, K, r, q, sigma, T):
    """Helper returning d1, d2. q is continuous carry/dividend (use funding rate)."""
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be > 0")
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def bs_price(S, K, r, q, sigma, T, option_type='c'):
    """Black-Scholes price with continuous carry q (q=0 for no carry).
       option_type: 'c' for call, 'p' for put.
    """
    d1, d2 = _d1d2(S, K, r, q, sigma, T)
    if option_type == 'c':
        return S * np.exp(-q*T) * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    elif option_type == 'p':
        return K * np.exp(-r*T) * norm.cdf(-d2) - S * np.exp(-q*T) * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'c' or 'p'")

def bs_greeks(S, K, r, q, sigma, T, option_type='c'):
    """Return Greeks: delta, gamma, theta (per day), vega (per 1% vol), rho (per 1% rate)."""
    d1, d2 = _d1d2(S, K, r, q, sigma, T)
    pdf_d1 = norm.pdf(d1)
    # common terms
    discount_S = np.exp(-q*T)
    discount_K = np.exp(-r*T)

    # Delta
    if option_type == 'c':
        delta = discount_S * norm.cdf(d1)
    else:
        delta = discount_S * (norm.cdf(d1) - 1)

    # Gamma (same for call/put)
    gamma = discount_S * pdf_d1 / (S * sigma * np.sqrt(T))

    # Vega (return per 1 vol point as percentage, e.g. per 1.0 = 100 vol points)
    vega = S * discount_S * pdf_d1 * np.sqrt(T)    # per 1 vol (i.e., 1.0 = 100 vol points)
    vega_per_1pct = vega * 0.01                     # per 1% change in sigma

    # Theta (per year -> convert to per day)
    if option_type == 'c':
        theta = (-S * discount_S * pdf_d1 * sigma / (2 * np.sqrt(T))
                 - r * K * discount_K * norm.cdf(d2)
                 + q * S * discount_S * norm.cdf(d1))
    else:
        theta = (-S * discount_S * pdf_d1 * sigma / (2 * np.sqrt(T))
                 + r * K * discount_K * norm.cdf(-d2)
                 - q * S * discount_S * norm.cdf(-d1))
    theta_per_day = theta / 365.0

    # Rho (sensitivity to r), per 1% (multiply by 0.01)
    if option_type == 'c':
        rho = K * T * discount_K * norm.cdf(d2)
    else:
        rho = -K * T * discount_K * norm.cdf(-d2)
    rho_per_1pct = rho * 0.01

    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega_per_1pct,         # per 1% vol
        'theta': theta_per_day,       # per calendar day
        'rho': rho_per_1pct           # per 1% rate
    }

def implied_vol(market_price, S, K, r, q, T, option_type='c', sigma_bounds=(1e-6, 5.0), tol=1e-8):
    """Solve for implied vol by Brent root-finding on bs_price - market_price = 0."""
    def objective(sigma):
        return bs_price(S, K, r, q, sigma, T, option_type) - market_price

    a, b = sigma_bounds
    fa, fb = objective(a), objective(b)
    if fa * fb > 0:
        # Try to expand bounds a bit if no sign change
        raise ValueError("Implied vol not bracketed. Try wider bounds or check market price.")
    return brentq(objective, a, b, xtol=tol)
# Example BTC option
S = 91771      # BTC spot price
K = 95000  # Strike price
r = 0.0371         # Risk-free rate
q = 0.0          # Carry/funding rate
T = 118/365       # Time to expiry in years
sigma = 0.6      # Volatility
option_type = 'c'  # 'c' for call, 'p' for put

# Compute option price
price = bs_price(S, K, r, q, sigma, T, option_type)
print("Option Price:", price)

# Compute Greeks
greeks = bs_greeks(S, K, r, q, sigma, T, option_type)
print("Greeks:", greeks)
