import numpy as np
from scipy.stats import norm

S0, K, R, SIGMA, T = 100.0, 105.0, 0.05, 0.2, 1.0
N_PRICE = 3
WIDTH = 3.0


def lognormal_params():
    return np.log(S0) + (R - 0.5 * SIGMA**2) * T, SIGMA * np.sqrt(T)


def discretize(n=N_PRICE, width=WIDTH):
    # 2^n equal-width bins in S_T over mu +/- width*s (log space); bin mass from the log-normal CDF
    mu, s = lognormal_params()
    edges = np.linspace(np.exp(mu - width * s), np.exp(mu + width * s), 2**n + 1)
    p = np.diff(norm.cdf((np.log(edges) - mu) / s))
    return 0.5 * (edges[:-1] + edges[1:]), p / p.sum()


def payoff(x):
    return np.maximum(x - K, 0.0)


def discount():
    return np.exp(-R * T)


def black_scholes():
    d1 = (np.log(S0 / K) + (R + 0.5 * SIGMA**2) * T) / (SIGMA * np.sqrt(T))
    d2 = d1 - SIGMA * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * discount() * norm.cdf(d2)


def amplitude(x, p):
    f = payoff(x)
    return float(p @ f / f.max()), float(f.max())


def to_price(a, fmax):
    return discount() * fmax * np.asarray(a)


def monte_carlo(x, p, n_samples, n_trials, rng):
    counts = rng.multinomial(n_samples, p, size=n_trials)
    return discount() * counts @ payoff(x) / n_samples
