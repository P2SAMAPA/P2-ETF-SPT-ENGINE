"""spt_strategies.py — Three SPT sub-portfolio strategies.

All three exploit the SPT Master Equation:
    g_p = g_market + gamma_p

where gamma_p (excess growth rate) is a free lunch from rebalancing.

Strategies
----------
1. DiversityWeighted  : w_i ∝ mu_i^p, p=0.5  (Fernholz diversity portfolio)
2. MaxEntropy         : maximise -Σ w_i log(w_i)  (entropy-weighted)
3. VolatilityHarvest  : maximise gamma_p = ½ Σ w_i (σ_i² - σ_p²)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import config


def _clip_normalise(w: np.ndarray) -> np.ndarray:
    w = np.clip(w, config.MIN_WEIGHT, config.MAX_WEIGHT)
    return w / w.sum()


# ── 1. Diversity-Weighted ─────────────────────────────────────────────────────


def diversity_weights(
    market_weights: np.ndarray,
    p: float = config.DIVERSITY_P,
) -> np.ndarray:
    """Fernholz diversity portfolio: w_i ∝ mu_i^p.

    For p < 1 this overweights smaller assets, generating a diversity premium.
    market_weights: capitalisation-proxy weights (equal for ETF universe).
    """
    w = market_weights**p
    return _clip_normalise(w)


# ── 2. Maximum Entropy ────────────────────────────────────────────────────────


def max_entropy_weights(n: int) -> np.ndarray:
    """Maximise portfolio entropy H = -Σ w_i log(w_i).

    Closed-form solution subject to min/max bounds is uniform within bounds,
    then renormalised. Entropy maximisation is equivalent to minimising
    -H subject to Σw=1, w∈[min,max].
    """
    w0 = np.ones(n) / n

    def _neg_entropy(w: np.ndarray) -> float:
        w_safe = np.clip(w, 1e-10, 1.0)
        return float(np.sum(w_safe * np.log(w_safe)))

    def _neg_entropy_grad(w: np.ndarray) -> np.ndarray:
        w_safe = np.clip(w, 1e-10, 1.0)
        return np.log(w_safe) + 1.0

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(config.MIN_WEIGHT, config.MAX_WEIGHT)] * n

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize(
            _neg_entropy,
            w0,
            jac=_neg_entropy_grad,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 300},
        )
    return _clip_normalise(result.x if result.success else w0)


# ── 3. Volatility Harvest ─────────────────────────────────────────────────────


def volatility_harvest_weights(variances: np.ndarray) -> np.ndarray:
    """Maximise excess growth rate gamma_p = ½ Σ w_i (σ_i² - σ_p²).

    Equivalently: maximise Σ w_i σ_i² - (Σ w_i σ_i)²
    which overweights high-variance assets to harvest the variance drag.

    variances: annualised per-asset variances (diagonal of cov matrix).
    """
    n = len(variances)
    w0 = np.ones(n) / n

    def _neg_gamma(w: np.ndarray) -> float:
        port_var = float(w @ variances)  # approx (ignores covariances for tractability)
        return -0.5 * float(w @ variances - port_var)

    def _gamma_exact(w: np.ndarray) -> float:
        # Exact: gamma = 0.5 * (Σ w_i σ_i² - Σ_ij w_i w_j σ_ij)
        # Using diagonal approx: gamma ≈ 0.5 * Σ w_i σ_i² * (1 - w_i)
        return -0.5 * float(np.sum(w * variances * (1.0 - w)))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(config.MIN_WEIGHT, config.MAX_WEIGHT)] * n

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize(
            _gamma_exact,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 300},
        )
    return _clip_normalise(result.x if result.success else w0)


# ── Excess Growth Rate (diagnostic) ──────────────────────────────────────────


def excess_growth_rate(
    weights: np.ndarray,
    cov: np.ndarray,
) -> float:
    """gamma_p = ½ (Σ w_i σ_i² - σ_p²) — the SPT 'free lunch'."""
    variances = np.diag(cov)
    weighted_var = float(weights @ variances)
    port_var = float(weights @ cov @ weights)
    return 0.5 * (weighted_var - port_var)


# ── Dynamic Sortino-based ensemble blender ────────────────────────────────────


def compute_sortino(returns: pd.Series, rf: float = 0.0) -> float:
    """Annualised Sortino ratio of a return series."""
    excess = returns - rf / 252.0
    mean_excess = float(excess.mean()) * 252
    downside = excess[excess < 0]
    ds_std = float(downside.std()) * np.sqrt(252) if len(downside) > 1 else 1e-8
    return mean_excess / (ds_std + 1e-8)


def blend_weights(
    w_div: np.ndarray,
    w_ent: np.ndarray,
    w_vol: np.ndarray,
    sortino_div: float,
    sortino_ent: float,
    sortino_vol: float,
) -> np.ndarray:
    """Blend three strategy weights by their trailing Sortino ratios.

    Negative Sortino strategies get zero allocation in the blend.
    Falls back to equal weight if all Sortinos are non-positive.
    """
    scores = np.array(
        [
            max(sortino_div, 0.0),
            max(sortino_ent, 0.0),
            max(sortino_vol, 0.0),
        ]
    )
    total = scores.sum()
    if total < 1e-8:
        # All strategies performing poorly — equal blend
        scores = np.ones(3) / 3.0
    else:
        scores /= total

    blended = scores[0] * w_div + scores[1] * w_ent + scores[2] * w_vol
    return _clip_normalise(blended)
