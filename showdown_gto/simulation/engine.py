"""
Monte Carlo simulation engine for player outcomes.

Generates correlated player scores using copula models and percentile interpolation.
Supports Gaussian copula (default) and Student-t copula for tail dependence.
"""

import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from typing import List, Optional, Callable, Literal
import logging

from ..types import ShowdownPlayer, quantize_score

logger = logging.getLogger(__name__)


CopulaType = Literal["gaussian", "t"]


def simulate_outcomes(
    players: List[ShowdownPlayer],
    n_sims: int,
    correlation_matrix: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    copula_type: CopulaType = "gaussian",
    copula_df: int = 5
) -> np.ndarray:
    """
    Generate correlated player outcome simulations.

    Uses a copula to generate correlated uniform samples,
    then transforms via inverse CDF built from player percentiles.

    Args:
        players: List of players to simulate
        n_sims: Number of simulations
        correlation_matrix: Optional NxN correlation matrix. If None, uses identity (independent).
        seed: Random seed for reproducibility
        copula_type: "gaussian" (default) or "t" (Student-t for tail dependence)
        copula_df: Degrees of freedom for t-copula (lower = heavier tails). Default 5.

    Returns:
        outcomes: [n_players, n_sims] int32 quantized scores (points × 10)
    """
    if seed is not None:
        np.random.seed(seed)

    n_players = len(players)

    if n_players == 0:
        return np.zeros((0, n_sims), dtype=np.int32)

    # Build inverse CDF for each player
    inverse_cdfs = [_build_inverse_cdf(player) for player in players]

    # Generate correlated uniform samples via copula
    uniform_samples = _generate_correlated_uniforms(
        n_players, n_sims, correlation_matrix,
        copula_type=copula_type, copula_df=copula_df
    )

    # Transform to scores via inverse CDF
    outcomes = np.zeros((n_players, n_sims), dtype=np.float64)
    for i, inv_cdf in enumerate(inverse_cdfs):
        outcomes[i, :] = inv_cdf(uniform_samples[i, :])

    # Quantize to int32 (points × 10)
    quantized = np.round(outcomes * 10).astype(np.int32)

    # Ensure non-negative
    quantized = np.maximum(quantized, 0)

    return quantized


def _build_inverse_cdf(player: ShowdownPlayer) -> Callable[[np.ndarray], np.ndarray]:
    """
    Build inverse CDF function from player percentiles.

    Uses linear interpolation between percentile points.
    Extrapolates beyond p99 using the tail slope.
    """
    percentiles = player.percentiles

    # Get available percentile points
    available_pcts = sorted([p for p in percentiles.keys() if percentiles.get(p, 0) > 0])

    if not available_pcts:
        # No valid percentiles - return constant at projection or 0
        proj = player.projection if player.projection > 0 else 0.0
        return lambda u: np.full_like(u, proj)

    # Build percentile -> score mapping
    # Add 0th percentile (assume 0 or small value)
    pct_points = [0.0]
    score_points = [0.0]

    for pct in available_pcts:
        pct_points.append(pct / 100.0)
        score_points.append(percentiles[pct])

    # Add 100th percentile (extrapolate from p99)
    if 99 in available_pcts and 95 in available_pcts:
        # Extrapolate using p95-p99 slope
        p95_score = percentiles[95]
        p99_score = percentiles[99]
        slope = (p99_score - p95_score) / 0.04  # 4% range
        p100_score = p99_score + slope * 0.01
    elif available_pcts:
        # Just extend p99 by 20%
        p100_score = percentiles.get(99, score_points[-1]) * 1.2
    else:
        p100_score = 0.0

    pct_points.append(1.0)
    score_points.append(max(p100_score, score_points[-1]))

    # Build interpolator
    # Use bounds_error=False and fill_value for edge cases
    inv_cdf = interp1d(
        pct_points,
        score_points,
        kind='linear',
        bounds_error=False,
        fill_value=(score_points[0], score_points[-1])
    )

    return inv_cdf


def _generate_correlated_uniforms(
    n_players: int,
    n_sims: int,
    correlation_matrix: Optional[np.ndarray] = None,
    copula_type: CopulaType = "gaussian",
    copula_df: int = 5
) -> np.ndarray:
    """
    Generate correlated uniform samples using the specified copula.

    Args:
        n_players: Number of players
        n_sims: Number of simulations
        correlation_matrix: NxN correlation matrix (defaults to identity)
        copula_type: "gaussian" or "t"
        copula_df: Degrees of freedom for t-copula

    Returns:
        [n_players, n_sims] uniform samples in (0, 1)
    """
    if correlation_matrix is None:
        # Independent samples
        return np.random.uniform(0, 1, (n_players, n_sims))

    # Ensure correlation matrix is valid
    corr = _ensure_valid_correlation_matrix(correlation_matrix)

    if copula_type == "t":
        return _generate_correlated_uniforms_t(n_players, n_sims, corr, copula_df)

    # Gaussian copula (default)
    return _generate_correlated_uniforms_gaussian(n_players, n_sims, corr)


def _generate_correlated_uniforms_gaussian(
    n_players: int,
    n_sims: int,
    corr: np.ndarray
) -> np.ndarray:
    """Generate correlated uniforms using Gaussian copula."""
    mean = np.zeros(n_players)
    z = np.random.multivariate_normal(mean, corr, size=n_sims).T  # [n_players, n_sims]

    # Transform to uniform via CDF
    u = stats.norm.cdf(z)

    # Clip to avoid exact 0 or 1 (causes issues with inverse CDF)
    u = np.clip(u, 1e-10, 1 - 1e-10)

    return u


def _generate_correlated_uniforms_t(
    n_players: int,
    n_sims: int,
    corr: np.ndarray,
    df: int
) -> np.ndarray:
    """
    Generate correlated uniforms using Student-t copula.

    The t-copula captures tail dependence: when one player booms,
    correlated players are more likely to also boom (and vice versa for busts).
    Lower df = heavier tails = more extreme joint outcomes.

    Sampling: t = z / sqrt(s / df), where z ~ N(0, corr), s ~ chi2(df)
    Then u = t_cdf(t, df) to get uniforms.
    """
    mean = np.zeros(n_players)

    # Draw correlated normals
    z = np.random.multivariate_normal(mean, corr, size=n_sims).T  # [n_players, n_sims]

    # Draw chi-squared for the t-distribution scaling
    # Shared across all players in a given sim (this is what creates tail dependence)
    s = np.random.chisquare(df, size=n_sims)  # [n_sims]

    # Scale: t = z / sqrt(s / df)
    scaling = np.sqrt(s / df)  # [n_sims]
    t_samples = z / scaling[np.newaxis, :]  # [n_players, n_sims]

    # Transform to uniform via t-distribution CDF
    u = stats.t.cdf(t_samples, df=df)

    # Clip to avoid exact 0 or 1
    u = np.clip(u, 1e-10, 1 - 1e-10)

    return u


def _ensure_valid_correlation_matrix(corr: np.ndarray) -> np.ndarray:
    """
    Ensure correlation matrix is valid (positive semi-definite).

    If not, apply nearest correlation matrix correction.
    """
    n = corr.shape[0]

    # Check if symmetric
    if not np.allclose(corr, corr.T):
        corr = (corr + corr.T) / 2

    # Check diagonal is 1
    np.fill_diagonal(corr, 1.0)

    # Check positive semi-definite
    eigenvalues = np.linalg.eigvalsh(corr)
    if np.min(eigenvalues) < -1e-10:
        # Apply Higham's nearest correlation matrix algorithm (simplified)
        logger.warning("Correlation matrix not PSD, applying correction")
        corr = _nearest_correlation_matrix(corr)

    return corr


def _nearest_correlation_matrix(corr: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """
    Find nearest valid correlation matrix (simplified Higham algorithm).
    """
    n = corr.shape[0]
    Y = corr.copy()

    for _ in range(max_iter):
        # Project onto positive semi-definite cone
        eigenvalues, eigenvectors = np.linalg.eigh(Y)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        Y = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        # Project onto unit diagonal
        np.fill_diagonal(Y, 1.0)

        # Check convergence
        if np.allclose(Y, corr, atol=1e-6):
            break

    return Y


def simulate_outcomes_independent(
    players: List[ShowdownPlayer],
    n_sims: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Simplified simulation without correlations (for testing).
    """
    return simulate_outcomes(players, n_sims, correlation_matrix=None, seed=seed)
