"""
Guaranteed score bounds computation.

Computes bounds that are guaranteed to contain all possible lineup scores
across all simulations.
"""

import numpy as np
from typing import Tuple


def compute_guaranteed_score_bounds(
    outcomes: np.ndarray,
    buffer_pct: float = 0.1
) -> Tuple[int, int]:
    """
    Compute bounds GUARANTEED to contain all possible lineup scores.

    Iterates ALL sims. Uses max + (sum - max) formula - no sort needed.

    For each simulation, the maximum possible lineup score is:
    - Take the top 6 players by score
    - Best player as CPT (1.5x) + other 5 as FLEX
    - max_cpt + sum(other 5)

    Args:
        outcomes: [n_players, n_sims] int32 quantized scores
        buffer_pct: Safety buffer percentage (default 10%)

    Returns:
        (min_score, max_score) tuple of quantized scores
    """
    n_players, n_sims = outcomes.shape

    if n_players == 0 or n_sims == 0:
        return (0, 0)

    global_max = 0

    for sim in range(n_sims):
        sim_outcomes = outcomes[:, sim]

        # Get top 6 players via partition (O(n_players))
        if n_players > 6:
            top_6_idx = np.argpartition(sim_outcomes, -6)[-6:]
            top_6 = sim_outcomes[top_6_idx]
        else:
            top_6 = sim_outcomes

        # SIMPLIFIED: max + (sum - max), no sort needed
        # The best lineup has the highest scorer as CPT
        cpt_base = int(top_6.max())
        flex_sum = int(top_6.sum()) - cpt_base

        # CPT score = base * 1.5 with rounding: (base * 15 + 5) // 10
        max_cpt = (cpt_base * 15 + 5) // 10
        max_score = max_cpt + flex_sum

        global_max = max(global_max, max_score)

    # Apply buffer
    max_with_buffer = int(global_max * (1 + buffer_pct))

    return (0, max_with_buffer)


def compute_bounds_vectorized(
    outcomes: np.ndarray,
    buffer_pct: float = 0.1
) -> Tuple[int, int]:
    """
    Vectorized version of bounds computation (faster for many sims).

    Note: Uses more memory but is faster for large n_sims.
    """
    n_players, n_sims = outcomes.shape

    if n_players == 0 or n_sims == 0:
        return (0, 0)

    # For each sim, get top 6 scores
    if n_players > 6:
        # Partition along player axis
        partitioned = np.partition(outcomes, -6, axis=0)[-6:, :]  # [6, n_sims]
    else:
        partitioned = outcomes

    # Max and sum for each sim
    cpt_base = partitioned.max(axis=0)  # [n_sims]
    total_sum = partitioned.sum(axis=0)  # [n_sims]
    flex_sum = total_sum - cpt_base

    # CPT score with rounding
    max_cpt = (cpt_base * 15 + 5) // 10
    max_scores = max_cpt + flex_sum

    global_max = int(max_scores.max())
    max_with_buffer = int(global_max * (1 + buffer_pct))

    return (0, max_with_buffer)
