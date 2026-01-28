"""
Approximate EV computation (vs field only).

Streaming computation to avoid memory explosion with many candidates.

Supports three field modes:
- fixed: Use a single pre-generated field for all sims (original behavior)
- resample_per_sim: Sample a new field histogram each sim via multinomial
"""

import numpy as np
from typing import Tuple, Optional, List, Literal
import logging

from ..types import LineupArrays, ContestStructure, ShowdownPlayer, FieldGenConfig
from ..scoring.histogram import ArrayHistogram
from ..scoring.payout import PayoutLookup, score_lineups_vectorized

logger = logging.getLogger(__name__)

FieldMode = Literal["fixed", "resample_per_sim"]


def compute_lineup_probabilities(
    candidate_arrays: LineupArrays,
    cpt_players: List[ShowdownPlayer],
    flex_players: List[ShowdownPlayer],
    config: Optional[FieldGenConfig] = None
) -> np.ndarray:
    """
    Compute probability distribution over candidate lineups.

    Uses same ownership-weighted logic as field generation.
    p(lineup) ∝ product of ownership^(1/temperature) for all players.

    Args:
        candidate_arrays: All candidate lineups
        cpt_players: CPT player pool
        flex_players: FLEX player pool
        config: Field generation config (for temperature, position multipliers)

    Returns:
        p_lineup: [n_candidates] probability for each lineup, sums to 1.0
    """
    if config is None:
        config = FieldGenConfig()

    n_candidates = len(candidate_arrays)
    if n_candidates == 0:
        return np.array([], dtype=np.float64)

    # Pre-compute ownership weights (with temperature and position multipliers)
    cpt_weights = np.array([
        _get_adjusted_ownership(p, config) ** (1.0 / config.temperature)
        for p in cpt_players
    ], dtype=np.float64)

    flex_weights = np.array([
        _get_adjusted_ownership(p, config) ** (1.0 / config.temperature)
        for p in flex_players
    ], dtype=np.float64)

    # Compute lineup weights: product of player weights
    # log-sum is more stable for products
    cpt_log = np.log(cpt_weights + 1e-10)
    flex_log = np.log(flex_weights + 1e-10)

    # Lineup log-weight = cpt_log[cpt_idx] + sum(flex_log[flex_idxs])
    lineup_log_weights = (
        cpt_log[candidate_arrays.cpt_idx] +
        flex_log[candidate_arrays.flex_idx].sum(axis=1)
    )

    # Softmax to get probabilities
    lineup_log_weights -= lineup_log_weights.max()  # numerical stability
    weights = np.exp(lineup_log_weights)
    p_lineup = weights / weights.sum()

    return p_lineup


def _get_adjusted_ownership(player: ShowdownPlayer, config: FieldGenConfig) -> float:
    """Get ownership with position-based multipliers applied."""
    own = max(player.ownership, 0.01)

    if player.position == 'DST':
        own *= config.dst_rate_multiplier
    elif player.position == 'K':
        own *= config.kicker_rate_multiplier

    return own


def compute_approx_lineup_evs(
    candidate_arrays: LineupArrays,
    field_arrays: LineupArrays,
    field_counts: np.ndarray,
    outcomes: np.ndarray,
    contest: ContestStructure,
    score_bounds: Tuple[int, int],
    chunk_size: int = 10000
) -> np.ndarray:
    """
    Streaming EV computation (vs field only, overestimates).

    Computes approximate EV by comparing each candidate against the field,
    without considering self-competition among selected lineups.

    Memory: O(n_field + chunk_size) per sim, not O(n_cand × n_sims)

    Args:
        candidate_arrays: Candidate lineups to evaluate
        field_arrays: Field lineups
        field_counts: Count of each field lineup
        outcomes: [n_players, n_sims] simulated outcomes
        contest: Contest structure with payout info
        score_bounds: (min, max) score bounds
        chunk_size: Number of candidates to process per chunk

    Returns:
        approx_evs: [n_candidates] approximate EV for each candidate
    """
    n_candidates = len(candidate_arrays)
    n_sims = outcomes.shape[1]

    if n_candidates == 0:
        return np.array([], dtype=np.float64)

    payout_lookup = PayoutLookup.from_contest(contest)
    ev_sum = np.zeros(n_candidates, dtype=np.float64)

    for sim in range(n_sims):
        sim_outcomes = outcomes[:, sim]

        # Score field and build histogram
        field_scores = score_lineups_vectorized(field_arrays, sim_outcomes)
        histogram = ArrayHistogram.from_scores_and_counts(
            field_scores, field_counts, score_bounds
        )

        # Stream candidate chunks
        for chunk_start in range(0, n_candidates, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_candidates)

            # Score chunk
            chunk_cpt = candidate_arrays.cpt_idx[chunk_start:chunk_end]
            chunk_flex = candidate_arrays.flex_idx[chunk_start:chunk_end]

            cpt_scores = (sim_outcomes[chunk_cpt] * 15 + 5) // 10
            flex_scores = sim_outcomes[chunk_flex].sum(axis=1)
            chunk_scores = (cpt_scores + flex_scores).astype(np.int32)

            # Rank lookup (explicit out-of-range handling)
            ranks, tied_in_field = histogram.batch_get_rank_and_ties(chunk_scores)

            # +1 for the candidate itself being added
            n_tied = tied_in_field + 1

            # Payout
            payouts = payout_lookup.batch_get_payout(ranks, n_tied)
            ev_sum[chunk_start:chunk_end] += payouts

        if (sim + 1) % 10000 == 0:
            logger.info(f"Approx EV: {sim + 1}/{n_sims} sims")

    return ev_sum / n_sims


def compute_approx_ev_single_lineup(
    lineup_arrays: LineupArrays,
    lineup_idx: int,
    field_arrays: LineupArrays,
    field_counts: np.ndarray,
    outcomes: np.ndarray,
    contest: ContestStructure,
    score_bounds: Tuple[int, int]
) -> float:
    """
    Compute approx EV for a single lineup.

    Useful for quick evaluation of specific lineups.
    """
    n_sims = outcomes.shape[1]
    payout_lookup = PayoutLookup.from_contest(contest)

    payout_sum = 0.0

    cpt_idx = lineup_arrays.cpt_idx[lineup_idx]
    flex_idx = lineup_arrays.flex_idx[lineup_idx]

    for sim in range(n_sims):
        sim_outcomes = outcomes[:, sim]

        # Score field and build histogram
        field_scores = score_lineups_vectorized(field_arrays, sim_outcomes)
        histogram = ArrayHistogram.from_scores_and_counts(
            field_scores, field_counts, score_bounds
        )

        # Score this lineup
        cpt_score = (sim_outcomes[cpt_idx] * 15 + 5) // 10
        flex_score = sim_outcomes[flex_idx].sum()
        lineup_score = int(cpt_score + flex_score)

        # Rank lookup
        ranks, n_tied = histogram.batch_get_rank_and_ties(
            np.array([lineup_score], dtype=np.int32)
        )

        # Payout (+1 for self)
        payout = payout_lookup.get_payout(int(ranks[0]), int(n_tied[0]) + 1)
        payout_sum += payout

    return payout_sum / n_sims


def compute_approx_lineup_evs_resampled(
    candidate_arrays: LineupArrays,
    p_lineup: np.ndarray,
    outcomes: np.ndarray,
    contest: ContestStructure,
    score_bounds: Tuple[int, int],
    field_size: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Approximate EV with resampled field histogram per simulation.

    Instead of using a fixed pre-generated field, this samples a new field
    histogram each simulation using multinomial(F, q) where q is the
    score distribution induced by p_lineup.

    This is Option 2 ("new field per sim") without generating actual lineups.

    Key insight: For payout/rank math, we only need the field score histogram,
    not the identity of field lineups. Per sim:
    1. Score all candidates → get score per lineup
    2. bincount(scores, weights=p_lineup) → probability mass by score q
    3. multinomial(F, q) → sample field counts per score bin
    4. Vectorized rank/payout lookup

    Args:
        candidate_arrays: Candidate lineups to evaluate
        p_lineup: [n_candidates] probability for each lineup (sum=1)
        outcomes: [n_players, n_sims] simulated outcomes
        contest: Contest structure with payout info
        score_bounds: (min, max) score bounds
        field_size: Number of opponent entries (F)
        seed: Random seed for reproducibility

    Returns:
        approx_evs: [n_candidates] approximate EV for each candidate
    """
    n_candidates = len(candidate_arrays)
    n_sims = outcomes.shape[1]

    if n_candidates == 0:
        return np.array([], dtype=np.float64)

    rng = np.random.default_rng(seed)
    payout_lookup = PayoutLookup.from_contest(contest)
    ev_sum = np.zeros(n_candidates, dtype=np.float64)

    min_score, max_score = score_bounds
    n_bins = max_score - min_score + 1

    for sim in range(n_sims):
        sim_outcomes = outcomes[:, sim]

        # 1) Score ALL candidates for this sim (vectorized)
        cand_scores = score_lineups_vectorized(candidate_arrays, sim_outcomes)

        # 2) Build q(score) via weighted bincount (probability mass by score)
        # Shift scores into [0, K-1] bin indices
        bins = cand_scores - min_score

        # Handle out-of-bounds (shouldn't happen with correct bounds, but be safe)
        valid_mask = (bins >= 0) & (bins < n_bins)
        if not valid_mask.all():
            # Clamp out-of-range to boundaries
            bins = np.clip(bins, 0, n_bins - 1)

        # Weighted bincount: q[k] = sum of p_lineup for lineups with score k
        q = np.bincount(bins, weights=p_lineup, minlength=n_bins).astype(np.float64)

        # Normalize (should already sum to 1, but ensure for numerical stability)
        q_sum = q.sum()
        if q_sum > 0:
            q = q / q_sum
        else:
            # Fallback to uniform if all zeros (shouldn't happen)
            q = np.ones(n_bins, dtype=np.float64) / n_bins

        # 3) Sample a NEW field score histogram for this sim
        count_at = rng.multinomial(field_size, q).astype(np.int32)

        # 4) Build entries_above (reverse cumsum): entries scoring strictly higher
        # entries_above[k] = sum of count_at[k+1:]
        suffix_sum = np.cumsum(count_at[::-1])[::-1]
        entries_above = np.empty(n_bins, dtype=np.int32)
        entries_above[:-1] = suffix_sum[1:]
        entries_above[-1] = 0

        # 5) Rank/tie lookup for ALL candidates (vectorized)
        # rank = entries_above[bin] + 1 (1-indexed)
        # tied_in_field = count_at[bin]
        ranks = entries_above[bins] + 1
        tied_in_field = count_at[bins]

        # +1 for the candidate itself being added to the field
        n_tied = tied_in_field + 1

        # 6) Payout (vectorized via prefix sums)
        payouts = payout_lookup.batch_get_payout(ranks, n_tied)

        # 7) Accumulate EV
        ev_sum += payouts

        if (sim + 1) % 10000 == 0:
            logger.info(f"Approx EV (resampled): {sim + 1}/{n_sims} sims")

    return ev_sum / n_sims
