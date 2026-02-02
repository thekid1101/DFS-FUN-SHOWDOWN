"""
Tournament-aligned metrics for GPP portfolio evaluation.

Computes top-heavy metrics that matter for tournament DFS:
- Top 1% rate, ceiling EV, win rate, composite score.

Reuses the same histogram scoring pattern as compute_true_portfolio_ev()
but tracks per-sim best_rank and total_payout for richer output.
"""

import numpy as np
from typing import Tuple, Dict
import logging

from ..types import LineupArrays, ContestStructure, TournamentMetrics
from ..scoring.histogram import ArrayHistogram, add_entries_to_histogram
from ..scoring.payout import PayoutLookup, score_lineups_vectorized

logger = logging.getLogger(__name__)

# Composite score weights
W_TOP_1PCT = 0.50
W_CEILING = 0.35
W_WIN_RATE = 0.10
W_ROI = 0.05


def compute_tournament_metrics(
    selected_arrays: LineupArrays,
    field_arrays: LineupArrays,
    field_counts: np.ndarray,
    outcomes: np.ndarray,
    contest: ContestStructure,
    score_bounds: Tuple[int, int],
    max_sims: int = 10000,
) -> TournamentMetrics:
    """
    Compute tournament-aligned metrics from a scored portfolio.

    Mirrors the scoring loop in compute_true_portfolio_ev() but additionally
    tracks per-sim best_rank (min rank across portfolio) and per-sim
    total_payout for computing tournament-specific metrics.

    Args:
        selected_arrays: Your selected lineups
        field_arrays: Opponent field lineups
        field_counts: Count of each field lineup
        outcomes: [n_players, n_sims] simulated outcomes
        contest: Contest structure
        score_bounds: (min, max) score bounds
        max_sims: Cap on simulations to use (default 10000)

    Returns:
        TournamentMetrics with top_1pct_rate, ceiling_ev, win_rate, roi_pct,
        composite_score
    """
    n_selected = len(selected_arrays)
    n_sims = min(outcomes.shape[1], max_sims)
    field_size = int(field_counts.sum())

    if n_selected == 0:
        return TournamentMetrics(
            top_1pct_rate=0.0, ceiling_ev=0.0, win_rate=0.0,
            roi_pct=0.0, composite_score=0.0, n_sims=0
        )

    payout_lookup = PayoutLookup.from_contest(contest)
    entry_cost = contest.entry_fee * n_selected

    best_ranks = np.empty(n_sims, dtype=np.int32)
    sim_payouts = np.empty(n_sims, dtype=np.float64)

    for sim in range(n_sims):
        sim_outcomes = outcomes[:, sim]

        # Build field histogram
        field_scores = score_lineups_vectorized(field_arrays, sim_outcomes)
        field_histogram = ArrayHistogram.from_scores_and_counts(
            field_scores, field_counts, score_bounds
        )

        # Add selected lineups to histogram
        selected_scores = score_lineups_vectorized(selected_arrays, sim_outcomes)
        selected_counts = np.ones(n_selected, dtype=np.int32)
        combined_histogram = add_entries_to_histogram(
            field_histogram, selected_scores, selected_counts
        )

        # Get ranks and payouts
        ranks, n_tied = combined_histogram.batch_get_rank_and_ties(selected_scores)
        payouts = payout_lookup.batch_get_payout(ranks, n_tied)

        best_ranks[sim] = ranks.min()
        sim_payouts[sim] = payouts.sum()

        if (sim + 1) % 10000 == 0:
            logger.info(f"Tournament metrics: {sim + 1}/{n_sims} sims")

    # Compute metrics
    total_entries = contest.total_entries
    top_1pct_threshold = max(1, int(total_entries * 0.01))

    top_1pct_mask = best_ranks <= top_1pct_threshold
    top_1pct_rate = float(top_1pct_mask.mean())

    ceiling_ev = float(sim_payouts[top_1pct_mask].mean()) if top_1pct_mask.any() else 0.0

    win_rate = float((best_ranks == 1).mean())

    mean_payout = float(sim_payouts.mean())
    roi_pct = float((mean_payout - entry_cost) / entry_cost * 100) if entry_cost > 0 else 0.0

    # Composite score: weighted combination of normalized metrics
    # Normalize ceiling_ev to per-dollar basis for comparability
    ceiling_ev_per_dollar = ceiling_ev / entry_cost if entry_cost > 0 else 0.0

    composite_score = (
        W_TOP_1PCT * top_1pct_rate
        + W_CEILING * min(ceiling_ev_per_dollar / 100, 1.0)  # Cap at 100x return
        + W_WIN_RATE * win_rate * 100  # Scale win_rate up (typically very small)
        + W_ROI * max(min(roi_pct / 100, 1.0), 0.0)  # Normalize ROI to 0-1 range
    )

    return TournamentMetrics(
        top_1pct_rate=top_1pct_rate,
        ceiling_ev=ceiling_ev,
        win_rate=win_rate,
        roi_pct=roi_pct,
        composite_score=composite_score,
        n_sims=n_sims,
    )
