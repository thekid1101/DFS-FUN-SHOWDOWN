"""
True portfolio EV computation with self-competition.

Your entries compete with each other, not just the field.

Supports both fixed field and resampled field modes.
"""

import numpy as np
from typing import Tuple, Dict, Optional
import logging

from ..types import LineupArrays, ContestStructure
from ..scoring.histogram import ArrayHistogram, add_entries_to_histogram
from ..scoring.payout import PayoutLookup, score_lineups_vectorized

logger = logging.getLogger(__name__)


def compute_true_portfolio_ev(
    selected_arrays: LineupArrays,
    field_arrays: LineupArrays,
    field_counts: np.ndarray,
    outcomes: np.ndarray,
    contest: ContestStructure,
    score_bounds: Tuple[int, int]
) -> Tuple[float, Dict]:
    """
    True portfolio EV with self-competition.

    Your entries compete with each other, not just the field.
    Includes invariant check: combined total = field + selected.

    Args:
        selected_arrays: Your selected lineups
        field_arrays: Opponent field lineups
        field_counts: Count of each field lineup
        outcomes: [n_players, n_sims] simulated outcomes
        contest: Contest structure
        score_bounds: (min, max) score bounds

    Returns:
        portfolio_ev: Total expected payout for portfolio
        diagnostics: Dict with detailed metrics
    """
    n_selected = len(selected_arrays)
    n_sims = outcomes.shape[1]
    field_size = int(field_counts.sum())

    if n_selected == 0:
        return 0.0, {'error': 'No lineups selected'}

    payout_lookup = PayoutLookup.from_contest(contest)

    total_payout_sum = 0.0
    per_lineup_payout_sum = np.zeros(n_selected, dtype=np.float64)
    cash_count = np.zeros(n_selected, dtype=np.int32)
    profit_count = 0
    entry_cost = contest.entry_fee * n_selected

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

        # INVARIANT CHECK: combined total must equal field + selected
        expected_total = field_size + n_selected
        if combined_histogram.total_entries != expected_total:
            raise AssertionError(
                f"Histogram total {combined_histogram.total_entries} != "
                f"expected {expected_total} (field={field_size}, selected={n_selected})"
            )

        # Get payouts from combined histogram
        ranks, n_tied = combined_histogram.batch_get_rank_and_ties(selected_scores)
        payouts = payout_lookup.batch_get_payout(ranks, n_tied)

        # Accumulate
        per_lineup_payout_sum += payouts
        sim_total = payouts.sum()
        total_payout_sum += sim_total
        cash_count += (payouts > 0).astype(np.int32)

        if sim_total > entry_cost:
            profit_count += 1

        if (sim + 1) % 10000 == 0:
            logger.info(f"True EV: {sim + 1}/{n_sims} sims")

    portfolio_ev = total_payout_sum / n_sims
    per_lineup_evs = per_lineup_payout_sum / n_sims

    diagnostics = {
        'true_portfolio_ev': float(portfolio_ev),
        'entry_cost': float(entry_cost),
        'expected_profit': float(portfolio_ev - entry_cost),
        'roi_pct': float((portfolio_ev - entry_cost) / entry_cost * 100) if entry_cost > 0 else 0.0,
        'p_profit': float(profit_count / n_sims),
        'per_lineup_evs': per_lineup_evs.tolist(),
        'per_lineup_cash_probs': (cash_count / n_sims).tolist(),
        'expected_cashes': float(cash_count.sum() / n_sims),
        'field_size': field_size,
        'n_selected': n_selected,
        'n_sims': n_sims,
    }

    return portfolio_ev, diagnostics


def compute_marginal_ev(
    existing_selected: LineupArrays,
    new_lineup_arrays: LineupArrays,
    new_lineup_idx: int,
    field_arrays: LineupArrays,
    field_counts: np.ndarray,
    outcomes: np.ndarray,
    contest: ContestStructure,
    score_bounds: Tuple[int, int]
) -> float:
    """
    Compute marginal EV of adding one lineup to existing selection.

    Marginal EV = EV(existing + new) - EV(existing)

    Useful for greedy portfolio construction.
    """
    # Get combined selection
    n_existing = len(existing_selected)

    if n_existing == 0:
        # First lineup - just compute its EV vs field
        single_arrays = LineupArrays(
            cpt_idx=new_lineup_arrays.cpt_idx[new_lineup_idx:new_lineup_idx + 1],
            flex_idx=new_lineup_arrays.flex_idx[new_lineup_idx:new_lineup_idx + 1],
            salary=new_lineup_arrays.salary[new_lineup_idx:new_lineup_idx + 1]
        )
        ev, _ = compute_true_portfolio_ev(
            single_arrays, field_arrays, field_counts,
            outcomes, contest, score_bounds
        )
        return ev - contest.entry_fee

    # Compute EV with new lineup added
    combined_cpt = np.concatenate([
        existing_selected.cpt_idx,
        new_lineup_arrays.cpt_idx[new_lineup_idx:new_lineup_idx + 1]
    ])
    combined_flex = np.concatenate([
        existing_selected.flex_idx,
        new_lineup_arrays.flex_idx[new_lineup_idx:new_lineup_idx + 1]
    ])
    combined_salary = np.concatenate([
        existing_selected.salary,
        new_lineup_arrays.salary[new_lineup_idx:new_lineup_idx + 1]
    ])

    combined_arrays = LineupArrays(
        cpt_idx=combined_cpt,
        flex_idx=combined_flex,
        salary=combined_salary
    )

    # EV of combined
    ev_combined, _ = compute_true_portfolio_ev(
        combined_arrays, field_arrays, field_counts,
        outcomes, contest, score_bounds
    )

    # EV of existing
    ev_existing, _ = compute_true_portfolio_ev(
        existing_selected, field_arrays, field_counts,
        outcomes, contest, score_bounds
    )

    # Marginal = combined - existing - entry_fee
    return (ev_combined - ev_existing) - contest.entry_fee


def compute_true_portfolio_ev_resampled(
    selected_arrays: LineupArrays,
    p_lineup: np.ndarray,
    candidate_arrays: LineupArrays,
    outcomes: np.ndarray,
    contest: ContestStructure,
    score_bounds: Tuple[int, int],
    field_size: int,
    seed: Optional[int] = None
) -> Tuple[float, Dict]:
    """
    True portfolio EV with self-competition and resampled field.

    Each sim samples a new field histogram via multinomial(F, q) where q is
    the score distribution induced by p_lineup. Then adds selected lineups
    to compute self-competition.

    Args:
        selected_arrays: Your selected lineups
        p_lineup: [n_candidates] probability for each lineup (sum=1)
        candidate_arrays: All candidate lineups (for computing score distribution)
        outcomes: [n_players, n_sims] simulated outcomes
        contest: Contest structure
        score_bounds: (min, max) score bounds
        field_size: Number of opponent entries (F)
        seed: Random seed for reproducibility

    Returns:
        portfolio_ev: Total expected payout for portfolio
        diagnostics: Dict with detailed metrics
    """
    n_selected = len(selected_arrays)
    n_sims = outcomes.shape[1]

    if n_selected == 0:
        return 0.0, {'error': 'No lineups selected'}

    rng = np.random.default_rng(seed)
    payout_lookup = PayoutLookup.from_contest(contest)

    min_score, max_score = score_bounds
    n_bins = max_score - min_score + 1

    total_payout_sum = 0.0
    per_lineup_payout_sum = np.zeros(n_selected, dtype=np.float64)
    cash_count = np.zeros(n_selected, dtype=np.int32)
    profit_count = 0
    entry_cost = contest.entry_fee * n_selected

    for sim in range(n_sims):
        sim_outcomes = outcomes[:, sim]

        # 1) Score ALL candidates to get score distribution
        cand_scores = score_lineups_vectorized(candidate_arrays, sim_outcomes)

        # 2) Build q(score) via weighted bincount
        bins = cand_scores - min_score
        bins = np.clip(bins, 0, n_bins - 1)  # Safety clamp
        q = np.bincount(bins, weights=p_lineup, minlength=n_bins).astype(np.float64)
        q_sum = q.sum()
        if q_sum > 0:
            q = q / q_sum
        else:
            q = np.ones(n_bins, dtype=np.float64) / n_bins

        # 3) Sample field histogram
        field_count_at = rng.multinomial(field_size, q).astype(np.int32)

        # 4) Score selected lineups
        selected_scores = score_lineups_vectorized(selected_arrays, sim_outcomes)
        selected_bins = selected_scores - min_score
        selected_bins = np.clip(selected_bins, 0, n_bins - 1)

        # 5) Add selected to histogram (self-competition)
        combined_count_at = field_count_at.copy()
        np.add.at(combined_count_at, selected_bins, 1)

        # 6) Build entries_above for combined histogram
        suffix_sum = np.cumsum(combined_count_at[::-1])[::-1]
        entries_above = np.empty(n_bins, dtype=np.int32)
        entries_above[:-1] = suffix_sum[1:]
        entries_above[-1] = 0

        # 7) Rank/tie lookup for selected lineups
        ranks = entries_above[selected_bins] + 1
        n_tied = combined_count_at[selected_bins]

        # 8) Payouts
        payouts = payout_lookup.batch_get_payout(ranks, n_tied)

        # Accumulate
        per_lineup_payout_sum += payouts
        sim_total = payouts.sum()
        total_payout_sum += sim_total
        cash_count += (payouts > 0).astype(np.int32)

        if sim_total > entry_cost:
            profit_count += 1

        if (sim + 1) % 10000 == 0:
            logger.info(f"True EV (resampled): {sim + 1}/{n_sims} sims")

    portfolio_ev = total_payout_sum / n_sims
    per_lineup_evs = per_lineup_payout_sum / n_sims

    diagnostics = {
        'true_portfolio_ev': float(portfolio_ev),
        'entry_cost': float(entry_cost),
        'expected_profit': float(portfolio_ev - entry_cost),
        'roi_pct': float((portfolio_ev - entry_cost) / entry_cost * 100) if entry_cost > 0 else 0.0,
        'p_profit': float(profit_count / n_sims),
        'per_lineup_evs': per_lineup_evs.tolist(),
        'per_lineup_cash_probs': (cash_count / n_sims).tolist(),
        'expected_cashes': float(cash_count.sum() / n_sims),
        'field_size': field_size,
        'n_selected': n_selected,
        'n_sims': n_sims,
        'field_mode': 'resample_per_sim',
    }

    return portfolio_ev, diagnostics
