"""
Enhanced portfolio diagnostics.

Full self-competition decomposition, near-duplicate detection,
profit covariance metrics, ownership leverage, tail decomposition,
and game-state coverage reporting.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

from .types import LineupArrays, ContestStructure, ShowdownPlayer, TournamentMetrics
from .scoring.payout import PayoutLookup, score_lineups_vectorized
from .metrics.tournament import compute_tournament_metrics
from .ev.game_states import (
    classify_sims_by_game_state,
    compute_game_state_coverage,
    GAME_STATES,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Self-Competition Decomposition
# =============================================================================

def decompose_self_competition(
    selected_arrays: LineupArrays,
    field_arrays: LineupArrays,
    field_counts: np.ndarray,
    outcomes: np.ndarray,
    contest: ContestStructure,
    score_bounds: Tuple[int, int],
    max_sims: int = 5000,
) -> Dict:
    """
    Decompose self-competition cost into exact ties, near ties, and displacement.

    For each sim:
    1. Score all selected lineups
    2. For each selected lineup:
       a. Compute payout WITH all other selected in histogram
       b. Compute payout WITHOUT other selected (field-only)
       c. Difference = total self-comp for lineup in this sim
    3. Exact ties: same bin as another selected lineup
    4. Near ties: within +/- 2 bins of another selected
    5. Remainder: general rank displacement

    Args:
        selected_arrays: Your selected lineups
        field_arrays: Opponent field lineups
        field_counts: Count of each field lineup
        outcomes: [n_players, n_sims] simulated outcomes
        contest: Contest structure
        score_bounds: (min, max) score bounds
        max_sims: Cap on sims for this intensive computation

    Returns:
        Dict with exact_tie_cost, near_tie_cost, displacement_cost (all in dollars)
    """
    min_score, max_score = score_bounds
    n_bins = max_score - min_score + 1
    n_selected = len(selected_arrays)
    n_sims = min(outcomes.shape[1], max_sims)

    payout_lookup = PayoutLookup.from_contest(contest)

    exact_tie_cost_sum = 0.0
    near_tie_cost_sum = 0.0
    displacement_cost_sum = 0.0
    total_self_comp_sum = 0.0

    for sim in range(n_sims):
        sim_outcomes = outcomes[:, sim]

        # Score field
        field_scores = score_lineups_vectorized(field_arrays, sim_outcomes)
        f_bins = np.clip(field_scores - min_score, 0, n_bins - 1)
        field_hist = np.bincount(
            f_bins, weights=field_counts.astype(np.float64), minlength=n_bins
        ).astype(np.int32)

        # Score selected
        selected_scores = score_lineups_vectorized(selected_arrays, sim_outcomes)
        selected_bins = np.clip(selected_scores - min_score, 0, n_bins - 1)

        # Build combined histogram
        combined_hist = field_hist.copy()
        np.add.at(combined_hist, selected_bins, 1)

        for i in range(n_selected):
            my_bin = selected_bins[i]

            # Payout WITH self-competition (combined histogram)
            suffix_sum = np.cumsum(combined_hist[::-1])[::-1]
            entries_above = np.zeros(n_bins, dtype=np.int32)
            entries_above[:-1] = suffix_sum[1:]
            rank_combined = entries_above[my_bin] + 1
            tied_combined = combined_hist[my_bin]
            payout_combined = payout_lookup.get_payout(rank_combined, tied_combined)

            # Payout WITHOUT self-competition (field-only + just this lineup)
            field_only_plus_me = field_hist.copy()
            field_only_plus_me[my_bin] += 1
            suffix_sum_fo = np.cumsum(field_only_plus_me[::-1])[::-1]
            entries_above_fo = np.zeros(n_bins, dtype=np.int32)
            entries_above_fo[:-1] = suffix_sum_fo[1:]
            rank_alone = entries_above_fo[my_bin] + 1
            tied_alone = field_only_plus_me[my_bin]
            payout_alone = payout_lookup.get_payout(rank_alone, tied_alone)

            self_comp_cost = payout_alone - payout_combined
            total_self_comp_sum += self_comp_cost

            # Classify the cost
            # Count other selected lineups at same bin (exact ties)
            other_at_same = sum(1 for j in range(n_selected)
                                if j != i and selected_bins[j] == my_bin)

            # Count other selected lineups within +/- 2 bins (near ties)
            other_near = sum(1 for j in range(n_selected)
                             if j != i and abs(int(selected_bins[j]) - int(my_bin)) <= 2
                             and selected_bins[j] != my_bin)

            if other_at_same > 0:
                # Attribute proportionally
                exact_frac = other_at_same / max(other_at_same + other_near, 1)
                exact_tie_cost_sum += self_comp_cost * exact_frac
                near_tie_cost_sum += self_comp_cost * (1 - exact_frac) * (other_near > 0)
                displacement_cost_sum += self_comp_cost * (1 - exact_frac) * (other_near == 0)
            elif other_near > 0:
                near_tie_cost_sum += self_comp_cost
            else:
                displacement_cost_sum += self_comp_cost

    return {
        'total_self_comp': float(total_self_comp_sum / n_sims),
        'exact_tie_cost': float(exact_tie_cost_sum / n_sims),
        'near_tie_cost': float(near_tie_cost_sum / n_sims),
        'displacement_cost': float(displacement_cost_sum / n_sims),
        'n_sims_used': n_sims,
    }


# =============================================================================
# Near-Duplicate Detection
# =============================================================================

def detect_near_duplicates(
    candidate_arrays: LineupArrays,
    selected_indices: List[int],
    threshold: int = 4
) -> Dict:
    """
    Flag lineup pairs sharing >= threshold of 6 roster slots.

    CPT match + FLEX overlap count.
    Near-duplicates have high profit correlation and are self-comp contributors.

    Args:
        candidate_arrays: All candidate lineups
        selected_indices: Indices of selected lineups
        threshold: Minimum shared slots to flag (default: 4 of 6)

    Returns:
        Dict with pair counts by overlap level and flagged pairs
    """
    n = len(selected_indices)
    overlap_counts = {6: 0, 5: 0, 4: 0}
    flagged_pairs = []

    for i in range(n):
        idx_i = selected_indices[i]
        cpt_i = int(candidate_arrays.cpt_idx[idx_i])
        flex_i = set(candidate_arrays.flex_idx[idx_i].tolist())

        for j in range(i + 1, n):
            idx_j = selected_indices[j]
            cpt_j = int(candidate_arrays.cpt_idx[idx_j])
            flex_j = set(candidate_arrays.flex_idx[idx_j].tolist())

            overlap = 0
            if cpt_i == cpt_j:
                overlap += 1
            overlap += len(flex_i & flex_j)

            if overlap >= threshold:
                if overlap in overlap_counts:
                    overlap_counts[overlap] += 1
                flagged_pairs.append((i, j, overlap))

    return {
        'pairs_6_shared': overlap_counts.get(6, 0),
        'pairs_5_shared': overlap_counts.get(5, 0),
        'pairs_4_shared': overlap_counts.get(4, 0),
        'total_flagged': len(flagged_pairs),
        'flagged_pairs': flagged_pairs[:20],  # Cap output
    }


# =============================================================================
# Profit Covariance Metrics
# =============================================================================

def compute_profit_covariance_metrics(
    selected_arrays: LineupArrays,
    field_arrays: LineupArrays,
    field_counts: np.ndarray,
    outcomes: np.ndarray,
    contest: ContestStructure,
    score_bounds: Tuple[int, int],
    max_sims: int = 5000,
) -> Dict:
    """
    Compute pairwise profit correlation metrics for the portfolio.

    Args:
        selected_arrays: Selected lineups
        field_arrays: Field lineups
        field_counts: Field counts
        outcomes: [n_players, n_sims] outcomes
        contest: Contest structure
        score_bounds: Score bounds
        max_sims: Max sims for computation

    Returns:
        Dict with mean_correlation, max_correlation, pct_above_0.8
    """
    min_score, max_score = score_bounds
    n_bins = max_score - min_score + 1
    n_selected = len(selected_arrays)
    n_sims = min(outcomes.shape[1], max_sims)

    payout_lookup = PayoutLookup.from_contest(contest)

    # Collect payout vectors for each selected lineup
    payout_matrix = np.zeros((n_selected, n_sims), dtype=np.float64)

    for sim in range(n_sims):
        sim_outcomes = outcomes[:, sim]

        field_scores = score_lineups_vectorized(field_arrays, sim_outcomes)
        f_bins = np.clip(field_scores - min_score, 0, n_bins - 1)
        field_hist = np.bincount(
            f_bins, weights=field_counts.astype(np.float64), minlength=n_bins
        ).astype(np.int32)

        selected_scores = score_lineups_vectorized(selected_arrays, sim_outcomes)
        selected_bins = np.clip(selected_scores - min_score, 0, n_bins - 1)

        combined_hist = field_hist.copy()
        np.add.at(combined_hist, selected_bins, 1)

        suffix_sum = np.cumsum(combined_hist[::-1])[::-1]
        entries_above = np.zeros(n_bins, dtype=np.int32)
        entries_above[:-1] = suffix_sum[1:]

        ranks = entries_above[selected_bins] + 1
        n_tied = combined_hist[selected_bins]
        payouts = payout_lookup.batch_get_payout(ranks, n_tied)

        payout_matrix[:, sim] = payouts

    # Compute pairwise correlations
    if n_selected <= 1:
        return {
            'mean_correlation': 0.0,
            'max_correlation': 0.0,
            'pct_above_08': 0.0,
        }

    corr_matrix = np.corrcoef(payout_matrix)
    # Extract upper triangle (excluding diagonal)
    upper_tri = corr_matrix[np.triu_indices(n_selected, k=1)]

    return {
        'mean_correlation': float(np.nanmean(upper_tri)),
        'max_correlation': float(np.nanmax(upper_tri)) if len(upper_tri) > 0 else 0.0,
        'pct_above_08': float(np.sum(upper_tri > 0.8) / max(len(upper_tri), 1) * 100),
        'n_pairs': len(upper_tri),
    }


# =============================================================================
# Player Exposure & Ownership Leverage
# =============================================================================

def compute_exposure_leverage(
    selected_arrays: LineupArrays,
    cpt_players: List[ShowdownPlayer],
    flex_players: List[ShowdownPlayer],
    field_arrays: Optional[LineupArrays] = None,
    field_counts: Optional[np.ndarray] = None,
) -> Dict:
    """
    Compute player exposure rates and ownership leverage.

    Leverage = portfolio_exposure / field_exposure (how much you overweight a player).

    Args:
        selected_arrays: Selected lineup arrays
        cpt_players: CPT player pool
        flex_players: FLEX player pool
        field_arrays: Optional field arrays for leverage computation
        field_counts: Optional field counts

    Returns:
        Dict with exposure data, unique CPTs, leverage metrics
    """
    n_selected = len(selected_arrays)

    # Portfolio exposure
    flex_exposure = np.zeros(len(flex_players), dtype=np.float64)
    cpt_exposure = np.zeros(len(cpt_players), dtype=np.float64)

    for i in range(n_selected):
        cpt_exposure[selected_arrays.cpt_idx[i]] += 1
        for fi in selected_arrays.flex_idx[i]:
            flex_exposure[fi] += 1

    flex_pct = flex_exposure / n_selected * 100
    cpt_pct = cpt_exposure / n_selected * 100

    unique_cpts = int(np.sum(cpt_exposure > 0))

    # Top 5 by FLEX exposure
    top5_flex = np.argsort(flex_pct)[-5:][::-1]
    top_exposures = []
    for idx in top5_flex:
        p = flex_players[idx]
        top_exposures.append({
            'name': p.name,
            'exposure_pct': float(flex_pct[idx]),
            'field_own_pct': float(p.ownership),
            'leverage': float(flex_pct[idx] / max(p.ownership, 0.01)),
        })

    # Aggregate leverage
    mean_port_own = float(np.mean(flex_pct[flex_pct > 0])) if np.any(flex_pct > 0) else 0
    mean_field_own = float(np.mean([p.ownership for p in flex_players]))
    overall_leverage = mean_port_own / max(mean_field_own, 0.01)

    return {
        'unique_cpts': unique_cpts,
        'top_exposures': top_exposures,
        'overall_leverage': overall_leverage,
        'mean_portfolio_own': mean_port_own,
        'mean_field_own': mean_field_own,
    }


# =============================================================================
# Tail Decomposition
# =============================================================================

def compute_tail_decomposition(
    per_lineup_evs: np.ndarray,
    contest: ContestStructure,
) -> Dict:
    """
    Decompose portfolio EV into tail contribution buckets.

    Shows what fraction of total EV comes from top-1%, top-10%, bottom-50% finishes.
    """
    total_ev = per_lineup_evs.sum()
    if total_ev <= 0:
        return {'top_1pct': 0, 'top_10pct': 0, 'bottom_50pct': 0}

    sorted_evs = np.sort(per_lineup_evs)[::-1]
    n = len(sorted_evs)

    top_1_n = max(1, int(n * 0.01))
    top_10_n = max(1, int(n * 0.10))
    bottom_50_start = int(n * 0.50)

    return {
        'top_1pct_ev_share': float(sorted_evs[:top_1_n].sum() / total_ev * 100),
        'top_10pct_ev_share': float(sorted_evs[:top_10_n].sum() / total_ev * 100),
        'bottom_50pct_ev_share': float(sorted_evs[bottom_50_start:].sum() / total_ev * 100),
    }


# =============================================================================
# Full Diagnostic Report
# =============================================================================

def compute_full_diagnostics(
    selected_arrays: LineupArrays,
    selected_indices: List[int],
    candidate_arrays: LineupArrays,
    field_arrays: LineupArrays,
    field_counts: np.ndarray,
    outcomes: np.ndarray,
    contest: ContestStructure,
    score_bounds: Tuple[int, int],
    cpt_players: List[ShowdownPlayer],
    flex_players: List[ShowdownPlayer],
    per_lineup_evs: np.ndarray,
    team_indices: Optional[Dict[str, list]] = None,
    game_total: float = 48.5,
    vegas_spread: float = 0.0,
) -> Dict:
    """
    Compute all enhanced diagnostics.

    Returns a comprehensive diagnostic dict with all metrics.
    """
    diag = {}

    # Player exposure & leverage
    diag['exposure'] = compute_exposure_leverage(
        selected_arrays, cpt_players, flex_players,
        field_arrays, field_counts
    )

    # Near-duplicate detection
    diag['near_duplicates'] = detect_near_duplicates(
        candidate_arrays, selected_indices
    )

    # Self-competition decomposition
    diag['self_comp_decomposition'] = decompose_self_competition(
        selected_arrays, field_arrays, field_counts,
        outcomes, contest, score_bounds
    )

    # Profit covariance metrics
    diag['profit_covariance'] = compute_profit_covariance_metrics(
        selected_arrays, field_arrays, field_counts,
        outcomes, contest, score_bounds
    )

    # Tail decomposition
    diag['tail_decomposition'] = compute_tail_decomposition(per_lineup_evs, contest)

    # Score distribution
    all_scores = []
    for sim in range(min(outcomes.shape[1], 5000)):
        scores = score_lineups_vectorized(selected_arrays, outcomes[:, sim])
        all_scores.append(scores)
    all_scores = np.concatenate(all_scores) / 10.0  # dequantize
    diag['score_distribution'] = {
        'floor': float(np.percentile(all_scores, 1)),
        'ceiling': float(np.percentile(all_scores, 99)),
        'std': float(all_scores.std()),
        'mean': float(all_scores.mean()),
    }

    # Game-state coverage (if team data available)
    if team_indices is not None and len(team_indices) >= 2:
        state_indices = classify_sims_by_game_state(
            outcomes, team_indices,
            game_total=game_total,
            vegas_spread=vegas_spread,
        )

        # Get best rank per sim for portfolio
        n_sims_gs = min(outcomes.shape[1], len(state_indices))
        best_ranks = np.full(n_sims_gs, contest.total_entries + 1, dtype=np.int32)

        min_score, max_score = score_bounds
        n_bins = max_score - min_score + 1
        payout_lookup = PayoutLookup.from_contest(contest)

        for sim in range(n_sims_gs):
            sim_outcomes = outcomes[:, sim]
            field_scores = score_lineups_vectorized(field_arrays, sim_outcomes)
            f_bins = np.clip(field_scores - min_score, 0, n_bins - 1)
            field_hist = np.bincount(
                f_bins, weights=field_counts.astype(np.float64), minlength=n_bins
            ).astype(np.int32)

            selected_scores = score_lineups_vectorized(selected_arrays, sim_outcomes)
            selected_bins = np.clip(selected_scores - min_score, 0, n_bins - 1)

            combined_hist = field_hist.copy()
            np.add.at(combined_hist, selected_bins, 1)

            suffix_sum = np.cumsum(combined_hist[::-1])[::-1]
            entries_above = np.zeros(n_bins, dtype=np.int32)
            entries_above[:-1] = suffix_sum[1:]

            ranks = entries_above[selected_bins] + 1
            best_ranks[sim] = ranks.min()

        diag['game_state_coverage'] = compute_game_state_coverage(
            state_indices[:n_sims_gs], best_ranks
        )

    # Tournament metrics
    tournament_metrics = compute_tournament_metrics(
        selected_arrays, field_arrays, field_counts,
        outcomes, contest, score_bounds
    )
    diag['tournament_metrics'] = tournament_metrics.to_dict()

    return diag


def format_diagnostics(diag: Dict, contest: ContestStructure) -> str:
    """Format diagnostics dict into readable console output."""
    lines = []
    lines.append("PORTFOLIO DIAGNOSTICS")
    lines.append("=" * 60)

    # Exposure
    if 'exposure' in diag:
        exp = diag['exposure']
        lines.append(f"\n  Player exposure (top 5):")
        for e in exp['top_exposures']:
            lines.append(
                f"    {e['name']:<25} {e['exposure_pct']:.0f}% exposure "
                f"({e['field_own_pct']:.0f}% field own, {e['leverage']:.1f}x leverage)"
            )
        lines.append(f"  Unique CPTs: {exp['unique_cpts']}")
        lines.append(f"  Ownership leverage: {exp['overall_leverage']:.2f}x")

    # Self-comp decomposition
    if 'self_comp_decomposition' in diag:
        sc = diag['self_comp_decomposition']
        total = sc['total_self_comp']
        if total > 0:
            lines.append(f"\n  Self-competition decomposition:")
            lines.append(f"    Total:     ${total:.0f}")
            lines.append(
                f"    Exact ties: ${sc['exact_tie_cost']:.0f} "
                f"({sc['exact_tie_cost']/total*100:.0f}%)"
            )
            lines.append(
                f"    Near ties:  ${sc['near_tie_cost']:.0f} "
                f"({sc['near_tie_cost']/total*100:.0f}%)"
            )
            lines.append(
                f"    Displacement: ${sc['displacement_cost']:.0f} "
                f"({sc['displacement_cost']/total*100:.0f}%)"
            )

    # Near duplicates
    if 'near_duplicates' in diag:
        nd = diag['near_duplicates']
        lines.append(f"\n  Near-duplicate lineups:")
        lines.append(
            f"    {nd['pairs_5_shared']} pairs share 5/6 | "
            f"{nd['pairs_4_shared']} pairs share 4/6"
        )

    # Profit covariance
    if 'profit_covariance' in diag:
        pc = diag['profit_covariance']
        lines.append(f"\n  Profit covariance metrics:")
        lines.append(
            f"    Mean pairwise corr: {pc['mean_correlation']:.2f} | "
            f"Max: {pc['max_correlation']:.2f} | "
            f"% pairs > 0.8: {pc['pct_above_08']:.1f}%"
        )

    # Score distribution
    if 'score_distribution' in diag:
        sd = diag['score_distribution']
        lines.append(
            f"\n  Score distribution: Floor={sd['floor']:.0f} | "
            f"Ceiling={sd['ceiling']:.0f} | Std={sd['std']:.1f}"
        )

    # Tail decomposition
    if 'tail_decomposition' in diag:
        td = diag['tail_decomposition']
        lines.append(
            f"\n  Tail decomposition: Top-1%={td['top_1pct_ev_share']:.0f}% of EV | "
            f"Top-10%={td['top_10pct_ev_share']:.0f}% | "
            f"Bottom-50%={td['bottom_50pct_ev_share']:.0f}%"
        )

    # Game-state coverage
    if 'game_state_coverage' in diag:
        lines.append(f"\n  Game-state coverage:")
        for state, data in diag['game_state_coverage'].items():
            lines.append(
                f"    {state:<25} {data['sim_share']*100:4.0f}% sims, "
                f"{data['top_n_share']*100:4.0f}% top-100 "
                f"({data['concentration_ratio']:.2f}x)"
            )

    # Tournament metrics
    if 'tournament_metrics' in diag:
        tm = diag['tournament_metrics']
        lines.append(f"\n  Tournament metrics:")
        lines.append(f"    Top 1% rate:    {tm['top_1pct_rate']:.2%}")
        lines.append(f"    Ceiling EV:     ${tm['ceiling_ev']:.2f}")
        lines.append(f"    Win rate:       {tm['win_rate']:.4%}")
        lines.append(f"    Composite:      {tm['composite_score']:.4f}")

    return "\n".join(lines)
