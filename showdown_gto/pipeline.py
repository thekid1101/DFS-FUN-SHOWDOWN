"""
Full pipeline orchestration for portfolio optimization.

Wires all modules together for end-to-end execution.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

from .types import (
    ShowdownLineup, LineupArrays, ContestStructure, FieldGenConfig, ProjectionsData
)
from .data.loader import load_projections
from .data.correlations import (
    CorrelationMatrix, ArchetypeCorrelationConfig,
    create_archetype_mapping_template, load_archetype_mapping
)
from .simulation.engine import simulate_outcomes
from .simulation.bounds import compute_guaranteed_score_bounds
from .candidates.enumeration import enumerate_lineups, lineup_to_names
from .field.generator import generate_field
from .ev.approx import (
    compute_approx_lineup_evs,
    compute_approx_lineup_evs_resampled,
    compute_lineup_probabilities,
    FieldMode
)
from .ev.portfolio import compute_true_portfolio_ev, compute_true_portfolio_ev_resampled

logger = logging.getLogger(__name__)


def run_portfolio_optimization(
    csv_path: str,
    contest: ContestStructure,
    n_select: int = 150,
    n_sims: int = 100000,
    correlation_config_path: Optional[str] = None,
    archetype_map_path: Optional[str] = None,
    correlation_matrix: Optional[np.ndarray] = None,
    field_config: Optional[FieldGenConfig] = None,
    salary_cap: int = 50000,
    min_ev_threshold: float = 0.0,
    seed: Optional[int] = None,
    verbose: bool = True,
    field_mode: FieldMode = "fixed"
) -> Dict:
    """
    Full portfolio optimization pipeline.

    Steps:
    1. Load data from CSV
    2. Simulate outcomes (with correlations if provided)
    3. Compute guaranteed bounds
    4. Enumerate candidates
    5. Generate field (or compute lineup probabilities for resampled mode)
    6. Compute approx EVs
    7. Select top N
    8. Compute true portfolio EV
    9. Return results

    Args:
        csv_path: Path to projections CSV
        contest: Contest structure (entry fee, payouts, etc.)
        n_select: Number of lineups to select
        n_sims: Number of simulations
        correlation_config_path: Path to correlation_config_v2.json
        archetype_map_path: Path to player archetype mapping JSON
        correlation_matrix: Optional pre-built correlation matrix (overrides config)
        field_config: Field generation config
        salary_cap: Salary cap (default $50,000)
        min_ev_threshold: Minimum approx EV to be considered
        seed: Random seed for reproducibility
        verbose: Whether to log progress
        field_mode: "fixed" (single field for all sims) or "resample_per_sim"
                    (new field histogram sampled each sim via multinomial)

    Returns:
        Dict with selected lineups, diagnostics, and metadata
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)

    # === LOAD DATA ===
    logger.info(f"Loading projections from {csv_path}...")
    data = load_projections(csv_path)
    logger.info(f"Loaded {data.n_cpt} CPT players, {data.n_flex} FLEX players")

    # === VALIDATE CONTEST ===
    _validate_contest_config(contest, n_select)
    field_size = contest.field_size
    logger.info(f"Contest: {contest.name}, field size: {field_size}")

    # === BUILD CORRELATION MATRIX ===
    corr_matrix = correlation_matrix
    if corr_matrix is None and correlation_config_path is not None:
        logger.info(f"Loading correlation config from {correlation_config_path}...")
        archetype_map = {}
        if archetype_map_path is not None:
            archetype_map = load_archetype_mapping(archetype_map_path)
            logger.info(f"Loaded {len(archetype_map)} archetype mappings")

        corr_obj = CorrelationMatrix.from_players_and_config_file(
            data.flex_players, correlation_config_path, archetype_map
        )
        corr_matrix = corr_obj.matrix
        logger.info(f"Built {corr_matrix.shape[0]}x{corr_matrix.shape[0]} correlation matrix")

    # === SIMULATE OUTCOMES ===
    logger.info(f"Simulating {n_sims} outcomes...")
    # Use FLEX players for simulation (CPT scores derived from these)
    outcomes = simulate_outcomes(
        data.flex_players, n_sims,
        correlation_matrix=corr_matrix,
        seed=seed
    )
    logger.info(f"Outcomes shape: {outcomes.shape}")

    # === COMPUTE BOUNDS ===
    logger.info("Computing guaranteed score bounds...")
    score_bounds = compute_guaranteed_score_bounds(outcomes)
    logger.info(f"Score bounds: {score_bounds}")

    # === ENUMERATE CANDIDATES ===
    logger.info("Enumerating candidate lineups...")
    candidates = enumerate_lineups(
        data.cpt_players, data.flex_players,
        salary_cap=salary_cap,
        cpt_to_flex_map=data.cpt_to_flex_map
    )
    logger.info(f"Enumerated {len(candidates)} candidates")

    if len(candidates) == 0:
        return {'error': 'No valid lineups found'}

    candidate_arrays = LineupArrays.from_lineups(candidates)

    # === FIELD MODE BRANCHING ===
    if field_mode == "resample_per_sim":
        # Resampled field: compute lineup probabilities, sample histogram each sim
        logger.info("Using resample_per_sim field mode...")
        logger.info("Computing lineup probabilities for all candidates...")
        p_lineup = compute_lineup_probabilities(
            candidate_arrays,
            data.cpt_players, data.flex_players,
            config=field_config
        )
        logger.info(f"p_lineup: min={p_lineup.min():.2e}, max={p_lineup.max():.2e}")

        # Compute approx EVs with resampled field
        logger.info(f"Computing approx EVs for {len(candidates)} candidates (resampled)...")
        approx_evs = compute_approx_lineup_evs_resampled(
            candidate_arrays, p_lineup,
            outcomes, contest, score_bounds,
            field_size=field_size,
            seed=seed
        )

        # No fixed field needed - we'll use resampled true EV as well
        field_arrays = None
        field_counts = None
        actual_field_size = field_size  # Expected field size

    else:
        # Fixed field: generate once, use for all sims
        logger.info(f"Generating {field_size} field lineups...")
        field_arrays, field_counts = generate_field(
            data.cpt_players, data.flex_players,
            n_field=field_size,
            config=field_config,
            cpt_to_flex_map=data.cpt_to_flex_map,
            salary_cap=salary_cap,
            seed=seed
        )
        actual_field_size = int(field_counts.sum())
        logger.info(f"Generated field: {len(field_arrays)} unique, {actual_field_size} total")

        # Compute approx EVs with fixed field
        logger.info(f"Computing approx EVs for {len(candidates)} candidates...")
        approx_evs = compute_approx_lineup_evs(
            candidate_arrays, field_arrays, field_counts,
            outcomes, contest, score_bounds
        )
        p_lineup = None  # Not needed for fixed mode

    # === SELECT TOP N ===
    logger.info(f"Selecting top {n_select} lineups...")
    qualified = [
        (i, approx_evs[i])
        for i in range(len(candidates))
        if approx_evs[i] >= min_ev_threshold
    ]
    qualified.sort(key=lambda x: -x[1])
    selected_indices = [i for i, _ in qualified[:n_select]]

    if len(selected_indices) < n_select:
        logger.warning(
            f"Only {len(selected_indices)} lineups met threshold "
            f"(requested {n_select})"
        )

    if len(selected_indices) == 0:
        return {'error': 'No lineups met EV threshold'}

    # === COMPUTE TRUE PORTFOLIO EV ===
    logger.info(f"Computing true portfolio EV for {len(selected_indices)} lineups...")
    selected_lineups = [candidates[i] for i in selected_indices]
    selected_arrays = LineupArrays.from_lineups(selected_lineups)

    if field_mode == "resample_per_sim":
        # Use resampled true EV (consistent with approx EV computation)
        true_ev, diagnostics = compute_true_portfolio_ev_resampled(
            selected_arrays, p_lineup, candidate_arrays,
            outcomes, contest, score_bounds,
            field_size=field_size,
            seed=seed
        )
    else:
        # Use fixed field
        true_ev, diagnostics = compute_true_portfolio_ev(
            selected_arrays, field_arrays, field_counts,
            outcomes, contest, score_bounds
        )

    # === BUILD RESULTS ===
    approx_sum = sum(approx_evs[i] for i in selected_indices)
    diagnostics['approx_ev_sum'] = float(approx_sum)
    diagnostics['self_competition_cost'] = float(approx_sum - true_ev)
    diagnostics['score_bounds'] = score_bounds
    diagnostics['bounds_method'] = 'all_sims_max'
    diagnostics['n_candidates'] = len(candidates)
    diagnostics['n_qualified'] = len(qualified)

    # Get human-readable lineup info
    selected_players = []
    for lineup in selected_lineups:
        names = lineup_to_names(lineup, data.cpt_players, data.flex_players)
        selected_players.append(names)

    results = {
        'selected_lineups': selected_lineups,
        'selected_indices': selected_indices,
        'selected_players': selected_players,
        'approx_evs': [approx_evs[i] for i in selected_indices],
        'diagnostics': diagnostics,
        'metadata': {
            'csv_path': csv_path,
            'n_sims': n_sims,
            'n_select': n_select,
            'salary_cap': salary_cap,
            'teams': data.teams,
            'field_mode': field_mode,
        }
    }

    logger.info(f"Portfolio EV: ${true_ev:.2f}")
    logger.info(f"Expected profit: ${diagnostics['expected_profit']:.2f}")
    logger.info(f"ROI: {diagnostics['roi_pct']:.2f}%")
    logger.info(f"Self-competition cost: ${diagnostics['self_competition_cost']:.2f}")

    return results


def _validate_contest_config(contest: ContestStructure, n_select: int):
    """Validate contest configuration matches selection."""
    if contest.your_entries != n_select:
        raise ValueError(
            f"Contest configured for {contest.your_entries} entries, "
            f"but selecting {n_select}. Update contest config."
        )


def select_portfolio_greedy(
    candidates: List[ShowdownLineup],
    field_arrays: LineupArrays,
    field_counts: np.ndarray,
    outcomes: np.ndarray,
    contest: ContestStructure,
    score_bounds: Tuple[int, int],
    n_select: int,
    approx_evs: Optional[np.ndarray] = None
) -> List[int]:
    """
    Greedy portfolio selection considering marginal EV.

    More accurate than top-N by approx EV, but slower.
    Useful for small portfolios where self-competition matters more.
    """
    from .ev.portfolio import compute_true_portfolio_ev

    candidate_arrays = LineupArrays.from_lineups(candidates)

    # Start with best approx EV lineup
    if approx_evs is None:
        approx_evs = compute_approx_lineup_evs(
            candidate_arrays, field_arrays, field_counts,
            outcomes, contest, score_bounds
        )

    selected_indices = [int(np.argmax(approx_evs))]

    while len(selected_indices) < n_select:
        best_idx = -1
        best_marginal_ev = float('-inf')

        # Get current selection
        current_lineups = [candidates[i] for i in selected_indices]
        current_arrays = LineupArrays.from_lineups(current_lineups)

        # Evaluate each remaining candidate
        remaining = [i for i in range(len(candidates)) if i not in selected_indices]

        for idx in remaining:
            # Compute EV with this candidate added
            test_lineups = current_lineups + [candidates[idx]]
            test_arrays = LineupArrays.from_lineups(test_lineups)

            ev, _ = compute_true_portfolio_ev(
                test_arrays, field_arrays, field_counts,
                outcomes, contest, score_bounds
            )

            # Marginal EV
            marginal = ev - (len(selected_indices) * contest.entry_fee) - contest.entry_fee

            if marginal > best_marginal_ev:
                best_marginal_ev = marginal
                best_idx = idx

        if best_idx >= 0:
            selected_indices.append(best_idx)
            logger.info(f"Added lineup {best_idx}, marginal EV: ${best_marginal_ev:.2f}")
        else:
            break

    return selected_indices
