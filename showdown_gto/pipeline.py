"""
Full pipeline orchestration for portfolio optimization.

Wires all modules together for end-to-end execution.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Literal
import logging

from .types import (
    ShowdownLineup, LineupArrays, ContestStructure, FieldGenConfig, ProjectionsData
)
from .data.loader import load_projections
from .data.correlations import (
    CorrelationMatrix, ArchetypeCorrelationConfig,
    create_archetype_mapping_template, load_archetype_mapping,
    _infer_archetype
)
from .simulation.engine import simulate_outcomes, CopulaType
from .simulation.bounds import compute_guaranteed_score_bounds
from .candidates.enumeration import enumerate_lineups, lineup_to_names
from .field.generator import generate_field
from .ev.approx import (
    compute_approx_lineup_evs,
    compute_approx_lineup_evs_resampled,
    compute_lineup_probabilities,
    FieldMode
)
from .ev.portfolio import (
    compute_true_portfolio_ev, compute_true_portfolio_ev_resampled,
    greedy_select_portfolio, greedy_select_portfolio_resampled
)

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
    field_mode: FieldMode = "fixed",
    copula_type: CopulaType = "gaussian",
    copula_df: int = 5,
    selection_method: Literal["top_n", "greedy_marginal"] = "top_n",
    shortlist_size: int = 2000,
    greedy_n_sims: Optional[int] = None,
    effects_path: Optional[str] = None,
    sim_config_path: Optional[str] = None,
    spread_str: Optional[str] = None,
    game_total: Optional[float] = None
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
    corr_config = None  # Keep reference for variance_decomposition
    archetype_map = {}
    if corr_matrix is None and correlation_config_path is not None:
        logger.info(f"Loading correlation config from {correlation_config_path}...")
        if archetype_map_path is not None:
            archetype_map = load_archetype_mapping(archetype_map_path)
            logger.info(f"Loaded {len(archetype_map)} archetype mappings")

        corr_config = ArchetypeCorrelationConfig.from_json(correlation_config_path)
        corr_obj = CorrelationMatrix.from_archetype_config(
            data.flex_players, corr_config, archetype_map
        )
        corr_matrix = corr_obj.matrix
        logger.info(f"Built {corr_matrix.shape[0]}x{corr_matrix.shape[0]} correlation matrix")

    # === APPLY PLAYER EFFECTS ===
    if effects_path is not None:
        from .data.effects import apply_player_effects, parse_game_context, sync_cpt_from_flex

        game_context = parse_game_context(spread_str, game_total, data.teams)

        data.flex_players, corr_matrix = apply_player_effects(
            players=data.flex_players,
            correlation_matrix=corr_matrix,
            effects_path=effects_path,
            sim_config_path=sim_config_path,
            archetype_map=archetype_map if archetype_map else None,
            game_context=game_context
        )

        data.cpt_players = sync_cpt_from_flex(
            data.cpt_players, data.flex_players, data.cpt_to_flex_map
        )

        logger.info("Applied player effects from %s", effects_path)

    # === BUILD GAME ENVIRONMENT PARAMS ===
    game_shares = None
    team_idx_map = None
    if corr_config is not None and corr_config.variance_decomposition:
        var_decomp = corr_config.variance_decomposition
        n_flex = len(data.flex_players)
        game_shares = np.zeros(n_flex, dtype=np.float64)

        # Default game_shares for archetypes not in variance_decomposition
        default_game_shares = {'K': 0.30, 'DST': 0.50}

        matched = 0
        for i, player in enumerate(data.flex_players):
            arch = archetype_map.get(player.name, _infer_archetype(player))
            if arch in var_decomp:
                game_shares[i] = var_decomp[arch]['game_share']
                matched += 1
            elif arch in default_game_shares:
                game_shares[i] = default_game_shares[arch]
                matched += 1

        # Build team -> player indices mapping
        team_idx_map = {}
        for i, player in enumerate(data.flex_players):
            if player.team not in team_idx_map:
                team_idx_map[player.team] = []
            team_idx_map[player.team].append(i)

        logger.info(
            "Game environment: %d/%d players matched, teams=%s",
            matched, n_flex, list(team_idx_map.keys())
        )
        logger.info(
            "Game shares: min=%.3f, max=%.3f, mean=%.3f",
            game_shares.min(), game_shares.max(), game_shares.mean()
        )

    # === SIMULATE OUTCOMES ===
    logger.info(f"Simulating {n_sims} outcomes...")
    # Use FLEX players for simulation (CPT scores derived from these)
    outcomes = simulate_outcomes(
        data.flex_players, n_sims,
        correlation_matrix=corr_matrix,
        seed=seed,
        copula_type=copula_type,
        copula_df=copula_df,
        game_shares=game_shares,
        team_indices=team_idx_map
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

    # === SELECT LINEUPS ===
    if selection_method == "greedy_marginal":
        # Greedy selection: pick lineup with highest marginal EV at each step
        logger.info(f"Greedy marginal selection of {n_select} lineups...")

        if shortlist_size < n_select:
            shortlist_size = n_select
            logger.warning(f"shortlist_size increased to {n_select} (must be >= n_select)")

        if field_mode == "resample_per_sim":
            selected_indices = greedy_select_portfolio_resampled(
                candidate_arrays, approx_evs, p_lineup,
                outcomes, contest, score_bounds,
                field_size=field_size,
                n_select=n_select,
                shortlist_size=shortlist_size,
                greedy_n_sims=greedy_n_sims,
                seed=seed
            )
        else:
            selected_indices = greedy_select_portfolio(
                candidate_arrays, approx_evs,
                field_arrays, field_counts,
                outcomes, contest, score_bounds,
                n_select=n_select,
                shortlist_size=shortlist_size,
                greedy_n_sims=greedy_n_sims,
                seed=seed
            )

        # For compatibility with diagnostics below
        qualified = [(i, approx_evs[i]) for i in range(len(candidates))]
    else:
        # Top-N selection: sort by approx EV and pick top N
        logger.info(f"Selecting top {n_select} lineups by approx EV...")
        qualified = [
            (i, approx_evs[i])
            for i in range(len(candidates))
            if approx_evs[i] >= min_ev_threshold
        ]
        qualified.sort(key=lambda x: -x[1])
        selected_indices = [i for i, _ in qualified[:n_select]]

    if len(selected_indices) < n_select:
        logger.warning(
            f"Only {len(selected_indices)} lineups selected "
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
            'copula_type': copula_type,
            'copula_df': copula_df if copula_type == 't' else None,
            'selection_method': selection_method,
            'shortlist_size': shortlist_size if selection_method == 'greedy_marginal' else None,
            'effects_path': effects_path,
            'sim_config_path': sim_config_path,
            'game_environment': game_shares is not None,
            'game_context': {
                'spread': spread_str,
                'game_total': game_total,
            } if spread_str or game_total else None,
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


