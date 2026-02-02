"""
Full pipeline orchestration for portfolio optimization.

Wires all modules together for end-to-end execution.
"""

import numpy as np
from numpy.random import SeedSequence
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
from .field.generator import generate_field, generate_field_simulated, compute_field_lineup_probabilities
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
from .field.robust import (
    generate_perturbed_fields,
    compute_robust_approx_evs,
    build_union_shortlist
)
from .metrics.tournament import compute_tournament_metrics

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
    game_total: Optional[float] = None,
    min_projection: float = 0.0,
    field_method: str = "simulated",
    field_sharpness: float = 5.0,
    ownership_power: float = 0.5,
    field_quality_sims: int = 1000,
    dro_perturbations: int = 0,
    dro_scale: float = 0.10,
    dro_hhi_scale: float = 0.15,
    dro_aggregation: str = "mean",
    dro_calibration_path: Optional[str] = None,
    dro_cvar_alpha: float = 0.20,
    dro_representative_k: int = 5,
    covariance_gamma: float = 0.0,
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
    6b. (Optional) DRO: generate perturbed fields + robust EVs
    7. Select lineups (top_n or greedy_marginal)
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
        field_method: "simulated" (quality x ownership from candidates) or
                      "ownership" (legacy player-by-player sampling)
        field_sharpness: How projection-aware the field is (0=pure ownership, 2+=optimizer)
        ownership_power: How ownership-driven duplication is (0=no ownership, 1=full)
        field_quality_sims: Number of sims used for field quality scoring

    Returns:
        Dict with selected lineups, diagnostics, and metadata
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)

    # === DERIVE INDEPENDENT RNG SEEDS ===
    if seed is not None:
        ss = SeedSequence(seed)
        sim_seed, field_seed = ss.spawn(2)
    else:
        sim_seed = field_seed = None

    # === LOAD DATA ===
    logger.info(f"Loading projections from {csv_path}...")
    data = load_projections(csv_path, min_projection=min_projection)
    logger.info(f"Loaded {data.n_cpt} CPT players, {data.n_flex} FLEX players")

    # === VALIDATE CONTEST ===
    _validate_contest_config(contest, n_select)
    field_size = contest.field_size
    logger.info(f"Contest: {contest.name}, field size: {field_size}")

    # === BUILD CORRELATION MATRIX ===
    corr_matrix = correlation_matrix
    corr_config = None
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
    outcomes = simulate_outcomes(
        data.flex_players, n_sims,
        correlation_matrix=corr_matrix,
        seed=sim_seed,
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
        if field_method == "simulated":
            logger.info(
                "Simulated field probabilities: sharpness=%.1f, ownership_power=%.1f, quality_sims=%d",
                field_sharpness, ownership_power, field_quality_sims
            )
            p_lineup = compute_field_lineup_probabilities(
                candidate_arrays, outcomes,
                data.cpt_players, data.flex_players,
                config=field_config,
                field_sharpness=field_sharpness,
                ownership_power=ownership_power,
                quality_sims=field_quality_sims,
                seed=field_seed
            )
        else:
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
        actual_field_size = field_size

    else:
        # Fixed field: generate once, use for all sims
        logger.info(f"Generating {field_size} field lineups...")
        if field_method == "simulated":
            logger.info(
                "Using simulated field (sharpness=%.1f, ownership_power=%.1f, quality_sims=%d)",
                field_sharpness, ownership_power, field_quality_sims
            )
            field_arrays, field_counts = generate_field_simulated(
                candidate_arrays, outcomes,
                data.cpt_players, data.flex_players,
                n_field=field_size,
                config=field_config,
                field_sharpness=field_sharpness,
                ownership_power=ownership_power,
                quality_sims=field_quality_sims,
                seed=field_seed
            )
        else:
            field_arrays, field_counts = generate_field(
                data.cpt_players, data.flex_players,
                n_field=field_size,
                config=field_config,
                cpt_to_flex_map=data.cpt_to_flex_map,
                salary_cap=salary_cap,
                seed=field_seed
            )
        actual_field_size = int(field_counts.sum())
        logger.info(f"Generated field: {len(field_arrays)} unique, {actual_field_size} total")

        # Compute approx EVs with fixed field
        logger.info(f"Computing approx EVs for {len(candidates)} candidates...")
        approx_evs = compute_approx_lineup_evs(
            candidate_arrays, field_arrays, field_counts,
            outcomes, contest, score_bounds
        )
        p_lineup = None

    # === DRO: ROBUST EVs (Optional) ===
    robust_evs = None
    if dro_perturbations > 0 and field_mode != "fixed":
        logger.warning(
            "DRO is only supported with field_mode='fixed'. "
            "DRO perturbations will be skipped for field_mode='%s'.",
            field_mode
        )
    if dro_perturbations > 0 and field_mode == "fixed":
        logger.info(
            f"DRO: Generating {dro_perturbations} perturbed fields "
            f"(scale={dro_scale}, hhi_scale={dro_hhi_scale})..."
        )
        field_variants = generate_perturbed_fields(
            data.cpt_players, data.flex_players,
            n_field=field_size,
            n_perturbations=dro_perturbations,
            perturbation_scale=dro_scale,
            hhi_scale=dro_hhi_scale,
            config=field_config,
            cpt_to_flex_map=data.cpt_to_flex_map,
            salary_cap=salary_cap,
            seed=seed,
            candidate_arrays=candidate_arrays,
            outcomes=outcomes,
            field_sharpness=field_sharpness,
            ownership_power=ownership_power,
            quality_sims=field_quality_sims,
            calibration_path=dro_calibration_path,
        )

        logger.info(f"Computing robust EVs (aggregation={dro_aggregation})...")
        robust_evs, all_variant_evs = compute_robust_approx_evs(
            candidate_arrays, field_variants,
            outcomes, contest, score_bounds,
            aggregation=dro_aggregation,
            cvar_alpha=dro_cvar_alpha,
            representative_k=dro_representative_k,
        )

        # Build union shortlist if using greedy
        if selection_method == "greedy_marginal":
            union_shortlist = build_union_shortlist(
                approx_evs, robust_evs, candidate_arrays,
                target_size=shortlist_size
            )
            logger.info(f"DRO union shortlist: {len(union_shortlist)} candidates")

        # Use robust EVs for selection ranking
        selection_evs = robust_evs
        logger.info(
            "DRO complete: nominal_mean=%.4f, robust_mean=%.4f, regret=%.4f",
            approx_evs.mean(), robust_evs.mean(),
            approx_evs.mean() - robust_evs.mean()
        )
    else:
        selection_evs = approx_evs

    # === SELECT LINEUPS ===
    if selection_method == "greedy_marginal":
        # Greedy selection: pick lineup with highest marginal EV at each step
        logger.info(f"Greedy marginal selection of {n_select} lineups...")

        if shortlist_size < n_select:
            shortlist_size = n_select
            logger.warning(f"shortlist_size increased to {n_select} (must be >= n_select)")

        if field_mode == "resample_per_sim":
            selected_indices = greedy_select_portfolio_resampled(
                candidate_arrays, selection_evs, p_lineup,
                outcomes, contest, score_bounds,
                field_size=field_size,
                n_select=n_select,
                shortlist_size=shortlist_size,
                greedy_n_sims=greedy_n_sims,
                seed=seed,
                covariance_gamma=covariance_gamma,
            )
        else:
            selected_indices = greedy_select_portfolio(
                candidate_arrays, selection_evs,
                field_arrays, field_counts,
                outcomes, contest, score_bounds,
                n_select=n_select,
                shortlist_size=shortlist_size,
                greedy_n_sims=greedy_n_sims,
                seed=seed,
                covariance_gamma=covariance_gamma,
            )

    else:
        # Top-N selection: sort by selection EVs and pick top N
        logger.info(f"Selecting top {n_select} lineups by {'robust' if robust_evs is not None else 'approx'} EV...")
        qualified = [
            (i, selection_evs[i])
            for i in range(len(candidates))
            if selection_evs[i] >= min_ev_threshold
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
        true_ev, diagnostics = compute_true_portfolio_ev_resampled(
            selected_arrays, p_lineup, candidate_arrays,
            outcomes, contest, score_bounds,
            field_size=field_size,
            seed=seed
        )
    else:
        true_ev, diagnostics = compute_true_portfolio_ev(
            selected_arrays, field_arrays, field_counts,
            outcomes, contest, score_bounds
        )

    # === DIAGNOSTICS ===
    approx_sum = sum(approx_evs[i] for i in selected_indices)
    diagnostics['approx_ev_sum'] = float(approx_sum)
    diagnostics['self_competition_cost'] = float(approx_sum - true_ev)
    diagnostics['score_bounds'] = score_bounds
    diagnostics['bounds_method'] = 'all_sims_max'
    diagnostics['n_candidates'] = len(candidates)

    # DRO diagnostics
    if robust_evs is not None:
        robust_sum = sum(robust_evs[i] for i in selected_indices)
        diagnostics['robust_ev_sum'] = float(robust_sum)
        diagnostics['dro_regret'] = float(approx_sum - robust_sum)
        diagnostics['dro_perturbations'] = dro_perturbations
        diagnostics['dro_aggregation'] = dro_aggregation

    # === TOURNAMENT METRICS ===
    tournament_metrics = None
    if field_mode == "fixed" and field_arrays is not None:
        logger.info("Computing tournament metrics...")
        tournament_metrics = compute_tournament_metrics(
            selected_arrays, field_arrays, field_counts,
            outcomes, contest, score_bounds
        )
        logger.info(
            "Tournament: top-1%%=%.2f%%, ceiling=$%.2f, win=%.4f%%, composite=%.4f",
            tournament_metrics.top_1pct_rate * 100,
            tournament_metrics.ceiling_ev,
            tournament_metrics.win_rate * 100,
            tournament_metrics.composite_score,
        )

    # === BUILD RESULTS ===
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
        'tournament_metrics': tournament_metrics.to_dict() if tournament_metrics else None,
        'metadata': {
            'csv_path': csv_path,
            'n_sims': n_sims,
            'n_select': n_select,
            'salary_cap': salary_cap,
            'teams': data.teams,
            'field_mode': field_mode,
            'field_method': field_method,
            'field_sharpness': field_sharpness if field_method == 'simulated' else None,
            'ownership_power': ownership_power if field_method == 'simulated' else None,
            'field_quality_sims': field_quality_sims if field_method == 'simulated' else None,
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
            'dro': {
                'perturbations': dro_perturbations,
                'scale': dro_scale,
                'hhi_scale': dro_hhi_scale,
                'aggregation': dro_aggregation,
            } if dro_perturbations > 0 else None,
            'covariance_gamma': covariance_gamma if covariance_gamma > 0 else None,
        }
    }

    logger.info(f"Portfolio EV: ${true_ev:.2f}")
    logger.info(f"Expected profit: ${diagnostics['expected_profit']:.2f}")
    logger.info(f"ROI: {diagnostics['roi_pct']:.2f}%")
    logger.info(f"Self-competition cost: ${diagnostics['self_competition_cost']:.2f}")

    return results


def run_multi_contest_optimization(
    csv_path: str,
    contests: List[ContestStructure],
    n_sims: int = 100000,
    correlation_config_path: Optional[str] = None,
    archetype_map_path: Optional[str] = None,
    correlation_matrix: Optional[np.ndarray] = None,
    field_config: Optional[FieldGenConfig] = None,
    salary_cap: int = 50000,
    seed: Optional[int] = None,
    verbose: bool = True,
    copula_type: CopulaType = "gaussian",
    copula_df: int = 5,
    selection_method: Literal["top_n", "greedy_marginal"] = "top_n",
    shortlist_size: int = 2000,
    greedy_n_sims: Optional[int] = None,
    effects_path: Optional[str] = None,
    sim_config_path: Optional[str] = None,
    spread_str: Optional[str] = None,
    game_total: Optional[float] = None,
    min_projection: float = 0.0,
    field_method: str = "simulated",
    field_sharpness: float = 5.0,
    ownership_power: float = 0.5,
    field_quality_sims: int = 1000,
    dro_perturbations: int = 0,
    dro_scale: float = 0.10,
    dro_hhi_scale: float = 0.15,
    dro_aggregation: str = "mean",
    covariance_gamma: float = 0.0,
) -> Dict:
    """
    Multi-contest portfolio optimization with shared computation.

    Shared computation (run ONCE):
      1. Load CSV, build correlations, apply effects
      2. Simulate outcomes
      3. Enumerate candidates
      4. Compute score bounds

    Per-contest (run N times):
      5. Generate field for THIS contest's field_size + sharpness
      6. Compute approx EVs for THIS contest's payout structure
      7. Run greedy selection
      8. Compute true portfolio EV
      9. Export results

    Args:
        csv_path: Path to projections CSV
        contests: List of contest structures to optimize for
        (all other args same as run_portfolio_optimization)

    Returns:
        Dict with per-contest results and aggregate metrics
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)

    # === DERIVE INDEPENDENT RNG SEEDS ===
    if seed is not None:
        ss = SeedSequence(seed)
        child_seeds = ss.spawn(2 + len(contests))
        sim_seed = child_seeds[0]
        enum_seed = child_seeds[1]
        contest_seeds = child_seeds[2:]
    else:
        sim_seed = None
        enum_seed = None
        contest_seeds = [None] * len(contests)

    # === SHARED: LOAD DATA ===
    logger.info(f"Loading projections from {csv_path}...")
    data = load_projections(csv_path, min_projection=min_projection)
    logger.info(f"Loaded {data.n_cpt} CPT players, {data.n_flex} FLEX players")

    # === SHARED: BUILD CORRELATION MATRIX ===
    corr_matrix = correlation_matrix
    corr_config = None
    archetype_map = {}
    if corr_matrix is None and correlation_config_path is not None:
        if archetype_map_path is not None:
            archetype_map = load_archetype_mapping(archetype_map_path)
        corr_config = ArchetypeCorrelationConfig.from_json(correlation_config_path)
        corr_obj = CorrelationMatrix.from_archetype_config(
            data.flex_players, corr_config, archetype_map
        )
        corr_matrix = corr_obj.matrix

    # === SHARED: APPLY PLAYER EFFECTS ===
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

    # === SHARED: BUILD GAME ENVIRONMENT PARAMS ===
    game_shares = None
    team_idx_map = None
    if corr_config is not None and corr_config.variance_decomposition:
        var_decomp = corr_config.variance_decomposition
        n_flex = len(data.flex_players)
        game_shares = np.zeros(n_flex, dtype=np.float64)
        default_game_shares = {'K': 0.30, 'DST': 0.50}
        for i, player in enumerate(data.flex_players):
            arch = archetype_map.get(player.name, _infer_archetype(player))
            if arch in var_decomp:
                game_shares[i] = var_decomp[arch]['game_share']
            elif arch in default_game_shares:
                game_shares[i] = default_game_shares[arch]
        team_idx_map = {}
        for i, player in enumerate(data.flex_players):
            if player.team not in team_idx_map:
                team_idx_map[player.team] = []
            team_idx_map[player.team].append(i)

    # === SHARED: SIMULATE OUTCOMES ===
    logger.info(f"Simulating {n_sims} outcomes...")
    outcomes = simulate_outcomes(
        data.flex_players, n_sims,
        correlation_matrix=corr_matrix,
        seed=sim_seed,
        copula_type=copula_type,
        copula_df=copula_df,
        game_shares=game_shares,
        team_indices=team_idx_map
    )

    # === SHARED: COMPUTE BOUNDS ===
    score_bounds = compute_guaranteed_score_bounds(outcomes)

    # === SHARED: ENUMERATE CANDIDATES ===
    logger.info("Enumerating candidate lineups...")
    candidates = enumerate_lineups(
        data.cpt_players, data.flex_players,
        salary_cap=salary_cap,
        cpt_to_flex_map=data.cpt_to_flex_map
    )
    if len(candidates) == 0:
        return {'error': 'No valid lineups found'}
    candidate_arrays = LineupArrays.from_lineups(candidates)
    logger.info(f"Enumerated {len(candidates)} candidates")

    # === TIER ASSIGNMENT & ORDERING ===
    def assign_contest_tier(c):
        if c.tier is not None:
            return c.tier
        # Auto-assign by field share
        field_share = c.your_entries / c.total_entries if c.total_entries > 0 else 0
        if field_share > 0.05:
            return 2  # High saturation — covariance penalty sufficient
        return 3  # Low saturation — diversity penalty

    tier_map = {}
    for c in contests:
        tier_map[c.name] = assign_contest_tier(c)

    # Processing order: tier 1 first, then by field_share * sqrt(dollars_at_risk)
    def contest_priority(c):
        tier = tier_map[c.name]
        priority = c.field_share * np.sqrt(c.dollars_at_risk)
        return (tier, -priority)  # lower tier first, higher priority first

    ordered_contests = sorted(contests, key=contest_priority)

    # Flagship = first contest (tier 1 if any, else highest priority)
    if not any(tier_map[c.name] == 1 for c in contests):
        # Auto-assign largest prizepool as tier 1
        flagship = max(contests, key=lambda c: sum(
            t.payout * (t.end_rank - t.start_rank + 1)
            for t in c.payout_tiers
        ))
        tier_map[flagship.name] = 1
        ordered_contests = sorted(contests, key=contest_priority)

    for c in ordered_contests:
        logger.info(
            "Contest order: %s (tier %d, field_share=%.1f%%)",
            c.name, tier_map[c.name], c.field_share * 100
        )

    # === PER-CONTEST OPTIMIZATION ===
    all_results = {}
    overlap_matrix = {}
    previously_selected = {}  # contest_name -> set of indices

    from .candidates.enumeration import lineup_to_names

    for idx, contest in enumerate(ordered_contests):
        # Find original index for seeding
        orig_idx = contests.index(contest)

        logger.info(f"\n{'='*60}")
        logger.info(
            f"Contest {idx+1}/{len(contests)}: {contest.name} (tier {tier_map[contest.name]})"
        )
        logger.info(f"{'='*60}")

        contest_sharpness = contest.field_sharpness_override or field_sharpness
        field_size = contest.field_size
        n_select = contest.your_entries

        # Validate
        _validate_contest_config(contest, n_select)

        # Generate field for this contest
        if field_method == "simulated":
            field_arrays, field_counts = generate_field_simulated(
                candidate_arrays, outcomes,
                data.cpt_players, data.flex_players,
                n_field=field_size,
                config=field_config,
                field_sharpness=contest_sharpness,
                ownership_power=ownership_power,
                quality_sims=field_quality_sims,
                seed=contest_seeds[orig_idx]
            )
        else:
            field_arrays, field_counts = generate_field(
                data.cpt_players, data.flex_players,
                n_field=field_size,
                config=field_config,
                cpt_to_flex_map=data.cpt_to_flex_map,
                salary_cap=salary_cap,
                seed=contest_seeds[orig_idx]
            )

        # Compute approx EVs
        approx_evs = compute_approx_lineup_evs(
            candidate_arrays, field_arrays, field_counts,
            outcomes, contest, score_bounds
        )

        # Apply cross-contest overlap penalty based on tier
        contest_tier = tier_map[contest.name]
        selection_evs = approx_evs.copy()

        if previously_selected and contest_tier >= 2:
            # Compute overlap penalty for candidates previously selected in other contests
            all_prior = set()
            for prior_set in previously_selected.values():
                all_prior.update(prior_set)

            if all_prior:
                overlap_penalty = 0.005 if contest_tier == 2 else 0.015
                for ci in all_prior:
                    if ci < len(selection_evs):
                        selection_evs[ci] -= overlap_penalty

                logger.info(
                    "Applied overlap penalty $%.3f to %d previously-selected lineups (tier %d)",
                    overlap_penalty, len(all_prior), contest_tier
                )

        # Select lineups
        if selection_method == "greedy_marginal":
            sl_size = max(shortlist_size, n_select)
            selected_indices = greedy_select_portfolio(
                candidate_arrays, selection_evs,
                field_arrays, field_counts,
                outcomes, contest, score_bounds,
                n_select=n_select,
                shortlist_size=sl_size,
                greedy_n_sims=greedy_n_sims,
                seed=contest_seeds[orig_idx],
                covariance_gamma=covariance_gamma,
            )
        else:
            qualified = sorted(
                [(i, selection_evs[i]) for i in range(len(candidates))
                 if selection_evs[i] >= 0.0],
                key=lambda x: -x[1]
            )
            selected_indices = [i for i, _ in qualified[:n_select]]

        previously_selected[contest.name] = set(selected_indices)

        # Compute true portfolio EV
        selected_lineups = [candidates[i] for i in selected_indices]
        selected_arrays = LineupArrays.from_lineups(selected_lineups)
        true_ev, diagnostics = compute_true_portfolio_ev(
            selected_arrays, field_arrays, field_counts,
            outcomes, contest, score_bounds
        )

        approx_sum = sum(approx_evs[i] for i in selected_indices)
        diagnostics['approx_ev_sum'] = float(approx_sum)
        diagnostics['self_competition_cost'] = float(approx_sum - true_ev)
        diagnostics['contest_tier'] = contest_tier

        selected_players = [
            lineup_to_names(lu, data.cpt_players, data.flex_players)
            for lu in selected_lineups
        ]

        all_results[contest.name] = {
            'selected_lineups': selected_lineups,
            'selected_indices': selected_indices,
            'selected_players': selected_players,
            'approx_evs': [approx_evs[i] for i in selected_indices],
            'diagnostics': diagnostics,
        }

        logger.info(f"Contest {contest.name}: EV=${true_ev:.2f}, ROI={diagnostics['roi_pct']:.2f}%")

    # Compute overlap matrix between contests
    contest_names = [c.name for c in contests]
    for i, name_i in enumerate(contest_names):
        for j, name_j in enumerate(contest_names):
            if i >= j:
                continue
            set_i = set(all_results[name_i]['selected_indices'])
            set_j = set(all_results[name_j]['selected_indices'])
            overlap = len(set_i & set_j)
            key = f"{name_i} x {name_j}"
            overlap_matrix[key] = {
                'shared_lineups': overlap,
                'pct_of_i': overlap / len(set_i) * 100 if set_i else 0,
                'pct_of_j': overlap / len(set_j) * 100 if set_j else 0,
            }

    return {
        'per_contest': all_results,
        'overlap_matrix': overlap_matrix,
        'metadata': {
            'csv_path': csv_path,
            'n_sims': n_sims,
            'n_contests': len(contests),
            'contest_names': contest_names,
            'copula_type': copula_type,
            'selection_method': selection_method,
            'field_method': field_method,
        }
    }


def _validate_contest_config(contest: ContestStructure, n_select: int):
    """Validate contest configuration matches selection."""
    if contest.your_entries != n_select:
        raise ValueError(
            f"Contest configured for {contest.your_entries} entries, "
            f"but selecting {n_select}. Update contest config."
        )
