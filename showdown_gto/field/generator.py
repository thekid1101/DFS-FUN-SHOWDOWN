"""
Field generation for opponent modeling.

Generates a distribution of opponent lineups based on ownership and soft priors.
Supports two methods:
- "ownership": Legacy player-by-player random sampling (ownership only)
- "simulated": Sample from candidate pool weighted by quality × ownership
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter
import logging

from ..types import (
    ShowdownPlayer, ShowdownLineup, LineupArrays, FieldGenConfig
)
from ..scoring.payout import score_lineups_vectorized

logger = logging.getLogger(__name__)


def generate_field(
    cpt_players: List[ShowdownPlayer],
    flex_players: List[ShowdownPlayer],
    n_field: int,
    config: Optional[FieldGenConfig] = None,
    cpt_to_flex_map: Optional[Dict[int, int]] = None,
    salary_cap: int = 50000,
    seed: Optional[int] = None
) -> Tuple[LineupArrays, np.ndarray]:
    """
    Generate field lineups with soft priors.

    Models opponent behavior using:
    - Ownership-weighted player selection
    - Team stacking tendencies
    - QB-pass catcher correlation
    - Bring-back (game stack) tendencies

    Args:
        cpt_players: CPT player pool
        flex_players: FLEX player pool
        n_field: Number of field entries to generate
        config: Field generation configuration
        cpt_to_flex_map: CPT to FLEX index mapping
        salary_cap: Salary cap
        seed: Random seed (accepts int or SeedSequence)

    Returns:
        field_arrays: LineupArrays of unique field lineups
        field_counts: Count of each unique lineup
    """
    rng = np.random.default_rng(seed)

    if config is None:
        config = FieldGenConfig()

    if cpt_to_flex_map is None:
        cpt_to_flex_map = {}

    # Get team information
    teams = list(set(p.team for p in flex_players if p.team))
    if len(teams) != 2:
        logger.warning(f"Expected 2 teams, got {len(teams)}: {teams}")

    # Compute adjusted ownership probabilities
    cpt_probs = _compute_ownership_probs(cpt_players, config)
    flex_probs = _compute_ownership_probs(flex_players, config)

    # Generate lineups
    lineup_hashes: Dict[tuple, int] = Counter()

    for _ in range(n_field):
        lineup = _sample_lineup(
            cpt_players, flex_players, cpt_probs, flex_probs,
            config, cpt_to_flex_map, salary_cap, teams, rng=rng
        )
        if lineup is not None:
            # Hash lineup for deduplication
            lineup_key = (lineup.cpt_player_idx, tuple(lineup.flex_player_idxs))
            lineup_hashes[lineup_key] += 1

    # Convert to arrays
    unique_lineups = []
    counts = []

    for (cpt_idx, flex_idxs), count in lineup_hashes.items():
        flex_salary = sum(flex_players[i].salary for i in flex_idxs)
        unique_lineups.append(ShowdownLineup(
            cpt_player_idx=cpt_idx,
            flex_player_idxs=list(flex_idxs),
            salary=cpt_players[cpt_idx].salary + flex_salary
        ))
        counts.append(count)

    field_arrays = LineupArrays.from_lineups(unique_lineups)
    field_counts = np.array(counts, dtype=np.int32)

    logger.info(f"Generated {n_field} field lineups ({len(unique_lineups)} unique)")

    return field_arrays, field_counts


def _compute_ownership_probs(
    players: List[ShowdownPlayer],
    config: FieldGenConfig
) -> np.ndarray:
    """
    Compute sampling probabilities from ownership.

    Applies temperature scaling and position multipliers.
    """
    n = len(players)
    if n == 0:
        return np.array([])

    # Base probabilities from ownership
    probs = np.array([max(p.ownership, 0.01) for p in players], dtype=np.float64)

    # Apply position multipliers
    for i, p in enumerate(players):
        if p.position == 'DST':
            probs[i] *= config.dst_rate_multiplier
        elif p.position == 'K':
            probs[i] *= config.kicker_rate_multiplier

    # Temperature scaling
    probs = probs ** (1.0 / config.temperature)

    # Normalize
    probs = probs / probs.sum()

    return probs


def _sample_lineup(
    cpt_players: List[ShowdownPlayer],
    flex_players: List[ShowdownPlayer],
    cpt_probs: np.ndarray,
    flex_probs: np.ndarray,
    config: FieldGenConfig,
    cpt_to_flex_map: Dict[int, int],
    salary_cap: int,
    teams: List[str],
    max_attempts: int = 100,
    rng: Optional[np.random.Generator] = None
) -> Optional[ShowdownLineup]:
    """
    Sample a single lineup respecting priors and constraints.

    Uses rejection sampling with ownership-weighted selection.
    """
    if rng is None:
        rng = np.random.default_rng()

    for _ in range(max_attempts):
        # Sample CPT
        cpt_idx = rng.choice(len(cpt_players), p=cpt_probs)
        cpt = cpt_players[cpt_idx]

        # Determine target team split
        split = _sample_team_split(config, rng=rng)
        cpt_team_count, opp_team_count = split

        # Get excluded flex (same player as CPT)
        excluded_flex = cpt_to_flex_map.get(cpt_idx)

        # Adjust flex probs for this lineup
        adjusted_probs = flex_probs.copy()
        if excluded_flex is not None:
            adjusted_probs[excluded_flex] = 0

        # Apply team split preference
        cpt_team = cpt.team
        opp_team = [t for t in teams if t != cpt_team][0] if len(teams) == 2 else None

        # Ensure required team split is possible given available flex players
        if len(teams) == 2:
            available_cpt_team = sum(
                1 for i, p in enumerate(flex_players)
                if i != excluded_flex and p.team == cpt_team
            )
            available_opp_team = sum(
                1 for i, p in enumerate(flex_players)
                if i != excluded_flex and p.team == opp_team
            )
            if (cpt_team_count > available_cpt_team or
                    opp_team_count > available_opp_team):
                continue

        # Apply QB pairing if CPT is QB
        if cpt.position == 'QB' and rng.random() < config.qb_pair_rate:
            # Boost same-team pass catchers
            for i, p in enumerate(flex_players):
                if p.team == cpt_team and p.position in ('WR', 'TE'):
                    adjusted_probs[i] *= 2.0

        # Apply bring-back preference
        if opp_team and rng.random() < config.bring_back_rate:
            # Boost opposite team players
            for i, p in enumerate(flex_players):
                if p.team == opp_team:
                    adjusted_probs[i] *= 1.5

        # Normalize
        if adjusted_probs.sum() > 0:
            adjusted_probs = adjusted_probs / adjusted_probs.sum()
        else:
            continue

        # Sample FLEX players
        remaining_cap = salary_cap - cpt.salary
        flex_selected = []
        used_indices = set()

        if excluded_flex is not None:
            used_indices.add(excluded_flex)

        for _ in range(5):
            # Mask already selected
            sample_probs = adjusted_probs.copy()
            for idx in used_indices:
                sample_probs[idx] = 0

            # Check salary constraint
            for i, p in enumerate(flex_players):
                if p.salary > remaining_cap:
                    sample_probs[i] = 0

            if sample_probs.sum() == 0:
                break

            sample_probs = sample_probs / sample_probs.sum()

            # Sample
            flex_idx = rng.choice(len(flex_players), p=sample_probs)
            flex_selected.append(flex_idx)
            used_indices.add(flex_idx)
            remaining_cap -= flex_players[flex_idx].salary

        if len(flex_selected) == 5:
            # Verify both teams represented + enforce team split when possible
            lineup_teams = {cpt.team}
            cpt_team_flex = 0
            opp_team_flex = 0
            for i in flex_selected:
                team = flex_players[i].team
                lineup_teams.add(team)
                if team == cpt_team:
                    cpt_team_flex += 1
                elif team == opp_team:
                    opp_team_flex += 1

            if len(lineup_teams) < 2 and len(teams) >= 2:
                continue  # Invalid - need both teams

            if len(teams) == 2:
                if cpt_team_flex != cpt_team_count or opp_team_flex != opp_team_count:
                    continue

            flex_salary = sum(flex_players[i].salary for i in flex_selected)
            return ShowdownLineup(
                cpt_player_idx=cpt_idx,
                flex_player_idxs=sorted(flex_selected),
                salary=cpt.salary + flex_salary
            )

    return None


def _sample_team_split(
    config: FieldGenConfig,
    rng: Optional[np.random.Generator] = None
) -> Tuple[int, int]:
    """
    Sample team split (CPT team count, opponent team count).

    Split format: '5-1' means 5 from CPT team (including CPT), 1 from opponent.
    Since CPT counts as 1, FLEX splits are actually (split[0]-1, split[1]).
    """
    if rng is None:
        rng = np.random.default_rng()

    splits = list(config.split_priors.keys())
    probs = np.array([config.split_priors[s] for s in splits])
    probs = probs / probs.sum()

    chosen = rng.choice(splits, p=probs)
    parts = chosen.split('-')

    cpt_team_total = int(parts[0])  # Includes CPT
    opp_team_total = int(parts[1])

    # FLEX counts (excluding CPT)
    return (cpt_team_total - 1, opp_team_total)


def generate_field_from_lineups(
    all_lineups: List[ShowdownLineup],
    cpt_players: List[ShowdownPlayer],
    flex_players: List[ShowdownPlayer],
    n_field: int,
    config: Optional[FieldGenConfig] = None,
    seed: Optional[int] = None
) -> Tuple[LineupArrays, np.ndarray]:
    """
    Generate field by sampling from pre-enumerated lineups.

    Weights by lineup ownership (product of player ownerships).
    """
    rng = np.random.default_rng(seed)

    if config is None:
        config = FieldGenConfig()

    if not all_lineups:
        return LineupArrays.from_lineups([]), np.array([], dtype=np.int32)

    # Compute lineup weights
    weights = np.zeros(len(all_lineups), dtype=np.float64)

    for i, lineup in enumerate(all_lineups):
        cpt_own = cpt_players[lineup.cpt_player_idx].ownership
        flex_own = np.prod([flex_players[j].ownership for j in lineup.flex_player_idxs])
        weights[i] = cpt_own * flex_own

    # Temperature scaling
    weights = weights ** (1.0 / config.temperature)
    weights = weights / weights.sum()

    # Sample lineups
    indices = rng.choice(len(all_lineups), size=n_field, p=weights)

    # Count occurrences
    lineup_counts = Counter(indices)

    # Build output
    unique_lineups = [all_lineups[i] for i in lineup_counts.keys()]
    counts = list(lineup_counts.values())

    field_arrays = LineupArrays.from_lineups(unique_lineups)
    field_counts = np.array(counts, dtype=np.int32)

    return field_arrays, field_counts


def _compute_quality_ownership_weights(
    candidate_arrays: LineupArrays,
    outcomes: np.ndarray,
    cpt_players: List[ShowdownPlayer],
    flex_players: List[ShowdownPlayer],
    field_sharpness: float = 5.0,
    ownership_power: float = 0.5,
    quality_sims: int = 1000,
    seed=None
) -> np.ndarray:
    """
    Compute quality × ownership weights for candidate lineups.

    Uses z-score softmax for quality: weight ∝ exp(sharpness × z_quality) × ownership^power.
    This gives exponential (Boltzmann) discrimination where sharpness controls how
    much better lineups dominate sampling, independent of score scale.

    Args:
        candidate_arrays: All enumerated candidate lineups
        outcomes: [n_players, n_sims] simulated outcomes
        cpt_players: CPT player pool
        flex_players: FLEX player pool
        field_sharpness: Quality discrimination (0=flat, 5=realistic, 8+=optimizer-heavy)
        ownership_power: Ownership influence (0=none, 0.5=moderate, 1.0=full)
        quality_sims: Number of sims to use for quality scoring
        seed: Random seed

    Returns:
        weights: [n_candidates] unnormalized weights
    """
    n_candidates = len(candidate_arrays)
    n_sims = outcomes.shape[1]

    # Subsample simulations for quality scoring
    if quality_sims < n_sims:
        rng = np.random.default_rng(seed)
        sim_indices = rng.choice(n_sims, size=quality_sims, replace=False)
    else:
        sim_indices = np.arange(n_sims)
        quality_sims = n_sims

    # Score all candidates across quality_sims simulations
    score_sum = np.zeros(n_candidates, dtype=np.float64)
    for sim_idx in sim_indices:
        sim_outcomes = outcomes[:, sim_idx]
        scores = score_lineups_vectorized(candidate_arrays, sim_outcomes)
        score_sum += scores.astype(np.float64)

    mean_scores = score_sum / quality_sims

    # Z-score normalize quality for scale-independent weighting
    score_mean = mean_scores.mean()
    score_std = mean_scores.std()
    if score_std > 0:
        z_quality = (mean_scores - score_mean) / score_std
    else:
        z_quality = np.zeros(n_candidates, dtype=np.float64)

    # Compute ownership product per lineup (in log space for stability)
    cpt_log_own = np.log(np.maximum(
        np.array([p.ownership for p in cpt_players], dtype=np.float64),
        0.01
    ))
    flex_log_own = np.log(np.maximum(
        np.array([p.ownership for p in flex_players], dtype=np.float64),
        0.01
    ))

    lineup_log_own = (
        cpt_log_own[candidate_arrays.cpt_idx] +
        flex_log_own[candidate_arrays.flex_idx].sum(axis=1)
    )

    # Combine: log_weight = sharpness * z_quality + ownership_power * log_ownership
    log_weight = field_sharpness * z_quality

    if ownership_power > 0:
        log_weight += ownership_power * lineup_log_own

    # Normalize for numerical stability
    log_weight -= log_weight.max()
    weights = np.exp(log_weight)

    return weights


def generate_field_simulated(
    candidate_arrays: LineupArrays,
    outcomes: np.ndarray,
    cpt_players: List[ShowdownPlayer],
    flex_players: List[ShowdownPlayer],
    n_field: int,
    config: Optional[FieldGenConfig] = None,
    field_sharpness: float = 5.0,
    ownership_power: float = 0.5,
    quality_sims: int = 1000,
    seed=None
) -> Tuple[LineupArrays, np.ndarray]:
    """
    Generate field by sampling from enumerated candidates weighted by quality × ownership.

    Instead of building lineups player-by-player, samples complete lineups from the
    candidate pool. Quality is measured by mean lineup score across simulations.
    This naturally produces realistic salary utilization and field quality.

    Args:
        candidate_arrays: All enumerated candidate lineups
        outcomes: [n_players, n_sims] simulated outcomes
        cpt_players: CPT player pool
        flex_players: FLEX player pool
        n_field: Number of field entries to generate
        config: Field generation config
        field_sharpness: How projection-aware the field is (0=pure ownership, 2+=optimizer)
        ownership_power: How ownership-driven duplication is (0=no ownership, 1=full)
        quality_sims: Number of sims to use for quality scoring
        seed: Random seed

    Returns:
        field_arrays: LineupArrays of unique field lineups
        field_counts: Count of each unique lineup
    """
    rng = np.random.default_rng(seed)

    if config is None:
        config = FieldGenConfig()

    n_candidates = len(candidate_arrays)
    if n_candidates == 0:
        return LineupArrays.from_lineups([]), np.array([], dtype=np.int32)

    # Compute weights
    weights = _compute_quality_ownership_weights(
        candidate_arrays, outcomes,
        cpt_players, flex_players,
        field_sharpness=field_sharpness,
        ownership_power=ownership_power,
        quality_sims=quality_sims,
        seed=rng.integers(0, 2**31)
    )

    # Normalize to probabilities
    probs = weights / weights.sum()

    # Sample n_field lineups with replacement
    indices = rng.choice(n_candidates, size=n_field, p=probs, replace=True)

    # Deduplicate
    lineup_counts = Counter(indices)

    # Build output arrays from the candidate pool subset
    unique_indices = sorted(lineup_counts.keys())
    counts = [lineup_counts[i] for i in unique_indices]
    unique_indices_arr = np.array(unique_indices, dtype=np.int32)

    field_arrays = LineupArrays(
        cpt_idx=candidate_arrays.cpt_idx[unique_indices_arr].copy(),
        flex_idx=candidate_arrays.flex_idx[unique_indices_arr].copy(),
        salary=candidate_arrays.salary[unique_indices_arr].copy()
    )
    field_counts = np.array(counts, dtype=np.int32)

    # Log field diagnostics
    total_field = int(field_counts.sum())
    avg_salary = float(np.average(field_arrays.salary, weights=field_counts))
    high_salary_mask = field_arrays.salary >= 49000
    high_salary_pct = float(
        field_counts[high_salary_mask].sum() / total_field * 100
    ) if total_field > 0 else 0.0

    logger.info(
        "Simulated field: %d unique, %d total, avg salary=$%d, "
        ">=49K: %.1f%%",
        len(field_arrays), total_field, int(avg_salary), high_salary_pct
    )

    return field_arrays, field_counts


def compute_field_lineup_probabilities(
    candidate_arrays: LineupArrays,
    outcomes: np.ndarray,
    cpt_players: List[ShowdownPlayer],
    flex_players: List[ShowdownPlayer],
    config: Optional[FieldGenConfig] = None,
    field_sharpness: float = 5.0,
    ownership_power: float = 0.5,
    quality_sims: int = 1000,
    seed=None
) -> np.ndarray:
    """
    Compute quality × ownership probability distribution over candidates.

    For resample_per_sim mode — same quality × ownership weighting as
    generate_field_simulated() but returns p_lineup instead of sampling.

    Args:
        candidate_arrays: All enumerated candidate lineups
        outcomes: [n_players, n_sims] simulated outcomes
        cpt_players: CPT player pool
        flex_players: FLEX player pool
        config: Field generation config
        field_sharpness: How projection-aware the field is
        ownership_power: How ownership-driven duplication is
        quality_sims: Number of sims to use for quality scoring
        seed: Random seed

    Returns:
        p_lineup: [n_candidates] probabilities summing to 1.0
    """
    if config is None:
        config = FieldGenConfig()

    n_candidates = len(candidate_arrays)
    if n_candidates == 0:
        return np.array([], dtype=np.float64)

    weights = _compute_quality_ownership_weights(
        candidate_arrays, outcomes,
        cpt_players, flex_players,
        field_sharpness=field_sharpness,
        ownership_power=ownership_power,
        quality_sims=quality_sims,
        seed=seed
    )

    p_lineup = weights / weights.sum()
    return p_lineup
