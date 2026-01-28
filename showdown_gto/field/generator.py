"""
Field generation for opponent modeling.

Generates a distribution of opponent lineups based on ownership and soft priors.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter
import logging

from ..types import (
    ShowdownPlayer, ShowdownLineup, LineupArrays, FieldGenConfig
)

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
        seed: Random seed

    Returns:
        field_arrays: LineupArrays of unique field lineups
        field_counts: Count of each unique lineup
    """
    if seed is not None:
        np.random.seed(seed)

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
            config, cpt_to_flex_map, salary_cap, teams
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
    max_attempts: int = 100
) -> Optional[ShowdownLineup]:
    """
    Sample a single lineup respecting priors and constraints.

    Uses rejection sampling with ownership-weighted selection.
    """
    for _ in range(max_attempts):
        # Sample CPT
        cpt_idx = np.random.choice(len(cpt_players), p=cpt_probs)
        cpt = cpt_players[cpt_idx]

        # Determine target team split
        split = _sample_team_split(config)
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

        # Apply QB pairing if CPT is QB
        if cpt.position == 'QB' and np.random.random() < config.qb_pair_rate:
            # Boost same-team pass catchers
            for i, p in enumerate(flex_players):
                if p.team == cpt_team and p.position in ('WR', 'TE'):
                    adjusted_probs[i] *= 2.0

        # Apply bring-back preference
        if opp_team and np.random.random() < config.bring_back_rate:
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
            flex_idx = np.random.choice(len(flex_players), p=sample_probs)
            flex_selected.append(flex_idx)
            used_indices.add(flex_idx)
            remaining_cap -= flex_players[flex_idx].salary

        if len(flex_selected) == 5:
            # Verify both teams represented
            lineup_teams = {cpt.team}
            for i in flex_selected:
                lineup_teams.add(flex_players[i].team)

            if len(lineup_teams) < 2 and len(teams) >= 2:
                continue  # Invalid - need both teams

            flex_salary = sum(flex_players[i].salary for i in flex_selected)
            return ShowdownLineup(
                cpt_player_idx=cpt_idx,
                flex_player_idxs=sorted(flex_selected),
                salary=cpt.salary + flex_salary
            )

    return None


def _sample_team_split(config: FieldGenConfig) -> Tuple[int, int]:
    """
    Sample team split (CPT team count, opponent team count).

    Split format: '5-1' means 5 from CPT team (including CPT), 1 from opponent.
    Since CPT counts as 1, FLEX splits are actually (split[0]-1, split[1]).
    """
    splits = list(config.split_priors.keys())
    probs = np.array([config.split_priors[s] for s in splits])
    probs = probs / probs.sum()

    chosen = np.random.choice(splits, p=probs)
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
    if seed is not None:
        np.random.seed(seed)

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
    indices = np.random.choice(len(all_lineups), size=n_field, p=weights)

    # Count occurrences
    lineup_counts = Counter(indices)

    # Build output
    unique_lineups = [all_lineups[i] for i in lineup_counts.keys()]
    counts = list(lineup_counts.values())

    field_arrays = LineupArrays.from_lineups(unique_lineups)
    field_counts = np.array(counts, dtype=np.int32)

    return field_arrays, field_counts
