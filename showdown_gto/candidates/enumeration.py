"""
Lineup enumeration for DraftKings Showdown.

Generates all valid lineups (1 CPT + 5 FLEX) within salary constraints.
"""

import itertools
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

from ..types import ShowdownPlayer, ShowdownLineup, LineupArrays

logger = logging.getLogger(__name__)


def enumerate_lineups(
    cpt_players: List[ShowdownPlayer],
    flex_players: List[ShowdownPlayer],
    salary_cap: int = 50000,
    cpt_to_flex_map: Optional[Dict[int, int]] = None,
    max_lineups: Optional[int] = None,
    min_salary: Optional[int] = None,
    require_both_teams: bool = True
) -> List[ShowdownLineup]:
    """
    Enumerate all valid showdown lineups.

    Constraints:
    - Exactly 1 CPT (captain)
    - Exactly 5 FLEX
    - Total salary <= salary_cap
    - CPT player cannot appear in FLEX (no same player twice)
    - Must have at least 1 player from each team (if require_both_teams=True)

    Args:
        cpt_players: Pool of players eligible for CPT slot
        flex_players: Pool of players eligible for FLEX slots
        salary_cap: Maximum total salary (default $50,000)
        cpt_to_flex_map: Maps cpt_idx -> flex_idx for same player
        max_lineups: Optional limit on number of lineups to generate
        min_salary: Optional minimum salary to use
        require_both_teams: If True, lineup must have players from both teams

    Returns:
        List of valid ShowdownLineup objects
    """
    if cpt_to_flex_map is None:
        cpt_to_flex_map = {}

    lineups = []
    n_flex = len(flex_players)

    # Pre-compute flex salaries for pruning
    flex_salaries = np.array([p.salary for p in flex_players], dtype=np.int32)

    # Sort flex by salary ascending for better pruning
    flex_sorted_idx = np.argsort(flex_salaries)
    flex_salaries_sorted = flex_salaries[flex_sorted_idx]

    # Minimum salary for 5 cheapest flex
    min_5_flex_salary = int(flex_salaries_sorted[:5].sum()) if n_flex >= 5 else 0

    # Get unique teams for validation
    all_teams = set(p.team for p in cpt_players) | set(p.team for p in flex_players)

    for cpt_idx, cpt in enumerate(cpt_players):
        # Check if this CPT can possibly form a valid lineup
        remaining_cap = salary_cap - cpt.salary
        if remaining_cap < min_5_flex_salary:
            continue

        # Get excluded flex index (same player as CPT)
        excluded_flex_idx = cpt_to_flex_map.get(cpt_idx)

        # Build valid flex indices (excluding same player)
        valid_flex_indices = [i for i in range(n_flex) if i != excluded_flex_idx]

        # Generate 5-combinations
        for flex_combo in itertools.combinations(valid_flex_indices, 5):
            flex_salary = sum(flex_players[i].salary for i in flex_combo)
            total_salary = cpt.salary + flex_salary

            if total_salary > salary_cap:
                continue

            if min_salary is not None and total_salary < min_salary:
                continue

            # Check both teams represented
            if require_both_teams and len(all_teams) >= 2:
                lineup_teams = {cpt.team}
                for i in flex_combo:
                    lineup_teams.add(flex_players[i].team)

                if len(lineup_teams) < 2:
                    continue  # Skip - must have players from both teams

            lineups.append(ShowdownLineup(
                cpt_player_idx=cpt_idx,
                flex_player_idxs=sorted(flex_combo),
                salary=total_salary
            ))

            if max_lineups is not None and len(lineups) >= max_lineups:
                logger.info(f"Reached max_lineups limit: {max_lineups}")
                return lineups

    logger.info(f"Enumerated {len(lineups)} valid lineups")
    return lineups


def enumerate_lineups_optimized(
    cpt_players: List[ShowdownPlayer],
    flex_players: List[ShowdownPlayer],
    salary_cap: int = 50000,
    cpt_to_flex_map: Optional[Dict[int, int]] = None,
    max_lineups: Optional[int] = None
) -> List[ShowdownLineup]:
    """
    Optimized enumeration with salary-based pruning.

    Uses branch-and-bound style pruning to skip infeasible combinations early.
    """
    if cpt_to_flex_map is None:
        cpt_to_flex_map = {}

    lineups = []
    n_flex = len(flex_players)

    if n_flex < 5:
        logger.warning(f"Not enough FLEX players ({n_flex}) to form lineups")
        return lineups

    # Sort flex by salary for pruning
    flex_with_idx = [(i, flex_players[i].salary) for i in range(n_flex)]
    flex_with_idx.sort(key=lambda x: x[1])

    # Pre-compute minimum salary to complete lineup from position k
    # min_to_complete[k] = min salary for (5-k) cheapest remaining players
    min_to_complete = np.zeros(6, dtype=np.int32)
    for k in range(5):
        remaining_needed = 5 - k
        if remaining_needed > 0:
            min_to_complete[k] = sum(s for _, s in flex_with_idx[:remaining_needed])

    for cpt_idx, cpt in enumerate(cpt_players):
        remaining_cap = salary_cap - cpt.salary
        excluded_flex_idx = cpt_to_flex_map.get(cpt_idx)

        # Filter and sort valid flex for this CPT
        valid_flex = [(i, flex_players[i].salary)
                      for i in range(n_flex)
                      if i != excluded_flex_idx]
        valid_flex.sort(key=lambda x: x[1])

        n_valid = len(valid_flex)
        if n_valid < 5:
            continue

        # Recursive enumeration with pruning
        _enumerate_recursive(
            valid_flex=valid_flex,
            remaining_cap=remaining_cap,
            current_combo=[],
            start_idx=0,
            cpt_idx=cpt_idx,
            cpt_salary=cpt.salary,
            lineups=lineups,
            max_lineups=max_lineups
        )

        if max_lineups is not None and len(lineups) >= max_lineups:
            return lineups

    logger.info(f"Enumerated {len(lineups)} valid lineups (optimized)")
    return lineups


def _enumerate_recursive(
    valid_flex: List[Tuple[int, int]],
    remaining_cap: int,
    current_combo: List[int],
    start_idx: int,
    cpt_idx: int,
    cpt_salary: int,
    lineups: List[ShowdownLineup],
    max_lineups: Optional[int]
) -> bool:
    """
    Recursive helper for optimized enumeration.

    Returns True if should stop (max_lineups reached).
    """
    needed = 5 - len(current_combo)

    if needed == 0:
        # Complete lineup
        total_salary = cpt_salary + sum(s for _, s in
                                         [(i, valid_flex[j][1]) for j, (i, _) in enumerate(valid_flex)
                                          if i in [valid_flex[k][0] for k in current_combo]])
        # Recalculate properly
        flex_salary = sum(valid_flex[k][1] for k in current_combo)
        lineups.append(ShowdownLineup(
            cpt_player_idx=cpt_idx,
            flex_player_idxs=sorted([valid_flex[k][0] for k in current_combo]),
            salary=cpt_salary + flex_salary
        ))
        return max_lineups is not None and len(lineups) >= max_lineups

    n_valid = len(valid_flex)

    for i in range(start_idx, n_valid - needed + 1):
        flex_idx, flex_sal = valid_flex[i]

        # Pruning: check if we can complete with remaining budget
        if flex_sal > remaining_cap:
            break  # Sorted by salary, so no point continuing

        # Check if minimum possible completion fits
        # Minimum = current + (needed-1) cheapest remaining
        if needed > 1:
            min_remaining = sum(valid_flex[j][1] for j in range(i + 1, i + needed))
            if i + needed > n_valid:
                break
            if flex_sal + min_remaining > remaining_cap:
                continue

        current_combo.append(i)
        if _enumerate_recursive(
            valid_flex, remaining_cap - flex_sal, current_combo, i + 1,
            cpt_idx, cpt_salary, lineups, max_lineups
        ):
            return True
        current_combo.pop()

    return False


def filter_lineups_by_projection(
    lineups: List[ShowdownLineup],
    cpt_players: List[ShowdownPlayer],
    flex_players: List[ShowdownPlayer],
    min_projection: float
) -> List[ShowdownLineup]:
    """
    Filter lineups by total projected points.
    """
    filtered = []
    for lineup in lineups:
        cpt_proj = cpt_players[lineup.cpt_player_idx].projection * 1.5
        flex_proj = sum(flex_players[i].projection for i in lineup.flex_player_idxs)
        total_proj = cpt_proj + flex_proj

        if total_proj >= min_projection:
            filtered.append(lineup)

    return filtered


def get_lineup_players(
    lineup: ShowdownLineup,
    cpt_players: List[ShowdownPlayer],
    flex_players: List[ShowdownPlayer]
) -> Tuple[ShowdownPlayer, List[ShowdownPlayer]]:
    """Get actual player objects for a lineup."""
    cpt = cpt_players[lineup.cpt_player_idx]
    flex = [flex_players[i] for i in lineup.flex_player_idxs]
    return cpt, flex


def lineup_to_names(
    lineup: ShowdownLineup,
    cpt_players: List[ShowdownPlayer],
    flex_players: List[ShowdownPlayer]
) -> List[str]:
    """Get player names for a lineup as [CPT_name, FLEX1, FLEX2, ...]."""
    cpt, flex = get_lineup_players(lineup, cpt_players, flex_players)
    return [f"{cpt.name} (CPT)"] + [p.name for p in flex]
