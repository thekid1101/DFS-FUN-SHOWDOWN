"""
Pre-processing player effects layer.

Loads unified_player_effects.json and simulation_config_v3.json,
then applies per-player correlation, projection, and distribution
modifiers before simulation.
"""

import copy
import json
import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

from ..types import ShowdownPlayer

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class GameContext:
    """Vegas-derived game context for conditional correlation modifiers."""
    spread: Dict[str, float]               # team -> spread (guide: negative=underdog)
    game_total: float
    team_implied_totals: Dict[str, float]  # team -> implied points


@dataclass
class RoleConfig:
    """Per-role distribution parameters from simulation_config_v3.json."""
    cv_default: float
    floor_pct: float
    ceiling_pct: float


# ============================================================================
# V2 -> V3 Archetype Fallback Mapping
# ============================================================================

V2_TO_V3_FALLBACK = {
    'DUAL_THREAT_QB': 'SCRAMBLER_QB',
    'POCKET_QB': 'POCKET_QB',
    'BELLCOW_EARLY_RB': 'BELLCOW_EARLY_RB',
    'BELLCOW_RECEIVING_RB': 'BELLCOW_RECEIVING_RB',
    'COMMITTEE_RB': 'TIMESHARE_RB',
    'SATELLITE_RB': 'SATELLITE_RB',
    'ALPHA_OUTSIDE_WR': 'ALPHA_OUTSIDE_WR',
    'DEEP_THREAT_WR': 'DEEP_THREAT_WR',
    'SECONDARY_OUTSIDE_WR': 'ALPHA_OUTSIDE_WR',
    'ALPHA_SLOT_WR': 'SECONDARY_SLOT_WR',
    'SECONDARY_SLOT_WR': 'SECONDARY_SLOT_WR',
    'DEPTH_WR': 'SECONDARY_SLOT_WR',
    'ELITE_TE': 'ELITE_TE',
    'RECEIVING_TE': 'ELITE_TE',
    'BLOCKING_TE': 'BLOCKING_TE',
    'K': None,
    'DST': None,
}


# ============================================================================
# Loading Functions
# ============================================================================

def load_player_effects(path: str) -> Dict[str, Dict[str, Any]]:
    """Load unified_player_effects.json. Returns the player_effects dict."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data.get('player_effects', {})


def load_role_configs(path: str) -> Dict[str, RoleConfig]:
    """Load simulation_config_v3.json, return archetype -> RoleConfig."""
    with open(path, 'r') as f:
        data = json.load(f)

    configs = {}
    for role, params in data.get('distribution_parameters', {}).items():
        configs[role] = RoleConfig(
            cv_default=params.get('cv_default', 0.35),
            floor_pct=params.get('floor_pct', 0.20),
            ceiling_pct=params.get('ceiling_pct', 2.50),
        )
    return configs


def parse_game_context(
    spread_str: Optional[str],
    game_total: Optional[float],
    teams: List[str]
) -> Optional[GameContext]:
    """
    Parse CLI spread and game total into a GameContext.

    Args:
        spread_str: e.g. "LAR -3.5" (standard Vegas: negative = favorite)
        game_total: e.g. 48.5
        teams: The two teams from the slate

    Returns:
        GameContext or None if neither spread nor game_total provided.
    """
    if spread_str is None and game_total is None:
        return None

    if game_total is None:
        game_total = 45.0  # Default if not provided

    spread_dict = {}
    if spread_str is not None:
        parts = spread_str.strip().split()
        if len(parts) >= 2:
            team_a = parts[0].upper()
            try:
                vegas_spread = float(parts[1])
            except ValueError:
                logger.warning("Could not parse spread value: %s", spread_str)
                vegas_spread = 0.0

            # Convert Vegas convention (negative=favorite) to guide convention
            # (negative=underdog). Guide: spread > 0 means favorite.
            # Vegas: -3.5 means favorite -> guide: +3.5
            guide_spread_a = -vegas_spread
            spread_dict[team_a] = guide_spread_a

            # Find the other team
            other_teams = [t for t in teams if t.upper() != team_a]
            if other_teams:
                team_b = other_teams[0]
                spread_dict[team_b] = -guide_spread_a
        else:
            logger.warning("Spread format should be 'TEAM VALUE' (e.g., 'LAR -3.5')")

    # If only game_total provided with no spread, use 0 spread for all teams
    if not spread_dict:
        for team in teams:
            spread_dict[team] = 0.0

    # Compute implied totals: team_total = (game_total - guide_spread) / 2
    # guide_spread > 0 means favorite -> higher implied total
    implied = {}
    for team, sp in spread_dict.items():
        implied[team] = (game_total + sp) / 2.0

    return GameContext(
        spread=spread_dict,
        game_total=game_total,
        team_implied_totals=implied
    )


# ============================================================================
# Helper Functions: Find Special Player Indices
# ============================================================================

def _find_team_qb_index(
    players: List[ShowdownPlayer],
    team: str
) -> Optional[int]:
    """Find the QB index for a team. Returns highest-projection QB if multiple."""
    best_idx = None
    best_proj = -1.0
    for idx, p in enumerate(players):
        if p.team == team and p.position.upper() == 'QB':
            if p.projection > best_proj:
                best_proj = p.projection
                best_idx = idx
    return best_idx


def _find_team_dst_index(
    players: List[ShowdownPlayer],
    team: str
) -> Optional[int]:
    """Find the DST index for a team."""
    for idx, p in enumerate(players):
        if p.team == team and p.position.upper() in ('DST', 'DEF'):
            return idx
    return None


def _find_team_rb1_index(
    players: List[ShowdownPlayer],
    team: str
) -> Optional[int]:
    """Find the RB1 (highest-projection RB) for a team."""
    best_idx = None
    best_proj = -1.0
    for idx, p in enumerate(players):
        if p.team == team and p.position.upper() == 'RB':
            if p.projection > best_proj:
                best_proj = p.projection
                best_idx = idx
    return best_idx


def _get_opponent_team(player: ShowdownPlayer, teams: List[str]) -> Optional[str]:
    """Get the opposing team for a player."""
    for t in teams:
        if t != player.team:
            return t
    return None


# ============================================================================
# Distribution Effects
# ============================================================================

def _resolve_role_config(
    player: ShowdownPlayer,
    archetype_map: Optional[Dict[str, str]],
    role_configs: Optional[Dict[str, RoleConfig]]
) -> Optional[RoleConfig]:
    """Resolve which RoleConfig applies to a player via V2->V3 mapping."""
    if role_configs is None:
        return None

    # Get V2 archetype
    arch = None
    if archetype_map:
        arch = archetype_map.get(player.name)

    if arch is None:
        # Infer from position
        from .correlations import _infer_archetype
        arch = _infer_archetype(player)

    # Try direct V3 lookup
    if arch in role_configs:
        return role_configs[arch]

    # Try V2->V3 fallback
    v3_arch = V2_TO_V3_FALLBACK.get(arch)
    if v3_arch and v3_arch in role_configs:
        return role_configs[v3_arch]

    return None


def _apply_projection_modifiers(
    player: ShowdownPlayer,
    effects: Dict[str, Any]
) -> None:
    """Scale projection and all percentiles by boost/penalty. Modifies in-place."""
    boost = effects.get('projection_boost_pct', 0.0)
    penalty = effects.get('projection_penalty_pct', 0.0)
    # Skip projection_adjustment (TD regression) per user decision

    combined = boost + penalty
    if abs(combined) < 1e-10:
        return

    # Clamp to prevent extreme distortions
    scale = max(0.5, min(1.5, 1.0 + combined))

    player.projection *= scale
    player.std *= scale
    for pct in player.percentiles:
        player.percentiles[pct] *= scale


def _apply_cv_boost(
    player: ShowdownPlayer,
    effects: Dict[str, Any],
    role_config: Optional[RoleConfig]
) -> None:
    """Widen/narrow percentile spread around median. Modifies in-place."""
    cv_boost = effects.get('cv_boost', 0.0)
    if abs(cv_boost) < 1e-10:
        return

    if player.projection <= 0:
        return

    p50 = player.percentiles.get(50, 0.0)
    if p50 <= 0:
        return

    # Determine base CV
    if role_config is not None:
        cv_base = role_config.cv_default
    elif player.std > 0:
        cv_base = player.std / player.projection
    else:
        return

    if cv_base <= 0:
        return

    cv_new = cv_base + cv_boost
    cv_scale = cv_new / cv_base

    # Scale each percentile's distance from the median
    for pct in player.percentiles:
        old_val = player.percentiles[pct]
        player.percentiles[pct] = p50 + (old_val - p50) * cv_scale

    # Update std
    player.std = player.projection * cv_new


def _apply_floor_ceiling(
    player: ShowdownPlayer,
    effects: Dict[str, Any],
    role_config: Optional[RoleConfig]
) -> None:
    """Clamp percentiles to role-adjusted floor/ceiling. Modifies in-place."""
    if role_config is None:
        return

    if player.projection <= 0:
        return

    floor_boost = effects.get('floor_boost_pct', 0.0)
    floor_penalty = effects.get('floor_penalty_pct', 0.0)
    ceiling_boost = effects.get('ceiling_boost_pct', 0.0)
    ceiling_penalty = effects.get('ceiling_penalty_pct', 0.0)

    # Check if any adjustments needed
    if (abs(floor_boost) < 1e-10 and abs(floor_penalty) < 1e-10
            and abs(ceiling_boost) < 1e-10 and abs(ceiling_penalty) < 1e-10):
        return

    floor = player.projection * role_config.floor_pct * (1.0 + floor_boost + floor_penalty)
    ceiling = player.projection * role_config.ceiling_pct * (1.0 + ceiling_boost + ceiling_penalty)

    floor = max(floor, 0.0)
    ceiling = max(ceiling, floor)

    # Clamp percentile values
    for pct in player.percentiles:
        player.percentiles[pct] = max(floor, min(ceiling, player.percentiles[pct]))

    # Enforce monotonicity
    _enforce_percentile_monotonicity(player)


def _enforce_percentile_monotonicity(player: ShowdownPlayer) -> None:
    """Ensure percentile values are non-decreasing."""
    ordered_pcts = [25, 50, 75, 85, 95, 99]
    for i in range(1, len(ordered_pcts)):
        prev = ordered_pcts[i - 1]
        curr = ordered_pcts[i]
        if prev in player.percentiles and curr in player.percentiles:
            if player.percentiles[curr] < player.percentiles[prev]:
                player.percentiles[curr] = player.percentiles[prev]


def apply_distribution_effects(
    player: ShowdownPlayer,
    effects: Dict[str, Any],
    role_config: Optional[RoleConfig]
) -> ShowdownPlayer:
    """
    Apply all distribution modifiers in correct order.
    Returns a modified copy of the player.
    """
    player = copy.deepcopy(player)
    _apply_projection_modifiers(player, effects)
    _apply_cv_boost(player, effects, role_config)
    _apply_floor_ceiling(player, effects, role_config)
    return player


# ============================================================================
# Correlation Effects
# ============================================================================

def _apply_always_on_correlation_modifiers(
    matrix: np.ndarray,
    players: List[ShowdownPlayer],
    player_effects: Dict[str, Dict[str, Any]],
    teams: List[str]
) -> int:
    """
    Apply always-on correlation modifiers to the NxN matrix.
    Returns count of modifiers applied.
    """
    n_applied = 0

    # Pre-compute special indices per team
    team_qb = {}
    team_dst = {}
    team_rb1 = {}
    for team in teams:
        team_qb[team] = _find_team_qb_index(players, team)
        team_dst[team] = _find_team_dst_index(players, team)
        team_rb1[team] = _find_team_rb1_index(players, team)

    for i, player in enumerate(players):
        name = player.name
        if name not in player_effects:
            continue

        e = player_effects[name]
        team = player.team
        opp_team = _get_opponent_team(player, teams)

        # QB correlation boost + reduction (net together)
        qb_idx = team_qb.get(team)
        if qb_idx is not None and qb_idx != i:
            net_qb = (e.get('correlation_boost_qb', 0.0)
                       + e.get('correlation_reduction_qb', 0.0))
            if abs(net_qb) > 1e-10:
                matrix[i, qb_idx] += net_qb
                matrix[qb_idx, i] += net_qb
                n_applied += 1

        # Opposing QB correlation (game stacks / bring-backs)
        if opp_team and 'correlation_boost_opp_qb' in e:
            opp_qb_idx = team_qb.get(opp_team)
            if opp_qb_idx is not None:
                val = e['correlation_boost_opp_qb']
                matrix[i, opp_qb_idx] += val
                matrix[opp_qb_idx, i] += val
                n_applied += 1

        # DST correlation (with opposing team's DST)
        if opp_team and 'correlation_boost_dst' in e:
            dst_idx = team_dst.get(opp_team)
            if dst_idx is not None:
                val = e['correlation_boost_dst']
                matrix[i, dst_idx] += val
                matrix[dst_idx, i] += val
                n_applied += 1

        # Goal-line RB cannibalization (penalty with same-team RBs)
        if 'correlation_penalty_glrb' in e:
            val = e['correlation_penalty_glrb']
            for j, other in enumerate(players):
                if other.team == team and other.position.upper() == 'RB' and j != i:
                    matrix[i, j] += val
                    matrix[j, i] += val
                    n_applied += 1

        # Unusual WR<->RB1 correlation
        if 'correlation_boost_rb1' in e:
            rb1_idx = team_rb1.get(team)
            if rb1_idx is not None and rb1_idx != i:
                val = e['correlation_boost_rb1']
                matrix[i, rb1_idx] += val
                matrix[rb1_idx, i] += val
                n_applied += 1

    return n_applied


def _apply_conditional_correlation_modifiers(
    matrix: np.ndarray,
    players: List[ShowdownPlayer],
    player_effects: Dict[str, Dict[str, Any]],
    teams: List[str],
    game_context: GameContext
) -> int:
    """
    Apply game-context-dependent correlation modifiers.
    Returns count of modifiers applied.
    """
    n_applied = 0

    # Pre-compute QB indices
    team_qb = {}
    for team in teams:
        team_qb[team] = _find_team_qb_index(players, team)

    for i, player in enumerate(players):
        name = player.name
        if name not in player_effects:
            continue

        e = player_effects[name]
        team = player.team
        qb_idx = team_qb.get(team)

        if qb_idx is None or qb_idx == i:
            continue

        spread = game_context.spread.get(team, 0.0)
        implied = game_context.team_implied_totals.get(team, 24.0)

        # Trailing modifiers (guide: spread > 0 means underdog = trailing)
        if spread > 0:
            spread_scale = min(abs(spread) / 7.0, 1.0)
            trailing = max(
                e.get('correlation_boost_trailing_script', 0.0),
                e.get('correlation_boost_trailing', 0.0)
            )
            if trailing > 0:
                val = trailing * spread_scale
                matrix[i, qb_idx] += val
                matrix[qb_idx, i] += val
                n_applied += 1

        # Winning/leading modifiers (guide: spread < 0 means favorite = leading)
        if spread < 0:
            spread_scale = min(abs(spread) / 7.0, 1.0)
            winning = max(
                e.get('correlation_boost_winning_script', 0.0),
                e.get('correlation_boost_leading_script', 0.0)
            )
            if winning > 0:
                val = winning * spread_scale
                matrix[i, qb_idx] += val
                matrix[qb_idx, i] += val
                n_applied += 1

        # Shootout modifier
        if game_context.game_total >= 44.0:
            total_scale = min((game_context.game_total - 44.0) / 8.0, 1.0)
            shootout = e.get('correlation_boost_shootout', 0.0)
            if shootout > 0 and total_scale > 0:
                val = shootout * total_scale
                matrix[i, qb_idx] += val
                matrix[qb_idx, i] += val
                n_applied += 1

        # Volume modifiers
        if implied >= 27.0:
            volume = max(
                e.get('correlation_boost_qb_volume', 0.0),
                e.get('correlation_boost_pass_volume', 0.0)
            )
            if volume > 0:
                matrix[i, qb_idx] += volume
                matrix[qb_idx, i] += volume
                n_applied += 1

    return n_applied


def _validate_correlation_matrix(matrix: np.ndarray) -> np.ndarray:
    """Clamp, enforce symmetry, set diagonal to 1. PSD handled by engine."""
    # Clamp off-diagonal
    np.clip(matrix, -0.95, 0.95, out=matrix)
    np.fill_diagonal(matrix, 1.0)

    # Enforce symmetry
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 1.0)

    return matrix


# ============================================================================
# CPT Sync
# ============================================================================

def sync_cpt_from_flex(
    cpt_players: List[ShowdownPlayer],
    flex_players: List[ShowdownPlayer],
    cpt_to_flex_map: Dict[int, int]
) -> List[ShowdownPlayer]:
    """
    After modifying FLEX players, sync CPT projection values.
    Returns a new list of CPT players with updated projections.
    """
    new_cpt = []
    for cpt_idx, cpt in enumerate(cpt_players):
        flex_idx = cpt_to_flex_map.get(cpt_idx)
        if flex_idx is not None and flex_idx < len(flex_players):
            cpt = copy.deepcopy(cpt)
            cpt.projection = flex_players[flex_idx].projection * 1.5
        new_cpt.append(cpt)
    return new_cpt


# ============================================================================
# Top-Level Orchestrator
# ============================================================================

def apply_player_effects(
    players: List[ShowdownPlayer],
    correlation_matrix: Optional[np.ndarray],
    effects_path: str,
    sim_config_path: Optional[str] = None,
    archetype_map: Optional[Dict[str, str]] = None,
    game_context: Optional[GameContext] = None
) -> Tuple[List[ShowdownPlayer], Optional[np.ndarray]]:
    """
    Apply all player effects from unified_player_effects.json.

    Args:
        players: FLEX player list
        correlation_matrix: NxN base correlation matrix (or None)
        effects_path: Path to unified_player_effects.json
        sim_config_path: Optional path to simulation_config_v3.json
        archetype_map: player name -> V2 archetype
        game_context: Optional game context for conditional modifiers

    Returns:
        (modified_players, modified_correlation_matrix)
    """
    # Load effects
    player_effects = load_player_effects(effects_path)
    logger.info("Loaded %d player effects entries", len(player_effects))

    # Load role configs (optional)
    role_configs = None
    if sim_config_path is not None:
        role_configs = load_role_configs(sim_config_path)
        logger.info("Loaded %d role configs from %s", len(role_configs), sim_config_path)

    # Get teams from players
    teams = list(set(p.team for p in players if p.team))

    # === DISTRIBUTION EFFECTS ===
    n_dist_applied = 0
    modified_players = []
    for player in players:
        if player.name in player_effects and player.projection > 0:
            e = player_effects[player.name]
            role_config = _resolve_role_config(player, archetype_map, role_configs)
            modified = apply_distribution_effects(player, e, role_config)
            modified_players.append(modified)
            n_dist_applied += 1
        else:
            modified_players.append(copy.deepcopy(player))

    logger.info("Applied distribution effects to %d/%d players", n_dist_applied, len(players))

    # === CORRELATION EFFECTS ===
    if correlation_matrix is not None:
        matrix = correlation_matrix.copy()

        n_always = _apply_always_on_correlation_modifiers(
            matrix, modified_players, player_effects, teams
        )
        logger.info("Applied %d always-on correlation modifiers", n_always)

        if game_context is not None:
            n_cond = _apply_conditional_correlation_modifiers(
                matrix, modified_players, player_effects, teams, game_context
            )
            logger.info(
                "Applied %d conditional correlation modifiers "
                "(spread=%s, total=%.1f)",
                n_cond,
                {t: f"{s:+.1f}" for t, s in game_context.spread.items()},
                game_context.game_total
            )

        matrix = _validate_correlation_matrix(matrix)
        logger.info(
            "Correlation matrix after effects: min=%.3f, max=%.3f (off-diagonal)",
            np.min(matrix[~np.eye(matrix.shape[0], dtype=bool)]),
            np.max(matrix[~np.eye(matrix.shape[0], dtype=bool)])
        )
    else:
        matrix = None
        if any(
            k.startswith('correlation_')
            for name in player_effects
            for k in player_effects[name]
            if isinstance(player_effects[name].get(k), (int, float))
        ):
            logger.warning(
                "Player effects contain correlation modifiers but no "
                "correlation matrix was provided. Use --correlation-config "
                "to enable correlation effects."
            )

    return modified_players, matrix
