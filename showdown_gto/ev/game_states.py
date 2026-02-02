"""
Game-state classification and coverage diagnostic.

Gap-free classification with favorite/underdog awareness and vegas spread.
Seven states covering all possible game outcomes.
"""

import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Game State Classification
# =============================================================================

GAME_STATES = [
    "shootout",
    "competitive",
    "blowout_fav",
    "upset",
    "defensive_close",
    "defensive_blowout_fav",
    "defensive_upset",
]


def classify_game_state(
    total_score: float,
    differential: float,
    game_total: float,
    vegas_spread: float
) -> str:
    """
    Gap-free classification with favorite/underdog awareness.

    Seven states covering all game outcomes:
    - shootout: high-scoring (total > game_total + 7)
    - competitive: normal scoring, close game (|diff| <= 7)
    - blowout_fav: normal scoring, favorite dominated
    - upset: normal scoring, underdog won
    - defensive_close: low-scoring, close game
    - defensive_blowout_fav: low-scoring, favorite dominated
    - defensive_upset: low-scoring, underdog won

    Args:
        total_score: Combined score of both teams
        differential: team_A_score - team_B_score (positive = team A won)
        game_total: Vegas total (e.g., 48.5)
        vegas_spread: Vegas spread from team A perspective
                      (negative = team A favored, e.g., -3.5)

    Returns:
        Game state string
    """
    abs_diff = abs(differential)

    # Did the favorite cover?
    # If vegas_spread = -3.5 (team A favored), team A needs diff < spread (more negative)
    # i.e., team A wins by > 3.5 points
    favorite_covered = (differential < vegas_spread) if vegas_spread < 0 else (differential > vegas_spread)

    if total_score > game_total + 7:
        return "shootout"
    elif total_score >= game_total - 3:
        # Normal scoring
        if abs_diff <= 7:
            return "competitive"
        elif favorite_covered:
            return "blowout_fav"
        else:
            return "upset"
    else:
        # Low-scoring
        if abs_diff <= 7:
            return "defensive_close"
        elif favorite_covered:
            return "defensive_blowout_fav"
        else:
            return "defensive_upset"


def classify_sims_by_game_state(
    outcomes: np.ndarray,
    team_indices: Dict[str, list],
    game_total: float = 48.5,
    vegas_spread: float = 0.0,
    spread_team: Optional[str] = None,
) -> np.ndarray:
    """
    Classify each simulation into a game state.

    Args:
        outcomes: [n_players, n_sims] quantized scores (points x 10)
        team_indices: team_name -> list of player indices
        game_total: Vegas game total
        vegas_spread: Spread from spread_team perspective (negative = favored)
        spread_team: Which team the spread applies to (first team if not specified)

    Returns:
        state_labels: [n_sims] array of game state string indices
    """
    teams = list(team_indices.keys())
    if len(teams) < 2:
        return np.full(outcomes.shape[1], 0, dtype=np.int32)

    team_a, team_b = teams[0], teams[1]
    if spread_team is not None and spread_team == team_b:
        team_a, team_b = team_b, team_a

    n_sims = outcomes.shape[1]

    # Compute team total scores per sim (sum of player scores, dequantized)
    team_a_scores = outcomes[team_indices[team_a]].sum(axis=0) / 10.0  # dequantize
    team_b_scores = outcomes[team_indices[team_b]].sum(axis=0) / 10.0

    total_scores = team_a_scores + team_b_scores
    differentials = team_a_scores - team_b_scores

    state_indices = np.zeros(n_sims, dtype=np.int32)
    for sim in range(n_sims):
        state = classify_game_state(
            total_scores[sim], differentials[sim], game_total, vegas_spread
        )
        state_indices[sim] = GAME_STATES.index(state)

    return state_indices


def compute_game_state_coverage(
    state_indices: np.ndarray,
    lineup_ranks: np.ndarray,
    top_n: int = 100,
) -> Dict[str, Dict]:
    """
    Compute game-state coverage diagnostic.

    For each game state, reports:
    - Share of simulations
    - Share of top-N finishes
    - Concentration ratio (top-N share / sim share)

    Args:
        state_indices: [n_sims] game state indices
        lineup_ranks: [n_sims] best rank of any portfolio lineup per sim
        top_n: Top-N threshold for finish share

    Returns:
        Dict of state -> {sim_share, top_n_share, concentration_ratio}
    """
    n_sims = len(state_indices)
    top_mask = lineup_ranks <= top_n

    coverage = {}
    for state_idx, state_name in enumerate(GAME_STATES):
        state_mask = state_indices == state_idx
        n_state = state_mask.sum()
        n_top_in_state = (state_mask & top_mask).sum()

        sim_share = n_state / n_sims if n_sims > 0 else 0
        top_share = n_top_in_state / top_mask.sum() if top_mask.sum() > 0 else 0
        concentration = top_share / sim_share if sim_share > 0 else 0

        coverage[state_name] = {
            'sim_share': float(sim_share),
            'top_n_share': float(top_share),
            'concentration_ratio': float(concentration),
            'n_sims': int(n_state),
            'n_top_finishes': int(n_top_in_state),
        }

    return coverage
