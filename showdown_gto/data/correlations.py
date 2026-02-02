"""
Correlation matrix handling for player outcome simulation.

Supports archetype-based correlations with same-team and opponent relationships.
"""

import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from ..types import ShowdownPlayer


# =============================================================================
# Archetype Definitions
# =============================================================================

PLAYER_ARCHETYPES = [
    'DUAL_THREAT_QB', 'POCKET_QB',
    'BELLCOW_RECEIVING_RB', 'COMMITTEE_RB', 'BELLCOW_EARLY_RB', 'SATELLITE_RB',
    'DEPTH_WR', 'ALPHA_OUTSIDE_WR', 'DEEP_THREAT_WR', 'SECONDARY_OUTSIDE_WR',
    'ALPHA_SLOT_WR', 'SECONDARY_SLOT_WR',
    'ELITE_TE', 'BLOCKING_TE', 'RECEIVING_TE',
    'K', 'DST'
]


@dataclass
class ArchetypeCorrelationConfig:
    """
    Archetype-based correlation configuration.

    Loaded from correlation_config_v2.json format.
    """
    same_team: Dict[str, Dict[str, float]]  # archetype -> archetype -> correlation
    opponent: Dict[str, Dict[str, float]]   # archetype -> archetype -> correlation
    variance_decomposition: Dict[str, Dict[str, float]]  # archetype -> {game_share, player_share, ...}
    metadata: Dict

    @classmethod
    def from_json(cls, path: str) -> 'ArchetypeCorrelationConfig':
        """Load from correlation config JSON."""
        with open(path, 'r') as f:
            data = json.load(f)

        return cls(
            same_team=data.get('same_team_correlations', {}),
            opponent=data.get('opponent_correlations', {}),
            variance_decomposition=data.get('variance_decomposition', {}),
            metadata=data.get('metadata', {})
        )

    def get_correlation(
        self,
        archetype1: str,
        archetype2: str,
        same_team: bool
    ) -> float:
        """Get correlation between two archetypes."""
        corr_dict = self.same_team if same_team else self.opponent

        # Try direct lookup
        if archetype1 in corr_dict and archetype2 in corr_dict[archetype1]:
            return corr_dict[archetype1][archetype2]

        # Try reverse lookup
        if archetype2 in corr_dict and archetype1 in corr_dict[archetype2]:
            return corr_dict[archetype2][archetype1]

        # Default correlations for archetypes not in config
        if same_team:
            return 0.05  # Small positive for same team
        else:
            return 0.02  # Smaller for opponents


@dataclass
class CorrelationMatrix:
    """
    Player correlation matrix for simulation.
    """
    player_ids: List[str]
    matrix: np.ndarray

    def __post_init__(self):
        n = len(self.player_ids)
        if self.matrix.shape != (n, n):
            raise ValueError(
                f"Matrix shape {self.matrix.shape} doesn't match "
                f"{n} player IDs"
            )

    @classmethod
    def from_archetype_config(
        cls,
        players: List[ShowdownPlayer],
        config: ArchetypeCorrelationConfig,
        player_archetypes: Dict[str, str]
    ) -> 'CorrelationMatrix':
        """
        Build correlation matrix from archetype configuration.

        Applies nearest-PSD projection (Higham algorithm) if the resulting
        matrix is not positive semi-definite, and ridge regularization if
        the condition number exceeds the threshold.

        Args:
            players: List of players
            config: Archetype correlation config
            player_archetypes: Mapping of player name -> archetype
        """
        import logging
        logger = logging.getLogger(__name__)

        n = len(players)
        matrix = np.eye(n, dtype=np.float64)

        for i in range(n):
            for j in range(i + 1, n):
                p1, p2 = players[i], players[j]

                # Get archetypes
                arch1 = player_archetypes.get(p1.name, _infer_archetype(p1))
                arch2 = player_archetypes.get(p2.name, _infer_archetype(p2))

                # Determine if same team
                same_team = p1.team == p2.team

                # Get correlation
                corr = config.get_correlation(arch1, arch2, same_team)

                matrix[i, j] = corr
                matrix[j, i] = corr

        # Nearest-PSD projection if needed
        eigenvalues = np.linalg.eigvalsh(matrix)
        if eigenvalues.min() < -1e-10:
            logger.warning(
                "Correlation matrix not PSD (min eigenvalue=%.6e), "
                "applying nearest-PSD projection (Higham)",
                eigenvalues.min()
            )
            matrix = _nearest_psd_higham(matrix)

        # Conditional ridge regularization if condition number is too high
        eigenvalues = np.linalg.eigvalsh(matrix)
        min_eig = max(eigenvalues.min(), 1e-15)
        max_eig = eigenvalues.max()
        cond = max_eig / min_eig

        if cond > 10000:
            epsilon = 1e-6
            logger.warning(
                "Correlation matrix condition number %.0f exceeds threshold 10000, "
                "applying ridge regularization (epsilon=%.1e)",
                cond, epsilon
            )
            matrix = matrix + epsilon * np.eye(n)
            # Re-normalize diagonal to 1.0
            d = np.sqrt(np.diag(matrix))
            matrix = matrix / np.outer(d, d)
        elif cond > 1000:
            logger.info(
                "Correlation matrix condition number %.0f (elevated but within bounds)",
                cond
            )

        player_ids = [p.id for p in players]
        return cls(player_ids=player_ids, matrix=matrix)

    @classmethod
    def from_players_and_config_file(
        cls,
        players: List[ShowdownPlayer],
        config_path: str,
        archetype_map: Optional[Dict[str, str]] = None
    ) -> 'CorrelationMatrix':
        """
        Convenience method to build matrix from config file.

        Args:
            players: List of players
            config_path: Path to correlation_config_v2.json
            archetype_map: Optional player name -> archetype mapping.
                          If not provided, archetypes are inferred from position.
        """
        config = ArchetypeCorrelationConfig.from_json(config_path)

        if archetype_map is None:
            archetype_map = {}

        return cls.from_archetype_config(players, config, archetype_map)

    def reorder_for_players(self, players: List[ShowdownPlayer]) -> np.ndarray:
        """Return matrix reordered to match player list order."""
        n = len(players)
        result = np.eye(n, dtype=np.float64)

        id_to_idx = {pid: idx for idx, pid in enumerate(self.player_ids)}

        for i, p1 in enumerate(players):
            if p1.id not in id_to_idx:
                continue
            idx1 = id_to_idx[p1.id]

            for j, p2 in enumerate(players):
                if i == j or p2.id not in id_to_idx:
                    continue
                idx2 = id_to_idx[p2.id]
                result[i, j] = self.matrix[idx1, idx2]

        return result


def _nearest_psd_higham(matrix: np.ndarray, max_iter: int = 100, tol: float = 1e-8) -> np.ndarray:
    """
    Nearest correlation matrix via Higham's alternating projections.

    Projects onto the intersection of:
    - S_U: symmetric matrices with unit diagonal
    - S_PSD: positive semi-definite matrices

    This correctly handles PSD (not just PD) — e.g., DST = -1 x team
    scoring can create legitimate zero eigenvalues.

    Args:
        matrix: Input symmetric matrix
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        Nearest PSD correlation matrix
    """
    n = matrix.shape[0]
    Y = matrix.copy()
    dS = np.zeros_like(matrix)

    for iteration in range(max_iter):
        R = Y - dS

        # Project onto PSD cone
        eigenvalues, eigenvectors = np.linalg.eigh(R)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        X = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        # Dykstra correction
        dS = X - R

        # Project onto unit diagonal
        Y_new = X.copy()
        np.fill_diagonal(Y_new, 1.0)

        # Symmetrize
        Y_new = (Y_new + Y_new.T) / 2.0

        # Check convergence
        if np.linalg.norm(Y_new - Y, 'fro') < tol:
            Y = Y_new
            break
        Y = Y_new

    # Final PSD check and clamp
    eigenvalues = np.linalg.eigvalsh(Y)
    if eigenvalues.min() < 0:
        eigenvalues_full, eigenvectors = np.linalg.eigh(Y)
        eigenvalues_full = np.maximum(eigenvalues_full, 0.0)
        Y = eigenvectors @ np.diag(eigenvalues_full) @ eigenvectors.T
        np.fill_diagonal(Y, 1.0)
        Y = (Y + Y.T) / 2.0

    return Y


def _infer_archetype(player: ShowdownPlayer) -> str:
    """
    Infer player archetype from position and stats.

    This is a heuristic fallback when explicit archetype mapping isn't provided.
    """
    pos = player.position.upper()

    if pos == 'QB':
        # Default to POCKET_QB; could use rush stats if available
        return 'POCKET_QB'

    elif pos == 'RB':
        # Use projection as proxy for role
        if player.projection >= 15:
            return 'BELLCOW_EARLY_RB'
        elif player.projection >= 8:
            return 'COMMITTEE_RB'
        else:
            return 'SATELLITE_RB'

    elif pos == 'WR':
        # Use projection/salary as proxy for role
        if player.projection >= 15:
            return 'ALPHA_OUTSIDE_WR'
        elif player.projection >= 10:
            return 'SECONDARY_OUTSIDE_WR'
        elif player.projection >= 5:
            return 'DEPTH_WR'
        else:
            return 'DEPTH_WR'

    elif pos == 'TE':
        if player.projection >= 10:
            return 'ELITE_TE'
        elif player.projection >= 5:
            return 'RECEIVING_TE'
        else:
            return 'BLOCKING_TE'

    elif pos == 'K':
        return 'K'

    elif pos == 'DST' or pos == 'DEF':
        return 'DST'

    else:
        return 'DEPTH_WR'  # Fallback


def load_archetype_mapping(path: str) -> Dict[str, str]:
    """
    Load player name -> archetype mapping from JSON.

    Expected format:
    {
        "Puka Nacua": "ALPHA_OUTSIDE_WR",
        "Matthew Stafford": "POCKET_QB",
        ...
    }
    """
    with open(path, 'r') as f:
        return json.load(f)


def create_archetype_mapping_template(
    players: List[ShowdownPlayer],
    output_path: str
):
    """
    Create a template JSON for archetype mapping.

    User can fill in the archetypes manually.
    """
    # Get unique player names
    names = sorted(set(p.name for p in players))

    mapping = {}
    for name in names:
        # Find player to get position
        player = next((p for p in players if p.name == name), None)
        if player:
            # Pre-fill with inferred archetype
            mapping[name] = _infer_archetype(player)
        else:
            mapping[name] = "UNKNOWN"

    with open(output_path, 'w') as f:
        json.dump(mapping, f, indent=2)

    print(f"Archetype mapping template saved to {output_path}")
    print("Edit the file to assign correct archetypes to each player.")


# =============================================================================
# Default K and DST correlations (not in the config)
# =============================================================================

DEFAULT_K_DST_CORRELATIONS = {
    # K correlations
    ('K', 'POCKET_QB', True): 0.20,
    ('K', 'DUAL_THREAT_QB', True): 0.15,
    ('K', 'K', True): -0.50,  # Only one K per team
    ('K', 'DST', True): 0.10,
    ('K', 'DST', False): -0.15,

    # DST correlations
    ('DST', 'POCKET_QB', False): -0.35,
    ('DST', 'DUAL_THREAT_QB', False): -0.30,
    ('DST', 'ALPHA_OUTSIDE_WR', False): -0.25,
    ('DST', 'ELITE_TE', False): -0.20,
    ('DST', 'BELLCOW_EARLY_RB', False): -0.25,
    ('DST', 'DST', False): 0.15,  # Game script
}


def get_k_dst_correlation(arch1: str, arch2: str, same_team: bool) -> Optional[float]:
    """Get correlation involving K or DST."""
    key = (arch1, arch2, same_team)
    if key in DEFAULT_K_DST_CORRELATIONS:
        return DEFAULT_K_DST_CORRELATIONS[key]

    # Try reverse
    key_rev = (arch2, arch1, same_team)
    if key_rev in DEFAULT_K_DST_CORRELATIONS:
        return DEFAULT_K_DST_CORRELATIONS[key_rev]

    return None


# Extend ArchetypeCorrelationConfig.get_correlation to handle K/DST
_original_get_correlation = ArchetypeCorrelationConfig.get_correlation

def _extended_get_correlation(self, archetype1: str, archetype2: str, same_team: bool) -> float:
    """Extended correlation lookup that includes K and DST."""
    # Check K/DST first
    if archetype1 in ('K', 'DST') or archetype2 in ('K', 'DST'):
        k_dst_corr = get_k_dst_correlation(archetype1, archetype2, same_team)
        if k_dst_corr is not None:
            return k_dst_corr

    # Fall back to original
    return _original_get_correlation(self, archetype1, archetype2, same_team)

ArchetypeCorrelationConfig.get_correlation = _extended_get_correlation
