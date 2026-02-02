"""
Core data structures for DFS Showdown GTO Portfolio Builder.

Matches PRD v2.5.1 specifications.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np


# Score precision: 0.1 point precision (multiply by 10, store as int)
SCORE_PRECISION = 10


@dataclass
class ShowdownPlayer:
    """
    A player available for showdown lineups.

    Attributes:
        id: DraftKings player ID (DFS ID from CSV)
        name: Player name
        team: Team abbreviation (e.g., 'LAR', 'SEA')
        position: Position (QB, RB, WR, TE, K, DST)
        salary: Salary for this slot (CPT or FLEX)
        is_cpt: True if this is a CPT slot entry
        percentiles: Dict mapping percentile (25, 50, 75, 85, 95, 99) to score
        std: Standard deviation of projections
        ownership: Projected ownership percentage (0-100)
        projection: Mean projection (dk_points or My Proj)
        flex_player_idx: For CPT entries, index of corresponding FLEX entry
    """
    id: str
    name: str
    team: str
    position: str
    salary: int
    is_cpt: bool
    percentiles: Dict[int, float]
    std: float
    ownership: float
    projection: float
    flex_player_idx: Optional[int] = None


@dataclass
class ShowdownLineup:
    """
    A single showdown lineup (1 CPT + 5 FLEX).

    Attributes:
        cpt_player_idx: Index into CPT player pool
        flex_player_idxs: 5 indices into FLEX player pool, sorted
        salary: Total salary used
    """
    cpt_player_idx: int
    flex_player_idxs: List[int]  # Length 5, sorted
    salary: int


@dataclass
class LineupArrays:
    """
    Vectorized lineup representation for fast scoring.

    Use this instead of List[ShowdownLineup] in hot paths.
    """
    cpt_idx: np.ndarray    # [n_lineups] int32
    flex_idx: np.ndarray   # [n_lineups, 5] int32
    salary: np.ndarray     # [n_lineups] int32

    def __len__(self) -> int:
        return len(self.cpt_idx)

    @classmethod
    def from_lineups(cls, lineups: List[ShowdownLineup]) -> 'LineupArrays':
        """Convert list of lineups to array representation."""
        if not lineups:
            return cls(
                cpt_idx=np.array([], dtype=np.int32),
                flex_idx=np.zeros((0, 5), dtype=np.int32),
                salary=np.array([], dtype=np.int32)
            )
        return cls(
            cpt_idx=np.array([lu.cpt_player_idx for lu in lineups], dtype=np.int32),
            flex_idx=np.array([lu.flex_player_idxs for lu in lineups], dtype=np.int32),
            salary=np.array([lu.salary for lu in lineups], dtype=np.int32)
        )


@dataclass
class PayoutTier:
    """A payout tier covering a range of ranks."""
    start_rank: int
    end_rank: int
    payout: float


@dataclass
class ContestStructure:
    """
    Contest configuration.

    Attributes:
        name: Contest name/identifier
        entry_fee: Cost per entry
        total_entries: Total entries in contest (including yours)
        your_entries: Number of entries you're submitting
        payout_tiers: List of payout tiers
        tier: Optional contest tier for cross-contest diversity (1=flagship, 2=grinder, 3=filler)
        field_sharpness_override: Optional per-contest field sharpness
    """
    name: str
    entry_fee: float
    total_entries: int
    your_entries: int
    payout_tiers: List[PayoutTier]
    tier: Optional[int] = None
    field_sharpness_override: Optional[float] = None

    @property
    def field_size(self) -> int:
        """Number of opponent entries (total - yours)."""
        return self.total_entries - self.your_entries

    @property
    def dollars_at_risk(self) -> float:
        """Total dollars at risk in this contest."""
        return self.entry_fee * self.your_entries

    @property
    def field_share(self) -> float:
        """Fraction of field that is your entries."""
        return self.your_entries / self.total_entries if self.total_entries > 0 else 0.0


@dataclass
class FieldGenConfig:
    """
    Field generation configuration (soft priors).

    These priors model typical opponent behavior in DFS contests.
    """
    temperature: float = 1.0
    qb_pair_rate: float = 0.80
    bring_back_rate: float = 0.65
    dst_rate_multiplier: float = 0.7
    kicker_rate_multiplier: float = 0.8
    split_priors: Dict[str, float] = field(default_factory=lambda: {
        '5-1': 0.08,
        '4-2': 0.32,
        '3-3': 0.35,
        '2-4': 0.20,
        '1-5': 0.05
    })
    field_sharpness: float = 5.0
    ownership_power: float = 0.5
    quality_sims: int = 1000
    field_method: str = "simulated"

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError(
                "FieldGenConfig.temperature must be positive to avoid "
                "division by zero in ownership weighting."
            )


@dataclass
class ProjectionsData:
    """
    Container for loaded projections data.

    Separates CPT and FLEX player pools with linking information.
    """
    cpt_players: List[ShowdownPlayer]
    flex_players: List[ShowdownPlayer]
    cpt_to_flex_map: Dict[int, int]  # cpt_idx -> flex_idx for same player
    teams: List[str]  # Unique teams in the slate

    @property
    def n_cpt(self) -> int:
        return len(self.cpt_players)

    @property
    def n_flex(self) -> int:
        return len(self.flex_players)


@dataclass
class TournamentMetrics:
    """
    Tournament-aligned metrics computed from a scored portfolio.

    These metrics emphasize top-heavy outcomes that matter for GPP tournaments,
    complementing the standard EV-based diagnostics.

    Attributes:
        top_1pct_rate: Fraction of sims where best portfolio rank is in top 1%
        ceiling_ev: E[total_payout | best_rank in top 1%] — conditional upside
        win_rate: Fraction of sims where best portfolio rank == 1
        roi_pct: Standard ROI (sanity check, mirrors diagnostics)
        composite_score: Weighted combination (0.50*top1 + 0.35*ceiling + 0.10*win + 0.05*roi)
        n_sims: Number of simulations used
    """
    top_1pct_rate: float
    ceiling_ev: float
    win_rate: float
    roi_pct: float
    composite_score: float
    n_sims: int

    def to_dict(self) -> Dict:
        """Convert to serializable dict."""
        return {
            'top_1pct_rate': self.top_1pct_rate,
            'ceiling_ev': self.ceiling_ev,
            'win_rate': self.win_rate,
            'roi_pct': self.roi_pct,
            'composite_score': self.composite_score,
            'n_sims': self.n_sims,
        }


def quantize_score(float_score: float) -> int:
    """Quantize float score to int (points × 10)."""
    return int(round(float_score * SCORE_PRECISION))


def dequantize_score(int_score: int) -> float:
    """Convert quantized int score back to float."""
    return int_score / SCORE_PRECISION
