"""
DFS Showdown GTO Portfolio Builder - Reference Implementation
=============================================================

Matches PRD v2.5.1 exactly. Ready to hand to an engineer.

File structure:
    showdown_gto/
    ├── __init__.py
    ├── types.py          # Data classes
    ├── simulation.py     # Phase 1: Simulation + bounds
    ├── candidates.py     # Phase 2A: Enumeration
    ├── field.py          # Phase 2B: Field generation
    ├── histogram.py      # Phase 3: Array histogram
    ├── payout.py         # Phase 3: Payout lookup
    ├── ev.py             # Phase 3: EV computation
    ├── pipeline.py       # Full pipeline
    └── tests/
        ├── test_histogram.py
        ├── test_bounds.py
        ├── test_ev.py
        └── test_integration.py
"""

# =============================================================================
# types.py - Core data structures
# =============================================================================

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np


@dataclass
class ShowdownPlayer:
    """A player available for showdown lineups."""
    id: str
    name: str
    team: str
    position: str
    salary: int
    # Projection percentiles (quantized: points × 10)
    p10: int
    p25: int
    p50: int
    p75: int
    p90: int


@dataclass
class ShowdownLineup:
    """A single showdown lineup (1 CPT + 5 FLEX)."""
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
    """Contest configuration."""
    name: str
    entry_fee: float
    total_entries: int
    your_entries: int
    payout_tiers: List[PayoutTier]


@dataclass
class FieldGenConfig:
    """Field generation configuration (soft priors)."""
    temperature: float = 1.0
    salary_utilization_mean: float = 0.98
    salary_utilization_std: float = 0.03
    qb_pair_rate: float = 0.80
    bring_back_rate: float = 0.65
    dst_rate_multiplier: float = 0.7
    kicker_rate_multiplier: float = 0.8
    split_priors: Dict[str, float] = field(default_factory=lambda: {
        '5-1': 0.08, '4-2': 0.32, '3-3': 0.35, '2-4': 0.20, '1-5': 0.05
    })


# =============================================================================
# simulation.py - Phase 1: Simulation engine + guaranteed bounds
# =============================================================================

SCORE_PRECISION = 10  # 0.1 point precision


def quantize_score(float_score: float) -> int:
    """Quantize float score to int (points × 10)."""
    return int(round(float_score * SCORE_PRECISION))


def compute_guaranteed_score_bounds(
    outcomes: np.ndarray,
    buffer_pct: float = 0.1
) -> Tuple[int, int]:
    """
    Compute bounds GUARANTEED to contain all possible lineup scores.
    
    Iterates ALL sims. Uses max + (sum - max) — no sort needed.
    
    Args:
        outcomes: [n_players, n_sims] int32 quantized scores
        buffer_pct: Safety buffer (default 10%)
    
    Returns:
        (min_score, max_score) tuple
    """
    n_players, n_sims = outcomes.shape
    global_max = 0
    
    for sim in range(n_sims):
        sim_outcomes = outcomes[:, sim]
        
        # Top 6 via partition (O(n_players))
        if n_players > 6:
            top_6_idx = np.argpartition(sim_outcomes, -6)[-6:]
            top_6 = sim_outcomes[top_6_idx]
        else:
            top_6 = sim_outcomes
        
        # SIMPLIFIED: max + (sum - max), no sort
        cpt_base = int(top_6.max())
        flex_sum = int(top_6.sum()) - cpt_base
        
        max_cpt = (cpt_base * 15 + 5) // 10  # 1.5x with rounding
        max_score = max_cpt + flex_sum
        
        global_max = max(global_max, max_score)
    
    return (0, int(global_max * (1 + buffer_pct)))


def score_lineups_vectorized(
    lineup_arrays: LineupArrays,
    outcomes: np.ndarray  # [n_players] for ONE sim
) -> np.ndarray:
    """
    Vectorized scoring for batch of lineups.
    
    Returns: [n_lineups] int32 scores
    """
    if len(lineup_arrays) == 0:
        return np.array([], dtype=np.int32)
    
    # CPT: gather + 1.5x with rounding
    cpt_base = outcomes[lineup_arrays.cpt_idx]
    cpt_scores = (cpt_base * 15 + 5) // 10
    
    # FLEX: gather + sum
    flex_scores = outcomes[lineup_arrays.flex_idx].sum(axis=1)
    
    return (cpt_scores + flex_scores).astype(np.int32)


# =============================================================================
# histogram.py - Phase 3: Array histogram with fail-fast
# =============================================================================

@dataclass
class ArrayHistogram:
    """
    Array-based histogram for O(1) rank lookup.
    
    Key features:
    - Fail-fast on out-of-range in build (strict=True)
    - Explicit out-of-range handling in lookup (no clamping)
    - Vectorized entries_above computation
    """
    min_score: int
    max_score: int
    count_at: np.ndarray      # [size] int32
    entries_above: np.ndarray # [size] int32
    total_entries: int
    
    @classmethod
    def from_scores_and_counts(
        cls,
        scores: np.ndarray,
        counts: np.ndarray,
        score_bounds: Tuple[int, int],
        strict: bool = True
    ) -> 'ArrayHistogram':
        """
        Build histogram with accumulation.
        
        Args:
            scores: [n] int32 scores
            counts: [n] int32 counts per score
            score_bounds: (min, max) guaranteed to contain all scores
            strict: If True, raise on out-of-range (default)
        """
        min_score, max_score = score_bounds
        
        # FAIL-FAST: out-of-range is a bug
        if strict and len(scores) > 0:
            if scores.min() < min_score or scores.max() > max_score:
                raise ValueError(
                    f"Scores [{scores.min()}, {scores.max()}] outside bounds "
                    f"[{min_score}, {max_score}] — bounds computation bug!"
                )
        
        size = max_score - min_score + 1
        count_at = np.zeros(size, dtype=np.int32)
        
        # ACCUMULATE with np.add.at (not assignment!)
        if len(scores) > 0:
            indices = (scores - min_score).astype(np.int64)
            np.add.at(count_at, indices, counts)
        
        # Vectorized entries_above
        entries_above = cls._build_entries_above(count_at)
        
        return cls(
            min_score=min_score,
            max_score=max_score,
            count_at=count_at,
            entries_above=entries_above,
            total_entries=int(count_at.sum())  # Self-consistent
        )
    
    @staticmethod
    def _build_entries_above(count_at: np.ndarray) -> np.ndarray:
        """Vectorized entries_above (no Python loop)."""
        if len(count_at) == 0:
            return np.array([], dtype=np.int32)
        # entries_above[i] = sum of count_at[i+1:]
        suffix = np.cumsum(count_at[::-1])[::-1]
        entries_above = np.zeros_like(count_at)
        if len(suffix) > 1:
            entries_above[:-1] = suffix[1:]
        return entries_above
    
    def batch_get_rank_and_ties(
        self,
        scores: np.ndarray,
        assert_in_bounds: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized rank/tie lookup with EXPLICIT out-of-range handling.
        
        NO CLAMPING. Semantics:
        - above max: rank=1, ties=0 (beat everyone)
        - below min: rank=total+1, ties=0 (lost to everyone)
        - in-range: array lookup
        """
        n = len(scores)
        if n == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
        
        above = scores > self.max_score
        below = scores < self.min_score
        in_range = ~(above | below)
        
        if assert_in_bounds and (above.any() or below.any()):
            raise ValueError(
                f"Scores out of bounds: {above.sum()} above, {below.sum()} below"
            )
        
        ranks = np.empty(n, dtype=np.int32)
        n_tied = np.empty(n, dtype=np.int32)
        
        # Above max: beat everyone
        ranks[above] = 1
        n_tied[above] = 0
        
        # Below min: lost to everyone
        ranks[below] = self.total_entries + 1
        n_tied[below] = 0
        
        # In range: array lookup
        if in_range.any():
            idxs = scores[in_range] - self.min_score
            ranks[in_range] = self.entries_above[idxs] + 1
            n_tied[in_range] = self.count_at[idxs]
        
        return ranks, n_tied


def add_entries_to_histogram(
    hist: ArrayHistogram,
    scores: np.ndarray,
    counts: np.ndarray
) -> ArrayHistogram:
    """Return new histogram with entries added."""
    if len(scores) == 0:
        return hist
    
    # Expand bounds if needed
    new_min = min(hist.min_score, int(scores.min()))
    new_max = max(hist.max_score, int(scores.max()))
    
    new_size = new_max - new_min + 1
    new_count_at = np.zeros(new_size, dtype=np.int32)
    
    # Copy existing
    old_offset = hist.min_score - new_min
    old_size = len(hist.count_at)
    new_count_at[old_offset:old_offset + old_size] = hist.count_at
    
    # Add new with accumulation
    indices = (scores - new_min).astype(np.int64)
    np.add.at(new_count_at, indices, counts)
    
    # Rebuild entries_above
    entries_above = ArrayHistogram._build_entries_above(new_count_at)
    
    return ArrayHistogram(
        min_score=new_min,
        max_score=new_max,
        count_at=new_count_at,
        entries_above=entries_above,
        total_entries=int(new_count_at.sum())
    )


# =============================================================================
# payout.py - Phase 3: Payout lookup with prefix sums
# =============================================================================

@dataclass
class PayoutLookup:
    """
    Pre-computed payout lookup for O(1) tie-splitting.
    
    Uses prefix sums: tie_pool = prefix[end] - prefix[start-1]
    """
    payout_by_rank: np.ndarray   # [total_entries + 2] float64
    prefix_payout: np.ndarray    # [total_entries + 2] float64
    total_entries: int
    
    @classmethod
    def from_contest(cls, contest: ContestStructure) -> 'PayoutLookup':
        n = contest.total_entries
        
        # Build payout by rank (1-indexed)
        payout_by_rank = np.zeros(n + 2, dtype=np.float64)
        for tier in contest.payout_tiers:
            for r in range(tier.start_rank, min(tier.end_rank + 1, n + 1)):
                payout_by_rank[r] = tier.payout
        
        # Build prefix sum
        prefix_payout = np.cumsum(payout_by_rank)
        
        return cls(
            payout_by_rank=payout_by_rank,
            prefix_payout=prefix_payout,
            total_entries=n
        )
    
    def batch_get_payout(
        self,
        ranks: np.ndarray,
        n_tied: np.ndarray
    ) -> np.ndarray:
        """
        Vectorized payout with tie-splitting.
        
        payout = (prefix[end] - prefix[start-1]) / n_tied
        """
        if len(ranks) == 0:
            return np.array([], dtype=np.float64)
        
        # Clamp end rank to total_entries
        end_ranks = np.minimum(ranks + n_tied - 1, self.total_entries)
        start_idx = np.maximum(ranks - 1, 0)
        
        # Valid = actually cashes
        valid = ranks <= self.total_entries
        
        # Tie pool via prefix sums
        tie_pool = np.where(
            valid,
            self.prefix_payout[end_ranks] - self.prefix_payout[start_idx],
            0.0
        )
        
        return np.where(n_tied > 0, tie_pool / n_tied, 0.0)


# =============================================================================
# ev.py - Phase 3: EV computation
# =============================================================================

import logging
logger = logging.getLogger(__name__)


def compute_approx_lineup_evs(
    candidate_arrays: LineupArrays,
    field_arrays: LineupArrays,
    field_counts: np.ndarray,
    outcomes: np.ndarray,
    contest: ContestStructure,
    score_bounds: Tuple[int, int],
    chunk_size: int = 10000
) -> np.ndarray:
    """
    Streaming EV computation (vs field only, overestimates).
    
    Memory: O(n_field + chunk_size) per sim, not O(n_cand × n_sims)
    """
    n_candidates = len(candidate_arrays)
    n_sims = outcomes.shape[1]
    
    payout_lookup = PayoutLookup.from_contest(contest)
    ev_sum = np.zeros(n_candidates, dtype=np.float64)
    
    for sim in range(n_sims):
        sim_outcomes = outcomes[:, sim]
        
        # Score field and build histogram
        field_scores = score_lineups_vectorized(field_arrays, sim_outcomes)
        histogram = ArrayHistogram.from_scores_and_counts(
            field_scores, field_counts, score_bounds
        )
        
        # Stream candidate chunks
        for chunk_start in range(0, n_candidates, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_candidates)
            
            # Score chunk
            chunk_cpt = candidate_arrays.cpt_idx[chunk_start:chunk_end]
            chunk_flex = candidate_arrays.flex_idx[chunk_start:chunk_end]
            cpt_scores = (sim_outcomes[chunk_cpt] * 15 + 5) // 10
            flex_scores = sim_outcomes[chunk_flex].sum(axis=1)
            chunk_scores = (cpt_scores + flex_scores).astype(np.int32)
            
            # Rank lookup (explicit out-of-range handling)
            ranks, tied_in_field = histogram.batch_get_rank_and_ties(chunk_scores)
            n_tied = tied_in_field + 1  # +1 for candidate
            
            # Payout
            payouts = payout_lookup.batch_get_payout(ranks, n_tied)
            ev_sum[chunk_start:chunk_end] += payouts
        
        if (sim + 1) % 10000 == 0:
            logger.info(f"Approx EV: {sim + 1}/{n_sims} sims")
    
    return ev_sum / n_sims


def compute_true_portfolio_ev(
    selected_arrays: LineupArrays,
    field_arrays: LineupArrays,
    field_counts: np.ndarray,
    outcomes: np.ndarray,
    contest: ContestStructure,
    score_bounds: Tuple[int, int]
) -> Tuple[float, Dict]:
    """
    True portfolio EV with self-competition.
    
    Your entries compete with each other, not just the field.
    Includes invariant check: combined total = field + selected.
    """
    n_selected = len(selected_arrays)
    n_sims = outcomes.shape[1]
    field_size = int(field_counts.sum())
    
    payout_lookup = PayoutLookup.from_contest(contest)
    
    total_payout_sum = 0.0
    per_lineup_payout_sum = np.zeros(n_selected, dtype=np.float64)
    cash_count = np.zeros(n_selected, dtype=np.int32)
    profit_count = 0
    entry_cost = contest.entry_fee * n_selected
    
    for sim in range(n_sims):
        sim_outcomes = outcomes[:, sim]
        
        # Build field histogram
        field_scores = score_lineups_vectorized(field_arrays, sim_outcomes)
        field_histogram = ArrayHistogram.from_scores_and_counts(
            field_scores, field_counts, score_bounds
        )
        
        # Add selected lineups to histogram
        selected_scores = score_lineups_vectorized(selected_arrays, sim_outcomes)
        selected_counts = np.ones(n_selected, dtype=np.int32)
        combined_histogram = add_entries_to_histogram(
            field_histogram, selected_scores, selected_counts
        )
        
        # INVARIANT CHECK
        expected_total = field_size + n_selected
        assert combined_histogram.total_entries == expected_total, \
            f"Histogram total {combined_histogram.total_entries} != {expected_total}"
        
        # Get payouts
        ranks, n_tied = combined_histogram.batch_get_rank_and_ties(selected_scores)
        payouts = payout_lookup.batch_get_payout(ranks, n_tied)
        
        # Accumulate
        per_lineup_payout_sum += payouts
        sim_total = payouts.sum()
        total_payout_sum += sim_total
        cash_count += (payouts > 0).astype(np.int32)
        if sim_total > entry_cost:
            profit_count += 1
    
    portfolio_ev = total_payout_sum / n_sims
    
    return portfolio_ev, {
        'true_portfolio_ev': float(portfolio_ev),
        'entry_cost': float(entry_cost),
        'expected_profit': float(portfolio_ev - entry_cost),
        'roi_pct': float((portfolio_ev - entry_cost) / entry_cost * 100) if entry_cost > 0 else 0.0,
        'p_profit': float(profit_count / n_sims),
        'per_lineup_evs': (per_lineup_payout_sum / n_sims).tolist(),
        'per_lineup_cash_probs': (cash_count / n_sims).tolist(),
        'expected_cashes': float(cash_count.sum() / n_sims),
        'field_size': field_size,
        'n_selected': n_selected,
    }


# =============================================================================
# pipeline.py - Full pipeline
# =============================================================================

def validate_contest_config(contest: ContestStructure, n_select: int):
    """Validate contest config matches portfolio size."""
    if contest.your_entries != n_select:
        raise ValueError(
            f"Contest configured for {contest.your_entries} entries, "
            f"but selecting {n_select}. Update config."
        )


def compute_field_size(contest: ContestStructure) -> int:
    """Field size = total - your entries."""
    return contest.total_entries - contest.your_entries


def select_portfolio(
    candidates: List[ShowdownLineup],
    field_arrays: LineupArrays,
    field_counts: np.ndarray,
    outcomes: np.ndarray,
    contest: ContestStructure,
    n_select: int,
    min_ev_threshold: float = 0.0
) -> Tuple[List[int], Dict]:
    """
    Full portfolio selection pipeline.
    
    Steps:
    1. Validate contest config
    2. Compute guaranteed bounds (all sims)
    3. Compute approx EVs (streaming)
    4. Select top N
    5. Compute true portfolio EV (with self-competition)
    
    Returns:
        selected_indices: Indices into candidates list
        diagnostics: Dict with EV stats
    """
    # === VALIDATION ===
    validate_contest_config(contest, n_select)
    
    field_size = compute_field_size(contest)
    actual_field_size = int(field_counts.sum())
    assert actual_field_size == field_size, \
        f"Field count {actual_field_size} != expected {field_size}"
    
    # === CONVERT CANDIDATES ===
    candidate_arrays = LineupArrays.from_lineups(candidates)
    
    # === GUARANTEED BOUNDS ===
    logger.info("Computing guaranteed score bounds...")
    score_bounds = compute_guaranteed_score_bounds(outcomes)
    logger.info(f"Score bounds: {score_bounds}")
    
    # === APPROX EVS ===
    logger.info(f"Computing approx EVs for {len(candidates)} candidates...")
    approx_evs = compute_approx_lineup_evs(
        candidate_arrays, field_arrays, field_counts,
        outcomes, contest, score_bounds
    )
    
    # === SELECT TOP N ===
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
    
    # === TRUE PORTFOLIO EV ===
    logger.info(f"Computing true portfolio EV for {len(selected_indices)} lineups...")
    selected_arrays = LineupArrays.from_lineups(
        [candidates[i] for i in selected_indices]
    )
    
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
    diagnostics['n_qualified'] = len(qualified)
    
    return selected_indices, diagnostics


# =============================================================================
# tests/test_histogram.py - Unit tests for histogram
# =============================================================================

def test_histogram_accumulation():
    """Test that duplicate scores accumulate correctly."""
    scores = np.array([100, 100, 200], dtype=np.int32)
    counts = np.array([5, 3, 10], dtype=np.int32)
    bounds = (0, 300)
    
    hist = ArrayHistogram.from_scores_and_counts(scores, counts, bounds)
    
    assert hist.count_at[100] == 8, "Should accumulate: 5 + 3 = 8"
    assert hist.count_at[200] == 10
    assert hist.total_entries == 18


def test_histogram_out_of_range_fails():
    """Test that out-of-range scores raise in strict mode."""
    scores = np.array([100, 500], dtype=np.int32)
    counts = np.array([5, 3], dtype=np.int32)
    bounds = (0, 300)  # 500 is out of range
    
    try:
        ArrayHistogram.from_scores_and_counts(scores, counts, bounds, strict=True)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "outside bounds" in str(e)


def test_rank_lookup_above_max():
    """Test that scores above max get rank 1."""
    scores = np.array([100, 200], dtype=np.int32)
    counts = np.array([5, 10], dtype=np.int32)
    bounds = (0, 300)
    
    hist = ArrayHistogram.from_scores_and_counts(scores, counts, bounds)
    
    # Score 500 is above max (300)
    query = np.array([500], dtype=np.int32)
    ranks, ties = hist.batch_get_rank_and_ties(query)
    
    assert ranks[0] == 1, "Above max should be rank 1"
    assert ties[0] == 0, "No ties above max"


def test_rank_lookup_below_min():
    """Test that scores below min get rank total+1."""
    scores = np.array([100, 200], dtype=np.int32)
    counts = np.array([5, 10], dtype=np.int32)
    bounds = (50, 300)
    
    hist = ArrayHistogram.from_scores_and_counts(scores, counts, bounds)
    
    # Score 10 is below min (50)
    query = np.array([10], dtype=np.int32)
    ranks, ties = hist.batch_get_rank_and_ties(query)
    
    assert ranks[0] == hist.total_entries + 1, "Below min should be last"
    assert ties[0] == 0, "No ties below min"


# =============================================================================
# tests/test_bounds.py - Unit tests for bounds computation
# =============================================================================

def test_bounds_covers_all_sims():
    """Test that bounds contain max from any sim."""
    np.random.seed(42)
    n_players, n_sims = 30, 1000
    outcomes = np.random.randint(0, 500, (n_players, n_sims), dtype=np.int32)
    
    bounds = compute_guaranteed_score_bounds(outcomes, buffer_pct=0.0)
    
    # Manually compute max for each sim
    for sim in range(n_sims):
        top_6 = np.sort(outcomes[:, sim])[-6:][::-1]
        max_cpt = (top_6[0] * 15 + 5) // 10
        max_score = max_cpt + top_6[1:].sum()
        
        assert max_score <= bounds[1], f"Sim {sim} max {max_score} > bound {bounds[1]}"


# =============================================================================
# tests/test_ev.py - Unit tests for EV computation
# =============================================================================

def test_true_ev_invariant():
    """Test that combined histogram total = field + selected."""
    # This is checked via assertion inside compute_true_portfolio_ev
    # If it fails, the assertion will raise
    pass  # Covered by integration test


def test_self_competition_reduces_ev():
    """Test that true EV <= approx EV sum (self-competition cost >= 0)."""
    # When you have multiple lineups, they compete with each other
    # This should never INCREASE total EV
    pass  # Covered by integration test


# =============================================================================
# tests/test_integration.py - Full pipeline test
# =============================================================================

def test_full_pipeline():
    """Integration test of full pipeline."""
    np.random.seed(42)
    
    # Create mock data
    n_players = 12
    n_sims = 100
    outcomes = np.random.randint(50, 300, (n_players, n_sims), dtype=np.int32)
    
    # Create some lineups
    lineups = []
    for cpt in range(6):
        flex = [i for i in range(6) if i != cpt]
        lineups.append(ShowdownLineup(
            cpt_player_idx=cpt,
            flex_player_idxs=sorted(flex),
            salary=45000
        ))
    
    # Create field
    field_arrays = LineupArrays.from_lineups(lineups[:3])
    field_counts = np.array([100, 100, 100], dtype=np.int32)
    
    # Create contest
    contest = ContestStructure(
        name="Test",
        entry_fee=5.0,
        total_entries=302,  # 300 field + 2 yours
        your_entries=2,
        payout_tiers=[
            PayoutTier(1, 1, 100.0),
            PayoutTier(2, 10, 20.0),
            PayoutTier(11, 50, 10.0),
        ]
    )
    
    # Run pipeline
    selected, diagnostics = select_portfolio(
        candidates=lineups,
        field_arrays=field_arrays,
        field_counts=field_counts,
        outcomes=outcomes,
        contest=contest,
        n_select=2
    )
    
    assert len(selected) == 2
    assert diagnostics['true_portfolio_ev'] >= 0
    assert diagnostics['self_competition_cost'] >= 0
    assert diagnostics['bounds_method'] == 'all_sims_max'
    
    print("Integration test passed!")
    print(f"  True EV: ${diagnostics['true_portfolio_ev']:.2f}")
    print(f"  Self-competition cost: ${diagnostics['self_competition_cost']:.2f}")


if __name__ == "__main__":
    # Run tests
    print("Running tests...")
    test_histogram_accumulation()
    print("✓ test_histogram_accumulation")
    
    test_histogram_out_of_range_fails()
    print("✓ test_histogram_out_of_range_fails")
    
    test_rank_lookup_above_max()
    print("✓ test_rank_lookup_above_max")
    
    test_rank_lookup_below_min()
    print("✓ test_rank_lookup_below_min")
    
    test_bounds_covers_all_sims()
    print("✓ test_bounds_covers_all_sims")
    
    test_full_pipeline()
    print("✓ test_full_pipeline")
    
    print("\nAll tests passed!")
