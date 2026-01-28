"""
Payout lookup with prefix sums for O(1) tie-splitting.

Computes expected payout for a given rank with tied entries.
"""

import numpy as np
from dataclasses import dataclass
from typing import List

from ..types import ContestStructure, PayoutTier


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
        """
        Build payout lookup from contest structure.

        Args:
            contest: Contest configuration with payout tiers
        """
        n = contest.total_entries

        # Build payout by rank (1-indexed, so size n+2 for safety)
        payout_by_rank = np.zeros(n + 2, dtype=np.float64)

        for tier in contest.payout_tiers:
            for r in range(tier.start_rank, min(tier.end_rank + 1, n + 1)):
                payout_by_rank[r] = tier.payout

        # Build prefix sum for O(1) tie pool computation
        prefix_payout = np.cumsum(payout_by_rank)

        return cls(
            payout_by_rank=payout_by_rank,
            prefix_payout=prefix_payout,
            total_entries=n
        )

    @classmethod
    def from_tiers(cls, tiers: List[PayoutTier], total_entries: int) -> 'PayoutLookup':
        """Build from list of payout tiers."""
        contest = ContestStructure(
            name="custom",
            entry_fee=0.0,
            total_entries=total_entries,
            your_entries=0,
            payout_tiers=tiers
        )
        return cls.from_contest(contest)

    def get_payout(self, rank: int, n_tied: int) -> float:
        """
        Get payout for a single entry.

        Tie-splitting: sum payouts from rank to rank+n_tied-1, divide by n_tied.

        Args:
            rank: 1-indexed rank
            n_tied: Number of entries tied at this score

        Returns:
            Payout amount
        """
        if rank > self.total_entries:
            return 0.0

        if n_tied <= 0:
            n_tied = 1

        # Clamp end rank to total_entries
        end_rank = min(rank + n_tied - 1, self.total_entries)
        start_idx = max(rank - 1, 0)

        # Tie pool via prefix sums
        tie_pool = self.prefix_payout[end_rank] - self.prefix_payout[start_idx]

        return tie_pool / n_tied

    def batch_get_payout(
        self,
        ranks: np.ndarray,
        n_tied: np.ndarray
    ) -> np.ndarray:
        """
        Vectorized payout with tie-splitting.

        payout = (prefix[end] - prefix[start-1]) / n_tied

        Args:
            ranks: [n] int32 ranks (1-indexed)
            n_tied: [n] int32 number of ties at each rank

        Returns:
            payouts: [n] float64 payout amounts
        """
        if len(ranks) == 0:
            return np.array([], dtype=np.float64)

        # Ensure n_tied is at least 1
        n_tied = np.maximum(n_tied, 1)

        # Clamp end rank to total_entries
        end_ranks = np.minimum(ranks + n_tied - 1, self.total_entries)
        start_idx = np.maximum(ranks - 1, 0)

        # Valid = actually cashes (rank <= total_entries)
        valid = ranks <= self.total_entries

        # Tie pool via prefix sums
        tie_pool = np.where(
            valid,
            self.prefix_payout[end_ranks] - self.prefix_payout[start_idx],
            0.0
        )

        return np.where(n_tied > 0, tie_pool / n_tied, 0.0)


def score_lineups_vectorized(
    lineup_arrays,  # LineupArrays
    outcomes: np.ndarray  # [n_players] for ONE sim
) -> np.ndarray:
    """
    Vectorized scoring for batch of lineups.

    CPT score = (base * 15 + 5) // 10  (1.5x with rounding)
    FLEX score = sum of base scores

    Args:
        lineup_arrays: LineupArrays with cpt_idx and flex_idx
        outcomes: [n_players] quantized scores for one simulation

    Returns:
        scores: [n_lineups] int32 lineup scores
    """
    if len(lineup_arrays) == 0:
        return np.array([], dtype=np.int32)

    # CPT: gather + 1.5x with rounding
    cpt_base = outcomes[lineup_arrays.cpt_idx]
    cpt_scores = (cpt_base * 15 + 5) // 10

    # FLEX: gather + sum
    flex_scores = outcomes[lineup_arrays.flex_idx].sum(axis=1)

    return (cpt_scores + flex_scores).astype(np.int32)
