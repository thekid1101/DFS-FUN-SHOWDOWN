"""
Array-based histogram for O(1) rank lookup.

Implements fail-fast semantics on out-of-range scores during build,
and explicit handling during rank lookup (no clamping).
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


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
    count_at: np.ndarray      # [size] int32 - count at each score
    entries_above: np.ndarray  # [size] int32 - entries with score > this
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

        Raises:
            ValueError: If strict=True and scores are outside bounds
        """
        min_score, max_score = score_bounds

        # FAIL-FAST: out-of-range is a bug in bounds computation
        if strict and len(scores) > 0:
            scores_min = int(scores.min())
            scores_max = int(scores.max())
            if scores_min < min_score or scores_max > max_score:
                raise ValueError(
                    f"Scores [{scores_min}, {scores_max}] outside bounds "
                    f"[{min_score}, {max_score}] — bounds computation bug!"
                )

        size = max_score - min_score + 1
        count_at = np.zeros(size, dtype=np.int32)

        # ACCUMULATE with np.add.at (not assignment!)
        # This correctly handles duplicate scores
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
        """
        Vectorized entries_above computation.

        entries_above[i] = sum of count_at[i+1:]
        """
        if len(count_at) == 0:
            return np.array([], dtype=np.int32)

        # Suffix sum: cumsum of reversed, then reverse back
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

        Args:
            scores: [n] int32 scores to look up
            assert_in_bounds: If True, raise on out-of-range

        Returns:
            ranks: [n] int32 ranks (1-indexed)
            n_tied: [n] int32 number of entries at same score
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
    """
    Return new histogram with entries added.

    Expands bounds if needed to accommodate new scores.
    """
    if len(scores) == 0:
        return hist

    # Expand bounds if needed
    new_min = min(hist.min_score, int(scores.min()))
    new_max = max(hist.max_score, int(scores.max()))

    new_size = new_max - new_min + 1
    new_count_at = np.zeros(new_size, dtype=np.int32)

    # Copy existing counts
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
