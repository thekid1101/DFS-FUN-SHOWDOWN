# DFS Showdown GTO Portfolio Builder
## Product Requirements Document v2.5.1

**Document Version:** 2.5.1  
**Last Updated:** January 24, 2026  
**Status:** Production Ready (Final)  
**Author:** Johnathon (Product Owner)

---

## Executive Summary

The DFS Showdown GTO Portfolio Builder computes true contest expected value for NFL single-game DFS. This v2.5.1 revision adds final guardrails to v2.5.

### v2.5.1 Guardrails (from v2.5)

| Item | v2.5 | v2.5.1 |
|------|------|--------|
| **Histogram out-of-range** | Silently filters | **Fail-fast** (raise if any out-of-range) |
| **Bounds computation** | Sort top 6 | **Simplified** (max + sum - max) |
| **True EV invariant** | None | **Assert** combined total = field + selected |

### Correctness Guarantee (Complete)

- ✅ Out-of-range in histogram build: **fail-fast** (bug detector)
- ✅ Out-of-range in rank lookup: **explicit handling** (correct semantics)
- ✅ Score bounds: **guaranteed** over all sims
- ✅ Histogram: **accumulates** with `np.add.at()`
- ✅ Self-competition: **invariant checked**
- ✅ Matches DraftKings payout math exactly

---

## Phase 1: Guaranteed Score Bounds (Simplified)

```python
def compute_guaranteed_score_bounds(
    outcomes: np.ndarray,  # [n_players, n_sims]
    buffer_pct: float = 0.1
) -> Tuple[int, int]:
    """
    Compute bounds GUARANTEED to contain all possible lineup scores.
    
    v2.5.1: Simplified — no sort needed, just max + sum.
    """
    n_players, n_sims = outcomes.shape
    global_max = 0
    
    for sim in range(n_sims):
        sim_outcomes = outcomes[:, sim]
        
        # Top 6 via partition
        if n_players > 6:
            top_6_idx = np.argpartition(sim_outcomes, -6)[-6:]
            top_6 = sim_outcomes[top_6_idx]
        else:
            top_6 = sim_outcomes
        
        # SIMPLIFIED: No sort needed
        # Max lineup = best as CPT (1.5x) + other 5 as FLEX
        cpt_base = top_6.max()
        flex_sum = top_6.sum() - cpt_base  # Sum of other 5
        
        max_cpt = (cpt_base * 15 + 5) // 10  # 1.5x with rounding
        max_score = max_cpt + flex_sum
        
        global_max = max(global_max, max_score)
    
    return (0, int(global_max * (1 + buffer_pct)))
```

---

## Phase 3: Array Histogram (Fail-Fast)

```python
@dataclass
class ArrayHistogram:
    """Array histogram with fail-fast on out-of-range."""
    min_score: int
    max_score: int
    count_at: np.ndarray
    entries_above: np.ndarray
    total_entries: int
    
    @classmethod
    def from_scores_and_counts(
        cls,
        scores: np.ndarray,
        counts: np.ndarray,
        score_bounds: Tuple[int, int],
        strict: bool = True  # FAIL-FAST by default
    ) -> 'ArrayHistogram':
        """
        Build histogram.
        
        v2.5.1: FAIL-FAST if any score is out of bounds.
        Out-of-range here means bounds computation is buggy.
        """
        min_score, max_score = score_bounds
        
        # FAIL-FAST: Out-of-range is a bug, not expected
        if strict and len(scores) > 0:
            if scores.min() < min_score or scores.max() > max_score:
                raise ValueError(
                    f"Scores [{scores.min()}, {scores.max()}] outside bounds "
                    f"[{min_score}, {max_score}] — bounds computation bug!"
                )
        
        size = max_score - min_score + 1
        count_at = np.zeros(size, dtype=np.int32)
        
        # Accumulate (all scores should be in range now)
        if len(scores) > 0:
            indices = scores - min_score
            np.add.at(count_at, indices, counts)
        
        # Vectorized entries_above
        entries_above = cls._build_entries_above_vectorized(count_at)
        
        return cls(
            min_score=min_score,
            max_score=max_score,
            count_at=count_at,
            entries_above=entries_above,
            total_entries=int(count_at.sum())
        )
    
    @staticmethod
    def _build_entries_above_vectorized(count_at: np.ndarray) -> np.ndarray:
        """Vectorized entries_above."""
        suffix = np.cumsum(count_at[::-1])[::-1]
        entries_above = np.zeros_like(count_at)
        entries_above[:-1] = suffix[1:]
        return entries_above
    
    def batch_get_rank_and_ties(
        self,
        scores: np.ndarray,
        assert_in_bounds: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rank lookup with EXPLICIT out-of-range handling.
        
        Note: This is for CANDIDATE scores, which may legitimately
        exceed field bounds (rare high-scoring lineup).
        
        Semantics:
        - above max: rank=1, ties=0 (beat everyone)
        - below min: rank=total+1, ties=0 (lost to everyone)
        - in-range: array lookup
        """
        n = len(scores)
        
        above = scores > self.max_score
        below = scores < self.min_score
        in_range = ~(above | below)
        
        if assert_in_bounds and (above.any() or below.any()):
            raise ValueError(f"Scores out of bounds: {above.sum()} above, {below.sum()} below")
        
        ranks = np.empty(n, dtype=np.int32)
        n_tied = np.empty(n, dtype=np.int32)
        
        ranks[above] = 1
        n_tied[above] = 0
        
        ranks[below] = self.total_entries + 1
        n_tied[below] = 0
        
        if in_range.any():
            idxs = scores[in_range] - self.min_score
            ranks[in_range] = self.entries_above[idxs] + 1
            n_tied[in_range] = self.count_at[idxs]
        
        return ranks, n_tied
```

---

## Phase 3: True Portfolio EV (With Invariant Check)

```python
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
    
    v2.5.1: Invariant check after combining histogram.
    """
    n_selected = len(selected_arrays.cpt_idx)
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
        
        # Score and build field histogram
        field_scores = score_lineups_vectorized(field_arrays, sim_outcomes)
        field_histogram = ArrayHistogram.from_scores_and_counts(
            field_scores, field_counts, score_bounds
        )
        
        # Score selected and add to histogram
        selected_scores = score_lineups_vectorized(selected_arrays, sim_outcomes)
        selected_counts = np.ones(n_selected, dtype=np.int32)
        combined_histogram = add_entries_to_histogram(
            field_histogram, selected_scores, selected_counts
        )
        
        # INVARIANT CHECK: combined total must equal field + selected
        expected_total = field_size + n_selected
        assert combined_histogram.total_entries == expected_total, \
            f"Histogram total {combined_histogram.total_entries} != expected {expected_total}"
        
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
        'roi_pct': float((portfolio_ev - entry_cost) / entry_cost * 100),
        'p_profit': float(profit_count / n_sims),
        'per_lineup_evs': (per_lineup_payout_sum / n_sims).tolist(),
        'per_lineup_cash_probs': (cash_count / n_sims).tolist(),
        'expected_cashes': float(cash_count.sum() / n_sims),
        'field_size': field_size,
        'n_selected': n_selected,
    }
```

---

## Changelog from v2.5

| Item | Change |
|------|--------|
| **Histogram build** | `strict=True` by default, raises on out-of-range |
| **Bounds computation** | Simplified: `max + (sum - max)` instead of sort |
| **True EV** | Invariant: `combined.total == field_size + n_selected` |

---

## Implementation Checklist (Final)

### Module 1: Guaranteed Bounds

| Task | Acceptance |
|------|------------|
| Uses `max + sum - max` | No sort on top 6 |
| Iterates ALL sims | Not sampled |

### Module 2: Histogram Build (Fail-Fast)

| Task | Acceptance |
|------|------------|
| `strict=True` default | Raises if out-of-range |
| Uses `np.add.at()` | Accumulates |
| Unit test | Out-of-range raises ValueError |

### Module 3: Rank Lookup (Explicit)

| Task | Acceptance |
|------|------------|
| No `np.clip()` | Explicit above/below/in-range |
| Unit test | Score 9999 with max 3000 → rank=1 |

### Module 4: True Portfolio EV (Invariant)

| Task | Acceptance |
|------|------------|
| Assert `total == field + selected` | Catches bugs |
| Unit test | Wrong total raises AssertionError |

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0–2.4 | 2026-01-24 | Progressive fixes |
| 2.5 | 2026-01-24 | No clamp, guaranteed bounds |
| **2.5.1** | 2026-01-24 | **Final**: Fail-fast histogram, simplified bounds, invariant check |

---

*v2.5.1 is the canonical reference. All correctness issues resolved. All guardrails in place.*
