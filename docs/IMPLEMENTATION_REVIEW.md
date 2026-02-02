# DFS Showdown GTO Portfolio Builder - Implementation Review

A deep technical review of the codebase covering correctness, performance, modeling fidelity, and architectural quality. This review examines every core module and identifies issues ranked by their impact on the system's ability to produce profitable, tournament-winning portfolios.

---

## Executive Summary

The system implements a well-structured Monte Carlo pipeline for DFS Showdown portfolio optimization. The mathematical foundations are mostly sound — quantized histograms with prefix-sum payout lookup, copula-based correlated simulation, and greedy marginal selection with covariance penalty are all reasonable choices. However, several issues range from **performance bottlenecks that make the system impractical at production scale** to **modeling assumptions that silently degrade portfolio quality**. The most critical finding is that the hot-path EV loops run in pure Python over 100k simulations, making the system orders of magnitude slower than necessary. The second most critical finding is that the greedy selection loop computes an approximation to marginal EV that may systematically misallocate lineup slots.

---

## Table of Contents

1. [Performance: The Python Loop Problem](#1-performance-the-python-loop-problem)
2. [Greedy Selection: Marginal EV Is Not Marginal](#2-greedy-selection-marginal-ev-is-not-marginal)
3. [Simulation: CPT Percentiles Are Loaded But Never Used](#3-simulation-cpt-percentiles-are-loaded-but-never-used)
4. [Simulation: Double-Counted Correlation Structure](#4-simulation-double-counted-correlation-structure)
5. [Inverse CDF: Poor Tail Representation](#5-inverse-cdf-poor-tail-representation)
6. [Field Model: Sampling Your Own Candidates Back At Yourself](#6-field-model-sampling-your-own-candidates-back-at-yourself)
7. [Enumeration: Brute-Force Despite Having an Optimized Version](#7-enumeration-brute-force-despite-having-an-optimized-version)
8. [Bounds: Slow Version Used Despite Vectorized Alternative](#8-bounds-slow-version-used-despite-vectorized-alternative)
9. [Approx EV: Inconsistent Tie Handling](#9-approx-ev-inconsistent-tie-handling)
10. [Monkey-Patched Correlation Method](#10-monkey-patched-correlation-method)
11. [The "GTO" Claim](#11-the-gto-claim)
12. [No Test Suite](#12-no-test-suite)
13. [DRO Limitations](#13-dro-limitations)
14. [Tournament Composite Score](#14-tournament-composite-score)
15. [Diagnostics: Redundant Simulation Work](#15-diagnostics-redundant-simulation-work)
16. [Minor Issues](#16-minor-issues)
17. [Recommendations Summary](#17-recommendations-summary)

---

## 1. Performance: The Python Loop Problem

**Severity: Critical**
**Files: `ev/portfolio.py`, `ev/approx.py`, `metrics/tournament.py`, `diagnostics.py`**

Every hot-path function in the system iterates over simulations in a pure Python `for` loop. This is the single biggest performance issue and makes the system impractical at production simulation counts.

### The Problem

In `compute_true_portfolio_ev()` (`ev/portfolio.py:61`), the loop:

```python
for sim in range(n_sims):   # n_sims = 100,000
    sim_outcomes = outcomes[:, sim]
    field_scores = score_lineups_vectorized(field_arrays, sim_outcomes)
    field_histogram = ArrayHistogram.from_scores_and_counts(...)
    selected_scores = score_lineups_vectorized(selected_arrays, sim_outcomes)
    combined_histogram = add_entries_to_histogram(...)
    ranks, n_tied = combined_histogram.batch_get_rank_and_ties(selected_scores)
    payouts = payout_lookup.batch_get_payout(ranks, n_tied)
```

This loop calls into NumPy for vectorized lineup scoring (good), but then builds a histogram object, adds entries, rebuilds suffix sums, and does rank lookups — all through Python object creation and method dispatch — **100,000 times**. Each iteration involves:

- 2 calls to `score_lineups_vectorized` (NumPy, fast)
- 1 `ArrayHistogram.from_scores_and_counts` (Python object + NumPy bincount)
- 1 `add_entries_to_histogram` (Python object + copy + add.at + suffix sum)
- 1 `batch_get_rank_and_ties` (NumPy, fast)
- 1 `batch_get_payout` (NumPy, fast)

The Python overhead of object creation, method dispatch, and intermediate array allocation dominates. This same pattern repeats in:
- `compute_approx_lineup_evs()` — another 100k-iteration loop
- `compute_tournament_metrics()` — another 10k-iteration loop
- `decompose_self_competition()` — another 5k-iteration loop
- `compute_profit_covariance_metrics()` — another 5k-iteration loop
- `compute_full_diagnostics()` — yet another 5k-iteration loop for game states

The greedy selection pre-compute phase (`ev/portfolio.py:378-391`) also loops over sims:

```python
for sim in range(n_greedy_sims):     # 10,000
    scores = score_lineups_vectorized(shortlist_arrays, outcomes[:, sim])
    # ...
    field_scores = score_lineups_vectorized(field_arrays, outcomes[:, sim])
```

This pre-computes scores into matrices, which is good — but the per-sim scoring itself could be fully vectorized.

### Impact

At 100k sims, the true EV computation alone likely takes several minutes. With greedy selection (evaluating each candidate at each step), DRO (50 perturbed fields x approx EV each), diagnostics, and tournament metrics, a full pipeline run could easily take 30+ minutes. This makes iterative parameter tuning effectively impossible.

### Recommended Changes

**Option A: Vectorize the histogram operation across sims.** Pre-compute all field scores and selected scores as `[n_lineups, n_sims]` matrices. Build per-sim histograms as a `[n_sims, n_bins]` 2D array in a single `np.apply_along_axis` or explicit broadcasting operation. The greedy loop already does this (`field_histograms` is `[n_greedy_sims, n_bins]`), proving the approach works — but `compute_true_portfolio_ev` and `compute_approx_lineup_evs` don't.

**Option B: Use Numba or Cython for the inner loop.** The histogram-build-rank-payout loop is a perfect candidate for `@numba.njit`. The entire inner loop is just integer arithmetic on arrays.

**Option C: Pre-compute all lineup scores as a `[n_lineups, n_sims]` matrix once**, then reuse across all downstream computations (approx EV, greedy, true EV, diagnostics, tournament metrics). Currently, `score_lineups_vectorized` is called redundantly across every function. This alone could cut wall time in half.

---

## 2. Greedy Selection: Marginal EV Is Not Marginal

**Severity: High**
**File: `ev/portfolio.py:502-620`**

### The Problem

The `_greedy_loop` function claims to select lineups by marginal EV, but what it actually computes is the **raw payout of the candidate in the current combined histogram**, not the true marginal EV `EV(portfolio + candidate) - EV(portfolio)`.

At each step, the loop:

1. Computes `entries_above` from the combined histogram (field + already-selected)
2. For each remaining candidate, looks up rank and payout **as if the candidate is added to the existing histogram**
3. Takes the mean payout across sims as `mean_payouts`
4. Picks the candidate with the highest (optionally covariance-adjusted) mean payout

The critical issue: when candidate C is added to the histogram, it displaces not only field entries but also already-selected portfolio entries. The payout **C receives** is computed, but the **change in payouts of already-selected lineups** due to C's presence is not computed. The true marginal is:

```
marginal_EV(C) = [payout(C | field+portfolio+C)] + [sum(payout(existing_i | field+portfolio+C)) - sum(payout(existing_i | field+portfolio))] - entry_fee
```

The second term (self-competition displacement on existing lineups) is completely missing. This means the greedy loop will systematically over-value candidates that cluster near already-selected lineups — exactly the self-competition the covariance penalty is trying to fix.

### Impact

The covariance penalty partially compensates, but it's a statistical proxy (payout covariance) for the exact displacement computation. The system effectively runs a noisy approximation to greedy marginal selection while claiming exact marginal EV. For a portfolio of 150 lineups in a field of 5,000, the self-competition displacement can be dollars per lineup, making this a materially inaccurate approximation.

### Recommended Changes

Compute true marginal EV by tracking the per-sim payout of every already-selected lineup before and after adding the candidate. This is `O(n_selected * n_sims)` per candidate per step — expensive, but the shortlist is only ~2000 candidates. Alternatively, batch-compute the displacement analytically: when C enters bin `b`, every existing lineup at bin `b` or `b+1` (or nearby) has its rank/tie-count changed, and you can compute the payout delta from the prefix-sum lookup in O(1) per affected lineup.

---

## 3. Simulation: CPT Percentiles Are Loaded But Never Used

**Severity: High**
**File: `simulation/engine.py`, `pipeline.py:217-225`**

### The Problem

The pipeline simulates outcomes for **FLEX players only**:

```python
outcomes = simulate_outcomes(
    data.flex_players, n_sims, ...
)
```

CPT scores are then derived mechanically from FLEX scores in `score_lineups_vectorized()`:

```python
cpt_scores = (outcomes[cpt_idx] * 15 + 5) // 10  # 1.5x of FLEX base
```

This means the CPT percentile distributions loaded from the CSV (`dk_25_percentile` through `dk_99_percentile` for CPT rows) are **completely unused**. The CPT simulation is a deterministic 1.5x transform of the FLEX simulation.

### Impact

In reality, the CPT slot applies a 1.5x multiplier to DraftKings scoring, so the CPT score IS 1.5x the base score. The percentile data in the CSV for CPT rows should already reflect this (they should be 1.5x the FLEX percentiles). So the approach is defensible IF the CSV generator provides CPT percentiles that are exactly 1.5x FLEX percentiles. But:

1. If the projection provider models CPT-specific effects (e.g., CPT concentration changes the variance structure), those are lost.
2. The CPT `ShowdownPlayer` objects are created with their own percentile dictionaries, std, and projection — all of which are loaded, stored in memory, and never used for simulation. This is misleading to anyone reading the code.
3. The correlation matrix is built from FLEX players only, meaning CPT-CPT or CPT-FLEX correlations are derived from the same FLEX-FLEX matrix implicitly.

### Recommended Changes

Either: (a) explicitly document that CPT simulation is derived from FLEX base scores and that CPT percentile data is not used, or (b) simulate a combined player universe and use CPT percentiles directly for CPT slots. Option (b) requires careful handling since CPT and FLEX for the same player are the same underlying score, just scaled differently.

---

## 4. Simulation: Double-Counted Correlation Structure

**Severity: Medium-High**
**File: `simulation/engine.py:82-86`**

### The Problem

When both a correlation matrix and game environment variance decomposition are provided, correlation is modeled twice:

1. **Copula correlations**: Players on the same team get positive correlation via the archetype-based correlation matrix (e.g., QB-WR same team = 0.65).
2. **Game environment**: After copula transformation, `_apply_game_environment()` applies an additive game factor that shifts all same-team players up or down together.

These two mechanisms both induce correlation between same-team players, and they're applied sequentially rather than decomposed into a coherent model. The game environment step modifies scores AFTER the copula has already produced correlated outcomes.

Looking at the math in `_apply_game_environment()`:

```python
player_component = proj + sqrt(1 - gs) * (outcomes[pidx] - proj)
game_component = sqrt(gs) * std * g
adjusted[pidx] = player_component + game_component
```

The player component compresses variance toward the projection, and the game component adds shared game-state noise. Total variance is preserved (`Var = std²`), but the inter-player correlation structure from the copula is modified. Players with high `game_share` will have their copula-induced correlations compressed (the `sqrt(1-gs)` factor reduces the copula signal) and replaced with game-factor correlation. This is somewhat principled but:

- The final correlation between two same-team players is no longer what the archetype config specifies.
- There's no documentation of the effective correlation after both transformations.
- Users tuning the archetype correlation config will get different actual correlations than they specify.

### Impact

Empirical Spearman correlations between simulated player outcomes will differ from the input correlation matrix. This makes it hard to validate the simulation or interpret the archetype correlation config. The system may produce more correlated outcomes than intended.

### Recommended Changes

Either: (a) use the copula correlation matrix as the ONLY source of inter-player correlation (remove game environment or use it only when no explicit correlations are provided), or (b) derive the copula correlation matrix from the game environment model (the effective correlation between two players should account for both the copula and game environment, requiring you to back-solve for what copula correlation produces the desired total correlation after the game environment transform), or (c) apply game environment WITHIN the copula (model it as a latent factor in the multivariate normal before CDF transformation).

---

## 5. Inverse CDF: Poor Tail Representation

**Severity: Medium**
**File: `simulation/engine.py:187-251`**

### The Problem

The inverse CDF is built from 6 percentile points (p25, p50, p75, p85, p95, p99) with piecewise linear interpolation and ad-hoc extrapolation:

- **Below p25**: Extrapolated using the p25→p50 slope, mirrored downward. Floored at 0.
- **Above p99**: Extrapolated using the p95→p99 slope, extended by 0.01 (1 percentile point).

Issues:

1. **The p0 extrapolation uses a linear mirror of p25-p50 slope.** This means the distribution below p25 is assumed to be symmetric with the p25-p50 region. In reality, DFS scoring floors at 0 (a player can't score negative), creating a left-truncated distribution. The linear extrapolation can produce negative scores that are then clipped to 0 by `np.maximum(quantized, 0)` in line 92, creating a point mass at 0 that distorts the lower tail.

2. **The p100 extrapolation adds only `slope * 0.01` above p99.** This is a mere 1 percentile point of extension. For a player with p95=20 and p99=30, the slope is 250 per unit, giving p100=32.5. The true tail extends much further — a player can score 40+ DFS points in outlier games. The ceiling is too compressed, which systematically undervalues high-ceiling players.

3. **Only 8 interpolation points** (p0, p25, p50, p75, p85, p95, p99, p100) define the entire distribution. The p0-p25 region (25% of probability mass) is represented by a single linear segment. The p85-p95 region (10% of mass) is also one segment. This is very coarse.

### Impact

For tournament DFS, the tails matter enormously. The top 1% of outcomes drive most of the EV in GPP tournaments. Compressing the right tail directly reduces the system's ability to identify high-ceiling lineups. Inflating the left tail with a point mass at 0 makes downside risk look worse than it is.

### Recommended Changes

1. Fit a parametric distribution (e.g., Johnson SU, skew-normal, or piecewise exponential) to the percentile points instead of using piecewise linear interpolation. This gives smooth, well-behaved tails.
2. At minimum, extend the right tail further. Use `p99 + 2 * (p99 - p95)` or similar to allow for genuine outlier games.
3. For the left tail, use a truncated distribution at 0 rather than linear extrapolation + clipping.
4. If the projection source provides `dk_std`, use it as a constraint on the fitted distribution's variance, creating a more accurate shape.

---

## 6. Field Model: Sampling Your Own Candidates Back At Yourself

**Severity: Medium**
**File: `field/generator.py:434-523`**

### The Problem

The default field generation method (`generate_field_simulated`) samples field lineups from the same enumerated candidate pool used for the user's own lineup selection:

```python
field_arrays, field_counts = generate_field_simulated(
    candidate_arrays, outcomes,    # YOUR candidate lineups
    cpt_players, flex_players,
    n_field=field_size, ...
)
```

The field is literally a weighted subsample of your own candidate space. The weights combine lineup quality (mean score across sims) with ownership, controlled by `field_sharpness` and `ownership_power`.

Issues:

1. **The field can only contain lineups YOU would consider.** If your enumeration excludes certain lineups (e.g., via `min_projection` filter or salary floor), the field also excludes them. Real opponents will have different filters.

2. **The quality scoring uses YOUR simulation outcomes.** If your correlation assumptions or percentile inputs are wrong, the field quality assessment inherits those errors. You're not modeling an independent opponent view — you're modeling an opponent who uses your exact projection system with different weighting.

3. **No structural modeling of opponent behavior.** Real DFS opponents don't uniformly sample from all valid lineups weighted by quality × ownership. They build around game narratives (e.g., "shoot-out game, stack QB+WR"), they have favorite players, they follow industry consensus. The legacy `generate_field()` method at least models QB pairing, bring-back, and team split priors. The simulated method loses all of this structural information.

4. **The field and your candidates are correlated.** Since the field is sampled from the same space, there's a systematic correlation between field composition and your candidate scores that wouldn't exist in reality. This can bias EV estimates.

### Impact

The field model determines the competitive environment against which all EV calculations are made. If the field is unrealistically similar to your own candidate space, the system will underestimate the benefit of contrarian plays and overestimate the benefit of chalk plays.

### Recommended Changes

1. Use the structural `generate_field()` method as the primary field generation, or blend it with the simulated method.
2. Add noise to the field generation that represents the diversity of opponent strategies — different projection sets, different correlation assumptions, different salary utilization preferences.
3. Consider a two-population model: sharp opponents (optimizers) + recreational opponents (random/narrative-driven), blended at some ratio.
4. At minimum, decouple the field quality scoring from the user's simulation outcomes by using a subset of sims or independent noise.

---

## 7. Enumeration: Brute-Force Despite Having an Optimized Version

**Severity: Medium**
**File: `candidates/enumeration.py:17-110`, `pipeline.py:235-239`**

### The Problem

The pipeline calls `enumerate_lineups()` (the brute-force version) rather than `enumerate_lineups_optimized()` (the branch-and-bound version):

```python
candidates = enumerate_lineups(
    data.cpt_players, data.flex_players,
    salary_cap=salary_cap,
    cpt_to_flex_map=data.cpt_to_flex_map
)
```

The brute-force version generates all `C(n_flex - 1, 5)` combinations for each CPT, then filters by salary cap. For a typical slate with ~25 FLEX players, this is `C(24, 5) = 42,504` combos per CPT × ~25 CPTs = ~1M combinations to evaluate. The salary filter rejects a large fraction.

The optimized version (`enumerate_lineups_optimized`) sorts by salary and prunes branches where the minimum remaining salary already exceeds the budget. This can prune large subtrees. However, the optimized version has a subtle bug in the salary computation:

```python
# Lines 199-201: Confusing and incorrect salary calculation
total_salary = cpt_salary + sum(s for _, s in
                                 [(i, valid_flex[j][1]) for j, (i, _) in enumerate(valid_flex)
                                  if i in [valid_flex[k][0] for k in current_combo]])
# Then immediately recalculated correctly:
flex_salary = sum(valid_flex[k][1] for k in current_combo)
```

The first calculation is garbled (list comprehension inside a generator inside a sum, with index confusion). The second calculation is correct and used for the actual lineup. The dead code on lines 199-201 is confusing but harmless.

Additionally, the optimized version doesn't enforce `require_both_teams`, which is on by default in the standard version.

### Impact

Enumeration is not the bottleneck (it runs once), but on larger slates or with no `min_projection` filter, it could be slow. More importantly, the existence of two divergent implementations with different feature sets is a maintenance hazard.

### Recommended Changes

1. Merge the salary pruning from `enumerate_lineups_optimized` into the main `enumerate_lineups` function.
2. Remove the dead code in the optimized version (lines 199-201).
3. Add `require_both_teams` support to the optimized version, or deprecate it.

---

## 8. Bounds: Slow Version Used Despite Vectorized Alternative

**Severity: Low-Medium**
**File: `simulation/bounds.py`, `pipeline.py:230`**

### The Problem

The pipeline calls `compute_guaranteed_score_bounds(outcomes)` which uses the Python for-loop version (`bounds.py:12-64`):

```python
for sim in range(n_sims):     # 100,000
    sim_outcomes = outcomes[:, sim]
    top_6 = ...
    # compute max score for this sim
```

A fully vectorized version `compute_bounds_vectorized()` exists in the same file and does the same computation in a single NumPy operation:

```python
partitioned = np.partition(outcomes, -6, axis=0)[-6:, :]
cpt_base = partitioned.max(axis=0)
# ...
```

The vectorized version is never called.

### Impact

At 100k sims, the loop version is noticeably slower (seconds vs. milliseconds). Not a huge deal in the overall pipeline, but it's free performance left on the table.

### Recommended Changes

Replace the call in `pipeline.py:230` with `compute_bounds_vectorized(outcomes)`. Or better, make the main function use the vectorized implementation internally.

---

## 9. Approx EV: Inconsistent Tie Handling

**Severity: Medium**
**File: `ev/approx.py:153-161`**

### The Problem

In `compute_approx_lineup_evs()`, each candidate's rank is looked up against the **field-only histogram**, but the tie count is incremented by 1 to account for the candidate itself:

```python
ranks, tied_in_field = histogram.batch_get_rank_and_ties(chunk_scores)
n_tied = tied_in_field + 1
payouts = payout_lookup.batch_get_payout(ranks, n_tied)
```

The issue: `ranks` is computed from the field-only histogram. If the field has 3 entries at score 150, and the candidate also scores 150, the field histogram says `entries_above[150] = X, count_at[150] = 3`. The rank would be `X + 1`, and the tied count would be `3 + 1 = 4`.

But the correct rank with the candidate included should reflect that the candidate is now one of the 4 entries at score 150. The rank remains `X + 1` (same number above), but the tie pool is now split 4 ways instead of 3. This IS what the code computes, so it's actually correct for the rank.

However, there's a subtlety: the `entries_above` computation doesn't include the candidate, so if the candidate scores ABOVE the field max, `entries_above` would say 0 (rank 1), but with the candidate added, the candidate should also be at rank 1 with 1 tie (itself). The `n_tied = tied_in_field + 1 = 0 + 1 = 1` handles this correctly.

On closer inspection, this is actually correct. The rank from field-only is the right rank (entries above are the same whether or not the candidate is present at a lower or equal score). The +1 to ties accounts for the candidate. **This is correct.** I retract this as an issue — but I'll note that the correctness is non-obvious and deserves a comment.

---

## 10. Monkey-Patched Correlation Method

**Severity: Medium**
**File: `data/correlations.py:406-419`**

### The Problem

The K/DST correlation handling is implemented by monkey-patching the `ArchetypeCorrelationConfig.get_correlation` method at module load time:

```python
_original_get_correlation = ArchetypeCorrelationConfig.get_correlation

def _extended_get_correlation(self, archetype1, archetype2, same_team):
    if archetype1 in ('K', 'DST') or archetype2 in ('K', 'DST'):
        k_dst_corr = get_k_dst_correlation(archetype1, archetype2, same_team)
        if k_dst_corr is not None:
            return k_dst_corr
    return _original_get_correlation(self, archetype1, archetype2, same_team)

ArchetypeCorrelationConfig.get_correlation = _extended_get_correlation
```

Issues:

1. **Fragile**: If `correlations.py` is imported in a different order, or if the method is called before the patch, behavior changes.
2. **Hard to discover**: Someone reading `ArchetypeCorrelationConfig.get_correlation` sees the original implementation (which returns defaults for K/DST), and would never know the behavior is different at runtime without finding the monkey-patch at the bottom of the file.
3. **Not subclass-safe**: Any subclass overriding `get_correlation` would lose the K/DST extension.

### Recommended Changes

Move the K/DST logic into the `get_correlation` method itself, checking for K/DST archetypes before the generic lookup. Or use a proper method override pattern (inheritance, decorator, or wrapper class).

---

## 11. The "GTO" Claim

**Severity: Conceptual/Medium**
**Affects: Project framing and user expectations**

### The Problem

The project is named "GTO Portfolio Builder" but does not implement game-theory optimal strategies. GTO in the poker sense means computing a Nash equilibrium — a strategy that cannot be exploited regardless of what opponents do. This system computes:

1. Expected value against a modeled field (not a best-response strategy)
2. Robustness via DRO against perturbed fields (uncertainty quantification, not equilibrium computation)
3. Diversification via covariance penalty (portfolio theory, not game theory)

None of these are GTO in any formal sense. The system is an **EV-maximizing optimizer with robustness** — which is a perfectly valid approach! — but calling it GTO may create false expectations.

True GTO for DFS would involve:
- Modeling all opponents as rational agents who also optimize
- Computing the fixed-point where no player can improve their portfolio given others' strategies
- This is computationally intractable for realistic DFS field sizes

### Impact

Users may believe their portfolios are "unexploitable" when they're actually just EV-maximized against a particular field model. If the field model is wrong (and it always will be), the portfolio could be significantly exploited by observant opponents.

### Recommendation

Consider renaming to "EV-Optimal" or "Robust Portfolio Builder" — or document clearly what "GTO" means in this context (i.e., robust EV optimization, not Nash equilibrium).

---

## 12. No Test Suite

**Severity: High**
**File: `showdown_gto/tests/` (empty)**

### The Problem

The `tests/` directory exists but contains no tests. The only validation mechanism is `showdown_gto_reference.py` (941 lines), which is a standalone reference implementation — not an automated test suite.

For a system that directly influences real-money decisions, the absence of tests is a significant risk. Key properties that should be tested:

- **Histogram invariant**: `combined.total_entries == field_size + n_selected` (tested at runtime but not in unit tests)
- **Payout correctness**: Tie-splitting produces correct payouts for known configurations
- **Score quantization round-trip**: `dequantize(quantize(x)) ≈ x`
- **Correlation matrix PSD guarantee**: After Higham projection, matrix is actually PSD
- **Simulation distribution**: Empirical percentiles match input percentiles
- **CPT scoring formula**: `(base * 15 + 5) // 10` gives correct 1.5x for all values
- **Bounds guarantee**: No lineup score ever exceeds computed bounds
- **Enumeration completeness**: All valid lineups are generated
- **EV sign**: Portfolio EV should be positive for a favorable contest structure

### Recommended Changes

Build a test suite covering at minimum:
1. Unit tests for scoring, histogram, payout lookup
2. Property tests for the simulation distribution (check that p50 of simulated outcomes matches input p50)
3. Integration tests for a small slate (2 players per team, few lineups)
4. Regression tests with fixed seeds to catch behavioral changes

---

## 13. DRO Limitations

**Severity: Medium**
**Files: `field/robust.py`, `pipeline.py:328-333`**

### Problems

1. **DRO only works with fixed field mode.** The pipeline explicitly skips DRO when `field_mode == "resample_per_sim"`:

   ```python
   if dro_perturbations > 0 and field_mode != "fixed":
       logger.warning("DRO is only supported with field_mode='fixed'...")
   ```

   This seems like an artificial limitation. The `generate_perturbed_p_lineups()` function in `robust.py` already supports generating perturbed probability vectors for the resample mode. It's just not wired into the pipeline.

2. **Layer 2 (HHI perturbation) requires a calibration file that is generated by `scripts/validate_ownership.py`.** If this file doesn't exist, Layer 2 is silently disabled. Users who don't know to run the calibration script get DRO with only Layer 1 (ownership perturbation), which is a significant reduction in robustness.

3. **The representative scenario selection (`select_representative_scenarios`) runs AFTER computing EVs for all perturbations.** It would be more efficient to select representative perturbations BEFORE computing EVs, reducing the number of expensive approx-EV computations from `n_perturbations` to `representative_k`.

4. **The FLEX ownership perturbation variance partition (global=20%, team=40%, role=20%, idio=20%) is hardcoded.** These proportions significantly affect the DRO behavior but cannot be tuned by the user.

### Recommended Changes

1. Wire DRO into `resample_per_sim` mode using `generate_perturbed_p_lineups`.
2. Add a warning or automatic calibration fallback if the HHI calibration file is missing.
3. Move representative scenario selection before EV computation.
4. Expose the variance partition as configurable parameters.

---

## 14. Tournament Composite Score

**Severity: Low-Medium**
**File: `metrics/tournament.py:117-122`**

### The Problem

The composite score normalization is ad-hoc and unit-inconsistent:

```python
composite_score = (
    W_TOP_1PCT * top_1pct_rate                        # 0.50 * [0, 1]
    + W_CEILING * min(ceiling_ev_per_dollar / 100, 1.0) # 0.35 * [0, 1]
    + W_WIN_RATE * win_rate * 100                      # 0.10 * [0, ~0.5]
    + W_ROI * max(min(roi_pct / 100, 1.0), 0.0)       # 0.05 * [0, 1]
)
```

Issues:

1. `win_rate * 100` for a portfolio of 150 lineups in a 5000-person contest is `~150/5000 * 100 = 3.0`, so this term contributes `0.10 * 3.0 = 0.30`, which can dominate the entire composite. The "scale win_rate up" comment suggests awareness of this, but the scaling is too aggressive.

2. The ceiling EV normalization divides by entry cost then by 100, capping at 1.0. For a $5 entry, ceiling_ev of $500 gives `500/5/100 = 1.0` (maxed out). This means any ceiling above $500 for a $5 entry is treated identically — no discrimination between $500 and $5000 ceiling.

3. The weights (50/35/10/5) are not derived from any principled analysis of what predicts tournament success. They're tuning parameters disguised as constants.

### Recommended Changes

1. Normalize all components to the same [0, 1] range using empirical percentiles or known distribution properties.
2. Use log-scale for ceiling EV to avoid the cap problem.
3. Document the weights as tunable and expose them as parameters.

---

## 15. Diagnostics: Redundant Simulation Work

**Severity: Medium**
**File: `diagnostics.py:389-499`**

### The Problem

`compute_full_diagnostics()` calls five separate functions, each of which independently loops over simulations and scores lineups from scratch:

1. `decompose_self_competition()` — 5000 sims, rebuilds field histogram per sim
2. `compute_profit_covariance_metrics()` — 5000 sims, rebuilds field histogram per sim
3. Score distribution loop — 5000 sims, scores selected lineups per sim
4. Game-state coverage loop — 5000 sims, scores field and selected per sim
5. `compute_tournament_metrics()` — 10,000 sims, rebuilds everything per sim

Each function independently calls `score_lineups_vectorized` for field and selected lineups, builds histograms, computes suffix sums, etc. If a single pass over sims collected all the data needed for every diagnostic, the total computation would be ~5x less.

### Recommended Changes

Create a single `_diagnostic_sim_pass()` function that iterates over sims once and collects:
- Per-lineup payouts (for covariance)
- Best rank per sim (for tournament metrics)
- Per-sim game state (for coverage)
- Per-lineup self-competition cost (for decomposition)

Then compute all diagnostics from the collected data.

---

## 16. Minor Issues

### 16a. No Parallelism
**File: Entire codebase**

The entire pipeline is single-threaded. Monte Carlo simulation, approx EV computation, and DRO field generation are all embarrassingly parallel. Even `multiprocessing.Pool` with chunked sim ranges would provide linear speedup.

### 16b. Ownership Floor at 0.01
**Files: Multiple**

Throughout the codebase, ownership is floored at 0.01 (or 0.01%). In logit space (used in DRO FLEX perturbation), `logit(0.01/100) = logit(0.0001) ≈ -9.2`. This extreme logit value means the Gaussian noise in logit space barely moves these players' ownership, effectively freezing near-zero-ownership players at near-zero regardless of perturbation scale. This may be intentional but should be documented.

### 16c. `FieldGenConfig.__post_init__` Only Validates Temperature
**File: `types.py:162-167`**

Only temperature is validated. Other fields like `field_sharpness` (should be non-negative), `ownership_power` (should be [0, 1] or at least non-negative), and `quality_sims` (should be positive) are not validated.

### 16d. Pipeline Function Signature Bloat
**File: `pipeline.py:45-81`**

`run_portfolio_optimization()` accepts 33 parameters. `run_multi_contest_optimization()` accepts 29. This is a code smell indicating that a configuration object pattern would be more appropriate. Some parameters like `dro_perturbations`, `dro_scale`, `dro_hhi_scale`, `dro_aggregation`, `dro_calibration_path`, `dro_cvar_alpha`, and `dro_representative_k` naturally group into a "DRO config" object.

### 16e. `compute_marginal_ev` Is Never Called
**File: `ev/portfolio.py:121-187`**

The function `compute_marginal_ev()` computes true marginal EV by calling `compute_true_portfolio_ev` twice — once with and once without the candidate. This is the correct way to compute marginal EV, but it's never called anywhere in the codebase. The greedy loop uses the approximate version instead. This function should either be the one used by the greedy loop (at higher cost), or it should be removed.

### 16f. Inconsistent Score Clamping
**File: `scoring/histogram.py`**

The `ArrayHistogram.from_scores_and_counts` uses strict mode by default (raises on out-of-range), but the greedy loop and diagnostics use `np.clip` to force bins into range:

```python
bins = np.clip(scores - min_score, 0, n_bins - 1)  # Silent clamp
```

This bypasses the fail-fast semantic that the histogram was designed to enforce.

### 16g. Log-Space Ownership Product Missing Temperature
**File: `field/generator.py:407-419`**

In `_compute_quality_ownership_weights`, the log-ownership is computed without the temperature scaling that's applied in the legacy ownership-based methods. The simulated method uses `ownership_power` as a direct exponent on ownership, while the legacy method uses `ownership^(1/temperature)`. These produce different field characteristics for the same ownership inputs.

### 16h. Seed Handling Inconsistency
**Files: `pipeline.py`, `field/robust.py`**

The pipeline derives independent seeds via `SeedSequence.spawn()`, but several functions (like `generate_perturbed_fields`) use `rng.integers(0, 2**31)` to derive child seeds. This is statistically inferior to `SeedSequence.spawn()` because it doesn't guarantee independence of the resulting RNG streams. The `SeedSequence` approach is used at the top level but not propagated through the function call tree.

---

## 17. Recommendations Summary

### Must-Fix (Impact on EV accuracy or usability)

| # | Issue | Impact | Effort |
|---|-------|--------|--------|
| 1 | **Vectorize sim loops** (approx EV, true EV, tournament metrics) | 10-100x speedup | Medium-High |
| 2 | **Fix greedy marginal** to compute true displacement on existing lineups | More accurate portfolio selection | Medium |
| 3 | **Build a test suite** | Prevent regressions, validate correctness | Medium |
| 4 | **Document/fix CPT simulation** (loaded but unused percentiles) | Reduce confusion, possibly improve accuracy | Low |

### Should-Fix (Quality improvements)

| # | Issue | Impact | Effort |
|---|-------|--------|--------|
| 5 | Improve inverse CDF tail representation | Better high-ceiling lineup identification | Medium |
| 6 | Single-pass diagnostic computation | ~5x faster diagnostics | Medium |
| 7 | Remove monkey-patching in correlations.py | Code clarity, maintainability | Low |
| 8 | Use vectorized bounds computation | Free speedup | Trivial |
| 9 | Wire DRO into resample_per_sim mode | Feature completeness | Low-Medium |
| 10 | Address double-counted correlations | Simulation fidelity | Medium |

### Nice-to-Have (Polish)

| # | Issue | Impact | Effort |
|---|-------|--------|--------|
| 11 | Add parallelism (multiprocessing or joblib) | Further speedup | Medium |
| 12 | Configuration object pattern for pipeline | Code cleanliness | Medium |
| 13 | Fix tournament composite score normalization | Better metric | Low |
| 14 | Merge enumeration implementations | Code cleanliness | Low |
| 15 | Consistent seed handling (SeedSequence throughout) | Statistical rigor | Low |
| 16 | Richer field model (sharp/recreational split) | More realistic opponent modeling | High |

---

## Appendix: What the System Does Well

For balance, here are the aspects of the implementation that are well-done:

1. **Quantized histogram with prefix-sum payout lookup** — The `ArrayHistogram` + `PayoutLookup` design is elegant. O(1) rank lookup with correct tie-splitting via prefix sums is the right approach. The fail-fast semantics on histogram construction catch bugs early.

2. **Copula-based simulation** — Supporting both Gaussian and Student-t copulas with configurable degrees of freedom is well-suited for modeling DFS outcomes. The t-copula tail dependence (shared chi-squared scaling) correctly models boom/bust scenarios.

3. **Streaming approx EV** — The chunked processing in `compute_approx_lineup_evs` avoids O(n_cand × n_sims) memory, which is critical for large candidate pools.

4. **DRO two-layer perturbation model** — The separation of CPT ownership (Dirichlet simplex) from FLEX ownership (correlated logit-space noise) is principled. The three-factor noise model (global + team + role + idiosyncratic) captures realistic ownership error structure.

5. **Greedy selection with covariance penalty** — The Markowitz-style covariance penalty is a reasonable proxy for self-competition reduction. The dynamic gamma scaling (`gamma_pct * max(marginals)`) is a practical choice that adapts to the current EV landscape.

6. **Self-competition invariant check** — The runtime assertion `combined_histogram.total_entries == field_size + n_selected` catches histogram accounting bugs immediately rather than silently producing wrong EVs.

7. **Module separation** — The code is well-organized into focused modules with clear responsibilities. The 5-phase pipeline architecture is easy to follow and modify.

8. **Profile system** — Named profiles (aggressive/balanced/robust/defensive) with consistent parameter sets reduce user cognitive load and promote reproducibility.
