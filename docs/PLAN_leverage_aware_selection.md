# Plan: Leverage-Aware Portfolio Selection

## Status: Ready for Implementation

## Context: What Changed and What We Learned

### Recent Change: Simulation-Based Field Generation (COMPLETED)

We replaced the ownership-only field generator with a projection-aware model (`field/generator.py`). The new `generate_field_simulated()` samples from the enumerated candidate pool weighted by `exp(sharpness * z_quality) * ownership^power`. With default `field_sharpness=5.0`, the field now produces:

- Avg salary $48,556 (97.1% utilization) vs old $46,080 (92.2%)
- 51% of field lineups at $49K+ vs old 26.6%
- Field quality at p99.7 of candidate pool vs old p96.1

Files modified: `types.py`, `field/generator.py`, `pipeline.py`, `cli.py`, `field/robust.py`, `field/__init__.py`, `config.py`. The `salary_utilization_mean`/`salary_utilization_std` dead fields were removed from `FieldGenConfig`.

### Diagnosis: The Real Problem is Portfolio Construction

With a realistic field, we ran diagnostics on the current top-150 portfolio and found **three structural failures that are slate-independent**:

#### 1. Zero Leverage (Pure Chalk)

The top-150 by approx EV is 100% Kenneth Walker III (47% ownership), 93% Jaxon Smith-Njigba (48% ownership), 89% Puka Nacua (46% ownership). Weighted average player ownership in the portfolio is 35.2% vs pool average 22.6% (1.56x overweight on chalk).

**Why this is bad**: In a 5,000-entry contest, ~2,350 opponents also have Kenneth Walker. When KW3 booms, we compete with half the field for payouts. When he busts, all 150 entries bust simultaneously. There is no scenario where our portfolio is differentiated from the field.

**Why it happens**: Top-N selection sorts by mean approx EV. The highest-mean-score lineups naturally contain the highest-projected players, who are also the highest-owned players. There's nothing in the objective function that penalizes high ownership.

#### 2. Extreme Score Band Concentration

Top-150 mean scores span only 54 quantized points (5.4 DFS points). Average score standard deviation is 209 (20.9 DFS points). The mean range / mean std ratio is 0.26 — meaning all 150 lineups are within ~0.26 standard deviations of each other.

**Why this is bad**: In a top-heavy payout structure (1st place pays 100x more than 50th), you want some lineups that can win outright (high ceiling, differentiated) and others that cash reliably (high floor, consistent). Having all 150 in the same score band means you either cash 150 or cash 0 — no barbell, no asymmetric upside.

**Why it happens**: Mean score is highly correlated across similar lineups. The top ~500 candidates by mean score are all slight permutations of the same core players. Top-N selection picks from this narrow band.

#### 3. Self-Competition Cost is 27% of Approx EV

Self-competition cost was $145 on $543 approx EV sum (26.7%). The theoretical expectation for 150 well-diversified entries in a 5,000-entry contest should be closer to 5-15%.

**Why**: When all 150 lineups score similarly (finding #2), they cluster in the same histogram bins. If you have 10 lineups tied at rank 50, they split the payout 10 ways instead of one lineup getting the full amount. High similarity = high tie frequency = high self-competition.

### Evidence These Problems Are Slate-Independent

These failures are **structural properties of top-N selection**, not artifacts of this particular slate:

- **Any** slate's top candidates by mean score will be dominated by highest-projected players
- **Any** slate's highest-projected players will be the highest-owned players (projections drive ownership)
- **Any** top-N selection will produce a tight score band (the top of a continuous distribution is always narrow)
- Self-competition cost scales with lineup similarity, which top-N maximizes

---

## The Plan: Three Changes

### Change 1: Leverage-Adjusted EV (Core Formula)

**What**: Modify the approx EV computation to penalize lineups with high-ownership players, reflecting that their wins are shared with more opponents.

**The Math**:

Standard approx EV computes: `EV(lineup) = mean(payout(rank_vs_field))`

This ignores that when your lineup wins, opponents with the same high-ownership players also win. The leverage adjustment models this:

```
leverage_adjusted_ev[i] = approx_ev[i] * ownership_deflator[i]
```

Where `ownership_deflator` accounts for how many field entries share key players:

```
# For each lineup, compute product of (1 - correlation_with_field) across players
# Simplified: use inverse of joint ownership probability
ownership_deflator[i] = 1.0 / (1.0 + leverage_penalty * mean_player_ownership[i])
```

A more principled version uses the **leverage score** concept from the DFS literature:

```
# Implied ownership: what ownership SHOULD be given projections
# Leverage = implied_ownership / projected_ownership
# Lineups with avg_leverage > 1 are underpriced by the field

mean_lineup_ownership[i] = mean(ownership[player] for player in lineup[i]) / 100
leverage_ev[i] = approx_ev[i] * (1 - leverage_lambda * mean_lineup_ownership[i])
```

Where `leverage_lambda` controls how much to penalize ownership:
- `leverage_lambda = 0`: No penalty (current behavior)
- `leverage_lambda = 0.5`: Moderate penalty (recommended default)
- `leverage_lambda = 1.0`: Heavy penalty (for large-field GPPs)

**Why this is the right approach**: Rather than trying to compute "what ownership should be" (which requires modeling all opponents), we use a simple penalty on actual ownership. High-ownership lineups get their EV reduced because their wins are shared with more opponents. Low-ownership lineups get relatively boosted because their wins are more exclusive.

**Where to implement**: `showdown_gto/ev/approx.py` — add a new function `compute_leverage_adjusted_evs()` that takes `approx_evs`, `candidate_arrays`, player ownership arrays, and `leverage_lambda`. This operates as a post-processing step on approx EVs, so it composes with all existing field modes (fixed, resampled), DRO, and multi-portfolio.

### Change 2: Exposure-Capped Selection

**What**: Add maximum per-player exposure constraints to all selection methods, preventing any single player from dominating the portfolio.

**The Problem**: Even with leverage-adjusted EVs, top-N selection can still produce 80%+ exposure for top players because the EV ordering doesn't change dramatically. We need a hard constraint.

**The Implementation**:

Add an `exposure_cap` parameter (default: 0.6 = 60% of lineups) to the selection pipeline. During selection, track per-player exposure counts and skip lineups that would push any player over the cap.

```python
def select_with_exposure_cap(
    candidate_indices_sorted_by_ev: List[int],
    candidate_arrays: LineupArrays,
    cpt_to_flex_map: Dict[int, int],
    n_select: int,
    n_flex: int,
    exposure_cap: float = 0.6  # max fraction of lineups containing any single player
) -> List[int]:
    max_count = int(n_select * exposure_cap)
    player_counts = np.zeros(n_flex, dtype=np.int32)
    selected = []

    for idx in candidate_indices_sorted_by_ev:
        # Check all players in this lineup against cap
        players_in_lineup = get_player_indices(candidate_arrays, idx, cpt_to_flex_map)
        if all(player_counts[p] < max_count for p in players_in_lineup):
            selected.append(idx)
            for p in players_in_lineup:
                player_counts[p] += 1
            if len(selected) == n_select:
                break

    return selected
```

**Key design decision**: The exposure cap is on FLEX player indices (since CPT and FLEX for the same player share the same underlying player). The `cpt_to_flex_map` handles the CPT→FLEX mapping.

**Where to implement**: `showdown_gto/ev/approx.py` or a new module `showdown_gto/ev/selection.py`. Applied after EV ranking but before true portfolio EV computation. Works with top-N, and can be composed with DPP/barbell/greedy (applied as a filter on their shortlists or outputs).

**Parameters**:
- `--exposure-cap`: float, default 0.6 (60%)
- Setting to 1.0 disables the cap (backward compatible)

### Change 3: Portfolio Diagnostic Framework

**What**: Add a standardized diagnostic suite that runs after portfolio selection on any slate, producing metrics that can be tracked across slates to detect overfitting or structural problems.

**The Metrics** (all computed in a single post-selection pass):

```python
@dataclass
class PortfolioDiagnostics:
    # Leverage metrics
    mean_player_ownership: float      # Weighted avg ownership of selected players
    pool_mean_ownership: float        # Avg ownership across all players
    ownership_ratio: float            # mean_player_ownership / pool_mean_ownership (target: <1.2)
    max_player_exposure: float        # Highest single-player exposure (target: <0.6)
    top5_player_exposures: List[Tuple[str, float, float]]  # (name, exposure%, ownership%)

    # Diversity metrics
    mean_score_range: float           # Max - min of lineup mean scores
    mean_score_std: float             # Std of lineup mean scores
    score_band_ratio: float           # mean_score_range / avg_lineup_std (target: >1.0)
    unique_cpts_used: int             # Number of distinct CPT players (target: 5+)

    # Self-competition metrics
    self_comp_ratio: float            # self_comp_cost / approx_ev_sum (target: <0.15)

    # Quality metrics
    portfolio_mean_score: float       # Avg mean score of selected lineups
    portfolio_score_percentile: float # Percentile of portfolio avg score in candidate pool
    field_quality_percentile: float   # Percentile of field avg score in candidate pool
```

**Where to implement**: New function `compute_portfolio_diagnostics()` in `showdown_gto/ev/portfolio.py` (or new file `showdown_gto/diagnostics.py`). Called from `pipeline.py` after selection, results added to the diagnostics dict.

**CLI output**: After the existing portfolio results display, add a "PORTFOLIO HEALTH" section:

```
PORTFOLIO HEALTH
  Ownership ratio:     1.12x  (target: <1.2)  OK
  Max exposure:        58%    (target: <60%)   OK
  Score band ratio:    1.34   (target: >1.0)   OK
  Unique CPTs:         8      (target: 5+)     OK
  Self-comp ratio:     11.2%  (target: <15%)   OK
```

**Generalizability**: These thresholds are derived from structural properties:
- `ownership_ratio < 1.2`: Portfolio should not be >20% overweight vs pool (any slate)
- `max_exposure < 0.6`: No player in >60% of lineups (prevents single-player dependence)
- `score_band_ratio > 1.0`: Lineups should span at least 1 std of their own variance (any slate)
- `unique_cpts >= 5`: At least 5 different CPT players used (any 2-team showdown has ~15-22 CPT options)
- `self_comp_ratio < 0.15`: Self-competition should not exceed 15% of approx EV (any contest size)

---

## File-by-File Implementation Details

### File 1: `showdown_gto/ev/approx.py`

**Add function: `compute_leverage_adjusted_evs()`**

```python
def compute_leverage_adjusted_evs(
    approx_evs: np.ndarray,
    candidate_arrays: LineupArrays,
    cpt_players: List[ShowdownPlayer],
    flex_players: List[ShowdownPlayer],
    cpt_to_flex_map: Dict[int, int],
    leverage_lambda: float = 0.5
) -> np.ndarray:
    """
    Adjust approx EVs to penalize high-ownership lineups.

    leverage_adjusted_ev[i] = approx_ev[i] * (1 - lambda * mean_own[i])

    where mean_own[i] is the average ownership (0-1) of all 6 players in lineup i.

    Args:
        approx_evs: [n_candidates] raw approx EVs
        candidate_arrays: Lineup arrays for index lookups
        cpt_players: CPT player pool (for ownership)
        flex_players: FLEX player pool (for ownership)
        cpt_to_flex_map: CPT→FLEX mapping
        leverage_lambda: Penalty strength (0=none, 0.5=moderate, 1.0=heavy)

    Returns:
        leverage_evs: [n_candidates] adjusted EVs
    """
```

Implementation notes:
- Compute mean ownership vectorized: use `flex_players[i].ownership / 100` for each player
- For CPT, map to FLEX index via `cpt_to_flex_map` to get the underlying player ownership
- Mean across 6 players (1 CPT + 5 FLEX) per lineup
- Multiply approx EV by `(1 - leverage_lambda * mean_ownership)`
- Lineups with 50% avg ownership get a 25% penalty at lambda=0.5
- Lineups with 10% avg ownership get a 5% penalty at lambda=0.5

### File 2: `showdown_gto/ev/selection.py` (NEW FILE)

**Add function: `select_with_exposure_cap()`**

```python
def select_with_exposure_cap(
    ranked_indices: List[int],
    candidate_arrays: LineupArrays,
    cpt_to_flex_map: Dict[int, int],
    n_flex: int,
    n_select: int,
    exposure_cap: float = 0.6
) -> List[int]:
    """
    Select lineups from pre-ranked list respecting per-player exposure caps.

    Iterates through ranked_indices in order, skipping any lineup that would
    push a player's exposure above the cap.

    Args:
        ranked_indices: Candidate indices sorted by EV (best first)
        candidate_arrays: All candidate lineups
        cpt_to_flex_map: CPT→FLEX index mapping
        n_flex: Total number of FLEX players
        n_select: Number of lineups to select
        exposure_cap: Max fraction of portfolio for any player (0.0-1.0)

    Returns:
        selected_indices: List of selected candidate indices
    """
```

Implementation notes:
- `max_count = int(n_select * exposure_cap)` — max times any player can appear
- For each candidate, extract FLEX indices: the 5 flex slots + the CPT mapped to FLEX
- Track `player_counts[flex_idx]` — increment when lineup selected
- Skip lineup if ANY player would exceed `max_count`
- If we can't fill `n_select` lineups, log a warning and return what we have
- Helper: `get_flex_player_indices(candidate_arrays, idx, cpt_to_flex_map) -> List[int]`

### File 3: `showdown_gto/diagnostics.py` (NEW FILE)

**Add function: `compute_portfolio_diagnostics()`**

```python
def compute_portfolio_diagnostics(
    selected_indices: List[int],
    candidate_arrays: LineupArrays,
    outcomes: np.ndarray,
    cpt_players: List[ShowdownPlayer],
    flex_players: List[ShowdownPlayer],
    cpt_to_flex_map: Dict[int, int],
    approx_evs: np.ndarray,
    true_ev: float,
    contest: ContestStructure,
    n_diagnostic_sims: int = 1000
) -> Dict:
    """
    Compute slate-independent portfolio health metrics.

    Returns dict with:
        ownership_ratio, max_exposure, score_band_ratio,
        unique_cpts, self_comp_ratio, top5_exposures, etc.
    """
```

Implementation notes:
- Player exposure: iterate selected lineups, count per-player appearances using `cpt_to_flex_map`
- Score stats: score selected lineups across `n_diagnostic_sims` sims, compute mean/std per lineup
- `score_band_ratio = (max(lineup_means) - min(lineup_means)) / mean(lineup_stds)`
- `ownership_ratio = weighted_avg_ownership / pool_avg_ownership`
- `self_comp_ratio = (sum(approx_evs[selected]) - true_ev) / sum(approx_evs[selected])`
- Return as flat dict for easy JSON serialization

**Add function: `format_portfolio_health(diagnostics: Dict) -> str`**

Formats the health report as a string for CLI display with pass/fail indicators.

### File 4: `showdown_gto/pipeline.py`

**Changes to `run_portfolio_optimization()`:**

1. Add parameters:
   - `leverage_lambda: float = 0.5`
   - `exposure_cap: float = 0.6`

2. After computing `approx_evs` (and optionally `robust_evs` from DRO), apply leverage adjustment:
   ```python
   if leverage_lambda > 0:
       from .ev.approx import compute_leverage_adjusted_evs
       selection_evs = compute_leverage_adjusted_evs(
           selection_evs, candidate_arrays,
           data.cpt_players, data.flex_players,
           data.cpt_to_flex_map, leverage_lambda
       )
   ```

3. After selection (all methods), apply exposure cap if `exposure_cap < 1.0`:
   ```python
   if exposure_cap < 1.0:
       from .ev.selection import select_with_exposure_cap
       selected_indices = select_with_exposure_cap(
           selected_indices, candidate_arrays,
           data.cpt_to_flex_map, data.n_flex,
           n_select, exposure_cap
       )
   ```

4. After computing true portfolio EV, run diagnostics:
   ```python
   from .diagnostics import compute_portfolio_diagnostics
   health = compute_portfolio_diagnostics(
       selected_indices, candidate_arrays, outcomes,
       data.cpt_players, data.flex_players,
       data.cpt_to_flex_map, approx_evs, true_ev, contest
   )
   diagnostics['portfolio_health'] = health
   ```

**Same changes to `run_multi_portfolio_optimization()`** — apply leverage adjustment to `multi_evs[k]` per portfolio, apply exposure cap per portfolio selection.

### File 5: `showdown_gto/cli.py`

**Add CLI options:**

```python
@click.option('--leverage-lambda', type=float, default=0.5,
    help='Ownership penalty strength: 0=none, 0.5=moderate (default), 1.0=heavy')
@click.option('--exposure-cap', type=float, default=0.6,
    help='Max single-player exposure: 0.6=60% (default), 1.0=no cap')
```

**Add to display output** (after portfolio results):

```python
if 'portfolio_health' in diag:
    from .diagnostics import format_portfolio_health
    click.echo(format_portfolio_health(diag['portfolio_health']))
```

**Pass new params through** to `run_portfolio_optimization()` and `_run_multi_portfolio()`.

### File 6: `showdown_gto/ev/__init__.py`

Export new function:
```python
from .selection import select_with_exposure_cap
```

---

## Parameter Guidance

### leverage_lambda

| Value | Effect | Use Case |
|-------|--------|----------|
| 0.0 | No ownership penalty (current behavior) | Cash games, double-ups |
| 0.3 | Light penalty | Small-field GPPs (<1000 entries) |
| 0.5 | Moderate penalty (DEFAULT) | Standard GPPs (1000-10000 entries) |
| 0.75 | Heavy penalty | Large-field GPPs (>10000 entries) |
| 1.0 | Maximum penalty | Ultra-large field, max differentiation |

**Intuition**: In a 5000-entry contest, a lineup with 40% avg ownership shares its wins with ~2000 opponents. A lineup with 10% avg ownership shares with ~500. The lambda controls how much we discount for this sharing.

### exposure_cap

| Value | Effect | Use Case |
|-------|--------|----------|
| 0.4 | Very diverse | Large portfolios (150+ entries), large fields |
| 0.6 | Moderate diversity (DEFAULT) | Standard 150-entry portfolios |
| 0.8 | Light constraint | Small portfolios (<50 entries) |
| 1.0 | No constraint (backward compatible) | Cash games, legacy behavior |

**Intuition**: With 150 lineups and cap=0.6, no player appears in more than 90 lineups. This forces the optimizer to find alternative builds featuring different core players.

---

## Interaction with Existing Features

### DRO (Wasserstein Robust)

Leverage adjustment is applied AFTER DRO robust EVs are computed. The robust EV accounts for field uncertainty; the leverage penalty accounts for ownership sharing. They're orthogonal:

```
DRO robust EVs → leverage adjustment → selection with exposure cap
```

### DPP (Quality-Diversity)

DPP already provides structural diversity via its kernel. Leverage adjustment modifies the quality scores fed into the DPP kernel, making low-ownership lineups relatively more attractive. Exposure cap acts as a safety net on top of DPP's implicit diversity.

### Barbell (Floor/Ceiling)

Leverage adjustment is applied to the EVs that barbell uses for floor and ceiling tier selection. The exposure cap operates on the final combined portfolio. Floor lineups may still be chalk-heavy (higher floor); ceiling lineups should naturally be more contrarian (higher variance usually means lower ownership).

### Greedy Marginal

Greedy already accounts for self-competition. Leverage adjustment modifies the EV rankings used for the initial shortlist. Exposure cap is applied after greedy finishes (may trim some high-exposure lineups and require fallback filling from the shortlist).

### Multi-Portfolio (Dirichlet)

Each portfolio gets its own leverage-adjusted EVs (since each portfolio has different Dirichlet-weighted EVs). Exposure cap is applied per-portfolio independently.

---

## Testing Strategy (Slate-Independent)

### Test 1: Leverage Reduces Ownership Concentration

```
FOR any slate:
  portfolio_legacy = run_pipeline(leverage_lambda=0, exposure_cap=1.0)
  portfolio_new = run_pipeline(leverage_lambda=0.5, exposure_cap=0.6)

  ASSERT portfolio_new.ownership_ratio < portfolio_legacy.ownership_ratio
  ASSERT portfolio_new.max_exposure < portfolio_legacy.max_exposure
  ASSERT portfolio_new.ownership_ratio < 1.3
  ASSERT portfolio_new.max_exposure < 0.65
```

### Test 2: Score Band Diversity Improves

```
FOR any slate:
  ASSERT portfolio_new.score_band_ratio > portfolio_legacy.score_band_ratio
  ASSERT portfolio_new.score_band_ratio > 0.8
  ASSERT portfolio_new.unique_cpts >= 5
```

### Test 3: Self-Competition Cost Decreases

```
FOR any slate:
  ASSERT portfolio_new.self_comp_ratio < portfolio_legacy.self_comp_ratio
  ASSERT portfolio_new.self_comp_ratio < 0.20
```

### Test 4: Backward Compatibility

```
portfolio_compat = run_pipeline(leverage_lambda=0, exposure_cap=1.0, field_method='ownership')
ASSERT portfolio_compat == portfolio_original  # bit-for-bit identical
```

### Test 5: Parameter Monotonicity

```
FOR lambda in [0.0, 0.25, 0.5, 0.75, 1.0]:
  portfolio = run_pipeline(leverage_lambda=lambda)
  # Higher lambda should decrease ownership_ratio monotonically

FOR cap in [1.0, 0.8, 0.6, 0.4]:
  portfolio = run_pipeline(exposure_cap=cap)
  # Lower cap should decrease max_exposure monotonically
```

### Test 6: Composability

```
# All combinations should run without error and produce valid results
FOR field_method in ['simulated', 'ownership']:
  FOR selection_method in ['top_n', 'greedy_marginal', 'dpp', 'barbell']:
    FOR dro in [True, False]:
      portfolio = run_pipeline(
          field_method=field_method,
          selection_method=selection_method,
          dro_enabled=dro,
          leverage_lambda=0.5,
          exposure_cap=0.6
      )
      ASSERT 'error' not in portfolio
      ASSERT portfolio.diagnostics.self_comp_ratio < 0.30
```

---

## Expected Outcomes (Based on Current Slate)

With `leverage_lambda=0.5, exposure_cap=0.6` on the LAR-SEA slate:

| Metric | Before | Expected After |
|--------|--------|----------------|
| KW3 exposure | 100% | ~55-60% |
| JSN exposure | 93% | ~55-60% |
| Puka exposure | 89% | ~55-60% |
| Ownership ratio | 1.56x | ~1.1-1.2x |
| Score band ratio | 0.26 | ~1.0-1.5 |
| Self-comp ratio | 27% | ~10-15% |
| Unique CPTs | ~3 | ~6-8 |
| True portfolio EV | $398 | Likely slightly lower |
| ROI | -46.9% | Likely slightly more negative |

**Note on ROI**: The ROI may appear worse because we're choosing more contrarian lineups that don't "win" as often in simulation. But in a real contest with real opponents, the leverage advantage means our wins are less shared — the **real** ROI should be better, but this can't be measured in simulation alone.

---

## Implementation Order

1. **`diagnostics.py`** — Build the measurement framework first so we can validate each subsequent change
2. **`ev/approx.py`** — Add `compute_leverage_adjusted_evs()`
3. **`ev/selection.py`** — Add `select_with_exposure_cap()`
4. **`pipeline.py`** — Wire everything together (both single and multi-portfolio)
5. **`cli.py`** — Add CLI options and display
6. **`ev/__init__.py`** — Update exports
7. **Run all tests** — Verify metrics improve across the board

---

## What This Does NOT Address (Future Work)

1. **Fitted distributions**: The simulation engine still uses linear percentile interpolation instead of the Gamma/LogNormal distributions specified in config. This is a separate improvement to simulation accuracy, not portfolio construction.

2. **Copula defaults**: The Gaussian copula's lack of tail dependence underestimates boom/bust scenarios. Making t-copula the default (or auto-selecting based on contest type) is a separate change.

3. **DPP payout-aware features**: The DPP kernel uses structural features (player exposure) not payout-space features. Adding payout-awareness to DPP is a deeper refactor.

4. **Barbell cluster validation**: The K-means game-state clustering should be validated as predictive. This requires backtesting infrastructure.

5. **Contest-specific field calibration**: Different contest types (small-field, large-field, high-stakes) should have different field sharpness. This requires contest metadata.

6. **True leverage scoring**: Instead of using raw ownership as a proxy, compute implied ownership from simulation outcomes (probability of appearing on the winning lineup). This is the Haugh-Singal approach and requires an additional simulation pass.
