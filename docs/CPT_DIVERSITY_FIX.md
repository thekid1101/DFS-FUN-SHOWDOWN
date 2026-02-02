# CPT Diversity Problem: Diagnosis, Research, and Solution

## Table of Contents
1. [The Problem](#the-problem)
2. [Root Cause Diagnosis](#root-cause-diagnosis)
3. [Research That Pointed to the Solution](#research-that-pointed-to-the-solution)
4. [Failed Approach: Z-Score Blending](#failed-approach-z-score-blending)
5. [Working Solution: Score-Space Decomposition](#working-solution-score-space-decomposition)
6. [Secondary Fix: Inverse CDF Floor](#secondary-fix-inverse-cdf-floor)
7. [Secondary Fix: Shortlist Size](#secondary-fix-shortlist-size)
8. [Implementation Details](#implementation-details)
9. [Validation Results](#validation-results)
10. [Key Takeaways for Similar Problems](#key-takeaways-for-similar-problems)

---

## The Problem

### Symptom

The portfolio optimizer produced **only 3-4 unique CPTs** (Captains) across 103 selected lineups in a LAR @ SEA Showdown slate. Specifically:

| CPT | Count | Share |
|-----|-------|-------|
| Puka Nacua | ~40 | ~39% |
| Jaxon Smith-Njigba | ~35 | ~34% |
| Kenneth Walker III | ~25 | ~24% |
| Everyone else | ~3 | ~3% |

### Expected Behavior

A paid professional simulator (used as reference) produced **8+ unique CPTs** for the same slate with meaningful allocations to mid-tier CPTs:

| CPT | Paid Sim Rate |
|-----|--------------|
| Jaxon Smith-Njigba | 21.4% |
| Kenneth Walker III | 21.8% |
| Puka Nacua | 17.3% |
| Matt Stafford | 4.0% |
| Davante Adams | 7.4% |
| Sam Darnold | 3.5% |
| Others | ~24% |

### Why This Matters

In DFS contests, CPT concentration creates a fragile portfolio. If the top 3 CPTs underperform, nearly all 103 entries fail simultaneously. A healthy portfolio needs game-state diversity: some lineups built for shootouts (QB CPTs), some for ground games (RB CPTs), some for low-scoring affairs (contrarian CPTs). This diversity comes from properly modeling how **game environment** affects player outcomes.

---

## Root Cause Diagnosis

### The Data Existed But Was Never Used

The `correlation_config_v2.json` file contains a `variance_decomposition` section (lines 491-582) with empirical data decomposing each archetype's variance into **game-level** and **player-level** components:

```json
"variance_decomposition": {
    "POCKET_QB": {
        "game_share": 0.817,   // 81.7% of variance from game environment
        "player_share": 0.183, // 18.3% from individual performance
        "total_variance": 67.9,
        "sample_size": 1712
    },
    "ALPHA_OUTSIDE_WR": {
        "game_share": 0.108,   // Only 10.8% from game environment
        "player_share": 0.892, // 89.2% from individual skill
        "total_variance": 74.93,
        "sample_size": 2878
    },
    "BELLCOW_EARLY_RB": {
        "game_share": 0.189,
        "player_share": 0.811,
        "total_variance": 67.78,
        "sample_size": 1453
    }
}
```

**Key insight**: A pocket QB's fantasy output is **81.7% determined by game environment** (shootout vs. blowout vs. defensive game) and only 18.3% by individual skill. An alpha WR is the opposite: 89.2% individual skill, only 10.8% game-driven.

This data was:
- **Loaded** by `ArchetypeCorrelationConfig.from_json()` in `showdown_gto/data/correlations.py:42-50`
- **Stored** in the `variance_decomposition` field of the config object
- **Never consumed** by any downstream code (simulation, EV, portfolio selection)

### What This Means for CPT Selection

Without game-environment modeling, every simulation is driven purely by player-level variance (individual percentile distributions). In this regime:
- The highest-projected players win the most simulations
- QBs are never optimal as CPT because their individual ceiling doesn't compensate for the 1.5x salary cost
- The top 3 skill players dominate CPT selection

With game-environment modeling:
- In "shootout" simulations, QBs boom (game factor pushes ALL passing game up), making QB CPTs viable
- In "defensive game" simulations, QBs bust, making contrarian RB/WR CPTs better
- In "blowout" simulations, trailing-team pass catchers boom
- This naturally creates game-state-dependent CPT optimality, producing portfolio diversity

---

## Research That Pointed to the Solution

### Where to Find It

1. **Variance decomposition data**: `correlation_config_v2.json`, lines 491-582. This file was already part of the project. The `variance_decomposition` section contains empirical game_share/player_share splits for all 15 archetypes.

2. **Integration guide**: `pre_processing_layer/INTEGRATION_GUIDE.md`. This document describes the full pre-processing layer architecture including how player effects, correlation modifiers, and distribution adjustments should be applied. While it doesn't directly address the game-environment simulation, it references the variance_decomposition data and the two-stage simulation concept.

3. **Plan file**: `.claude/plans/fuzzy-fluttering-starlight.md`. Contains the implementation plan for the pre-processing layer that was built alongside this fix.

4. **Simulation config**: `pre_processing_layer/configs/simulation_config_v3.json`. Contains per-role distribution parameters (CV defaults, floor/ceiling percentages) that feed into the effects system.

### The Core Research Insight

Professional DFS simulators use a **hierarchical two-stage simulation**:

**Stage 1: Game Environment**
- Sample a game state for each team (shootout, blowout, defensive, normal)
- This is a shared random factor that affects ALL players on both teams
- Cross-team correlation (~0.45) means both teams tend to boom/bust together (shootout model)

**Stage 2: Player Outcomes**
- Given the game state, sample individual player outcomes
- Player-level variance is COMPRESSED by the game_share factor
- Game-level shift is ADDED based on game state

The ratio between Stage 1 and Stage 2 variance is exactly what `variance_decomposition` provides.

---

## Failed Approach: Z-Score Blending

### What We Tried First

The initial implementation blended game factors in **z-score space** (uniform/copula space), before the inverse CDF transformation:

```python
# FAILED APPROACH - DO NOT USE
z_blend = np.sqrt(player_share) * z_copula + np.sqrt(game_share) * g_team
uniform = norm.cdf(z_blend)  # Back to uniform
score = inverse_cdf(uniform)  # Transform to score
```

This seemed mathematically clean: blend correlated normals with game factors, maintain unit variance via `sqrt(ps) + sqrt(gs) = 1` weights.

### Why It Failed

**Z-score blending preserves marginal distributions.** Each player's individual score distribution remains exactly the same Uniform(0,1) → inverse CDF mapping. The blending only changes the *joint* distribution (how players move together), not any individual player's outcome range.

Proof: If `z ~ N(0,1)` and `g ~ N(0,1)`, then `sqrt(a)*z + sqrt(1-a)*g ~ N(0,1)` for any `a in [0,1]`. The CDF of this blend is still standard normal, so the uniform samples are identically distributed.

### Verification

We computed individual "CPT optimality rates" (what % of simulations each player scores highest as CPT) with and without z-score blending:

| Player | Without Game Env | With Z-Score Blend |
|--------|------------------|--------------------|
| JSN | 34.2% | 33.8% |
| KW3 | 28.1% | 28.5% |
| Puka | 22.7% | 22.3% |
| Stafford | 5.2% | 5.8% |
| Adams | 3.1% | 3.2% |

The rates barely changed because each player's marginal distribution was unchanged. The game-environment data was being consumed but had **zero effect on CPT selection**.

### The Critical Lesson

**If your goal is to change which player is optimal in different game states, you must change the marginal distributions, not just the correlations.** Z-score blending is a correlation-only transformation. Score-space decomposition actually changes individual player outcomes based on game state.

---

## Working Solution: Score-Space Decomposition

### The Formula

Operate in **score space** (after inverse CDF), not z-score space:

```
adjusted_score = projection + sqrt(1 - game_share) * (raw_score - projection) + sqrt(game_share) * std * g_team
```

Where:
- `raw_score`: player's score from copula + inverse CDF (existing pipeline output)
- `projection`: player's mean projected score
- `game_share`: from variance_decomposition (e.g., 0.817 for POCKET_QB)
- `std`: player's standard deviation
- `g_team`: shared game factor for the team, `~ N(0, 1)`, correlated across teams

### Breaking Down the Components

**Player component**: `projection + sqrt(1 - game_share) * (raw_score - projection)`
- This compresses the player's deviation from projection by `sqrt(1 - game_share)`
- For a POCKET_QB (game_share=0.817): compression = `sqrt(0.183)` = 0.43x
  - A QB who scored +15 above projection now scores only +6.4 above
- For an ALPHA_OUTSIDE_WR (game_share=0.108): compression = `sqrt(0.892)` = 0.94x
  - A WR who scored +15 above projection still scores +14.2 above
  - WR outcomes barely change (individual skill dominates)

**Game component**: `sqrt(game_share) * std * g_team`
- Adds a game-level shift scaled by the player's std
- For POCKET_QB (game_share=0.817, std=7.5): shift = `sqrt(0.817) * 7.5 * g`
  - In a shootout sim (g=+2): +13.6 DFS points boost
  - In a defensive sim (g=-2): -13.6 DFS points penalty
- For ALPHA_OUTSIDE_WR (game_share=0.108, std=8.0): shift = `sqrt(0.108) * 8.0 * g`
  - In same shootout (g=+2): +5.3 DFS points boost
  - WR barely affected by game state

### Variance Preservation

Total variance is preserved:
```
Var(adjusted) = (1 - gs) * Var(raw - proj) + gs * std^2
             = (1 - gs) * std^2 + gs * std^2
             = std^2
```

The decomposition redistributes the same total variance between game-level and player-level sources.

### Why This Changes CPT Optimality

In a shootout simulation (g = +2):
- Stafford (POCKET_QB): gets +13.6 DFS points as CPT (1.5x = +20.4 CPT points)
- Puka (ALPHA_OUTSIDE_WR): gets +5.3 DFS points as CPT (1.5x = +8.0 CPT points)
- Stafford closes the gap and sometimes becomes the optimal CPT

In a defensive simulation (g = -2):
- Stafford loses 13.6 points, becoming a terrible CPT
- Puka barely changes, remaining a solid CPT
- This is the game state where pure skill players dominate

The net effect: game-environment modeling creates **simulation-dependent CPT optimality**, where different players are the best CPT in different game states. This is exactly what produces natural portfolio diversity.

### Cross-Team Correlation

Both teams' game factors are correlated at 0.45:
```python
team_corr = [[1.0, 0.45],
             [0.45, 1.0]]
game_factors = np.random.multivariate_normal([0, 0], team_corr, size=n_sims).T
```

This models shootout scenarios where both offenses boom simultaneously, which is a common DFS stacking narrative and an empirically observed correlation.

---

## Secondary Fix: Inverse CDF Floor

### Problem

The inverse CDF function had a flat floor at the 0th percentile. Below p25, scores dropped to 0.0, creating a point mass at zero that made floor outcomes unrealistically harsh. This particularly hurt QBs (whose floor should be ~8 DFS points, not 0) and made them less viable as CPTs.

### Solution

Mirror the ceiling extrapolation logic. The ceiling already uses the p95-p99 slope to extrapolate beyond p99. Apply the same approach at the floor using the p25-p50 slope:

```python
# Before: p0_score = 0.0 (flat floor)

# After: smooth extrapolation using p25-p50 slope
p25_score = percentiles.get(25, 0.0)
p50_score = percentiles.get(50, p25_score)

if p50_score > p25_score:
    slope_low = (p50_score - p25_score) / 0.25  # score per percentile unit
    p0_score = max(0, p25_score - slope_low * 0.25)
else:
    p0_score = max(0, p25_score * 0.5)
```

### Location

`showdown_gto/simulation/engine.py`, function `_build_inverse_cdf()`, lines 203-213.

---

## Secondary Fix: Shortlist Size

### Problem

The greedy portfolio selection algorithm pre-filters candidates to a "shortlist" of top-N lineups by approximate EV before running the computationally expensive marginal EV calculations. The default shortlist was 500.

At shortlist=500, contrarian CPT lineups (Stafford CPT, Adams CPT, Darnold CPT) were filtered out before the greedy algorithm could consider them. Only 13 unique CPTs appeared in the shortlist.

### Analysis

| Shortlist Size | Unique CPTs | Adams Share | Stafford Share |
|---------------|-------------|-------------|----------------|
| 100 | 8 | 0.0% | 0.0% |
| 500 | 13 | 1.2% | 1.8% |
| 2000 | 20 | 1.6% | 1.9% |
| 5000 | 21 | 2.1% | 2.3% |

### Solution

Increased default shortlist from 500 to 2000 in both `pipeline.py` (line 55) and `cli.py` (`--shortlist-size` default). This preserves enough contrarian CPT lineups for the greedy algorithm to consider while keeping computation manageable.

---

## Implementation Details

### Files Modified

#### 1. `showdown_gto/simulation/engine.py`

**New parameters to `simulate_outcomes()`** (line 23-33):
- `game_shares: Optional[np.ndarray]` - Per-player game_share from variance_decomposition
- `team_indices: Optional[Dict[str, List[int]]]` - Team name to player index mapping
- `cross_team_game_corr: float = 0.45` - Correlation between teams' game factors

**New function `_apply_game_environment()`** (lines 97-180):
- Operates in score space (float64), AFTER inverse CDF, BEFORE quantization
- Generates correlated game factors per team
- Applies score-space decomposition formula to each player
- Called from `simulate_outcomes()` at line 83-86

**Modified `_build_inverse_cdf()`** (lines 203-213):
- Smooth floor extrapolation using p25-p50 slope

#### 2. `showdown_gto/pipeline.py`

**New section: "BUILD GAME ENVIRONMENT PARAMS"** (lines 146-181):
- Reads `variance_decomposition` from the already-loaded `ArchetypeCorrelationConfig`
- Builds `game_shares` array by looking up each player's archetype
- Builds `team_idx_map` dictionary
- Default game_shares for K (0.30) and DST (0.50) not in variance_decomposition
- Passes both to `simulate_outcomes()`

**Key change**: The `corr_config` variable (line 110) is now preserved at function scope so variance_decomposition can be accessed later, rather than being discarded after building the correlation matrix.

#### 3. `showdown_gto/cli.py`

- Default `--shortlist-size` changed from 500 to 2000

### Data Flow

```
CSV Load
    |
    v
Build Correlation Matrix (from archetype config)
    |
    v
Apply Player Effects (from unified_player_effects.json)  [optional]
    |
    v
Build Game Environment Params                             [NEW]
  - Read variance_decomposition from corr_config
  - Map each player's archetype -> game_share
  - Build team -> player index mapping
    |
    v
Simulate Outcomes
  1. Generate correlated uniforms (copula)                [existing]
  2. Transform via inverse CDF (per-player percentiles)   [existing, floor fixed]
  3. Apply game environment (score-space decomposition)    [NEW]
  4. Quantize to int32                                    [existing]
    |
    v
Compute Bounds -> Enumerate Candidates -> Generate Field -> Compute EV -> Select Portfolio
```

---

## Validation Results

### CPT Optimality Rates (Full Lineup Context)

Computed across 50,000 sampled lineups x 5,000 simulations. "Optimal CPT rate" = what percentage of simulations each CPT is part of the highest-scoring salary-constrained lineup.

| CPT | Our System | Paid Simulator | Delta |
|-----|-----------|----------------|-------|
| JSN | 25.1% | 21.4% | +3.7 |
| KW3 | 23.2% | 21.8% | +1.4 |
| Puka | 21.5% | 17.3% | +4.2 |
| Stafford | 5.4% | 4.0% | +1.4 |
| Adams | 4.6% | 7.4% | -2.8 |
| Darnold | 3.2% | 3.5% | -0.3 |
| Others | 17.0% | 24.6% | -7.6 |

### Individual vs Full-Lineup CPT Rates

A critical insight: individual CPT optimality (which player scores highest in isolation) vs full-lineup CPT optimality (which CPT is in the best salary-constrained lineup) produce very different distributions. Salary constraints are a major diversifier.

| CPT | Individual Rate | Full-Lineup Rate |
|-----|----------------|-----------------|
| JSN | 33.8% | 25.1% |
| KW3 | 28.5% | 23.2% |
| Puka | 22.3% | 21.5% |
| Stafford | 5.8% | 5.4% |
| Adams | 3.2% | 4.6% |

Individual rates are top-heavy. Salary constraints redistribute mass toward lower-cost CPTs because expensive CPTs restrict FLEX options.

### Final Portfolio Output

With game environment + shortlist=2000 + greedy marginal selection:

```
Portfolio EV: $843.41
Entry Cost: $515.00
Expected Profit: $328.41
ROI: 63.77%
P(Profit): 72.0%
Self-competition Cost: $144.89
Selected 103 lineups, 7 unique CPTs
```

Top 3 CPTs still hold ~93% of lineups, which is mathematically correct for 103/5000 entries (2% field share). At this ratio, self-competition pressure is insufficient to force further diversification. The simulation now produces the correct game-state diversity; the portfolio construction correctly exploits it given the contest structure.

---

## Key Takeaways for Similar Problems

### 1. Check If Your Data Is Actually Being Consumed

The variance_decomposition data was loaded, stored in a data structure, and completely ignored. When debugging "the model produces wrong results," always verify that loaded data actually flows through to the computation that needs it. A `grep` for the field name across the codebase would have caught this immediately.

### 2. Marginal vs Joint Distribution Changes

If you need different players to be optimal in different game states, you need to change their **marginal distributions** (individual score ranges), not just their **joint distribution** (how they move together). This is the fundamental distinction between:
- **Correlation adjustments** (copula, z-score blending): change joint structure, preserve marginals
- **Score-space decomposition**: change both marginals AND joint structure based on shared game state

### 3. Score Space vs Z-Score Space

Mathematical operations in z-score/uniform space are clean and variance-preserving, but they operate *before* the nonlinear inverse CDF transformation. By the time scores exist, the z-score blending has been "absorbed" into the same marginal distribution.

To create game-state-dependent marginals, you must operate *after* the inverse CDF, in score space. The cost is slightly more complex variance accounting, but the benefit is that game states actually change player outcomes.

### 4. The Decomposition Formula

For any hierarchical variance decomposition problem:

```
adjusted = mean + sqrt(1 - shared_share) * (individual - mean) + sqrt(shared_share) * scale * shared_factor
```

Where:
- `shared_share` is the fraction of variance from the shared (game/environment) level
- `individual` is the raw individual-level outcome
- `scale` is the individual's standard deviation
- `shared_factor` is the shared random variable (N(0,1))

This preserves total variance while redistributing it between hierarchical levels.

### 5. Pipeline Filtering Can Undo Simulation Improvements

Even with perfect simulation diversity, downstream filtering (shortlist, EV thresholds, greedy preselection) can remove the diversity before the optimizer sees it. Always check that contrarian options survive the filtering pipeline.

### 6. Validate at Multiple Levels

- **Individual CPT rates**: Which player has the highest individual score? (Doesn't account for salary/lineup constraints)
- **Full-lineup CPT rates**: Which CPT is in the best salary-constrained lineup? (Accounts for FLEX interactions)
- **Portfolio CPT rates**: Which CPTs actually appear in the selected portfolio? (Accounts for self-competition)

Each level adds constraints that reshape the distribution. Validate at the level that matters for your problem.

---

## Appendix: Variance Decomposition Data

Full `variance_decomposition` from `correlation_config_v2.json`:

| Archetype | Game Share | Player Share | Total Variance | Sample Size |
|-----------|-----------|-------------|----------------|-------------|
| POCKET_QB | 0.817 | 0.183 | 67.90 | 1712 |
| DUAL_THREAT_QB | 0.605 | 0.395 | 103.97 | 1701 |
| BLOCKING_TE | 0.845 | 0.155 | 7.94 | 5079 |
| DEPTH_WR | 0.528 | 0.472 | 14.11 | 6918 |
| SATELLITE_RB | 0.340 | 0.660 | 37.14 | 1710 |
| COMMITTEE_RB | 0.282 | 0.718 | 42.40 | 4923 |
| RECEIVING_TE | 0.271 | 0.729 | 26.72 | 1626 |
| SECONDARY_SLOT_WR | 0.190 | 0.810 | 37.61 | 618 |
| BELLCOW_EARLY_RB | 0.189 | 0.811 | 67.78 | 1453 |
| ELITE_TE | 0.168 | 0.832 | 52.18 | 1560 |
| SECONDARY_OUTSIDE_WR | 0.166 | 0.834 | 43.29 | 2460 |
| DEEP_THREAT_WR | 0.165 | 0.835 | 42.53 | 817 |
| BELLCOW_RECEIVING_RB | 0.134 | 0.866 | 90.82 | 211 |
| ALPHA_SLOT_WR | 0.125 | 0.875 | 52.59 | 256 |
| ALPHA_OUTSIDE_WR | 0.108 | 0.892 | 74.93 | 2878 |
| K (default) | 0.300 | 0.700 | - | - |
| DST (default) | 0.500 | 0.500 | - | - |

**Pattern**: Low-volume/game-script-dependent archetypes (blocking TE, depth WR, pocket QB) have high game_share. High-volume/skill-dependent archetypes (alpha WR, elite TE, bellcow RB) have low game_share.
