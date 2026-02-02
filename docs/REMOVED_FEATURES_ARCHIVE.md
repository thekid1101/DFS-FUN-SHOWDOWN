# Removed Features Archive

This document archives features that have been removed from the DFS Showdown GTO Portfolio Builder pipeline. Each section contains sufficient detail (function signatures, formulas, algorithm descriptions, rationale) to reconstruct the feature if needed in the future.

---

## 1. Leverage-Adjusted EV

### Formula

```
leverage_adjusted_ev[i] = approx_ev[i] * (1 - lambda * mean_own[i])
```

where `mean_own[i]` is the average ownership (0-1 scale) of all 6 players in lineup `i` (1 CPT + 5 FLEX).

### Function Signature

```python
def compute_leverage_adjusted_evs(
    approx_evs: np.ndarray,          # [n_candidates] raw approx EVs
    candidate_arrays: np.ndarray,     # [n_candidates, 6] player indices per lineup
    cpt_players: List[Player],        # CPT player pool
    flex_players: List[Player],       # FLEX player pool
    cpt_to_flex_map: Dict[int, int],  # CPT index -> FLEX index mapping
    leverage_lambda: float = 0.5      # penalty strength (0=none, 0.5=moderate, 1.0=heavy)
) -> np.ndarray:                      # [n_candidates] leverage-adjusted EVs
```

Located in `ev/approx.py`.

### Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `leverage_lambda` | 0.5 | 0.0 - 1.0 | Strength of ownership penalty. 0 = no adjustment (identical to raw approx EV). 0.5 = moderate default. 1.0 = maximum penalty where a 100% owned lineup gets EV reduced to zero. |

### Algorithm

1. For each candidate lineup, extract the 6 player indices (1 CPT + 5 FLEX).
2. Look up each player's ownership from `cpt_players` and `flex_players` using `cpt_to_flex_map` to resolve CPT players to their base ownership.
3. Compute `mean_own[i]` as the arithmetic mean of the 6 player ownerships (each in 0-1 range).
4. Apply the multiplicative penalty: `adjusted_ev[i] = approx_ev[i] * (1 - lambda * mean_own[i])`.

### Rationale

High-ownership lineups share wins with more opponents, reducing the exclusive value of each win. In a contest of N entries, if your lineup scores highest but 5% of the field has the same players, the payout is split more ways in expectation. Penalizing chalk lineup EV should push selection toward lineups with unique upside that capture more exclusive payout when they hit.

### Why Removed

The default `lambda=0.5` was too aggressive, causing significant EV distortion. The penalty altered lineup rankings in ways that reduced true portfolio EV rather than improving it. Specifically:

- A lineup with mean ownership of 20% would have its EV reduced by 10%, which is a massive re-ranking effect.
- Lineups that were genuinely EV-optimal got pushed down in favor of contrarian lineups with lower raw EV.
- The field model already accounts for opponent ownership through the field generation process (`field/generator.py`), making this adjustment partially redundant and potentially double-counting the ownership effect.

### When to Revisit

If field modeling proves insufficient to capture ownership overlap effects, consider reintroducing with a much lighter lambda (0.05-0.1). At lambda=0.05, a 20% mean ownership lineup only loses 1% of its EV, which is a more defensible adjustment magnitude.

### Cross-Reference

- `docs/PLAN_leverage_aware_selection.md` -- Original plan and design rationale for leverage-aware selection.

---

## 2. Exposure Cap

### Function Signatures

```python
def select_with_exposure_cap(
    ranked_indices: List[int],             # candidate indices sorted by EV (best first)
    candidate_arrays: np.ndarray,          # [n_candidates, 6] player indices per lineup
    cpt_to_flex_map: Dict[int, int],       # CPT index -> FLEX index mapping
    n_flex: int,                           # number of FLEX players in pool
    n_select: int,                         # target portfolio size
    exposure_cap: float = 0.6             # max fraction of portfolio any player can appear in
) -> List[int]:                            # selected candidate indices
```

```python
def _get_flex_player_indices(
    candidate_arrays: np.ndarray,          # [n_candidates, 6] player indices per lineup
    idx: int,                              # candidate index to inspect
    cpt_to_flex_map: Dict[int, int]        # CPT index -> FLEX index mapping
) -> List[int]:                            # list of FLEX-space player indices for this lineup
```

Located in `ev/selection.py`.

### Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `exposure_cap` | 0.6 | 0.0 - 1.0 | Maximum fraction of the portfolio that any single player may appear in. At 0.6 with n_select=150, no player can appear in more than 90 lineups. |

### Algorithm

1. Compute `max_count = int(n_select * exposure_cap)`.
2. Initialize a player exposure counter dictionary (keyed by FLEX-space player index).
3. Iterate through `ranked_indices` (pre-sorted by EV, best first):
   a. For each candidate, call `_get_flex_player_indices()` to get all 6 players mapped to FLEX-space indices.
   b. Check if adding this lineup would push any player above `max_count`.
   c. If no player exceeds the cap, add the lineup to the selected set and increment all 6 player counters.
   d. If any player would exceed the cap, skip this lineup.
4. Continue until `n_select` lineups are selected or candidates are exhausted.

### Helper: `_get_flex_player_indices`

Maps all 6 positions in a lineup to their FLEX-space player indices. The CPT slot (index 0 in `candidate_arrays`) is mapped through `cpt_to_flex_map` so that CPT and FLEX appearances of the same player are counted together.

### Why Removed

Forces suboptimal lineups into the portfolio. If the EV-optimal portfolio has high exposure to certain players (e.g., the slate's best QB appears in 80% of top lineups), capping at 60% artificially forces the inclusion of lower-EV lineups. The greedy marginal selection method already handles self-competition: if too many lineups share the same players, the marginal EV of adding another overlapping lineup naturally decreases because those lineups compete against each other in the same simulations.

### When to Revisit

- If a specific contest format has explicit rules limiting player exposure per account.
- If extreme concentration is observed after further testing and greedy marginal proves insufficient to diversify.
- If implementing for a platform other than DraftKings where different contest rules apply.

---

## 3. DPP (Determinantal Point Process) Selection

### File

`ev/dpp.py`

### Entry Point

```python
def dpp_select_portfolio(
    candidate_arrays: np.ndarray,          # [n_candidates, 6] player indices per lineup
    approx_evs: np.ndarray,               # [n_candidates] approx EV scores
    outcomes: np.ndarray,                  # [n_players, n_sims] simulated outcomes (int32)
    n_cpt: int,                            # number of CPT players
    n_flex: int,                           # number of FLEX players
    n_select: int,                         # target portfolio size
    shortlist_size: int = 500,             # pre-filter to top N by EV before DPP
    quality_exponent: float = 1.0,         # controls quality vs diversity tradeoff
    feature_sims: Optional[int] = None     # subsample sims for feature computation
) -> List[int]:                            # selected candidate indices
```

### Core Functions

#### `build_lineup_features`

```python
def build_lineup_features(
    candidate_arrays: np.ndarray,          # [n_candidates, 6] player indices
    outcomes: np.ndarray,                  # [n_players, n_sims] simulated outcomes
    n_cpt: int,
    n_flex: int,
    shortlist_indices: List[int],          # indices into candidate_arrays to build features for
    feature_sims: Optional[int] = None     # subsample sims (None = use all)
) -> np.ndarray:                           # [len(shortlist_indices), n_features] feature matrix
```

Constructs a 3-segment feature vector for each lineup:

1. **Player exposure** (`n_cpt + n_flex` binary dimensions): A one-hot vector indicating which players appear in the lineup. Each dimension corresponds to a player in the CPT or FLEX pool. This ensures lineups sharing players have high similarity.

2. **Score profile** (3 dimensions, normalized): Mean, standard deviation, and skewness of the lineup's score distribution across simulations. Normalized to zero mean and unit variance across the shortlist. Captures the distributional character of each lineup.

3. **Game-state features** (2 dimensions):
   - CPT contribution fraction: `mean(cpt_score) / mean(lineup_score)` -- how much the lineup depends on the captain.
   - FLEX concentration: Herfindahl index of FLEX player score contributions -- whether the lineup's FLEX value is spread evenly or concentrated in one player.

#### `build_dpp_kernel`

```python
def build_dpp_kernel(
    features: np.ndarray,                  # [N, d] feature matrix from build_lineup_features
    quality_scores: np.ndarray,            # [N] quality scores (typically approx EVs)
    quality_exponent: float = 1.0,         # exponent applied to quality scores
    jitter: float = 0.01                   # diagonal regularization for numerical stability
) -> np.ndarray:                           # [N, N] L-ensemble kernel matrix
```

Constructs the L-ensemble kernel:

```
phi_i = features[i] / ||features[i]||     (unit-normalized feature vector)
q_i = quality_scores[i] ** quality_exponent
L[i, j] = q_i * (phi_i . phi_j) * q_j
L[i, i] += jitter                          (numerical stability)
```

The L-ensemble assigns probability `P(S) proportional to det(L_S)` to subset S. The determinant naturally balances:
- **Quality**: High `q_i` values increase the diagonal, making item `i` more likely to be selected.
- **Diversity**: High similarity `phi_i . phi_j` between items `i` and `j` reduces the determinant when both are selected, discouraging redundancy.

#### `dpp_greedy_select`

```python
def dpp_greedy_select(
    L: np.ndarray,                         # [N, N] L-ensemble kernel
    k: int                                 # number of items to select
) -> List[int]:                            # selected indices into the shortlist
```

Greedy MAP inference with incremental Cholesky decomposition. Time complexity: O(N * k^2).

**Algorithm:**
1. Initialize empty selection set S.
2. For each of k rounds:
   a. For each candidate `i` not yet in S, compute the marginal gain `log det(L_{S+i}) - log det(L_S)`.
   b. Using incremental Cholesky: maintain the Cholesky factor of `L_S`. For candidate `i`, compute the new row/column contribution and the resulting increase in log-determinant.
   c. Select the candidate with the highest marginal gain.
   d. Update the Cholesky factor to include the new item.
3. If at any point all remaining candidates have non-positive marginal gain (kernel has exhausted diversity budget), fall back to filling remaining slots with the highest-EV candidates not yet selected.

### Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `shortlist_size` | 500 | 100 - 5000 | Pre-filter candidates to top N by approx EV. Reduces kernel size from potentially 100K+ to manageable N. |
| `quality_exponent` | 1.0 | 0.1 - 5.0 | Controls quality vs diversity. 1.0 = balanced. 2.0 = quality-heavy (squares quality scores, making high-EV lineups dominate). 0.5 = diversity-heavy (compresses quality differences, letting diversity drive selection). |
| `feature_sims` | None | 100 - n_sims | Subsample simulations for feature computation. Reduces memory and compute for large sim counts. |

### Fallback Behavior

If the DPP kernel exhausts its diversity budget (all remaining marginal gains are non-positive) before selecting `n_select` lineups, the remaining slots are filled by the highest-EV candidates from the shortlist that were not already selected by the DPP phase.

### Why Removed

Added complexity without proven EV improvement over `top_n` + `greedy_marginal`. The kernel construction requires tuning `quality_exponent`, which is slate-dependent -- a value that works well for a 2-QB slate may be wrong for a run-heavy game. The shortlist pre-filtering also means the DPP can only diversify within the top 500 lineups by EV, limiting its ability to find truly different lineups.

### When to Revisit

If `top_n` consistently produces portfolios where multiple lineups correlate strongly (high overlap in both players and score outcomes), DPP provides a principled quality-diversity tradeoff grounded in probability theory. The key advantage is that DPP diversity is automatic and principled, unlike ad-hoc exposure caps.

---

## 4. Barbell Selection (Floor/Ceiling Tiers)

### File

`ev/barbell.py`

### Entry Point

```python
def barbell_select_portfolio(
    candidate_arrays: np.ndarray,          # [n_candidates, 6] player indices
    approx_evs: np.ndarray,               # [n_candidates] approx EV scores
    outcomes: np.ndarray,                  # [n_players, n_sims] simulated outcomes
    contest: Contest,                      # contest structure (entries, payouts, etc.)
    score_bounds: Tuple[int, int],         # (min_score, max_score) for histogram range
    n_select: int,                         # target portfolio size
    field_arrays: Optional[np.ndarray] = None,    # field lineup arrays
    field_counts: Optional[np.ndarray] = None,    # field lineup counts
    team_indices: Optional[Dict] = None,          # player -> team mapping
    floor_fraction: float = 0.65,          # fraction of portfolio for floor tier
    n_clusters: int = 5,                   # number of game-state clusters
    ceiling_strategy: str = "best_cluster",# "best_cluster" or "top_2_clusters"
    profile_sims: Optional[int] = None,    # subsample sims for profiling
    seed: Optional[int] = None             # RNG seed for reproducibility
) -> Tuple[List[int], Dict]:               # (selected indices, metadata dict)
```

### Internal Functions

#### `compute_lineup_profiles`

```python
def compute_lineup_profiles(
    candidate_arrays: np.ndarray,
    outcomes: np.ndarray,
    shortlist_indices: List[int],
    profile_sims: Optional[int] = None
) -> Dict[str, np.ndarray]:
```

Computes distributional statistics for each lineup:
- `mean`: mean lineup score across simulations
- `std`: standard deviation of lineup score
- `skew`: skewness of lineup score distribution
- `p10`: 10th percentile score
- `p90`: 90th percentile score
- `top_decile_mean`: mean score in the top 10% of simulations (ceiling measure)

#### `classify_lineup_tiers`

```python
def classify_lineup_tiers(
    profiles: Dict[str, np.ndarray],
    approx_evs: np.ndarray,
    shortlist_indices: List[int]
) -> Tuple[List[int], List[int]]:          # (floor_eligible, ceiling_eligible)
```

Classifies lineups into tiers:
- **Floor-eligible**: High probability of cashing (P(cash) estimated from score distribution relative to payout cutoff) combined with decent EV. These are reliable, lower-variance lineups.
- **Ceiling-eligible**: High variance (`std` above median) combined with strong upside (`top_decile_mean` above median). These are boom-or-bust lineups that can finish near the top.

A lineup can be eligible for both tiers.

#### `identify_game_state_clusters`

```python
def identify_game_state_clusters(
    outcomes: np.ndarray,
    team_indices: Dict,
    n_clusters: int = 5,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:        # (cluster_labels [n_sims], centroids [n_clusters, 2])
```

Uses `scipy.cluster.vq.kmeans2` to cluster simulations into game states based on two features:
1. **Total score**: Sum of all player outcomes in that simulation (proxy for game total).
2. **Team differential**: Difference between team A and team B total scores (proxy for game flow).

Each simulation is assigned to a cluster, representing a distinct game scenario (e.g., "high-scoring blowout for Team A", "low-scoring close game", etc.).

#### `compute_cluster_conditional_evs`

```python
def compute_cluster_conditional_evs(
    candidate_arrays: np.ndarray,
    outcomes: np.ndarray,
    cluster_labels: np.ndarray,
    n_clusters: int,
    shortlist_indices: List[int],
    field_arrays: Optional[np.ndarray],
    field_counts: Optional[np.ndarray],
    contest: Contest,
    score_bounds: Tuple[int, int]
) -> np.ndarray:                           # [n_clusters, len(shortlist_indices)] conditional EVs
```

For each cluster (game state), computes the EV of each candidate lineup using **only** the simulations assigned to that cluster. This reveals which lineups are best in which game scenario.

### Full Algorithm

1. Pre-filter to shortlist (top candidates by approx EV).
2. Compute lineup profiles via `compute_lineup_profiles()`.
3. Classify into floor and ceiling tiers via `classify_lineup_tiers()`.
4. **Floor selection**: Select top `n_floor = int(n_select * floor_fraction)` lineups by approx EV among floor-eligible candidates.
5. Cluster simulations into game states via `identify_game_state_clusters()`.
6. Compute cluster-conditional EVs via `compute_cluster_conditional_evs()`.
7. **Ceiling selection**:
   - Identify the cluster with the highest mean payout across all candidates ("best cluster").
   - If `ceiling_strategy == "best_cluster"`: Select top `n_ceiling = n_select - n_floor` lineups by conditional EV in the best cluster, from ceiling-eligible candidates not already selected.
   - If `ceiling_strategy == "top_2_clusters"`: Split ceiling slots between the two best clusters.
8. Return combined floor + ceiling indices and metadata dict with tier assignments, cluster info, and diagnostics.

### Theoretical Rationale

Top-heavy contest payouts create a convex payout function (payout grows super-linearly as rank improves toward 1st place). By Jensen's inequality, for a convex function f:

```
E[f(X)] >= f(E[X])
```

This means higher-variance portfolios have higher expected payout when the payout function is convex. The barbell approach exploits this by:
- Using floor lineups to secure minimum cash rate (protecting downside).
- Concentrating ceiling lineups on the same game state so that when the favorable scenario occurs, multiple lineups finish near the top simultaneously, capturing multiple high-payout positions.

### Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `floor_fraction` | 0.65 | 0.0 - 1.0 | Fraction of n_select allocated to floor tier. 0.65 means 65% floor, 35% ceiling. |
| `n_clusters` | 5 | 2 - 20 | Number of game-state clusters. More clusters = finer game-state resolution but noisier conditional EVs. |
| `ceiling_strategy` | "best_cluster" | "best_cluster" or "top_2_clusters" | Whether to concentrate ceiling on 1 or 2 game states. |

### Why Removed

Classification thresholds in `classify_lineup_tiers()` were fragile -- small changes in the threshold logic could significantly change which lineups were floor vs ceiling eligible. The cluster-conditional EV computation was expensive (recomputing EVs for each cluster subset of simulations). Most importantly, the improvement over `greedy_marginal` was not clearly demonstrated in testing.

### When to Revisit

For very top-heavy contests (e.g., DraftKings Milly Maker where 1st place pays $1M+ but minimum cash is only 2-3x entry fee) where ceiling payouts dominate expected value. In such contests, the convexity argument is strongest and the barbell's game-state concentration could genuinely improve outcomes.

---

## 5. Multi-Portfolio Diversification

### File

`ev/diversity.py`

### Core Functions

#### `generate_dirichlet_weights`

```python
def generate_dirichlet_weights(
    n_portfolios: int,                     # number of distinct portfolios to generate
    n_sims: int,                           # total number of simulations
    alpha: float = 1.0,                    # diversity control parameter
    n_blocks: Optional[int] = None,        # block bootstrap block count (unused in final version)
    seed: Optional[int] = None             # RNG seed
) -> np.ndarray:                           # [n_portfolios, n_sims] weight matrix
```

Generates weight matrices using subsampling bootstrap (Politis-Romano-Wolf 1999). Each portfolio receives uniform weights on a random subset of simulations, creating distinct "views" of the outcome space.

**Weight computation:**
```python
effective_sims = min(n_sims, max(100, int(n_sims * alpha / 20.0)))
```

For each portfolio:
1. Randomly sample `effective_sims` simulation indices (without replacement).
2. Set weight = `1.0 / effective_sims` for selected sims, 0 for others.

**Alpha interpretation:**
| alpha | effective_sims (10K total) | Diversity |
|-------|----------------------------|-----------|
| 1.0   | 500 (5%)                   | Maximum -- each portfolio sees a very different subset |
| 5.0   | 2500 (25%)                 | Moderate |
| 20.0  | 10000 (100%)               | None -- all portfolios see the same data |

#### `compute_overlap_matrix`

```python
def compute_overlap_matrix(
    portfolios: List[List[int]]            # list of portfolios, each a list of candidate indices
) -> np.ndarray:                           # [n_portfolios, n_portfolios] pairwise overlap counts
```

Computes the number of shared lineups between each pair of portfolios. Useful for verifying that multi-portfolio diversification is actually producing distinct portfolios.

### Pipeline Function

```python
def run_multi_portfolio_optimization(
    candidate_arrays, field_arrays, field_counts, outcomes, contest, score_bounds,
    n_select, n_portfolios, diversity_alpha, selection_method, seed, ...
) -> List[PortfolioResult]:
```

This was a top-level orchestration function in `pipeline.py` (entire function removed). It:

1. Called `generate_dirichlet_weights()` to create `n_portfolios` weight matrices.
2. Called `compute_approx_lineup_evs_multi()` to compute weighted EVs for all portfolios simultaneously.
3. For each portfolio, selected lineups using the specified selection method with that portfolio's weighted EVs.
4. Computed the overlap matrix across all portfolios.
5. Optionally ran true portfolio EV on each portfolio.
6. Returned a list of `PortfolioResult` objects.

### Multi-EV Functions (ev/approx.py)

#### `compute_approx_lineup_evs_multi`

```python
def compute_approx_lineup_evs_multi(
    candidate_arrays: np.ndarray,          # [n_candidates, 6] player indices
    field_arrays: np.ndarray,              # [n_field, 6] field lineup player indices
    field_counts: np.ndarray,              # [n_field] count of each field lineup
    outcomes: np.ndarray,                  # [n_players, n_sims] simulated outcomes
    contest: Contest,                      # contest parameters
    score_bounds: Tuple[int, int],         # histogram score range
    sim_weights: np.ndarray,               # [n_portfolios, n_sims] from generate_dirichlet_weights
    chunk_size: int = 10000                # process candidates in chunks
) -> np.ndarray:                           # [n_portfolios, n_candidates] weighted approx EVs
```

Key optimization: The expensive per-simulation work (computing lineup scores, field histogram, rank lookup) is done **once** per simulation. Only the final weight multiplication differs per portfolio. This avoids an `n_portfolios` multiplier on the core computation.

#### `compute_approx_lineup_evs_resampled_multi`

```python
def compute_approx_lineup_evs_resampled_multi(
    candidate_arrays: np.ndarray,
    p_lineup: np.ndarray,                  # [n_field_lineups] lineup probabilities
    outcomes: np.ndarray,
    contest: Contest,
    score_bounds: Tuple[int, int],
    field_size: int,                       # total field entries to simulate
    sim_weights: np.ndarray,               # [n_portfolios, n_sims]
    seed: Optional[int] = None
) -> np.ndarray:                           # [n_portfolios, n_candidates]
```

Same as above but uses resampled field histograms (drawing field lineups from `p_lineup` probabilities each simulation) rather than fixed field arrays.

### CLI Integration

- Function `_run_multi_portfolio()` in `cli.py` handled the multi-portfolio code path.
- CLI arguments:
  - `--n-portfolios N`: Number of portfolios to generate (default: 1, meaning single portfolio mode).
  - `--diversity-alpha FLOAT`: Diversity parameter for weight generation (default: 1.0).

### Why Removed

The subsampling approach creates noisy EV estimates because each portfolio only sees a fraction of the simulations. This noise can lead to suboptimal individual portfolios -- each portfolio may be worse than the single best portfolio computed on all simulations. The overlap matrix is interesting diagnostically but does not justify the pipeline complexity for production use.

### When to Revisit

- When contest rules allow submitting different lineups to different account entries (separate entry pools).
- For hedge/insurance portfolio strategies where you want multiple uncorrelated portfolios.
- If a more sophisticated weighting scheme (e.g., importance sampling rather than subsampling) can reduce the noise problem.

---

## 6. Portfolio Diagnostics

### File

`diagnostics.py`

### Functions

#### `compute_portfolio_diagnostics`

```python
def compute_portfolio_diagnostics(
    selected_indices: List[int],           # indices of selected lineups
    candidate_arrays: np.ndarray,          # [n_candidates, 6] player indices
    outcomes: np.ndarray,                  # [n_players, n_sims] simulated outcomes
    cpt_players: List[Player],             # CPT player pool
    flex_players: List[Player],            # FLEX player pool
    cpt_to_flex_map: Dict[int, int],       # CPT -> FLEX index mapping
    approx_evs: np.ndarray,               # [n_candidates] approx EV scores
    true_ev: float,                        # true portfolio EV (from compute_true_portfolio_ev)
    contest: Contest,                      # contest parameters
    n_diagnostic_sims: int = 1000          # sims to use for diagnostic computations
) -> Dict:                                 # dictionary of diagnostic metrics
```

#### `format_portfolio_health`

```python
def format_portfolio_health(
    diagnostics: Dict                      # output from compute_portfolio_diagnostics
) -> str:                                  # formatted multi-line string for CLI display
```

### Metrics Computed

| Metric | Formula | Target | Description |
|--------|---------|--------|-------------|
| `ownership_ratio` | `mean(portfolio_player_own) / mean(pool_player_own)` | < 1.2x | How chalky the portfolio is relative to the full player pool. Above 1.2x indicates over-indexing on popular players. |
| `max_exposure` | `max(player_appearances / n_select)` | < 60% | Maximum single-player exposure fraction. Above 60% means one player dominates the portfolio. |
| `score_band_ratio` | `mean_score_range / avg_lineup_std` | > 1.0 | Ratio of the range of mean scores across selected lineups to the average within-lineup standard deviation. Above 1.0 means lineup means are well-spread. |
| `unique_cpts` | `len(set(lineup[0] for lineup in selected))` | 5+ | Number of distinct CPT players used across the portfolio. Low values indicate CPT concentration. |
| `self_comp_ratio` | `(sum(approx_evs[selected]) - true_ev) / sum(approx_evs[selected])` | < 15% | Fraction of gross EV lost to self-competition. Above 15% means the portfolio is cannibalizing itself. |
| `portfolio_mean_score` | `mean(lineup_means)` | -- | Average expected score across selected lineups. |
| `portfolio_score_percentile` | percentile of `portfolio_mean_score` in full candidate pool | -- | Where the portfolio's average score falls in the candidate distribution. |
| `avg_lineup_std` | `mean(lineup_stds)` | -- | Average volatility of selected lineups. |
| `top5_exposures` | sorted list of `(player_name, exposure_fraction, ownership, leverage)` | -- | Top 5 most-exposed players. Leverage = `exposure / ownership`. |

### Display Format

`format_portfolio_health()` produced formatted CLI output with pass/fail indicators:

```
Portfolio Health:
  Ownership ratio:  1.15x  [OK]
  Max exposure:     0.58   [OK]
  Score band ratio: 1.23   [OK]
  Unique CPTs:      7      [OK]
  Self-comp ratio:  0.12   [OK]

Top Exposures:
  Player A:  55% exposure / 22% own (2.5x leverage)
  Player B:  48% exposure / 30% own (1.6x leverage)
  ...
```

Indicators: `[OK]` when within target, `[HIGH]` or `[LOW]` when outside target range.

### Why Removed

The thresholds were arbitrary and the metrics created pressure to optimize for diagnostic scores rather than true EV. Some metrics (`ownership_ratio`, `max_exposure`) were effectively being enforced by leverage adjustment and exposure cap, both of which are also being removed. The diagnostics module added a separate optimization target that could conflict with the actual goal (maximizing true portfolio EV).

### When to Revisit

Once the clean pipeline produces baseline results, add back specific diagnostics that prove actionable. In particular:
- `self_comp_ratio` is always useful to report as a sanity check -- it should just be a single-line output, not a full diagnostics framework with arbitrary thresholds.
- `unique_cpts` is a useful quick check but does not need pass/fail logic.
- Consider reporting diagnostics as informational only (no thresholds, no pass/fail) to avoid creating implicit optimization targets.

---

## 7. Wasserstein DRO (Field Robustness) Pipeline Integration

### Files Affected

- `field/robust.py` -- **kept on disk** but unwired from the pipeline
- `pipeline.py` -- DRO routing code removed
- `cli.py` -- DRO CLI arguments removed

### Pipeline Code Removed

The following call sites in `pipeline.py` were removed:

```python
# DRO field perturbation routing
if config.dro_enabled:
    perturbed_fields = generate_perturbed_fields(
        field_arrays, field_counts, ownership_probs,
        n_perturbations=config.dro_perturbations,
        scale=config.dro_scale,
        seed=config.seed
    )
    # or for resampled field mode:
    perturbed_p_lineups = generate_perturbed_p_lineups(
        p_lineup, ownership_probs,
        n_perturbations=config.dro_perturbations,
        scale=config.dro_scale,
        seed=config.seed
    )

    robust_evs = compute_robust_approx_evs(
        candidate_arrays, perturbed_fields, perturbed_field_counts,
        outcomes, contest, score_bounds,
        aggregation=config.dro_aggregation,
        lam=config.dro_lambda
    )
    # or resampled variant:
    robust_evs = compute_robust_approx_evs_resampled(
        candidate_arrays, perturbed_p_lineups, outcomes,
        contest, score_bounds, field_size,
        aggregation=config.dro_aggregation,
        lam=config.dro_lambda
    )
```

### Functions in `field/robust.py` (Preserved on Disk)

```python
def generate_perturbed_fields(
    field_arrays, field_counts, ownership_probs,
    n_perturbations=5, scale=0.15, seed=None
) -> List[Tuple[np.ndarray, np.ndarray]]:

def generate_perturbed_p_lineups(
    p_lineup, ownership_probs,
    n_perturbations=5, scale=0.15, seed=None
) -> List[np.ndarray]:

def compute_robust_approx_evs(
    candidate_arrays, perturbed_fields, perturbed_field_counts,
    outcomes, contest, score_bounds,
    aggregation="cvar", lam=0.5
) -> np.ndarray:

def compute_robust_approx_evs_resampled(
    candidate_arrays, perturbed_p_lineups, outcomes,
    contest, score_bounds, field_size,
    aggregation="cvar", lam=0.5
) -> np.ndarray:
```

### CLI Arguments Removed

| Argument | Default | Description |
|----------|---------|-------------|
| `--dro / --no-dro` | `--no-dro` | Enable/disable DRO field robustness |
| `--dro-perturbations` | 5 | Number of perturbed opponent fields to generate |
| `--dro-scale` | 0.15 | Scale of log-normal ownership perturbation noise |
| `--dro-aggregation` | "cvar" | Aggregation method: `mean`, `cvar`, or `mean_minus_std` |
| `--dro-lambda` | 0.5 | Blending parameter: 0 = pure mean EV, 1 = pure robust EV |

### DRO Aggregation Methods

- **`mean`**: Simple average of EVs across all perturbed fields. Treats all field scenarios equally.
- **`cvar`** (Conditional Value at Risk): Average of the worst `alpha` fraction of EVs. Focuses on downside protection against adversarial fields.
- **`mean_minus_std`**: `mean(EVs) - lambda * std(EVs)`. Penalizes lineups whose EV is volatile across field scenarios.

### Note

`field/robust.py` remains on disk for future reference. Only the pipeline wiring (calls from `pipeline.py` and CLI arguments in `cli.py`) has been removed. The file can be re-integrated by restoring the pipeline routing code and CLI arguments documented above.

### Why Removed

Added significant complexity to the pipeline (multiple code paths for DRO vs non-DRO, resampled vs fixed field variants) for marginal improvement. The default field model is already reasonably robust to ownership misspecification because it uses soft priors with ownership-weighted sampling rather than point estimates.

### When to Revisit

When opponent field modeling becomes more sophisticated (e.g., modeling correlation in opponent construction strategies, or when field data from past contests becomes available for calibration), DRO provides principled insurance against field misspecification. The Wasserstein ball around the nominal field distribution is a mathematically clean way to express uncertainty about the opponent field.

---

## Cross-References

- `docs/PLAN_leverage_aware_selection.md` -- Original design plan for leverage-aware selection, including motivation and alternative approaches considered.
- `docs/CPT_DIVERSITY_FIX.md` -- CPT diversity analysis that motivated the exposure cap feature.
- **Git history**:
  - Commit `ec2e3df` -- "Add t-copula and greedy marginal portfolio selection" -- contains the addition of multi-portfolio diversification and greedy marginal selection.
  - Commit `716a1a1` -- "Add hierarchical game-environment simulation and player effects layer" -- contains hierarchical simulation and several of the features documented here.
