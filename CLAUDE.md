# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DFS Showdown GTO Portfolio Builder computes true contest expected value for NFL single-game DFS (DraftKings Showdown format: 1 CPT at 1.5x + 5 FLEX). The system models self-competition among your entries and uses Monte Carlo simulation to compute portfolio-level EV.

## Commands

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run CLI:**
```bash
python -m showdown_gto.cli "NFL_2026-01-25-530pm_DK_SHOWDOWN_LAR-@-SEA.csv" --n-select 150 --n-sims 10000
```

**With contest preset:**
```bash
python -m showdown_gto.cli projections.csv --contest-preset dk_showdown_5 --n-select 150
```

**With custom contest JSON:**
```bash
python -m showdown_gto.cli projections.csv --contest-file contest.json --output results.json
```

**Run reference tests:**
```bash
python showdown_gto_reference.py
```

## Architecture

### Module Structure

```
showdown_gto/
├── types.py              # Core data structures
├── config.py             # Contest presets, field config defaults
├── diagnostics.py        # Full portfolio diagnostics (self-comp decomposition, near-dupes, etc.)
├── data/
│   ├── loader.py         # CSV parsing for DK Showdown format
│   └── correlations.py   # Correlation matrix handling (nearest-PSD projection)
├── simulation/
│   ├── engine.py         # Monte Carlo simulation from percentiles
│   └── bounds.py         # Guaranteed score bounds
├── candidates/
│   └── enumeration.py    # Lineup enumeration with salary constraints
├── field/
│   ├── generator.py      # Field generation with soft priors
│   └── robust.py         # DRO: two-layer field perturbation (ownership + condensation)
├── scoring/
│   ├── histogram.py      # ArrayHistogram (fail-fast)
│   └── payout.py         # PayoutLookup with prefix sums
├── ev/
│   ├── approx.py         # Streaming approx EV
│   ├── portfolio.py      # True portfolio EV with self-competition + covariance penalty
│   └── game_states.py    # Game-state classification and coverage diagnostic
├── pipeline.py           # Full orchestration (single + multi-contest)
└── cli.py                # CLI entry point
```

### Core Data Flow (5 Phases)

**Phase 1: Simulation & Bounds**
- Load CSV with `data.loader.load_projections()` - separates CPT/FLEX pools
- Simulate outcomes via `simulation.engine.simulate_outcomes()` using Gaussian or t-copula
- Outcomes: `[n_players, n_sims]` int32 (quantized: points x 10)
- Compute bounds: `simulation.bounds.compute_guaranteed_score_bounds()`

**Phase 2: Candidate & Field Generation**
- Enumerate lineups: `candidates.enumeration.enumerate_lineups()`
- Generate field: `field.generator.generate_field_simulated()` (quality x ownership) or `generate_field()` (legacy ownership-only)

**Phase 3: EV Computation & DRO**
- Approx EV: `ev.approx.compute_approx_lineup_evs()` (streaming, chunked)
- Optional DRO: `field.robust.generate_perturbed_fields()` → `compute_robust_approx_evs()` → `build_union_shortlist()`
- DRO uses two-layer perturbations: CPT Dirichlet + FLEX three-factor noise + HHI-space condensation

**Phase 4: Selection**
- Selection: `top_n` (sort by robust/approx EV, pick top N) or `greedy_marginal` (sequentially pick highest marginal EV with Markowitz-style profit covariance penalty)
- Covariance penalty: `adjusted_marginal = base_marginal - γ_dynamic × cov(candidate, portfolio)`

**Phase 5: True EV & Diagnostics**
- True EV: `ev.portfolio.compute_true_portfolio_ev()` (with self-competition)
- Diagnostics: `diagnostics.compute_full_diagnostics()` (self-comp decomposition, near-duplicates, game-state coverage)

### Key Invariants (v2.5.1)

1. **Fail-fast histogram build**: Out-of-range scores raise `ValueError`
2. **Explicit rank lookup**: No clamping; above max -> rank=1, below min -> rank=total+1
3. **Self-competition check**: `combined_histogram.total_entries == field_size + n_selected`

### Scoring Math

- CPT score: `(base_score * 15 + 5) // 10` (1.5x with rounding)
- FLEX score: sum of base scores
- Lineup score: CPT score + FLEX scores

## Key Design Decisions

- **Quantized scores**: All scores are `int32` (points x 10) for histogram indexing
- **Streaming EV**: Process candidates in chunks to avoid O(n_cand x n_sims) memory
- **No clamping**: Out-of-range handling is explicit, not hidden by `np.clip()`
- **Prefix sums for payouts**: O(1) tie-splitting via `prefix[end] - prefix[start-1]`
- **Gaussian/t-copula**: Correlated simulations from percentile projections
- **Game-environment variance decomposition**: Hierarchical simulation with shared game-state factors

## CSV Format

The loader expects DraftKings Showdown CSV with columns:
- `DFS ID`, `Name`, `Pos`, `Team`, `Opp`, `Salary`
- `dk_25_percentile`, `dk_50_percentile`, `dk_75_percentile`, `dk_85_percentile`, `dk_95_percentile`, `dk_99_percentile`
- `dk_std`, `Adj Own` (or `My Own`)

CPT entries have 1.5x salary vs FLEX for the same player.

**Player filtering**: `--min-projection <value>` removes players below the given median projection (`dk_50_percentile`) from both CPT and FLEX pools before enumeration.

## Contest JSON Format

```json
{
  "name": "Contest Name",
  "entry_fee": 5.0,
  "total_entries": 5000,
  "your_entries": 150,
  "payout_tiers": [
    {"start_rank": 1, "end_rank": 1, "payout": 1000.0},
    {"start_rank": 2, "end_rank": 5, "payout": 100.0}
  ]
}
```

## Correlation System

The system uses archetype-based correlations from `correlation_config_v2.json`:

**Player Archetypes:**
- QBs: `DUAL_THREAT_QB`, `POCKET_QB`
- RBs: `BELLCOW_RECEIVING_RB`, `COMMITTEE_RB`, `BELLCOW_EARLY_RB`, `SATELLITE_RB`
- WRs: `ALPHA_OUTSIDE_WR`, `DEEP_THREAT_WR`, `SECONDARY_OUTSIDE_WR`, `ALPHA_SLOT_WR`, `SECONDARY_SLOT_WR`, `DEPTH_WR`
- TEs: `ELITE_TE`, `RECEIVING_TE`, `BLOCKING_TE`
- Special: `K`, `DST`

**Usage with correlations:**
```bash
python -m showdown_gto.cli projections.csv \
    --correlation-config correlation_config_v2.json \
    --archetype-map archetype_map_LAR_SEA.json \
    --n-select 150
```

**Archetype map format** (`archetype_map_LAR_SEA.json`):
```json
{
  "Puka Nacua": "ALPHA_OUTSIDE_WR",
  "Matthew Stafford": "POCKET_QB",
  "Kenneth Walker III": "BELLCOW_EARLY_RB"
}
```

If no archetype map is provided, archetypes are inferred from position and projection.

## Selection Methods

### top_n (default)

Sorts all candidates by approximate EV and selects the top N. Fast and straightforward.

### greedy_marginal

Sequentially picks the lineup with highest marginal EV at each step, accounting for self-competition among already-selected lineups. More accurate but slower.

```bash
python -m showdown_gto.cli projections.csv --selection-method greedy_marginal --shortlist-size 2000
```

## Field Generation Methods

### simulated (default)

Quality x ownership field: scores candidates across a subset of simulations, then samples field lineups proportional to quality^sharpness * ownership^power.

```bash
python -m showdown_gto.cli projections.csv --field-method simulated --field-sharpness 5.0 --ownership-power 0.5
```

### ownership (legacy)

Player-by-player ownership-weighted sampling with structural priors (QB pairing, bring-back).

```bash
python -m showdown_gto.cli projections.csv --field-method ownership
```

## DRO (Distributionally Robust Optimization)

Tests lineups against perturbed opponent fields. Prefers lineups robust across plausible fields.

**Two-layer perturbation model:**
- Layer 1a: CPT ownership — Dirichlet simplex (alpha floored at 0.01)
- Layer 1b: FLEX ownership — Three-factor correlated noise (global 20% + team 40% + role 20% + idiosyncratic 20%) in logit space
- Layer 2: Condensation — Perturb in HHI space, invert to sharpness via calibration curve

**Aggregation**: mean (default), CVaR, or mean_minus_std. No lambda blending.

```bash
python -m showdown_gto.cli projections.csv \
    --selection-method greedy_marginal \
    --dro-perturbations 50 \
    --dro-scale 0.10 \
    --dro-hhi-scale 0.15 \
    --dro-aggregation mean
```

**Files**: `field/robust.py`, `pipeline.py`

## Profit Covariance Penalty (Markowitz-style)

Reduces self-competition by penalizing candidates whose profits correlate with the existing portfolio.

```
adjusted_marginal = base_marginal - γ_dynamic × cov(candidate_profit, portfolio_profit)
```

- **Dynamic gamma**: `gamma_dynamic = gamma_pct × max(base_marginals)` — scales with current step's EV range
- Default `gamma_pct = 0.05` (5% of max marginal)
- Only active during `greedy_marginal` selection

```bash
python -m showdown_gto.cli projections.csv \
    --selection-method greedy_marginal \
    --covariance-gamma 0.05
```

**Files**: `ev/portfolio.py` (`_greedy_loop()`)

## Multi-Contest Mode

Accepts multiple contests, shares simulation/enumeration, runs per-contest field/EV/selection.

**Tiered priority system**:
- Tier 1 (flagship): Pure greedy, no diversity penalty. Auto-assigned to largest prizepool.
- Tier 2 (high saturation): field_share > 5%. Covariance penalty + light overlap penalty ($0.005).
- Tier 3 (low saturation): field_share ≤ 5%. Covariance + stronger overlap penalty ($0.015).

```bash
python -m showdown_gto.cli projections.csv \
    --contest-file main_contest.json \
    --multi-contest side1.json --multi-contest side2.json \
    --selection-method greedy_marginal
```

**Outputs**: Per-contest CSV (`portfolio_{name}_{n}.csv`), summary JSON (`portfolio_summary.json`)

**Files**: `pipeline.py` (`run_multi_contest_optimization`), `cli.py`, `types.py`

## Diagnostics

Full portfolio diagnostics including self-competition decomposition:

- **Self-comp decomposition**: exact ties, near ties (±2 bins), general displacement
- **Near-duplicate detection**: pairs sharing ≥ 4 of 6 roster slots
- **Profit covariance metrics**: mean/max pairwise correlation, % pairs > 0.8
- **Exposure leverage**: player exposure rates vs field ownership
- **Tail decomposition**: top-1%, top-10%, bottom-50% EV share
- **Game-state coverage**: 7 gap-free states with concentration ratios

**Files**: `diagnostics.py`, `ev/game_states.py`

## Validation Scripts

Standalone diagnostic scripts (not wired into the pipeline):

- `scripts/validate_ownership.py` — Player exposure, salary utilization, CPT frequency, stack structure, sharpness-to-HHI calibration curve
- `scripts/validate_correlations.py` — PSD check (Higham projection), structural validation, condition number, empirical Spearman check

## Correlation Matrix Validation

`data/correlations.py` applies nearest-PSD projection (Higham's alternating projections) when the correlation matrix has negative eigenvalues. Ridge regularization (`matrix += ε*I`, ε=1e-6) is applied only when condition number exceeds 10,000.

## Removed Features

Several features were removed to restore the clean EV-optimal pipeline. See `docs/REMOVED_FEATURES_ARCHIVE.md` for full documentation including function signatures, rationale, and conditions for revisiting:

- Leverage-adjusted EV (`compute_leverage_adjusted_evs`)
- Exposure cap (`select_with_exposure_cap`)
- DPP selection (`dpp_select_portfolio`)
- Barbell floor/ceiling selection (`barbell_select_portfolio`)
- Multi-portfolio diversification (`run_multi_portfolio_optimization`)
- Portfolio diagnostics (`compute_portfolio_diagnostics`)
