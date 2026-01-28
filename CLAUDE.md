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
├── data/
│   ├── loader.py         # CSV parsing for DK Showdown format
│   └── correlations.py   # Correlation matrix handling
├── simulation/
│   ├── engine.py         # Monte Carlo simulation from percentiles
│   └── bounds.py         # Guaranteed score bounds
├── candidates/
│   └── enumeration.py    # Lineup enumeration with salary constraints
├── field/
│   └── generator.py      # Field generation with soft priors
├── scoring/
│   ├── histogram.py      # ArrayHistogram (fail-fast)
│   └── payout.py         # PayoutLookup with prefix sums
├── ev/
│   ├── approx.py         # Streaming approx EV
│   └── portfolio.py      # True portfolio EV with self-competition
├── pipeline.py           # Full orchestration
└── cli.py                # CLI entry point
```

### Core Data Flow (3 Phases)

**Phase 1: Simulation & Bounds**
- Load CSV with `data.loader.load_projections()` - separates CPT/FLEX pools
- Simulate outcomes via `simulation.engine.simulate_outcomes()` using Gaussian copula
- Outcomes: `[n_players, n_sims]` int32 (quantized: points × 10)
- Compute bounds: `simulation.bounds.compute_guaranteed_score_bounds()`

**Phase 2: Candidate & Field Generation**
- Enumerate lineups: `candidates.enumeration.enumerate_lineups()`
- Generate field: `field.generator.generate_field()` with ownership-weighted sampling

**Phase 3: EV Computation**
- Approx EV: `ev.approx.compute_approx_lineup_evs()` (streaming, chunked)
- True EV: `ev.portfolio.compute_true_portfolio_ev()` (with self-competition)

### Key Invariants (v2.5.1)

1. **Fail-fast histogram build**: Out-of-range scores raise `ValueError`
2. **Explicit rank lookup**: No clamping; above max → rank=1, below min → rank=total+1
3. **Self-competition check**: `combined_histogram.total_entries == field_size + n_selected`

### Scoring Math

- CPT score: `(base_score * 15 + 5) // 10` (1.5x with rounding)
- FLEX score: sum of base scores
- Lineup score: CPT score + FLEX scores

## Key Design Decisions

- **Quantized scores**: All scores are `int32` (points × 10) for histogram indexing
- **Streaming EV**: Process candidates in chunks to avoid O(n_cand × n_sims) memory
- **No clamping**: Out-of-range handling is explicit, not hidden by `np.clip()`
- **Prefix sums for payouts**: O(1) tie-splitting via `prefix[end] - prefix[start-1]`
- **Gaussian copula**: Correlated simulations from percentile projections

## CSV Format

The loader expects DraftKings Showdown CSV with columns:
- `DFS ID`, `Name`, `Pos`, `Team`, `Opp`, `Salary`
- `dk_25_percentile`, `dk_50_percentile`, `dk_75_percentile`, `dk_85_percentile`, `dk_95_percentile`, `dk_99_percentile`
- `dk_std`, `Adj Own` (or `My Own`)

CPT entries have 1.5x salary vs FLEX for the same player.

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
