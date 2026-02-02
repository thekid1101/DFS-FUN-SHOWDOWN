# DFS Showdown GTO Portfolio Builder

A portfolio optimizer for DraftKings NFL Showdown contests that models **self-competition** between your own entries to compute true contest expected value.

## The Problem

Most DFS optimizers treat each lineup independently: they score 150 lineups and call it done. But when you enter 150 lineups into the same contest, your own entries compete against each other for the same payout slots. A portfolio of 150 "best" lineups can lose 20-30% of its theoretical value to self-competition.

This tool models the full contest: your lineups + a simulated opponent field, scored across Monte Carlo simulations, with payouts computed using actual rank and tie-splitting logic. The result is **true portfolio EV** -- what you actually expect to win after accounting for your lineups displacing each other.

## Key Features

- **True Portfolio EV** -- Monte Carlo simulation with self-competition modeling
- **Greedy Marginal Selection** -- Sequentially picks the lineup that adds the most marginal EV at each step
- **Profit Covariance Penalty** -- Markowitz-style diversification to reduce correlated payoffs
- **Correlated Player Simulations** -- Archetype-based correlation matrices with Gaussian or t-copula
- **Distributionally Robust Optimization (DRO)** -- Tests lineups against perturbed opponent fields
- **Simulated Field Modeling** -- Quality-weighted field generation that models how opponents actually build lineups
- **Multi-Contest Optimization** -- Shared simulation with per-contest field and selection, tiered priority system
- **Game-State Coverage** -- Diagnostic showing how your portfolio performs across shootouts, blowouts, upsets, etc.

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url>
cd Showdown_gto
pip install -r requirements.txt

# 2. Run with a contest preset
python -m showdown_gto.cli projections.csv --contest-preset dk_showdown_5 --n-select 150

# 3. Run with greedy selection (recommended for serious use)
python -m showdown_gto.cli projections.csv --contest-preset dk_showdown_5 \
    --n-select 150 --selection-method greedy_marginal

# 4. Upload portfolio_lineups.csv to DraftKings
```

See [docs/QUICKSTART.md](docs/QUICKSTART.md) for a detailed walkthrough.

## CSV Input Format

Your projections CSV must include these columns:

| Column | Description | Example |
|---|---|---|
| `DFS ID` | DraftKings player ID | `12345` |
| `Name` | Player name | `Patrick Mahomes` |
| `Pos` | Position | `QB` |
| `Team` | Team abbreviation | `KC` |
| `Opp` | Opponent team | `BUF` |
| `Salary` | Salary (CPT entries have 1.5x salary) | `16200` |
| `dk_25_percentile` | 25th percentile projection | `10.5` |
| `dk_50_percentile` | 50th percentile (median) | `18.2` |
| `dk_75_percentile` | 75th percentile | `25.0` |
| `dk_85_percentile` | 85th percentile | `29.1` |
| `dk_95_percentile` | 95th percentile | `35.8` |
| `dk_99_percentile` | 99th percentile | `42.0` |
| `dk_std` | Standard deviation | `8.5` |
| `Adj Own` or `My Own` | Projected ownership % | `25.0` |

The CSV should contain both CPT and FLEX entries for each player. CPT entries are identified by having 1.5x the salary of the corresponding FLEX entry.

## Contest Configuration

### Presets

| Preset | Entry Fee | Field Size | Description |
|---|---|---|---|
| `dk_showdown_5` | $5 | 5,000 | Standard $5 Showdown |
| `dk_showdown_20` | $20 | 3,000 | Mid-stakes Showdown |
| `dk_showdown_milly` | $5 | 200,000 | Milly Maker |
| `dk_showdown_small` | $5 | 500 | Small field (20 entries) |

```bash
python -m showdown_gto.cli projections.csv --contest-preset dk_showdown_5
```

### Custom Contest JSON

For contests not covered by presets, create a JSON file:

```json
{
  "name": "Thursday Night $10",
  "entry_fee": 10.0,
  "total_entries": 8000,
  "your_entries": 150,
  "payout_tiers": [
    {"start_rank": 1, "end_rank": 1, "payout": 5000.0},
    {"start_rank": 2, "end_rank": 5, "payout": 500.0},
    {"start_rank": 6, "end_rank": 20, "payout": 100.0},
    {"start_rank": 21, "end_rank": 50, "payout": 50.0},
    {"start_rank": 51, "end_rank": 100, "payout": 25.0},
    {"start_rank": 101, "end_rank": 250, "payout": 15.0},
    {"start_rank": 251, "end_rank": 500, "payout": 12.0},
    {"start_rank": 501, "end_rank": 1000, "payout": 10.0}
  ]
}
```

```bash
python -m showdown_gto.cli projections.csv --contest-file my_contest.json
```

## Selection Methods

### `top_n` (default)

Sorts all candidates by approximate EV and picks the top N. Fast, but ignores self-competition during selection.

### `greedy_marginal` (recommended)

At each step, evaluates every shortlisted candidate and picks the one that adds the most marginal EV to the current portfolio. This directly accounts for self-competition: if the portfolio already has 50 lineups scoring in the same range, the next pick will favor a lineup that scores differently.

**The difference is significant.** On a representative DK Showdown $5 (5,000 entries, 150 selected):

| Method | Portfolio EV | ROI | Self-Comp Cost |
|---|---|---|---|
| `top_n` | $924 | 23.4% | $34.18 |
| `greedy_marginal` | $774 | 3.3% | $0.42 |

The top_n portfolio has inflated apparent EV because it clusters in the same score ranges. The greedy portfolio has lower nominal EV but dramatically less self-competition, yielding a realistic ROI estimate.

```bash
python -m showdown_gto.cli projections.csv --contest-preset dk_showdown_5 \
    --selection-method greedy_marginal --shortlist-size 2000
```

## Output

### Console

```
============================================================
PORTFOLIO RESULTS
============================================================

Portfolio EV: $774.15
Entry Cost: $750.00
Expected Profit: $24.15
ROI: 3.22%
P(Profit): 62.4%
Self-competition Cost: $0.42

Selected 150 lineups

Top 10 Lineups by Approx EV:
  1. $5.82 - Patrick Mahomes (CPT), Travis Kelce, ...
  2. $5.71 - ...
```

### Files

| File | Description |
|---|---|
| `portfolio_lineups.csv` | Lineup IDs and names, ready for DraftKings upload |
| `portfolio_summary.json` | Full diagnostics (if `-o results.json` specified) |

Multi-contest mode produces per-contest CSVs: `portfolio_{contest_name}_{count}.csv`

## Advanced Features

These features are covered in detail in the [Strategy Guide](docs/STRATEGY_GUIDE.md).

### Correlations & Archetypes

Model player correlations using archetype-based correlation matrices (QB-WR stacking, opposing QB bring-back, etc.):

```bash
python -m showdown_gto.cli projections.csv \
    --correlation-config correlation_config_v2.json \
    --archetype-map archetype_map_LAR_SEA.json
```

### t-Copula (Tail Dependence)

Use a Student-t copula instead of Gaussian for heavier tail dependence (extreme games happen together):

```bash
python -m showdown_gto.cli projections.csv --copula-type t --copula-df 5
```

### DRO (Distributionally Robust Optimization)

Test lineups against perturbed opponent fields to find lineups robust to ownership uncertainty:

```bash
python -m showdown_gto.cli projections.csv \
    --dro-perturbations 50 --dro-scale 0.10 --dro-hhi-scale 0.15
```

### Profit Covariance Penalty

Diversify the portfolio during greedy selection by penalizing candidates whose profits correlate with the existing portfolio:

```bash
python -m showdown_gto.cli projections.csv \
    --selection-method greedy_marginal --covariance-gamma 0.05
```

### Multi-Contest Optimization

Optimize across multiple contests simultaneously, with tiered priority and cross-contest overlap penalties:

```bash
python -m showdown_gto.cli projections.csv \
    --contest-file main_contest.json \
    --multi-contest side1.json --multi-contest side2.json \
    --selection-method greedy_marginal
```

### Field Modeling

Control how the simulated opponent field behaves:

```bash
python -m showdown_gto.cli projections.csv \
    --field-method simulated --field-sharpness 5.0 --ownership-power 0.5
```

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `csv_path` | *(required)* | Path to projections CSV |
| `--contest-preset`, `-p` | -- | Contest preset name |
| `--contest-file`, `-c` | -- | Custom contest JSON file |
| `--n-select`, `-n` | `150` | Number of lineups to select |
| `--n-sims`, `-s` | `100000` | Number of Monte Carlo simulations |
| `--selection-method` | `top_n` | `top_n` or `greedy_marginal` |
| `--shortlist-size` | `2000` | Candidates considered during greedy selection |
| `--greedy-sims` | `min(n_sims, 10000)` | Sims used during greedy loop |
| `--covariance-gamma` | `0.05` | Profit covariance penalty strength (0=disabled) |
| `--copula-type` | `gaussian` | `gaussian` or `t` |
| `--copula-df` | `5` | Degrees of freedom for t-copula |
| `--field-method` | `simulated` | `simulated` or `ownership` (legacy) |
| `--field-sharpness` | `5.0` | Field projection awareness (0=ownership-only, 8+=optimizer-heavy) |
| `--ownership-power` | `0.5` | Ownership influence on field duplication |
| `--field-quality-sims` | `1000` | Sims for field quality scoring |
| `--field-mode` | `fixed` | `fixed` or `resample_per_sim` |
| `--dro-perturbations` | `0` | Number of DRO perturbations (0=disabled, 50=recommended) |
| `--dro-scale` | `0.10` | DRO ownership perturbation scale |
| `--dro-hhi-scale` | `0.15` | DRO condensation perturbation scale |
| `--dro-aggregation` | `mean` | `mean`, `cvar`, or `mean_minus_std` |
| `--correlation-config` | -- | Path to correlation config JSON |
| `--archetype-map` | -- | Path to player archetype mapping JSON |
| `--spread` | -- | Vegas spread, e.g. `"LAR -3.5"` |
| `--game-total` | -- | Vegas game total, e.g. `48.5` |
| `--effects-file` | -- | Player effects JSON |
| `--sim-config` | -- | Simulation config JSON |
| `--multi-contest` | -- | Additional contest JSON (repeatable) |
| `--output`, `-o` | -- | Output file path (.csv or .json) |
| `--seed` | -- | Random seed for reproducibility |
| `--verbose/--quiet` | `--verbose` | Toggle logging output |

## Documentation

- **[Quick Start Guide](docs/QUICKSTART.md)** -- Get running from zero
- **[Strategy Guide](docs/STRATEGY_GUIDE.md)** -- DFS concepts, metric interpretation, and recommended workflows

## Requirements

- Python 3.9+
- numpy >= 1.24.0
- pandas >= 2.0.0
- scipy >= 1.10.0
- click >= 8.0.0
