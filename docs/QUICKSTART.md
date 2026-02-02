# Quick Start Guide

Get from zero to optimized DraftKings Showdown portfolio in five steps.

## Prerequisites

- **Python 3.9+** installed and available as `python` or `python3`
- **A projections CSV** for an upcoming NFL Showdown slate (see [CSV format](#step-2-prepare-your-csv) below)

## Step 1: Install

```bash
git clone <repo-url>
cd Showdown_gto
pip install -r requirements.txt
```

Dependencies: numpy, pandas, scipy, click.

## Step 2: Prepare Your CSV

Your projections CSV needs columns for each player in both CPT and FLEX slots. The loader identifies CPT entries by salary (1.5x the corresponding FLEX entry).

**Required columns:**

| Column | What it is |
|---|---|
| `DFS ID` | DraftKings player ID |
| `Name` | Player name (must match between CPT and FLEX rows) |
| `Pos` | Position (QB, RB, WR, TE, K, DST) |
| `Team` | Team abbreviation |
| `Salary` | Salary for this slot |
| `dk_25_percentile` | 25th percentile DK points projection |
| `dk_50_percentile` | 50th percentile (median) |
| `dk_75_percentile` | 75th percentile |
| `dk_85_percentile` | 85th percentile |
| `dk_95_percentile` | 95th percentile |
| `dk_99_percentile` | 99th percentile |
| `dk_std` | Standard deviation |
| `Adj Own` or `My Own` | Projected ownership (0-100) |

**Example rows:**

```
DFS ID,Name,Pos,Team,Opp,Salary,dk_25_percentile,dk_50_percentile,dk_75_percentile,dk_85_percentile,dk_95_percentile,dk_99_percentile,dk_std,Adj Own
12345,Patrick Mahomes,QB,KC,BUF,16200,12.5,20.3,28.1,32.4,38.5,45.2,9.2,28.0
12345,Patrick Mahomes,QB,KC,BUF,10800,12.5,20.3,28.1,32.4,38.5,45.2,9.2,28.0
12346,Travis Kelce,TE,KC,BUF,13500,8.2,14.1,20.5,24.0,30.2,36.8,7.5,35.0
12346,Travis Kelce,TE,KC,BUF,9000,8.2,14.1,20.5,24.0,30.2,36.8,7.5,35.0
```

Note: Each player appears twice -- once with CPT salary (higher), once with FLEX salary (lower). The tool handles the separation automatically.

## Step 3: Choose Your Contest

**Option A: Use a preset**

| Preset | Fee | Field | Use case |
|---|---|---|---|
| `dk_showdown_5` | $5 | 5,000 | Standard Showdown |
| `dk_showdown_20` | $20 | 3,000 | Mid-stakes |
| `dk_showdown_milly` | $5 | 200,000 | Milly Maker |
| `dk_showdown_small` | $5 | 500 | Small field (20 entries) |

```bash
python -m showdown_gto.cli projections.csv --contest-preset dk_showdown_5 --n-select 150
```

**Option B: Create a custom contest JSON**

Copy this template and fill in your contest details:

```json
{
  "name": "My Contest",
  "entry_fee": 5.0,
  "total_entries": 5000,
  "your_entries": 150,
  "payout_tiers": [
    {"start_rank": 1, "end_rank": 1, "payout": 1000.0},
    {"start_rank": 2, "end_rank": 5, "payout": 100.0},
    {"start_rank": 6, "end_rank": 20, "payout": 50.0},
    {"start_rank": 21, "end_rank": 50, "payout": 25.0},
    {"start_rank": 51, "end_rank": 100, "payout": 15.0},
    {"start_rank": 101, "end_rank": 250, "payout": 10.0},
    {"start_rank": 251, "end_rank": 500, "payout": 7.50},
    {"start_rank": 501, "end_rank": 1000, "payout": 6.00}
  ]
}
```

```bash
python -m showdown_gto.cli projections.csv --contest-file my_contest.json --n-select 150
```

Tip: You can find payout structures on the DraftKings contest lobby page. `total_entries` is the max entries, `your_entries` is how many you're entering.

## Step 4: Run

**Quick test** (fewer sims, fast feedback):

```bash
python -m showdown_gto.cli projections.csv --contest-preset dk_showdown_5 \
    --n-select 20 --n-sims 10000
```

**Production run** (full accuracy, greedy selection):

```bash
python -m showdown_gto.cli projections.csv --contest-preset dk_showdown_5 \
    --n-select 150 --n-sims 100000 --selection-method greedy_marginal
```

**Production with correlations** (recommended):

```bash
python -m showdown_gto.cli projections.csv --contest-preset dk_showdown_5 \
    --n-select 150 --n-sims 100000 --selection-method greedy_marginal \
    --correlation-config correlation_config_v2.json \
    --archetype-map archetype_map_LAR_SEA.json
```

## Step 5: Read the Output

The tool prints a results summary to the console:

```
Running optimization...
  CSV: projections.csv
  Contest: DraftKings Showdown $5
  Entry fee: $5.0
  Total entries: 5000
  Your entries: 150
  Simulations: 100000
  Field mode: fixed
  Field method: simulated
  Field sharpness: 5.0
  Ownership power: 0.5
  Field quality sims: 1000
  Copula: gaussian
  Selection: greedy_marginal (shortlist=2000)

============================================================
PORTFOLIO RESULTS
============================================================

Portfolio EV: $774.15          <-- Total expected winnings across all 150 entries
Entry Cost: $750.00            <-- 150 entries x $5
Expected Profit: $24.15        <-- EV minus cost
ROI: 3.22%                     <-- Profit / cost
P(Profit): 62.4%              <-- Probability of profiting across all sims
Self-competition Cost: $0.42   <-- Value lost to your own entries competing

Selected 150 lineups

Top 10 Lineups by Approx EV:
  1. $5.82 - Patrick Mahomes (CPT), Travis Kelce, Isiah Pacheco, ...
  2. $5.71 - Josh Allen (CPT), Stefon Diggs, James Cook, ...
  ...

Lineups exported to portfolio_lineups.csv
```

**Key metrics to look at:**

| Metric | What it means | Good range |
|---|---|---|
| ROI | Return on investment | Positive = profitable edge |
| P(Profit) | How often you profit | > 50% for consistent results |
| Self-Comp Cost | Value lost to self-competition | Lower is better; < $5 is good for 150 entries |

## Step 6: Upload to DraftKings

The output file `portfolio_lineups.csv` contains columns:

```
CPT_ID,FLEX1_ID,FLEX2_ID,FLEX3_ID,FLEX4_ID,FLEX5_ID,CPT_Name,FLEX1_Name,...
```

Use the DraftKings bulk lineup upload feature to import these lineups. The `CPT_ID` and `FLEX*_ID` columns contain the DFS IDs that DraftKings expects.

## What Next?

Once you're comfortable with basic runs:

1. **Switch to `greedy_marginal` selection** if you haven't already -- it's the most important upgrade for portfolio quality
2. **Add correlations** via `--correlation-config` and `--archetype-map` for more realistic player simulations
3. **Try DRO** with `--dro-perturbations 50` to hedge against ownership uncertainty
4. **Explore t-copula** with `--copula-type t` for heavier tail dependence
5. **Run multi-contest** if entering the same slate across several contests

See the [Strategy Guide](STRATEGY_GUIDE.md) for detailed explanations of each feature and recommended workflows.

## Troubleshooting

### "No valid players found in CSV"

The CSV is missing `dk_50_percentile` values or all values are zero/empty. Check that your CSV has percentile projections for each player.

### "No valid lineups found"

No lineup combination fits within the $50,000 salary cap. This can happen if the CSV only has high-salary players or if CPT/FLEX separation failed. Check that your CSV has both CPT (high salary) and FLEX (low salary) rows for each player.

### Slow runtime

- Reduce `--n-sims` (e.g. `10000` for testing)
- Reduce `--n-select` for quick tests
- For greedy selection, reduce `--shortlist-size`

### Out of memory

- Reduce `--n-sims`
- The tool uses streaming EV computation, but very large simulations can still use significant memory

### Greedy selection output looks different from top_n

This is expected. Greedy selection explicitly avoids clustering and may pick lineups with lower individual EV but better portfolio-level diversification. The true portfolio EV is the number that matters.
