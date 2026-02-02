# Strategy Guide

How the optimizer works, what the numbers mean, and how to use the features effectively.

## Table of Contents

- [The Self-Competition Problem](#the-self-competition-problem)
- [Understanding Key Metrics](#understanding-key-metrics)
- [Selection Methods](#selection-methods)
- [Correlation System & Archetypes](#correlation-system--archetypes)
- [Copula Choice](#copula-choice)
- [DRO: Distributionally Robust Optimization](#dro-distributionally-robust-optimization)
- [Field Modeling](#field-modeling)
- [Multi-Contest Optimization](#multi-contest-optimization)
- [Interpreting Diagnostics](#interpreting-diagnostics)
- [Recommended Workflows](#recommended-workflows)
- [FAQ](#faq)

---

## The Self-Competition Problem

When you enter 150 lineups into a 5,000-person contest, you own 3% of the field. Those 150 lineups aren't just competing against 4,850 opponents -- they're competing against each other.

**Concrete example:** Suppose you have two lineups that both score in the top 50. Without self-competition, each might rank ~25th and win $25. But with both in the contest, one ranks 25th ($25) and the other 26th ($25) -- or worse, they tie at the same score and split a lower total payout. If you had 10 lineups clustered in the top 50, the displacement becomes severe: they push each other down the leaderboard.

**The math:**

- A naive optimizer picks 150 lineups with the highest individual EV
- These lineups tend to cluster in the same score ranges (they're built from the same high-projection players)
- Self-competition cost for a clustered portfolio of 150 can exceed $30 on a $750 investment
- A diversified portfolio might have 3-5% lower individual lineup quality but saves $30+ in self-competition

This is why the tool's `greedy_marginal` selection method exists: it builds the portfolio one lineup at a time, always asking "what does adding this lineup actually do to total portfolio EV?"

---

## Understanding Key Metrics

### Portfolio EV

Total expected winnings across all your entries, computed via Monte Carlo simulation. This is the "true" number that accounts for self-competition. In each simulation, all your lineups and the entire opponent field are scored, ranks are assigned, and payouts are computed using the contest's actual payout structure with tie-splitting.

### Entry Cost

`n_select x entry_fee`. For 150 entries at $5, this is $750.

### Expected Profit

`Portfolio EV - Entry Cost`. Positive means you have an edge.

### ROI (Return on Investment)

`Expected Profit / Entry Cost x 100%`. A 3% ROI on a $750 investment means ~$22.50 expected profit. In DFS, consistent 3-5% ROI is strong.

### P(Profit)

Probability that total winnings exceed total cost. Computed across all simulations. A 60% P(Profit) means that in 60% of simulated outcomes, your portfolio made money.

Note: P(Profit) can be high even with negative ROI if most profitable outcomes are small gains and most losing outcomes are losing your full investment. ROI is the more complete metric.

### Self-Competition Cost

`Sum of individual approx EVs - True portfolio EV`. This is the dollar amount you lose because your lineups compete against each other. Lower is better.

- **< $1** for 150 entries: excellent diversification (greedy_marginal target)
- **$5-15**: moderate, typical of a reasonably diverse top_n portfolio
- **$20+**: significant clustering, consider greedy_marginal

### Approx EV (per lineup)

Each lineup's expected value computed against only the opponent field (ignoring your other entries). Useful for ranking candidates but overstates value when lineups are correlated.

---

## Selection Methods

### top_n

**How it works:** Score all enumerated lineups against the simulated field, sort by approx EV, pick the top N.

**Pros:**
- Fast
- Each individual lineup has the highest possible EV

**Cons:**
- Ignores self-competition entirely
- Lineups cluster in score ranges, inflating self-competition cost
- Reported portfolio EV is realistic (computed post-selection), but the portfolio could be better

**When to use:** Quick tests, very small portfolios (< 20 entries), or when time is limited.

### greedy_marginal (recommended)

**How it works:** Start with an empty portfolio. At each step, evaluate all candidates in a shortlist and pick the one that adds the most *marginal* EV -- the increase in true portfolio EV from adding that lineup. Repeat N times.

The marginal EV of a candidate accounts for:
- Its individual quality (approx EV)
- How much it overlaps with lineups already in the portfolio
- The self-competition it would create

**Pros:**
- Directly optimizes true portfolio EV
- Naturally diversifies: after picking several QB1-heavy lineups, the marginal value of another QB1 lineup drops, and a QB2 lineup becomes more attractive
- Self-competition costs are typically < $1

**Cons:**
- Slower (evaluates candidates at each of N steps)
- `--shortlist-size` limits which candidates are considered

**When to use:** Any serious production run. The quality improvement over top_n is significant.

```bash
python -m showdown_gto.cli projections.csv --contest-preset dk_showdown_5 \
    --selection-method greedy_marginal --shortlist-size 2000
```

### Profit Covariance Penalty

An additional layer during greedy selection. At each step, before picking the best marginal candidate, the tool penalizes candidates whose profits are correlated with the existing portfolio:

```
adjusted_marginal = base_marginal - gamma_dynamic * cov(candidate_profit, portfolio_profit)
```

Where `gamma_dynamic = gamma_pct * max(base_marginals)` scales with the current step's EV range.

**Effect:** Pushes the portfolio toward lineups that win in *different* scenarios than what's already selected. Reduces the concentration ratio of profit outcomes.

```bash
python -m showdown_gto.cli projections.csv \
    --selection-method greedy_marginal --covariance-gamma 0.05
```

Default `--covariance-gamma` is `0.05` (5% of max marginal). Set to `0` to disable.

---

## Correlation System & Archetypes

### Why Correlations Matter

Player scores aren't independent. When the QB throws a touchdown, the WR who caught it also scores. When the game script goes run-heavy, all pass catchers underperform together. Correlated simulations capture these relationships -- without them, the simulator treats a QB-WR stack as no more likely to boom together than a QB and an opposing DST.

### How It Works

The correlation system uses **player archetypes** rather than raw positions. Two alpha WR1s correlate differently with their QB than a slot receiver does. The archetypes are:

**Quarterbacks:**
- `DUAL_THREAT_QB` -- Rushing upside (higher floor, different ceiling shape)
- `POCKET_QB` -- Pass-first (higher pass volume correlation)

**Running Backs:**
- `BELLCOW_RECEIVING_RB` -- Catches passes (correlates with QB)
- `BELLCOW_EARLY_RB` -- Goal-line/early-down (negative game script correlation)
- `COMMITTEE_RB` -- Splits touches
- `SATELLITE_RB` -- Pass-catching specialist

**Wide Receivers:**
- `ALPHA_OUTSIDE_WR` -- Primary target, outside
- `DEEP_THREAT_WR` -- Boom/bust profile
- `SECONDARY_OUTSIDE_WR` -- WR2/WR3 outside
- `ALPHA_SLOT_WR` -- Primary slot target
- `SECONDARY_SLOT_WR` -- Secondary slot
- `DEPTH_WR` -- WR4+

**Tight Ends:**
- `ELITE_TE` -- Target hog (Kelce-type)
- `RECEIVING_TE` -- Route runner
- `BLOCKING_TE` -- Rarely scores DFS points

**Special:**
- `K` -- Kicker
- `DST` -- Team defense

### Usage

1. Create a `correlation_config_v2.json` defining archetype-to-archetype correlations (or use the provided one)
2. Create an `archetype_map.json` assigning each player to an archetype:

```json
{
  "Patrick Mahomes": "POCKET_QB",
  "Puka Nacua": "ALPHA_OUTSIDE_WR",
  "Kenneth Walker III": "BELLCOW_EARLY_RB",
  "Travis Kelce": "ELITE_TE"
}
```

3. Run with both flags:

```bash
python -m showdown_gto.cli projections.csv \
    --correlation-config correlation_config_v2.json \
    --archetype-map archetype_map_LAR_SEA.json
```

If no archetype map is provided, archetypes are inferred from position and projection (reasonable defaults, but less accurate for niche roles).

### Validation

Use the validation script to verify your correlation matrix:

```bash
python scripts/validate_correlations.py
```

This checks positive semi-definiteness, structural soundness (same-team QB-WR correlation > opposing team correlation), and condition number.

---

## Copula Choice

The copula determines how correlated random variables are drawn from the correlation matrix.

### Gaussian (default)

Standard multivariate normal distribution. Correlations are symmetric: players are equally likely to boom together as they are to bust together.

```bash
python -m showdown_gto.cli projections.csv --copula-type gaussian
```

### Student-t (tail dependence)

A t-copula with low degrees of freedom produces heavier tails: extreme events (all players booming or all busting) happen more frequently than under Gaussian. This models the real-world phenomenon where NFL games sometimes go completely sideways in correlated ways.

```bash
python -m showdown_gto.cli projections.csv --copula-type t --copula-df 5
```

**Degrees of freedom (`--copula-df`):**
- `3-4`: Very heavy tails. Extreme co-movements are common. Aggressive.
- `5`: Moderate heavy tails. The default when t-copula is selected.
- `8-10`: Mild heavy tails. Closer to Gaussian behavior.
- `30+`: Essentially Gaussian.

**When to use t-copula:**
- Large-field tournaments (Milly Maker) where tail outcomes drive value
- Games with high uncertainty (backup QB, weather games)
- When you believe correlated boom/bust is underpriced by the field

**When to stick with Gaussian:**
- Small fields where median outcomes matter more
- Cash games (if adapted for Showdown)
- When you prefer more conservative simulations

---

## DRO: Distributionally Robust Optimization

### What It Solves

Your opponent field model is an estimate. You don't know exactly what ownership percentages will be, and you don't know how sharp the field will play. DRO tests your lineups against multiple perturbed versions of the opponent field to find lineups that perform well across a range of plausible fields, not just your best guess.

### How It Works (Conceptual)

DRO generates N perturbed fields (default: 50), each created by:

1. **CPT ownership perturbation**: Perturb CPT ownership using a Dirichlet distribution
2. **FLEX ownership perturbation**: Apply correlated noise in logit space with three factors:
   - Global shift (20%): all players up or down
   - Team shift (40%): entire teams up or down
   - Role shift (20%): position groups up or down
   - Idiosyncratic (20%): individual player noise
3. **Condensation perturbation**: Vary how concentrated the field is (sharp vs. casual) by perturbing in HHI space

For each perturbed field, lineup EVs are recomputed. The final "robust EV" aggregates across all perturbations:
- `mean` (default): Average EV across perturbations
- `cvar`: Conditional Value at Risk -- focuses on worst-case perturbations
- `mean_minus_std`: Penalizes high-variance lineups

### When to Use

- When you're uncertain about ownership projections
- For large fields where small ownership shifts create big EV swings
- When entering a slate where sharp players are likely to have different ownership reads

### Usage

```bash
python -m showdown_gto.cli projections.csv \
    --dro-perturbations 50 \
    --dro-scale 0.10 \
    --dro-hhi-scale 0.15 \
    --dro-aggregation mean
```

**Parameters:**
- `--dro-perturbations`: Number of perturbed fields (50 is a good balance of accuracy vs. speed)
- `--dro-scale`: How much to perturb ownership (0.10 = 10% scale)
- `--dro-hhi-scale`: How much to perturb field condensation (0.15 = 15% scale)
- `--dro-aggregation`: How to combine across perturbations

DRO adds overhead proportional to `n_perturbations`. With 50 perturbations, expect the approx EV step to run ~50x longer.

---

## Field Modeling

The simulated opponent field is critical: your lineup EVs depend entirely on who you're competing against.

### simulated (default)

The simulated field method generates opponent lineups by:

1. Scoring all candidate lineups across a subset of simulations (quality scoring)
2. Computing a probability for each candidate: `p ~ quality^sharpness * ownership^power`
3. Sampling the field from this distribution

This produces a field that plays "like a real contest" -- sharp players build good lineups, casual players build ownership-weighted lineups.

**Key parameters:**

| Parameter | Default | What it controls |
|---|---|---|
| `--field-sharpness` | `5.0` | How projection-aware the field is |
| `--ownership-power` | `0.5` | How ownership-driven duplication is |
| `--field-quality-sims` | `1000` | Sims used for quality scoring |

**Sharpness intuition:**

| Sharpness | Field behavior |
|---|---|
| `0` | Pure ownership-weighted random lineups (casual field) |
| `2-3` | Mostly ownership, some quality awareness |
| `5.0` | Realistic mix of sharp and casual players (default) |
| `8-10` | Optimizer-heavy field (everyone using tools) |
| `15+` | Extremely sharp (unlikely in practice) |

Higher sharpness means the field builds better lineups, which makes it harder for your lineups to stand out. Err toward the default unless you have strong evidence about field sharpness.

### ownership (legacy)

Player-by-player sampling with structural priors (QB pairing, bring-back). Simpler and faster, but doesn't account for how ownership interacts with lineup quality.

```bash
python -m showdown_gto.cli projections.csv --field-method ownership
```

Use this as a sanity check or when you want a quick comparison.

---

## Multi-Contest Optimization

When entering the same slate across multiple contests (e.g., a $5 main, a $20 side, and a Milly Maker), the optimizer shares simulation and enumeration work across contests while running per-contest field generation and selection.

### Tier System

Contests are automatically assigned to tiers:

| Tier | Assignment | Selection behavior |
|---|---|---|
| **Tier 1** (flagship) | Auto-assigned to largest prizepool, or explicit `"tier": 1` in JSON | Pure greedy, no cross-contest penalty |
| **Tier 2** (high saturation) | `field_share > 5%` | Covariance penalty + light overlap penalty ($0.005 per shared lineup) |
| **Tier 3** (low saturation) | `field_share <= 5%` | Covariance penalty + stronger overlap penalty ($0.015 per shared lineup) |

The flagship contest gets first pick of the best lineups. Lower-tier contests are gently nudged toward different lineups to reduce cross-contest correlation.

### Usage

```bash
python -m showdown_gto.cli projections.csv \
    --contest-file main_5k.json \
    --multi-contest side_3k.json \
    --multi-contest milly.json \
    --selection-method greedy_marginal
```

### Output

Multi-contest mode produces:
- Per-contest CSV files: `portfolio_{contest_name}_{count}.csv`
- Summary JSON: `portfolio_summary.json`
- Console output with per-contest EV, ROI, and overlap matrix

---

## Interpreting Diagnostics

The full diagnostics report (available via `compute_full_diagnostics`) provides deep insight into portfolio quality.

### Self-Competition Decomposition

Breaks down self-competition cost into three categories:

| Category | What it means | Concern level |
|---|---|---|
| **Exact ties** | Two of your lineups score identically in a sim | High (they split the same payout) |
| **Near ties** | Within +/- 2 score bins of another lineup | Medium (rank displacement) |
| **General displacement** | Your lineups push each other down broadly | Low (unavoidable at scale) |

If exact ties dominate, you have near-duplicate lineups. If displacement dominates, the portfolio is diverse but large enough that some overlap is inevitable.

### Near-Duplicate Detection

Flags lineup pairs sharing 4+ of 6 roster slots. Near-duplicates almost always score similarly and create self-competition.

| Shared slots | Risk |
|---|---|
| 6/6 | Identical lineup (shouldn't happen) |
| 5/6 | Very high correlation, one should be cut |
| 4/6 | High correlation, worth monitoring |

### Profit Covariance Metrics

| Metric | What it means | Target |
|---|---|---|
| **Mean pairwise correlation** | Average profit correlation across all lineup pairs | < 0.3 is well diversified |
| **Max correlation** | Highest single pair | < 0.8 ideally |
| **% pairs > 0.8** | Fraction of highly correlated pairs | < 5% |

### Exposure & Leverage

**Exposure:** What percentage of your lineups contain a given player.

**Leverage:** `portfolio_exposure / field_ownership`. A leverage of 2.0x on a player means you own them at twice the rate of the field -- you're betting on that player outperforming expectations.

High leverage on high-projection players is expected. High leverage on low-projection players suggests the optimizer is finding value the field is missing.

### Tail Decomposition

| Metric | What it means |
|---|---|
| **Top-1% EV share** | What fraction of total EV comes from your best ~1-2 lineups |
| **Top-10% EV share** | Fraction from your best ~15 lineups |
| **Bottom-50% EV share** | Fraction from the weaker half of the portfolio |

A healthy tournament portfolio has significant EV concentration in the top: most lineups are "insurance" that cover different scenarios, while a few are the primary profit drivers.

### Game-State Coverage

Shows how your portfolio performs across seven game states:

| State | Description |
|---|---|
| `shootout` | High-scoring game (total > game_total + 7) |
| `competitive` | Normal scoring, close (diff <= 7) |
| `blowout_fav` | Favorite dominated |
| `upset` | Underdog won |
| `defensive_close` | Low scoring, close |
| `defensive_blowout_fav` | Low scoring, favorite dominated |
| `defensive_upset` | Low scoring, underdog won |

**Concentration ratio:** `top_N_share / sim_share`. A value of 2.0x for "shootout" means your portfolio finishes in the top 100 at twice the base rate when the game is a shootout. Values near 0 indicate blind spots.

Good portfolios have reasonable coverage across all states. If concentration is 0 for an entire state, you have no lineups that win in that scenario.

---

## Recommended Workflows

### Quick Test Run

Just checking that everything works and getting a feel for the slate:

```bash
python -m showdown_gto.cli projections.csv \
    --contest-preset dk_showdown_5 \
    --n-select 20 --n-sims 10000
```

### Standard Production Run

Full-accuracy single-contest optimization:

```bash
python -m showdown_gto.cli projections.csv \
    --contest-preset dk_showdown_5 \
    --n-select 150 --n-sims 100000 \
    --selection-method greedy_marginal \
    --correlation-config correlation_config_v2.json \
    --archetype-map archetype_map.json \
    --copula-type t --copula-df 5
```

### Production with DRO

Add robustness to ownership uncertainty:

```bash
python -m showdown_gto.cli projections.csv \
    --contest-preset dk_showdown_5 \
    --n-select 150 --n-sims 100000 \
    --selection-method greedy_marginal \
    --correlation-config correlation_config_v2.json \
    --archetype-map archetype_map.json \
    --dro-perturbations 50 --dro-scale 0.10 --dro-hhi-scale 0.15
```

### Multi-Contest Production

Entering the same slate across several contests:

```bash
python -m showdown_gto.cli projections.csv \
    --contest-file main_5k.json \
    --multi-contest side_3k.json \
    --multi-contest milly.json \
    --n-sims 100000 \
    --selection-method greedy_marginal \
    --correlation-config correlation_config_v2.json \
    --archetype-map archetype_map.json
```

### Reproducibility

Use `--seed` to get identical results across runs:

```bash
python -m showdown_gto.cli projections.csv --contest-preset dk_showdown_5 \
    --n-select 150 --n-sims 100000 --selection-method greedy_marginal --seed 42
```

---

## FAQ

### How many simulations should I run?

100,000 (the default) is a good production number. EV estimates stabilize well by 50,000 sims. For quick tests, 10,000 is sufficient to get directionally correct results.

### Why is my greedy_marginal ROI lower than top_n?

The top_n ROI is misleading. Top_n picks lineups with the highest individual EV, which inflates the sum of approx EVs. But when those lineups are placed in the same contest, self-competition erodes the actual value. The greedy ROI is closer to what you'll actually experience.

Look at self-competition cost: if top_n shows $30+ and greedy shows < $1, the greedy portfolio is genuinely better despite the lower headline number.

### What shortlist size should I use?

The default of 2,000 works well for most slates. This means greedy selection considers the top 2,000 candidates (by approx EV) at each step. Larger shortlists (3,000-5,000) give slightly better results but increase runtime proportionally. Below 1,000, you risk missing good candidates.

### Should I always use correlations?

Correlations improve simulation realism. Without them, player outcomes are independent, which underestimates the value of stacked lineups and overestimates the value of max-diversification. Use correlations whenever you have a reasonable archetype map for the slate.

### When should I use t-copula vs Gaussian?

Use t-copula for large-field tournaments where you need to model extreme co-movements (the whole game goes haywire). The effect is most noticeable in tail-heavy scoring environments. For small fields or when tail events are less relevant, Gaussian is simpler and sufficient.

### How does field sharpness affect results?

Higher sharpness means the simulated field builds better lineups. If you set sharpness too low, your lineups look artificially good against a weak field. If too high, you're modeling an unrealistically sharp field. The default of 5.0 is calibrated for typical DraftKings Showdown contests with a mix of casual and sharp players.

### What does DRO actually change in my portfolio?

DRO shifts selection toward lineups that are robust across different ownership scenarios. In practice, this means slightly less exposure to consensus plays and slightly more exposure to "contrarian" lineups that perform well even if ownership shifts. The effect is subtle: typically a few lineup swaps compared to non-DRO selection.

### Why does multi-contest mode penalize overlap?

If you enter the same 150 lineups in three contests, you're tripling your exposure to the same outcomes. The overlap penalty gently pushes lower-tier contests toward different lineups, reducing your overall variance without sacrificing much EV in any single contest.

### Can I use this for classic DFS (not Showdown)?

Not directly. The tool is built specifically for DraftKings Showdown format (1 CPT at 1.5x + 5 FLEX). Classic format has different roster construction, scoring, and salary rules. The underlying math (self-competition modeling, greedy selection, DRO) would apply, but the implementation would need significant changes.
