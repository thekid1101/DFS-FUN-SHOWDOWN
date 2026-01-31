# DFS Simulation Pre-Processing Layer: Integration Guide

## What This Document Is

This document tells a programmer how to integrate the **unified player effects system** (`unified_player_effects.json`) into an existing DFS Monte Carlo simulation engine. The current sim only consumes `correlation_config_v2.json` (role-to-role correlations). This new system adds **player-specific modifiers** that adjust correlations, projections, distributions, and floor/ceiling bounds on a per-player basis before simulations run.

---

## Architecture: Before vs. After

### BEFORE (Current Sim)

```
correlation_config_v2.json
         |
         v
   [ Role Lookup ]  -->  Player is "ALPHA_OUTSIDE_WR"
         |
         v
   [ Static Correlation Matrix ]  -->  QB-to-ALPHA_OUTSIDE_WR = 0.428
         |
         v
   [ Cholesky Decomposition + Sampling ]
         |
         v
   [ Lineup Optimization ]
```

Every `ALPHA_OUTSIDE_WR` gets identical treatment. CeeDee Lamb and a random WR2 who happens to classify as alpha-outside get the same QB correlation.

### AFTER (With Pre-Processing Layer)

```
correlation_config_v2.json          unified_player_effects.json
         |                                    |
         v                                    v
   [ Role Lookup ]                 [ Player-Specific Lookup ]
         |                                    |
         +------------------------------------+
         |
         v
   [ STEP 1: Base Correlation Matrix from roles ]
   [ STEP 2: Apply player-specific correlation modifiers ]
   [ STEP 3: Adjust distribution parameters (CV, shape) ]
   [ STEP 4: Adjust projection means ]
   [ STEP 5: Adjust floor/ceiling bounds ]
   [ STEP 6: Apply conditional modifiers (game script, matchup) ]
         |
         v
   [ Cholesky Decomposition + Sampling ]
         |
         v
   [ Lineup Optimization ]
```

Now CeeDee Lamb gets `correlation_boost_qb: +0.104` on top of his base role correlation, plus `cv_boost: +0.07`, `floor_boost_pct: +0.041`, `ceiling_penalty_pct: -0.098`, etc.

---

## File Reference

| File | What It Is | When To Load |
|------|-----------|--------------|
| `correlation_config_v2.json` | Base role-to-role correlations (same-team + opponent), variance decomposition | Once at startup |
| `simulation_config_v3.json` | Distribution parameters per role (type, skew, kurtosis, CV, floor/ceiling), game script modifiers | Once at startup |
| `unified_player_effects.json` | Player-specific modifiers (747 players, 83 modifier keys, 34 tags) | Once at startup, then lookup per player per slate |

---

## unified_player_effects.json: Format Specification

### Top-Level Structure

```json
{
  "metadata": {
    "description": "...",
    "usage": "Apply modifiers to projections, correlations, and distributions BEFORE sim engine",
    "tag_sources": [
      {"file": "weighted_player_tags.json", "description": "WEIGHTED: Talent/Role/Luck decay system"},
      {"file": "quantified_tag_effects.json", "description": "QB Relationship + Game Script + Matchup + Efficiency"},
      {"file": "scheme_environment_tags.json", "description": "Scheme + Environment + Clutch + Creator"},
      {"file": "hidden_correlations_tags.json", "description": "Quality of Target + Separation + Run Game + TD Tags"},
      {"file": "player_tags.json", "description": "Coverage Splits + Script Sensitivity + TD Variance"},
      {"file": "context_tags.json", "description": "Defensive Profiles + QB Pressure + Beta Pivots"},
      {"file": "advanced_player_tags.json", "description": "30+ Archetype Classifications"}
    ]
  },
  "effect_types": { ... },       // Human-readable descriptions of each modifier key
  "player_effects": {            // THE MAIN DATA - keyed by player display name
    "CeeDee Lamb": { ... },
    "Saquon Barkley": { ... },
    ...
  }
}
```

### Per-Player Object

Every player has some subset of these fields (not all players have all fields):

```json
{
  "tags": ["TAG_NAME_1", "TAG_NAME_2", ...],   // List of string labels
  "position": "WR",                              // WR, RB, TE (no QBs)
  "team": "DAL",                                 // Current team abbreviation

  // --- CORRELATION MODIFIERS (add to base matrix values) ---
  "correlation_boost_qb": 0.104,                 // Always-on: add to QB<->player cell
  "correlation_boost_opp_qb": 0.15,              // Add to opposing QB<->player cell (game stacks)
  "correlation_boost_dst": 0.08,                 // Add to DST<->player cell
  "correlation_reduction_qb": -0.062,            // Subtract from QB dependency (QB-proof players)
  "correlation_boost_qb_volume": 0.03,           // Add when QB has high pass volume expected
  "correlation_boost_trailing_script": 0.165,    // Add when team is expected to trail (negative spread)
  "correlation_boost_winning_script": 0.12,      // Add when team is expected to lead (positive spread)
  "correlation_boost_shootout": 0.097,           // Add when game total is high (50+)
  "correlation_boost_trailing": 0.195,           // General trailing boost (overlap with trailing_script)
  "correlation_boost_rb1": 0.09,                 // Unusual WR<->RB1 correlation (bring-back stacks)
  "correlation_penalty_glrb": -0.036,            // Penalty for goal-line RB cannibalization
  "correlation_boost_pass_volume": 0.075,        // Boost in high-pass-volume projections
  "correlation_boost_leading_script": 0.06,      // Add when team leading (rare, for front-runners)

  // --- DISTRIBUTION MODIFIERS ---
  "cv_boost": 0.07,                // Add to base CV from simulation_config_v3.json
  "distribution_type": "LogNormal", // Override the role's default distribution type
  "base_cv": 0.526,                // Observed CV (informational, use cv_boost to adjust)
  "fp_cv": 1.234,                  // Raw fantasy point CV (informational)

  // --- PROJECTION MEAN MODIFIERS ---
  "projection_adjustment": 0.039,    // General adjustment factor (TD regression)
  "projection_boost_pct": 0.025,     // Multiply projection by (1 + value)
  "projection_penalty_pct": -0.03,   // Multiply projection by (1 + value), already negative
  "projection_boost_vs_man": 0.05,   // Conditional: boost when facing man coverage
  "projection_penalty_vs_man": -0.03,// Conditional: penalty when facing man coverage
  "projection_boost_vs_zone": 0.04,  // Conditional: boost when facing zone coverage
  "projection_penalty_vs_zone": -0.02,// Conditional: penalty when facing zone coverage

  // --- FLOOR/CEILING MODIFIERS ---
  "floor_boost_pct": 0.041,          // Raise floor percentile by this fraction
  "floor_penalty_pct": -0.041,       // Lower floor percentile (boom-or-bust players)
  "floor_protection": 0.753,         // Score 0-1: how protected is the floor (informational)
  "floor_penalty_vs_elite_dl": -0.035, // Conditional: floor drops vs elite d-line
  "ceiling_boost_pct": 0.15,         // Raise ceiling percentile
  "ceiling_penalty_pct": -0.098,     // Lower ceiling percentile

  // --- TD-SPECIFIC MODIFIERS ---
  "td_variance_boost": 0.2,         // Increase TD Poisson lambda
  "td_luck_factor": 0.35,           // actual/expected ratio (< 1.0 = unlucky)
  "actual_tds": 9,                   // Raw TD count
  "expected_tds": 26.0,              // Expected TDs based on volume
  "vulture_susceptibility": 0.056,   // Risk of GL carries going elsewhere
  "td_reliability_penalty": -0.047,  // Reduce TD projection confidence

  // --- CONTEXT FLAGS ---
  "coaching_change_2025": true,      // Team has new HC/OC - ROLE tags use only 2025 data
  "matchup_sensitivity": 0.087,      // How much output varies by opponent

  // --- INFORMATIONAL (diagnostic, not directly applied) ---
  "target_share": 28.324,
  "snap_share": 78.424,
  "consistency": 0.673,
  "contested_rate": 0.0507,
  "adot": 11.5,
  "boom_ratio": 14.0,
  "creator_score": 0.469,
  "avg_separation": 1.99,
  "yc_per_touch": 1.19,
  "third_down_target_rate": 0.284,
  "end_zone_target_rate": 0.125,
  // ... and more
}
```

---

## Integration Steps: Detailed Pseudocode

### Step 0: Load All Configs

```python
import json
import numpy as np

# Existing config (unchanged)
with open('correlation_config_v2.json') as f:
    corr_config = json.load(f)

with open('simulation_config_v3.json') as f:
    sim_config = json.load(f)

# NEW: Load player effects
with open('unified_player_effects.json') as f:
    player_effects = json.load(f)['player_effects']
```

### Step 1: Build Base Correlation Matrix (Unchanged)

This step is identical to current behavior. For each player on the slate, look up their role and build the NxN correlation matrix from `correlation_config_v2.json`.

```python
# existing_logic: for each pair of same-team players
base_corr = corr_config['same_team_correlations'][role_a][role_b]
# existing_logic: for each pair of opponent players
opp_corr = corr_config['opponent_correlations'][role_a][role_b]
```

### Step 2: Apply Player-Specific Correlation Modifiers (NEW)

After building the base matrix, overlay player-specific adjustments.

```python
def apply_correlation_modifiers(corr_matrix, slate_players, player_effects, game_context):
    """
    Modify correlation matrix with player-specific effects.

    Args:
        corr_matrix: NxN numpy array (base correlations from role lookup)
        slate_players: list of dicts with keys: name, index, team, position, role, qb_index, dst_index
        player_effects: the player_effects dict from unified JSON
        game_context: dict per team with keys: spread, total, coverage_scheme
    """
    for player in slate_players:
        name = player['name']
        if name not in player_effects:
            continue

        e = player_effects[name]
        i = player['index']          # This player's row/col in the matrix
        qb_i = player['qb_index']    # Their team's QB index in the matrix

        # -------------------------------------------------------
        # ALWAYS-ON CORRELATION MODIFIERS
        # -------------------------------------------------------

        # QB<->Player correlation boost
        if 'correlation_boost_qb' in e and qb_i is not None:
            corr_matrix[qb_i][i] += e['correlation_boost_qb']
            corr_matrix[i][qb_i] += e['correlation_boost_qb']

        # QB-proof reduction (player produces independently of QB)
        if 'correlation_reduction_qb' in e and qb_i is not None:
            corr_matrix[qb_i][i] += e['correlation_reduction_qb']  # Value is already negative
            corr_matrix[i][qb_i] += e['correlation_reduction_qb']

        # Opposing QB correlation (for game stacks / bring-backs)
        if 'correlation_boost_opp_qb' in e:
            opp_qb_i = player.get('opp_qb_index')
            if opp_qb_i is not None:
                corr_matrix[opp_qb_i][i] += e['correlation_boost_opp_qb']
                corr_matrix[i][opp_qb_i] += e['correlation_boost_opp_qb']

        # DST correlation
        if 'correlation_boost_dst' in e:
            dst_i = player.get('opp_dst_index')
            if dst_i is not None:
                corr_matrix[dst_i][i] += e['correlation_boost_dst']
                corr_matrix[i][dst_i] += e['correlation_boost_dst']

        # Goal-line RB cannibalization
        if 'correlation_penalty_glrb' in e:
            # Apply to correlation with team's other RBs
            for other in slate_players:
                if other['team'] == player['team'] and other['position'] == 'RB' and other['index'] != i:
                    corr_matrix[i][other['index']] += e['correlation_penalty_glrb']
                    corr_matrix[other['index']][i] += e['correlation_penalty_glrb']

        # Unusual WR<->RB1 correlation (e.g., Tyreek + team RB in screen-heavy offense)
        if 'correlation_boost_rb1' in e:
            for other in slate_players:
                if other['team'] == player['team'] and other['position'] == 'RB':
                    corr_matrix[i][other['index']] += e['correlation_boost_rb1']
                    corr_matrix[other['index']][i] += e['correlation_boost_rb1']
                    break  # Only apply to RB1 (first RB found on team)

        # -------------------------------------------------------
        # CONDITIONAL CORRELATION MODIFIERS (require game context)
        # -------------------------------------------------------
        team = player['team']
        ctx = game_context.get(team, {})
        spread = ctx.get('spread', 0)       # Negative = underdog
        total = ctx.get('total', 45)        # Game total

        # Trailing script boost (team is underdog)
        if 'correlation_boost_trailing_script' in e and spread < -3:
            corr_matrix[qb_i][i] += e['correlation_boost_trailing_script']
            corr_matrix[i][qb_i] += e['correlation_boost_trailing_script']

        # Alternative trailing key (same concept, different source)
        if 'correlation_boost_trailing' in e and spread < -3:
            if qb_i is not None:
                corr_matrix[qb_i][i] += e['correlation_boost_trailing']
                corr_matrix[i][qb_i] += e['correlation_boost_trailing']

        # Winning script boost (team is favorite)
        if 'correlation_boost_winning_script' in e and spread > 3:
            if qb_i is not None:
                corr_matrix[qb_i][i] += e['correlation_boost_winning_script']
                corr_matrix[i][qb_i] += e['correlation_boost_winning_script']

        # Shootout boost (high game total)
        if 'correlation_boost_shootout' in e and total >= 50:
            if qb_i is not None:
                corr_matrix[qb_i][i] += e['correlation_boost_shootout']
                corr_matrix[i][qb_i] += e['correlation_boost_shootout']

        # High pass volume boost
        if 'correlation_boost_qb_volume' in e:
            # Apply when team implied total suggests heavy passing
            team_implied = ctx.get('team_total', 24)
            if team_implied >= 27:
                if qb_i is not None:
                    corr_matrix[qb_i][i] += e['correlation_boost_qb_volume']
                    corr_matrix[i][qb_i] += e['correlation_boost_qb_volume']

    # -------------------------------------------------------
    # CLAMP: Ensure matrix stays valid for Cholesky
    # -------------------------------------------------------
    np.fill_diagonal(corr_matrix, 1.0)
    corr_matrix = np.clip(corr_matrix, -0.95, 0.95)
    np.fill_diagonal(corr_matrix, 1.0)

    # Force symmetry
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    np.fill_diagonal(corr_matrix, 1.0)

    # Ensure positive semi-definite (required for Cholesky)
    # If eigenvalue repair is needed:
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
    if np.any(eigenvalues < 0):
        eigenvalues = np.maximum(eigenvalues, 1e-6)
        corr_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        # Re-normalize to correlation matrix
        d = np.sqrt(np.diag(corr_matrix))
        corr_matrix = corr_matrix / np.outer(d, d)
        np.fill_diagonal(corr_matrix, 1.0)

    return corr_matrix
```

### Step 3: Adjust Distribution Parameters (NEW)

Before sampling, modify each player's distribution shape.

```python
def get_player_distribution(player_name, player_role, sim_config, player_effects):
    """
    Get adjusted distribution parameters for a player.

    Returns dict: {type, cv, skew, kurtosis, floor_pct, ceiling_pct}
    """
    # Start with role defaults from simulation_config_v3.json
    role_params = sim_config['distribution_parameters'].get(player_role, {})

    dist = {
        'type': role_params.get('type', 'Normal'),
        'cv': role_params.get('cv_default', 0.35),
        'skew': role_params.get('skew', 0.0),
        'kurtosis': role_params.get('kurtosis', 0.0),
        'floor_pct': role_params.get('floor_pct', 0.20),
        'ceiling_pct': role_params.get('ceiling_pct', 2.50),
    }

    if player_name not in player_effects:
        return dist

    e = player_effects[player_name]

    # Override distribution type if specified
    if 'distribution_type' in e:
        dist['type'] = e['distribution_type']

    # Adjust CV
    if 'cv_boost' in e:
        dist['cv'] += e['cv_boost']

    # Adjust floor
    if 'floor_boost_pct' in e:
        dist['floor_pct'] *= (1 + e['floor_boost_pct'])
    if 'floor_penalty_pct' in e:
        dist['floor_pct'] *= (1 + e['floor_penalty_pct'])  # Value is negative

    # Adjust ceiling
    if 'ceiling_boost_pct' in e:
        dist['ceiling_pct'] *= (1 + e['ceiling_boost_pct'])
    if 'ceiling_penalty_pct' in e:
        dist['ceiling_pct'] *= (1 + e['ceiling_penalty_pct'])  # Value is negative

    return dist
```

### Step 4: Adjust Projection Means (NEW)

```python
def adjust_projection(base_projection, player_name, player_effects, game_context=None):
    """
    Apply player-specific projection modifiers.

    Args:
        base_projection: Float, the raw projection (e.g., 15.5 FP)
        player_name: String
        player_effects: The unified dict
        game_context: Optional dict with: coverage_scheme, opponent_dl_rank, spread

    Returns:
        Float: adjusted projection
    """
    if player_name not in player_effects:
        return base_projection

    e = player_effects[player_name]
    modifier = 1.0

    # General projection adjustments
    if 'projection_boost_pct' in e:
        modifier += e['projection_boost_pct']
    if 'projection_penalty_pct' in e:
        modifier += e['projection_penalty_pct']  # Already negative

    # TD regression adjustment (if not already baked into projections)
    # projection_adjustment is a multiplier for expected TD correction
    # Only apply if your projection source does NOT already account for TD regression
    if 'projection_adjustment' in e:
        modifier += e['projection_adjustment']

    # Conditional: coverage scheme matchup
    if game_context:
        scheme = game_context.get('coverage_scheme', 'unknown')
        if scheme == 'man':
            if 'projection_boost_vs_man' in e:
                modifier += e['projection_boost_vs_man']
            if 'projection_penalty_vs_man' in e:
                modifier += e['projection_penalty_vs_man']
        elif scheme == 'zone':
            if 'projection_boost_vs_zone' in e:
                modifier += e['projection_boost_vs_zone']
            if 'projection_penalty_vs_zone' in e:
                modifier += e['projection_penalty_vs_zone']

    return base_projection * modifier
```

### Step 5: Conditional Floor Penalty (Matchup-Specific)

```python
def apply_matchup_floor_penalty(dist_params, player_name, player_effects, opponent_info):
    """
    Apply conditional floor/ceiling adjustments based on matchup.

    Args:
        dist_params: dict from get_player_distribution()
        player_name: String
        opponent_info: dict with keys: dl_rank (1-32), coverage_type
    """
    if player_name not in player_effects:
        return dist_params

    e = player_effects[player_name]

    # Floor penalty vs elite defensive lines (top-8 ranked)
    if 'floor_penalty_vs_elite_dl' in e:
        dl_rank = opponent_info.get('dl_rank', 16)
        if dl_rank <= 8:
            dist_params['floor_pct'] *= (1 + e['floor_penalty_vs_elite_dl'])

    return dist_params
```

---

## Complete Integration Flow (Putting It All Together)

```python
def run_simulation(slate_players, game_contexts, n_sims=10000):
    """
    Full simulation flow with pre-processing layer.
    """
    # =============================================
    # LOAD CONFIGS (do once, cache globally)
    # =============================================
    corr_config = load_json('correlation_config_v2.json')
    sim_config = load_json('simulation_config_v3.json')
    effects = load_json('unified_player_effects.json')['player_effects']

    n_players = len(slate_players)

    # =============================================
    # STEP 1: BASE CORRELATION MATRIX (existing logic)
    # =============================================
    corr_matrix = np.eye(n_players)
    for i, p_a in enumerate(slate_players):
        for j, p_b in enumerate(slate_players):
            if i >= j:
                continue
            if p_a['team'] == p_b['team']:
                # Same team: lookup from corr_config
                corr_matrix[i][j] = corr_config['same_team_correlations'] \
                    .get(p_a['role'], {}).get(p_b['role'], 0.0)
            elif are_opponents(p_a, p_b):
                # Opponents: lookup from opponent correlations
                corr_matrix[i][j] = corr_config['opponent_correlations'] \
                    .get(p_a['role'], {}).get(p_b['role'], 0.0)
            corr_matrix[j][i] = corr_matrix[i][j]  # Symmetric

    # =============================================
    # STEP 2: PLAYER-SPECIFIC CORRELATION MODS (NEW)
    # =============================================
    corr_matrix = apply_correlation_modifiers(
        corr_matrix, slate_players, effects, game_contexts
    )

    # =============================================
    # STEP 3: PER-PLAYER DISTRIBUTIONS (NEW)
    # =============================================
    player_dists = []
    adjusted_projections = []

    for player in slate_players:
        name = player['name']
        role = player['role']
        team = player['team']
        ctx = game_contexts.get(team, {})

        # Get adjusted distribution
        dist = get_player_distribution(name, role, sim_config, effects)

        # Apply matchup-specific floor penalties
        opp_info = ctx.get('opponent_info', {})
        dist = apply_matchup_floor_penalty(dist, name, effects, opp_info)

        player_dists.append(dist)

        # Get adjusted projection
        base_proj = player['projection']
        adj_proj = adjust_projection(base_proj, name, effects, ctx)
        adjusted_projections.append(adj_proj)

    # =============================================
    # STEP 4: CHOLESKY + SAMPLING (existing logic, uses adjusted inputs)
    # =============================================
    L = np.linalg.cholesky(corr_matrix)

    results = np.zeros((n_sims, n_players))
    for sim in range(n_sims):
        z = np.random.standard_normal(n_players)
        correlated_z = L @ z

        for p_idx in range(n_players):
            proj = adjusted_projections[p_idx]
            dist = player_dists[p_idx]
            std = proj * dist['cv']

            # Sample based on distribution type
            if dist['type'] == 'Normal':
                raw = proj + correlated_z[p_idx] * std
            elif dist['type'] == 'LogNormal':
                sigma2 = np.log(1 + (std / proj) ** 2)
                mu = np.log(proj) - sigma2 / 2
                raw = np.exp(mu + np.sqrt(sigma2) * correlated_z[p_idx])
            elif dist['type'] == 'Gamma':
                # Map correlated normal to gamma via CDF inversion
                from scipy.stats import norm, gamma as gamma_dist
                u = norm.cdf(correlated_z[p_idx])
                shape = (proj / std) ** 2
                scale = std ** 2 / proj
                raw = gamma_dist.ppf(u, a=shape, scale=scale)
            else:
                raw = proj + correlated_z[p_idx] * std

            # Apply floor/ceiling bounds
            floor = proj * dist['floor_pct']
            ceiling = proj * dist['ceiling_pct']
            results[sim, p_idx] = np.clip(raw, floor, ceiling)

    return results
```

---

## Modifier Reference: What Each Key Means and When To Apply

### ALWAYS-ON Correlation Modifiers

These are applied to every simulation regardless of game context.

| Key | Type | Range | What It Does | Apply To |
|-----|------|-------|-------------|----------|
| `correlation_boost_qb` | float | 0 to +0.30 | Strengthens QB<->Player co-movement | `corr_matrix[qb][player]` both directions |
| `correlation_reduction_qb` | float | -0.10 to 0 | Weakens QB dependency (QB-proof) | `corr_matrix[qb][player]` both directions |
| `correlation_boost_opp_qb` | float | 0 to +0.15 | Strengthens bring-back / game stack | `corr_matrix[opp_qb][player]` both directions |
| `correlation_boost_dst` | float | 0 to +0.10 | Player correlates with opposing DST | `corr_matrix[opp_dst][player]` both directions |
| `correlation_penalty_glrb` | float | -0.05 to 0 | Goal-line RB cannibalizes teammates | `corr_matrix[player][same_team_RBs]` |
| `correlation_boost_rb1` | float | 0 to +0.10 | Unusual WR<->RB1 correlation | `corr_matrix[player][team_rb1]` |

### CONDITIONAL Correlation Modifiers

These require game context (Vegas lines, implied totals) to activate.

| Key | Type | Trigger Condition | What It Does |
|-----|------|-------------------|-------------|
| `correlation_boost_trailing_script` | float | Team spread < -3 | Boosts QB correlation when chasing |
| `correlation_boost_trailing` | float | Team spread < -3 | Same concept (from different tag source) |
| `correlation_boost_winning_script` | float | Team spread > +3 | Boosts correlation in positive game script |
| `correlation_boost_shootout` | float | Game total >= 50 | Boosts correlation in shootout games |
| `correlation_boost_qb_volume` | float | Team implied total >= 27 | Boosts correlation when high pass volume expected |
| `correlation_boost_pass_volume` | float | Team pass rate > 60% | Similar to qb_volume |
| `correlation_boost_leading_script` | float | Team spread > +3 | Rare, for run-first front-runners |

### Distribution Modifiers

| Key | Type | What It Does |
|-----|------|-------------|
| `cv_boost` | float (0 to +0.10) | Add to the role's default CV from `simulation_config_v3.json`. Higher CV = wider outcomes. |
| `distribution_type` | string | Override distribution shape. Values: `"Normal"`, `"LogNormal"`, `"Gamma"`, `"Poisson_Hybrid"`. LogNormal = right-skewed boom potential. |
| `base_cv` | float | **Informational only.** The player's observed historical CV. Do NOT apply directly. |
| `fp_cv` | float | **Informational only.** Raw fantasy point CV. |

### Projection Modifiers

| Key | Type | Condition | What It Does |
|-----|------|-----------|-------------|
| `projection_adjustment` | float | Always | TD regression correction. Apply ONLY if your projection source doesn't already adjust for expected TDs. |
| `projection_boost_pct` | float | Always | Multiply projection by `(1 + value)` |
| `projection_penalty_pct` | float (negative) | Always | Multiply projection by `(1 + value)` |
| `projection_boost_vs_man` | float | Opponent plays man coverage | Multiply projection by `(1 + value)` |
| `projection_penalty_vs_man` | float (negative) | Opponent plays man coverage | Multiply projection by `(1 + value)` |
| `projection_boost_vs_zone` | float | Opponent plays zone coverage | Multiply projection by `(1 + value)` |
| `projection_penalty_vs_zone` | float (negative) | Opponent plays zone coverage | Multiply projection by `(1 + value)` |

### Floor/Ceiling Modifiers

| Key | Type | Condition | What It Does |
|-----|------|-----------|-------------|
| `floor_boost_pct` | float (positive) | Always | Raise floor: `floor *= (1 + value)` |
| `floor_penalty_pct` | float (negative) | Always | Lower floor: `floor *= (1 + value)` |
| `floor_penalty_vs_elite_dl` | float (negative) | Opponent DL rank top-8 | Lower floor in tough matchups |
| `floor_protection` | float 0-1 | **Informational** | How safe the floor is. Higher = safer. Do NOT apply directly. |
| `ceiling_boost_pct` | float (positive) | Always | Raise ceiling: `ceiling *= (1 + value)` |
| `ceiling_penalty_pct` | float (negative) | Always | Lower ceiling: `ceiling *= (1 + value)` |

### TD-Specific Modifiers

| Key | Type | What It Does |
|-----|------|-------------|
| `td_variance_boost` | float | Increase Poisson lambda for TD sampling. If sim models TDs separately, add this to expected TD rate. |
| `td_luck_factor` | float | Ratio of actual/expected TDs. Values < 1.0 = unlucky, buy candidate. |
| `actual_tds` / `expected_tds` | int/float | Raw data. `expected_tds` is volume-based expectation. |
| `vulture_susceptibility` | float | Risk that goal-line TDs go to another player. Reduce TD expectation. |
| `td_reliability_penalty` | float (negative) | Player is inefficient near goal line. Reduce TD expectation. |

### Context Flags

| Key | Type | What It Does |
|-----|------|-------------|
| `coaching_change_2025` | bool | This player's team has a new HC or OC. ROLE-based tags only use post-change data. **No direct sim action needed** - the tags already account for this. But useful for flagging uncertainty. |
| `matchup_sensitivity` | float 0-1 | How much this player's output depends on matchup. Higher = more volatile vs. tough opponents. Can be used to scale matchup-based adjustments. |

### Informational Fields (Do NOT Apply Directly)

These exist for diagnostics, reporting, and manual review. The sim should ignore them:

`target_share`, `snap_share`, `consistency`, `contested_rate`, `adot`, `boom_ratio`, `creator_score`, `avg_separation`, `yc_per_touch`, `third_down_target_rate`, `end_zone_target_rate`, `avg_ez_targets`, `bring_back_score`, `first_down_rate`, `fourth_down_rate`, `receiving_share`, `shotgun_rate`, `shotgun_snap_rate`, `slot_rate`, `pass_snap_rate`, `workload_score`, `yac_score`, `yards_created_per_touch`, `burn_rate`, `burn_score`, `catch_vs_catchable_diff`, `drop_rate`, `man_yards_per_target`, `zone_yards_per_target`, `yprr`, `ypc`, `ypc_vs_box`, `blocked_yards_rate`, `light_box_rate`, `avg_box_count`, `pa_snap_rate`, `gl_efficiency`, `gl_opps_per_game`, `rz_opps_per_game`, `yards_per_td`, `danger_play_rate`, `fragility_score`

---

## Player Lookup: Name Matching

Players in `unified_player_effects.json` are keyed by **display name** (e.g., `"Ja'Marr Chase"`, `"Travis Kelce"`, `"Saquon Barkley"`). These must match whatever name format your projection source uses.

**Coverage**: 747 players (347 WR, 212 RB, 172 TE). No QBs are in the effects file - QBs are only referenced indirectly through correlation modifiers on skill players.

If a player is NOT in the effects file, use the base role parameters from `correlation_config_v2.json` and `simulation_config_v3.json` with zero adjustments. Missing = no special treatment.

---

## Tag List (34 Tags)

Tags are string labels attached to each player. They are **descriptive only** in the unified file - the numerical effects are already broken out into the modifier keys above. Tags can be useful for:
- Filtering/grouping players in analysis
- Display in lineup optimizer UI
- Manual review of why a modifier exists

| Tag | Category | Count | Meaning |
|-----|----------|-------|---------|
| THE_FIRST_READ | ROLE | 489 | High target share, QB's primary option |
| CONTESTED_CATCH_SAVANT | TALENT | 409 | Elevated contested catch rate |
| TD_REGRESSION_BUY | LUCK | 329 | Scored fewer TDs than volume suggests, positive regression expected |
| GARBAGE_TIME_HERO | LUCK | 346 | Elevated production in garbage time / negative game scripts |
| THE_BAILOUT_OPTION | ROLE | 345 | Targeted heavily on 3rd down / pressure situations |
| YARDS_CREATED_GOD | TALENT | 331 | Elite yards created per touch |
| CHAIN_MOVER | ROLE | 223 | High first-down conversion rate |
| END_ZONE_TARGET_HOG | LUCK | 162 | Elevated end-zone target rate |
| DROP_MACHINE | TALENT | 143 | High drop rate (negative) |
| DEEP_BALL_DEPENDENT | ROLE | 142 | High aDOT, needs deep shots to produce |
| TIGHT_WINDOW_GLUE | TALENT | 127 | Catches in tight windows, low separation |
| THE_PANIC_BUTTON | ROLE | 125 | RB/TE safety valve for QB under pressure |
| SHOTGUN_SATELLITE | ROLE | 102 | Elevated usage in shotgun formations |
| LIGHT_BOX_SLASHER | ROLE | 97 | Exploits light defensive boxes |
| LINE_DEPENDENT_GRINDER | ROLE | 90 | Production depends heavily on O-line |
| 4TH_QUARTER_CLOSER | LUCK | 83 | Elevated 4th quarter production |
| BELLCOW_RECEIVING | ROLE | 71 | High-usage RB with receiving role |
| ZONE_MERCHANT | TALENT | 69 | Dominates zone coverage, struggles in man |
| BELLCOW_EARLY | ROLE | 61 | High-usage early-down RB |
| TD_REGRESSION_CANDIDATE | LUCK | 55 | Scored more TDs than expected, negative regression |
| YAC_MONSTER | TALENT | 46 | Elite yards after catch |
| MAN_BEATER | TALENT | 39 | Dominates man coverage, struggles in zone |
| BURN_ARTIST | TALENT | 31 | Creates separation through route running |
| THE_FRONT_RUNNER | LUCK | ~30 | Elevated production when team is winning |
| PLAY_ACTION_MERCHANT | ROLE | ~25 | Usage spikes in play-action |
| BAD_BALL_ERASER | TALENT | ~25 | Catches poorly thrown balls |
| VULTURE_VICTIM | LUCK | ~23 | Loses goal-line work to another player |
| GL_STUFF_RISK | LUCK | ~23 | Inefficient at goal line despite volume |
| SLOT_SAFETY_VALVE | ROLE | ~20 | Slot receiver, short targets, safety valve |
| CONTACT_ABSORBER | TALENT | ~15 | Maintains production through contact |
| ROUTE_WINNER | TALENT | ~10 | Elite route win rate |
| HYPER_EFFICIENT_GHOST | LUCK | ~5 | Unsustainably high efficiency |
| EMPTY_CALORIES | LUCK | ~5 | Volume without efficiency |
| DANGER_PLAY_MAGNET | TALENT | ~2 | Involved in high-leverage plays |

---

## Coaching Change Flag

206 players (across 17+ teams) have `"coaching_change_2025": true`. This means:

- Their ROLE-based tags (target share, usage patterns, scheme fit) only use 2025 data
- Their TALENT-based tags (route running, contested catches, YAC) still use full 2020-2025 history
- Their LUCK-based tags (TD rate, efficiency) only use 2024-2025 data

**Teams affected**: ARI, ATL, BAL, BUF, CHI, CLE, DAL, DET, HOU, JAX, LAC, LV, MIA, NO, NYG, NYJ, PHI, PIT, TEN

The numerical effects in the file already account for this weighting. No additional sim logic is needed. The flag exists for transparency and UI display.

---

## Overlap Between correlation_boost_trailing and correlation_boost_trailing_script

These two keys represent the same concept (boosted correlation when team is trailing) but come from different tag sources. **Do NOT double-apply.** Use this logic:

```python
trailing_boost = max(
    e.get('correlation_boost_trailing_script', 0),
    e.get('correlation_boost_trailing', 0)
)
# Apply trailing_boost once, not both
```

---

## Priority Order: What Matters Most

If implementing incrementally, apply in this order of impact:

1. **`correlation_boost_qb`** (506 players) - The single most impactful modifier. Differentiates "Tyreek Hill + Tua" from "generic WR + QB".
2. **`correlation_boost_trailing_script`** / **`correlation_boost_trailing`** - Critical for underdog stacks.
3. **`cv_boost`** + **`distribution_type`** - Changes outcome shape. LogNormal players have true boom potential.
4. **`floor_boost_pct`** / **`ceiling_penalty_pct`** - Adjusts the range of sampled outcomes.
5. **`projection_adjustment`** - TD regression corrections.
6. **`correlation_boost_opp_qb`** - Enables realistic game stacks.
7. **`correlation_boost_shootout`** - Important for high-total slates.
8. Everything else (matchup-specific, DST, etc.) - Lower priority, conditional.

---

## Example: Full Player Walkthrough

### CeeDee Lamb (DAL, WR)

**Base role**: ALPHA_OUTSIDE_WR
- Base QB correlation from config v2: `0.428` (POCKET_QB to ALPHA_OUTSIDE_WR)

**Player-specific adjustments**:
- `correlation_boost_qb: +0.104` -> Adjusted QB corr: `0.428 + 0.104 = 0.532`
- `correlation_reduction_qb: -0.062` -> Not present for CeeDee (he IS QB-dependent)
- `cv_boost: +0.07` -> Base CV `0.40` + `0.07` = `0.47` (wider outcomes)
- `floor_boost_pct: +0.041` -> Floor raised 4.1% (high target share = safe floor)
- `ceiling_penalty_pct: -0.098` -> Ceiling lowered 9.8% (DAL coaching change uncertainty)
- `floor_penalty_pct: -0.041` -> Floor also penalized (drop rate concerns)
- `projection_adjustment: +0.039` -> TD regression buy (+3.9% to projection)
- `coaching_change_2025: true` -> New coaching staff, ROLE data limited to 2025 only

**If DAL is trailing (spread < -3)**:
- `correlation_boost_winning_script: +0.12` -> Does NOT activate (team trailing)
- No trailing_script modifier for CeeDee

**If DAL is in a shootout (total >= 50)**:
- No shootout modifier for CeeDee

**Net effect**: CeeDee gets higher QB correlation than a generic alpha WR, wider variance, raised floor from volume, but capped ceiling due to coaching change uncertainty, and a TD regression buy on his projection.

### Saquon Barkley (PHI, RB)

**Base role**: BELLCOW_EARLY_RB
- Base QB correlation from config v2: `0.150` (POCKET_QB to BELLCOW_EARLY_RB)

**Player-specific adjustments**:
- `correlation_boost_qb: +0.300` -> Adjusted QB corr: `0.150 + 0.300 = 0.450` (massive - he IS the PHI offense)
- `correlation_reduction_qb: -0.088` -> Net QB corr: `0.450 - 0.088 = 0.362`
- `correlation_boost_trailing_script: +0.165` -> If PHI trailing, QB corr becomes `0.362 + 0.165 = 0.527`
- `correlation_boost_shootout: +0.097` -> If high total, QB corr: `+0.097` more
- `correlation_boost_dst: +0.065` -> Correlates with opposing DST (game script)
- `correlation_penalty_glrb: -0.036` -> Slight penalty vs. other PHI RBs
- `floor_boost_pct: +0.059` -> Safe floor from volume
- `ceiling_penalty_pct: -0.120` -> Capped ceiling (TD dependent, efficiency questions)
- `matchup_sensitivity: 0.087` -> Output varies by opponent

---

## What NOT To Do

1. **Do NOT apply `base_cv` or `fp_cv` directly.** These are observed values for reference. Use `cv_boost` to adjust the role default.
2. **Do NOT apply `floor_protection` or `consistency` directly.** These are diagnostics. The actual adjustments are in `floor_boost_pct`.
3. **Do NOT double-apply trailing/trailing_script.** Take the max of the two, apply once.
4. **Do NOT apply `projection_adjustment` if your projections already account for TD regression.** This field corrects for the gap between actual TDs and expected TDs. If your projection source (e.g., FantasyPros, 4for4) already does this, skip it.
5. **Do NOT treat missing keys as zero.** If a player doesn't have `cv_boost`, it means no adjustment needed - use the role default as-is. Use `.get(key, 0)` for additive modifiers and `.get(key, None)` with a guard for multiplicative ones.
6. **Do NOT forget to re-validate the correlation matrix after modifications.** Player-specific boosts can push the matrix out of PSD bounds. Always run eigenvalue repair before Cholesky decomposition.
