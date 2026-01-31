# Pre-Processing Layer for DFS Simulation Engine

Drop this folder into your simulation project. Read `INTEGRATION_GUIDE.md` first.

## Folder Structure

```
pre_processing_layer/
│
├── INTEGRATION_GUIDE.md              <-- START HERE. Full integration spec with pseudocode.
├── README.md                         <-- You are here.
│
├── unified_player_effects.json       <-- THE MAIN FILE. Load this in the sim.
│                                         747 players, 83 modifier keys, 34 tags.
│                                         Keyed by player display name.
│
├── configs/                          <-- Base configs the sim already uses (reference copies)
│   ├── correlation_config_v2.json    <-- Role-to-role correlation matrix (15 roles)
│   └── simulation_config_v3.json     <-- Distribution params, game script mods, role definitions
│
├── tag_outputs/                      <-- Individual tag source files (merged into unified)
│   ├── weighted_player_tags.json     <-- 3-tier decay: Talent/Role/Luck weighted tags
│   ├── quantified_tag_effects.json   <-- QB relationship, game script, matchup, efficiency
│   ├── scheme_environment_tags.json  <-- Formation DNA, clutch, environmental, creator
│   ├── hidden_correlations_tags.json <-- Separation, quality of target, run game, TD tags
│   ├── player_tags.json              <-- Coverage splits, script sensitivity
│   ├── context_tags.json             <-- Defensive profiles, QB pressure, beta pivots
│   └── advanced_player_tags.json     <-- 30+ archetype classifications
│
└── scripts/                          <-- Python generators (for regenerating tags from DB)
    ├── weighted_decay_utils.py       <-- Core: 3-tier weight system + coaching changes
    ├── weighted_tag_generator.py     <-- Generates weighted_player_tags.json
    ├── tag_effects_quantifier.py     <-- Generates quantified_tag_effects.json
    ├── scheme_environment_tagger.py  <-- Generates scheme_environment_tags.json
    ├── hidden_correlations_tagger.py <-- Generates hidden_correlations_tags.json
    ├── player_tag_generator.py       <-- Generates player_tags.json
    ├── context_generator.py          <-- Generates context_tags.json
    ├── advanced_archetype_tagger.py  <-- Generates advanced_player_tags.json
    └── merge_all_tags.py             <-- Merges all 7 tag files -> unified_player_effects.json
```

## Quick Start

The sim only needs to load ONE new file:

```python
import json

with open('pre_processing_layer/unified_player_effects.json') as f:
    player_effects = json.load(f)['player_effects']

# Lookup a player
if 'CeeDee Lamb' in player_effects:
    e = player_effects['CeeDee Lamb']
    print(e['tags'])                        # ['TIGHT_WINDOW_GLUE', 'TD_REGRESSION_BUY', ...]
    print(e.get('correlation_boost_qb', 0)) # 0.104
    print(e.get('cv_boost', 0))             # 0.07
```

Then follow `INTEGRATION_GUIDE.md` for the 6-step integration process:
1. Load configs (add unified JSON alongside existing configs)
2. Build base correlation matrix from roles (unchanged)
3. Apply player-specific correlation modifiers (NEW)
4. Adjust distribution parameters per player (NEW)
5. Adjust projection means per player (NEW)
6. Feed adjusted inputs into Cholesky + sampling (unchanged)

## What the Sim Currently Uses vs. What This Adds

| | Current (correlation_config_v2) | New (unified_player_effects) |
|---|---|---|
| Granularity | Role-level (15 roles) | Player-level (747 players) |
| Correlations | Static per role pair | Base + player-specific boosts |
| Distributions | Same for all players in a role | Per-player CV, shape, floor/ceiling |
| Projections | Unmodified | TD regression, matchup adjustments |
| Game context | None | Trailing/winning script, shootout, pass volume |

## Notes for the Programmer

- Players are keyed by display name (e.g., `"Ja'Marr Chase"`, `"Travis Kelce"`)
- If a player isn't in the file, use base role defaults with zero adjustments
- Correlation modifiers are ADDITIVE to the base matrix values
- After modifying the correlation matrix, you MUST validate it is positive semi-definite before Cholesky decomposition (eigenvalue repair code is in the integration guide)
- The `configs/` folder contains reference copies of the existing configs - the sim should continue loading them from wherever it currently does
- The `scripts/` folder is for regenerating tags when the database updates - the sim does not need to run these
