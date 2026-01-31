"""
Merge All Tags - Unified Pre-Processing Layer for Simulation Engine

Combines all tag systems into a single JSON file:
1. quantified_tag_effects.json - QB relationship, game script, matchup, efficiency tags
2. scheme_environment_tags.json - Formation DNA, clutch, environmental, creator tags
3. player_tags.json - Coverage splits, game script sensitivity
4. context_tags.json - Defensive profiles, QB pressure, beta pivots
5. advanced_player_tags.json - 30+ archetype classifications

Output: unified_player_effects.json - Single source of truth for sim pre-processing
"""

import json
from pathlib import Path

BASE_PATH = Path(__file__).parent.parent


def merge_all_tags():
    """Merge all tag files into unified pre-processing layer"""

    print("=" * 80)
    print("MERGING ALL TAG SYSTEMS INTO UNIFIED PRE-PROCESSING LAYER")
    print("=" * 80)

    unified = {
        "metadata": {
            "description": "Unified player effects for DFS simulation pre-processing",
            "usage": "Apply modifiers to projections, correlations, and distributions BEFORE sim engine",
            "tag_sources": []
        },
        "effect_types": {},
        "player_effects": {}
    }

    # Load and merge each tag file
    tag_files = [
        ("weighted_player_tags.json", "WEIGHTED: Talent/Role/Luck decay system"),
        ("quantified_tag_effects.json", "QB Relationship + Game Script + Matchup + Efficiency"),
        ("scheme_environment_tags.json", "Scheme + Environment + Clutch + Creator"),
        ("hidden_correlations_tags.json", "Quality of Target + Separation + Run Game + TD Tags"),
        ("player_tags.json", "Coverage Splits + Script Sensitivity + TD Variance"),
        ("context_tags.json", "Defensive Profiles + QB Pressure + Beta Pivots"),
        ("advanced_player_tags.json", "30+ Archetype Classifications")
    ]

    for filename, description in tag_files:
        filepath = BASE_PATH / filename
        if filepath.exists():
            print(f"\n Loading {filename}...")
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Track source
            unified["metadata"]["tag_sources"].append({
                "file": filename,
                "description": description
            })

            # Merge effect types
            if "effect_types" in data:
                unified["effect_types"].update(data["effect_types"])

            # Merge player effects
            if "player_effects" in data:
                for player, effects in data["player_effects"].items():
                    if player not in unified["player_effects"]:
                        unified["player_effects"][player] = {"tags": []}

                    # Accumulate tags
                    if "tags" in effects:
                        unified["player_effects"][player]["tags"].extend(effects["tags"])
                    elif "tag" in effects:
                        unified["player_effects"][player]["tags"].append(effects["tag"])

                    # Merge other effects (skip signal to reduce size)
                    for key, value in effects.items():
                        if key not in ["tags", "tag", "signal"]:
                            unified["player_effects"][player][key] = value

                print(f"   Merged {len(data.get('player_effects', {}))} players")
        else:
            print(f"   {filename} not found, skipping...")

    # Deduplicate tags per player
    for player in unified["player_effects"]:
        unified["player_effects"][player]["tags"] = list(set(
            unified["player_effects"][player]["tags"]
        ))

    # Export unified file
    output_path = BASE_PATH / "unified_player_effects.json"
    with open(output_path, 'w') as f:
        json.dump(unified, f, indent=2)

    print("\n" + "=" * 80)
    print(f"UNIFIED: {len(unified['player_effects'])} players with effects")
    print(f"EFFECT TYPES: {len(unified['effect_types'])} modifier types")
    print("=" * 80)

    # Show sample players
    print("\nSAMPLE PLAYER PROFILES:")
    print("-" * 60)

    sample_players = ["CeeDee Lamb", "Austin Ekeler", "Derrick Henry", "Travis Kelce", "Tyreek Hill"]
    for player in sample_players:
        if player in unified["player_effects"]:
            effects = unified["player_effects"][player]
            print(f"\n{player}:")
            print(f"  Tags: {effects['tags']}")

            # Show key modifiers
            key_modifiers = [
                'correlation_boost_qb', 'correlation_boost_trailing_script',
                'cv_boost', 'floor_boost_pct', 'ceiling_penalty_pct',
                'projection_boost_pct', 'projection_penalty_pct'
            ]
            for mod in key_modifiers:
                if mod in effects:
                    print(f"  {mod}: {effects[mod]}")

    print(f"\n\nExported to: {output_path}")

    # Create a quick-reference summary
    print("\n" + "=" * 80)
    print("QUICK REFERENCE: How to Apply Effects")
    print("=" * 80)

    print("""
In main.py or config loader:

```python
import json

# Load unified effects
with open('unified_player_effects.json') as f:
    effects = json.load(f)['player_effects']

# For each player in slate:
player = "Austin Ekeler"
if player in effects:
    e = effects[player]

    # 1. ADJUST CORRELATION MATRIX
    if 'correlation_boost_qb' in e:
        corr_matrix[qb_idx][player_idx] += e['correlation_boost_qb']

    if 'correlation_boost_trailing_script' in e:
        # Apply when opponent is favored (negative game script expected)
        if game_script_projection < 0:
            corr_matrix[qb_idx][player_idx] += e['correlation_boost_trailing_script']

    # 2. ADJUST DISTRIBUTION PARAMETERS
    if 'cv_boost' in e:
        player_cv += e['cv_boost']

    if 'distribution_type' in e and e['distribution_type'] == 'LogNormal':
        use_lognormal_sampling = True

    # 3. ADJUST PROJECTIONS
    projection_modifier = 1.0
    if 'projection_boost_pct' in e:
        projection_modifier += e['projection_boost_pct']
    if 'projection_penalty_pct' in e:
        projection_modifier += e['projection_penalty_pct']  # Already negative

    adjusted_projection = base_projection * projection_modifier

    # 4. ADJUST FLOOR/CEILING
    if 'floor_boost_pct' in e:
        floor_percentile *= (1 + e['floor_boost_pct'])
    if 'ceiling_penalty_pct' in e:
        ceiling_percentile *= (1 + e['ceiling_penalty_pct'])  # Already negative
```
""")

    return unified


if __name__ == "__main__":
    merge_all_tags()
