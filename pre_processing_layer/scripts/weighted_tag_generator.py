"""
Weighted Tag Generator - Applies 3-Tier Decay System

Uses weighted decay to balance recency vs sample size:
- TALENT tags: 40% 2025, 30% 2024, 20% 2023, 10% older
- ROLE tags: 60% 2025, 30% 2024, 10% 2023 (or 100% 2025 if new coach)
- LUCK tags: 80% 2025, 20% 2024, 0% older

Outputs weighted_player_tags.json with confidence-adjusted scores.
"""

import sqlite3
import pandas as pd
import numpy as np
import json
from pathlib import Path
from weighted_decay_utils import (
    add_season_weights, calculate_weighted_multi_stats,
    get_tag_type, apply_confidence_adjustment, CURRENT_SEASON,
    COACHING_CHANGES_2025
)

DB_PATH = Path(__file__).parent.parent / "nfl_research.db"

def get_connection():
    return sqlite3.connect(DB_PATH)


# =============================================================================
# TALENT TAGS (High Stability - 6 year lookback)
# =============================================================================

def tag_talent_separation():
    """
    TALENT-based tags: Routes won, burns, contested catches
    Uses 40/30/20/10 weighting
    """
    conn = get_connection()

    query = """
    SELECT
        g.player_id, g.name, g.position, g.team, g.season,
        g.burns, g.routes_run, g.contested_catches, g.targets,
        g.routes_won, g.separation_at_target, g.fantasy_points
    FROM gamelog g
    WHERE g.position IN ('WR', 'TE')
        AND g.season >= 2020
        AND g.targets > 0
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Add weights for TALENT tier
    df = add_season_weights(df, 'TALENT')

    # Calculate per-game metrics
    df['burn_rate'] = df['burns'].fillna(0) / df['routes_run'].replace(0, np.nan)
    df['contested_rate'] = df['contested_catches'].fillna(0) / df['targets'].replace(0, np.nan)

    # Weighted aggregation
    results = []
    for (pid, name, pos), group in df.groupby(['player_id', 'name', 'position']):
        if group['weight'].sum() < 0.3:  # Need minimum weight
            continue

        # Get most recent team
        team = group.sort_values('season', ascending=False)['team'].iloc[0]
        total_weight = group['weight'].sum()
        total_games = len(group)

        # Weighted burn rate
        burn_weighted = (group['burn_rate'].fillna(0) * group['weight']).sum() / total_weight
        contested_weighted = (group['contested_rate'].fillna(0) * group['weight']).sum() / total_weight
        sep_weighted = (group['separation_at_target'].fillna(2.5) * group['weight']).sum() / total_weight

        # Apply confidence adjustment
        burn_score = np.clip(burn_weighted / 0.06, 0, 1)
        burn_score = apply_confidence_adjustment(burn_score, total_weight, total_games)

        contested_score = np.clip(contested_weighted / 0.15, 0, 1)
        contested_score = apply_confidence_adjustment(contested_score, total_weight, total_games)

        results.append({
            'name': name,
            'position': pos,
            'team': team,
            'total_weight': round(total_weight, 3),
            'games': total_games,
            'burn_rate': round(burn_weighted, 4),
            'burn_score': round(burn_score, 3),
            'contested_rate': round(contested_weighted, 4),
            'contested_score': round(contested_score, 3),
            'avg_separation': round(sep_weighted, 2)
        })

    return pd.DataFrame(results)


def tag_talent_creation():
    """
    TALENT-based: Yards created, evaded tackles, YAC
    """
    conn = get_connection()

    query = """
    SELECT
        g.player_id, g.name, g.position, g.team, g.season,
        g.yards_created_receiving, g.yards_created_rushing,
        g.evaded_tackles, g.total_touches, g.fantasy_points
    FROM gamelog g
    WHERE g.position IN ('WR', 'RB', 'TE')
        AND g.season >= 2020
        AND g.total_touches > 0
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    df = add_season_weights(df, 'TALENT')

    # Yards created per touch
    df['yards_created'] = df['yards_created_receiving'].fillna(0) + df['yards_created_rushing'].fillna(0)
    df['yc_per_touch'] = df['yards_created'] / df['total_touches'].replace(0, np.nan)
    df['evade_per_touch'] = df['evaded_tackles'].fillna(0) / df['total_touches'].replace(0, np.nan)

    results = []
    for (pid, name, pos), group in df.groupby(['player_id', 'name', 'position']):
        if group['weight'].sum() < 0.3:
            continue

        team = group.sort_values('season', ascending=False)['team'].iloc[0]
        total_weight = group['weight'].sum()
        total_games = len(group)

        yc_weighted = (group['yc_per_touch'].fillna(0) * group['weight']).sum() / total_weight
        evade_weighted = (group['evade_per_touch'].fillna(0) * group['weight']).sum() / total_weight

        # Creator score
        creator_score = np.clip(yc_weighted / 2.5, 0, 1) * 0.6 + np.clip(evade_weighted / 0.25, 0, 1) * 0.4
        creator_score = apply_confidence_adjustment(creator_score, total_weight, total_games)

        results.append({
            'name': name,
            'position': pos,
            'team': team,
            'total_weight': round(total_weight, 3),
            'games': total_games,
            'yc_per_touch': round(yc_weighted, 3),
            'evade_per_touch': round(evade_weighted, 3),
            'creator_score': round(creator_score, 3)
        })

    return pd.DataFrame(results)


# =============================================================================
# ROLE TAGS (Medium Stability - respects coaching changes)
# =============================================================================

def tag_role_usage():
    """
    ROLE-based: Target share, aDOT, slot rate, shotgun usage
    Uses 60/30/10 weighting, resets on coaching changes
    """
    conn = get_connection()

    query = """
    SELECT
        g.player_id, g.name, g.position, g.team, g.season,
        g.target_share, g.air_yards, g.targets,
        g.slot_snaps, g.snaps_out_wide, g.snaps,
        g.shotgun_snaps, g.fantasy_points
    FROM gamelog g
    WHERE g.position IN ('WR', 'TE', 'RB')
        AND g.season >= 2022
        AND g.snaps > 10
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Apply ROLE weights (respects coaching changes)
    df = add_season_weights(df, 'ROLE', team_col='team')

    # Calculate metrics
    df['adot'] = df['air_yards'].fillna(0) / df['targets'].replace(0, np.nan)
    df['slot_rate'] = df['slot_snaps'].fillna(0) / df['snaps'].replace(0, np.nan)
    df['outside_rate'] = df['snaps_out_wide'].fillna(0) / df['snaps'].replace(0, np.nan)
    df['shotgun_rate'] = df['shotgun_snaps'].fillna(0) / df['snaps'].replace(0, np.nan)

    results = []
    for (pid, name, pos), group in df.groupby(['player_id', 'name', 'position']):
        # Filter out zero-weight games (pre-coaching change)
        valid_games = group[group['weight'] > 0]
        if len(valid_games) < 4 or valid_games['weight'].sum() < 0.3:
            continue

        team = valid_games.sort_values('season', ascending=False)['team'].iloc[0]
        total_weight = valid_games['weight'].sum()
        total_games = len(valid_games)

        # Check if team had coaching change
        had_coaching_change = team in COACHING_CHANGES_2025

        # Weighted metrics
        target_share = (valid_games['target_share'].fillna(0) * valid_games['weight']).sum() / total_weight
        adot = (valid_games['adot'].fillna(10) * valid_games['weight']).sum() / total_weight
        slot_rate = (valid_games['slot_rate'].fillna(0) * valid_games['weight']).sum() / total_weight
        shotgun_rate = (valid_games['shotgun_rate'].fillna(0) * valid_games['weight']).sum() / total_weight

        results.append({
            'name': name,
            'position': pos,
            'team': team,
            'coaching_change': had_coaching_change,
            'total_weight': round(total_weight, 3),
            'games': total_games,
            'target_share': round(target_share, 3),
            'adot': round(adot, 1),
            'slot_rate': round(slot_rate, 3),
            'shotgun_rate': round(shotgun_rate, 3)
        })

    return pd.DataFrame(results)


def tag_role_rb_usage():
    """
    ROLE-based: RB snap share, receiving role, goal line work
    """
    conn = get_connection()

    query = """
    SELECT
        g.player_id, g.name, g.position, g.team, g.season,
        g.snap_share, g.targets, g.carries,
        g.goal_line_carries, g.red_zone_touches,
        g.shotgun_snaps, g.snaps, g.fantasy_points
    FROM gamelog g
    WHERE g.position = 'RB'
        AND g.season >= 2022
        AND g.snaps > 5
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    df = add_season_weights(df, 'ROLE', team_col='team')

    # Calculate RB role metrics
    df['total_touches'] = df['targets'].fillna(0) + df['carries'].fillna(0)
    df['receiving_share'] = df['targets'].fillna(0) / df['total_touches'].replace(0, np.nan)
    df['gl_rate'] = df['goal_line_carries'].fillna(0) / df['carries'].replace(0, np.nan)
    df['shotgun_rate'] = df['shotgun_snaps'].fillna(0) / df['snaps'].replace(0, np.nan)

    results = []
    for (pid, name, pos), group in df.groupby(['player_id', 'name', 'position']):
        valid_games = group[group['weight'] > 0]
        if len(valid_games) < 4 or valid_games['weight'].sum() < 0.3:
            continue

        team = valid_games.sort_values('season', ascending=False)['team'].iloc[0]
        total_weight = valid_games['weight'].sum()
        total_games = len(valid_games)

        snap_share = (valid_games['snap_share'].fillna(0) * valid_games['weight']).sum() / total_weight
        receiving_share = (valid_games['receiving_share'].fillna(0) * valid_games['weight']).sum() / total_weight
        gl_rate = (valid_games['gl_rate'].fillna(0) * valid_games['weight']).sum() / total_weight
        shotgun_rate = (valid_games['shotgun_rate'].fillna(0) * valid_games['weight']).sum() / total_weight

        # Classify RB role
        if snap_share > 0.60 and receiving_share < 0.20:
            role = 'BELLCOW_EARLY'
        elif snap_share > 0.55 and receiving_share >= 0.20:
            role = 'BELLCOW_RECEIVING'
        elif shotgun_rate > 0.75 and receiving_share > 0.25:
            role = 'SATELLITE'
        elif snap_share > 0.40:
            role = 'TIMESHARE'
        else:
            role = 'DEPTH'

        results.append({
            'name': name,
            'position': pos,
            'team': team,
            'total_weight': round(total_weight, 3),
            'games': total_games,
            'snap_share': round(snap_share, 3),
            'receiving_share': round(receiving_share, 3),
            'gl_rate': round(gl_rate, 3),
            'shotgun_rate': round(shotgun_rate, 3),
            'role_classification': role
        })

    return pd.DataFrame(results)


# =============================================================================
# LUCK TAGS (Low Stability - current year focus)
# =============================================================================

def tag_luck_efficiency():
    """
    LUCK-based: TD rate, fantasy efficiency (regresses fast)
    Uses 80/20/0 weighting
    """
    conn = get_connection()

    query = """
    SELECT
        g.player_id, g.name, g.position, g.team, g.season,
        g.receiving_touchdowns, g.rushing_touchdowns,
        g.receiving_yards, g.rushing_yards,
        g.targets, g.carries, g.fantasy_points
    FROM gamelog g
    WHERE g.position IN ('WR', 'RB', 'TE')
        AND g.season >= 2024
        AND (g.targets > 0 OR g.carries > 0)
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    df = add_season_weights(df, 'LUCK')

    # Calculate efficiency metrics
    df['total_tds'] = df['receiving_touchdowns'].fillna(0) + df['rushing_touchdowns'].fillna(0)
    df['total_yards'] = df['receiving_yards'].fillna(0) + df['rushing_yards'].fillna(0)
    df['total_opps'] = df['targets'].fillna(0) + df['carries'].fillna(0)

    df['td_rate'] = df['total_tds'] / df['total_opps'].replace(0, np.nan)
    df['yards_per_opp'] = df['total_yards'] / df['total_opps'].replace(0, np.nan)

    results = []
    for (pid, name, pos), group in df.groupby(['player_id', 'name', 'position']):
        if group['weight'].sum() < 0.3 or len(group) < 4:
            continue

        team = group.sort_values('season', ascending=False)['team'].iloc[0]
        total_weight = group['weight'].sum()
        total_games = len(group)

        # Weighted metrics (heavily favors 2025)
        td_rate = (group['td_rate'].fillna(0) * group['weight']).sum() / total_weight
        yards_per_opp = (group['yards_per_opp'].fillna(0) * group['weight']).sum() / total_weight
        total_tds = group['total_tds'].sum()
        total_yards = group['total_yards'].sum()

        # Expected TDs based on yards (1 TD per ~90 yards)
        expected_tds = total_yards / 90
        td_luck = total_tds / expected_tds if expected_tds > 0 else 1.0

        # Classify luck
        if td_luck > 1.3:
            luck_tag = 'TD_REGRESSION_CANDIDATE'
            projection_adj = -min((td_luck - 1.0) * 0.08, 0.12)
        elif td_luck < 0.7:
            luck_tag = 'TD_REGRESSION_BUY'
            projection_adj = min((1.0 - td_luck) * 0.06, 0.08)
        else:
            luck_tag = None
            projection_adj = 0

        results.append({
            'name': name,
            'position': pos,
            'team': team,
            'total_weight': round(total_weight, 3),
            'games': total_games,
            'td_rate': round(td_rate, 4),
            'yards_per_opp': round(yards_per_opp, 2),
            'actual_tds': int(total_tds),
            'expected_tds': round(expected_tds, 1),
            'td_luck_factor': round(td_luck, 2),
            'luck_tag': luck_tag,
            'projection_adjustment': round(projection_adj, 3)
        })

    return pd.DataFrame(results)


def tag_luck_game_script():
    """
    LUCK-based: Garbage time, front-runner tendencies
    """
    conn = get_connection()

    # Use fantasy points variance as proxy for game script dependency
    query = """
    SELECT
        g.player_id, g.name, g.position, g.team, g.season,
        g.fantasy_points, g.targets, g.carries, g.snap_share
    FROM gamelog g
    WHERE g.position IN ('WR', 'RB', 'TE')
        AND g.season >= 2024
        AND g.snaps > 10
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    df = add_season_weights(df, 'LUCK')

    results = []
    for (pid, name, pos), group in df.groupby(['player_id', 'name', 'position']):
        if group['weight'].sum() < 0.3 or len(group) < 4:
            continue

        team = group.sort_values('season', ascending=False)['team'].iloc[0]
        total_weight = group['weight'].sum()
        total_games = len(group)

        # Calculate boom/bust profile
        fp_mean = group['fantasy_points'].mean()
        fp_std = group['fantasy_points'].std()
        fp_max = group['fantasy_points'].max()
        fp_cv = fp_std / fp_mean if fp_mean > 0 else 0

        boom_ratio = fp_max / fp_mean if fp_mean > 0 else 1

        # High variance = garbage time hero or front runner (game script dependent)
        if boom_ratio > 2.5 and fp_cv > 0.6:
            script_tag = 'GARBAGE_TIME_HERO'
            corr_opp_qb = min(boom_ratio * 0.04, 0.15)
        elif boom_ratio < 1.8 and fp_cv < 0.4:
            script_tag = 'CONSISTENT_PRODUCER'
            corr_opp_qb = 0
        else:
            script_tag = None
            corr_opp_qb = 0

        results.append({
            'name': name,
            'position': pos,
            'team': team,
            'total_weight': round(total_weight, 3),
            'games': total_games,
            'fp_mean': round(fp_mean, 2),
            'fp_cv': round(fp_cv, 3),
            'boom_ratio': round(boom_ratio, 2),
            'script_tag': script_tag,
            'correlation_boost_opp_qb': round(corr_opp_qb, 3)
        })

    return pd.DataFrame(results)


# =============================================================================
# MAIN: Generate all weighted tags
# =============================================================================

def generate_weighted_tags():
    """Generate all tags with proper decay weighting"""

    print("=" * 80)
    print("WEIGHTED TAG GENERATOR - 3-Tier Decay System")
    print("=" * 80)
    print(f"\nCurrent Season: {CURRENT_SEASON}")
    print("\nWeight Tiers:")
    print("  TALENT: 40% 2025, 30% 2024, 20% 2023, 10% older")
    print("  ROLE:   60% 2025, 30% 2024, 10% 2023 (or 100% if new coach)")
    print("  LUCK:   80% 2025, 20% 2024, 0% older")

    all_players = {}

    # TALENT tags
    print("\n" + "-" * 40)
    print("TALENT TAGS (High Stability)")
    print("-" * 40)

    print("\n1. Separation/Contested tags...")
    sep_df = tag_talent_separation()
    for _, row in sep_df.iterrows():
        name = row['name']
        if name not in all_players:
            all_players[name] = {'tags': [], 'position': row['position'], 'team': row['team']}

        if row['burn_score'] > 0.3:
            all_players[name]['tags'].append('BURN_ARTIST')
            all_players[name]['burn_rate'] = row['burn_rate']
            all_players[name]['burn_score'] = row['burn_score']

        if row['contested_score'] > 0.3:
            all_players[name]['tags'].append('CONTESTED_CATCH_SAVANT')
            all_players[name]['contested_rate'] = row['contested_rate']

    print(f"   Processed {len(sep_df)} players")

    print("\n2. Yards created/YAC tags...")
    yc_df = tag_talent_creation()
    for _, row in yc_df.iterrows():
        name = row['name']
        if name not in all_players:
            all_players[name] = {'tags': [], 'position': row['position'], 'team': row['team']}

        if row['creator_score'] > 0.4:
            all_players[name]['tags'].append('YARDS_CREATED_GOD')
            all_players[name]['yc_per_touch'] = row['yc_per_touch']
            all_players[name]['creator_score'] = row['creator_score']
            all_players[name]['correlation_reduction_qb'] = round(-row['creator_score'] * 0.12, 3)

    print(f"   Processed {len(yc_df)} players")

    # ROLE tags
    print("\n" + "-" * 40)
    print("ROLE TAGS (Medium Stability)")
    print("-" * 40)

    print("\n3. WR/TE usage tags...")
    usage_df = tag_role_usage()
    for _, row in usage_df.iterrows():
        name = row['name']
        if name not in all_players:
            all_players[name] = {'tags': [], 'position': row['position'], 'team': row['team']}

        # Deep threat
        if row['adot'] > 12 and row['position'] == 'WR':
            all_players[name]['tags'].append('DEEP_BALL_DEPENDENT')
            all_players[name]['adot'] = row['adot']
            all_players[name]['cv_boost'] = round(min((row['adot'] - 10) * 0.02, 0.12), 3)
            all_players[name]['distribution_type'] = 'LogNormal'

        # First read
        if row['target_share'] > 0.22:
            all_players[name]['tags'].append('THE_FIRST_READ')
            all_players[name]['target_share'] = row['target_share']
            all_players[name]['correlation_boost_qb'] = round(min(row['target_share'] * 0.6, 0.18), 3)

        # Slot specialist
        if row['slot_rate'] > 0.55:
            all_players[name]['tags'].append('SLOT_SAFETY_VALVE')
            all_players[name]['slot_rate'] = row['slot_rate']

        # Track coaching change impact
        if row['coaching_change']:
            all_players[name]['coaching_change_2025'] = True

    print(f"   Processed {len(usage_df)} players")

    print("\n4. RB role tags...")
    rb_df = tag_role_rb_usage()
    for _, row in rb_df.iterrows():
        name = row['name']
        if name not in all_players:
            all_players[name] = {'tags': [], 'position': row['position'], 'team': row['team']}

        role = row['role_classification']
        if role != 'DEPTH':
            all_players[name]['tags'].append(role)
            all_players[name]['snap_share'] = row['snap_share']
            all_players[name]['receiving_share'] = row['receiving_share']

        if row['shotgun_rate'] > 0.70:
            all_players[name]['tags'].append('SHOTGUN_SATELLITE')
            all_players[name]['shotgun_rate'] = row['shotgun_rate']
            all_players[name]['correlation_boost_trailing'] = round(row['shotgun_rate'] * 0.25, 3)

    print(f"   Processed {len(rb_df)} players")

    # LUCK tags
    print("\n" + "-" * 40)
    print("LUCK TAGS (Low Stability - 2024-2025 only)")
    print("-" * 40)

    print("\n5. TD regression tags...")
    luck_df = tag_luck_efficiency()
    for _, row in luck_df.iterrows():
        name = row['name']
        if name not in all_players:
            all_players[name] = {'tags': [], 'position': row['position'], 'team': row['team']}

        if row['luck_tag']:
            all_players[name]['tags'].append(row['luck_tag'])
            all_players[name]['td_luck_factor'] = row['td_luck_factor']
            all_players[name]['actual_tds'] = row['actual_tds']
            all_players[name]['expected_tds'] = row['expected_tds']
            all_players[name]['projection_adjustment'] = row['projection_adjustment']

    print(f"   Processed {len(luck_df)} players")

    print("\n6. Game script tags...")
    script_df = tag_luck_game_script()
    for _, row in script_df.iterrows():
        name = row['name']
        if name not in all_players:
            all_players[name] = {'tags': [], 'position': row['position'], 'team': row['team']}

        if row['script_tag']:
            all_players[name]['tags'].append(row['script_tag'])
            all_players[name]['boom_ratio'] = row['boom_ratio']
            all_players[name]['fp_cv'] = row['fp_cv']
            if row['correlation_boost_opp_qb'] > 0:
                all_players[name]['correlation_boost_opp_qb'] = row['correlation_boost_opp_qb']

    print(f"   Processed {len(script_df)} players")

    # Remove players with no tags
    all_players = {k: v for k, v in all_players.items() if len(v['tags']) > 0}

    # Build output
    output = {
        "metadata": {
            "description": "Weighted player tags using 3-tier decay system",
            "current_season": CURRENT_SEASON,
            "weight_tiers": {
                "TALENT": "40% 2025, 30% 2024, 20% 2023, 10% older",
                "ROLE": "60% 2025, 30% 2024, 10% 2023 (100% if new coach)",
                "LUCK": "80% 2025, 20% 2024"
            },
            "coaching_changes": list(COACHING_CHANGES_2025.keys())
        },
        "player_effects": all_players
    }

    # Export
    output_path = Path(__file__).parent.parent / "weighted_player_tags.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 80)
    print(f"SUMMARY: {len(all_players)} players with weighted tags")
    print("=" * 80)

    # Show examples
    print("\nSAMPLE PLAYERS:")
    print("-" * 60)

    sample = list(all_players.items())[:6]
    for name, effects in sample:
        print(f"\n{name} ({effects['position']}, {effects['team']}):")
        print(f"  Tags: {effects['tags']}")
        for k, v in effects.items():
            if k not in ['tags', 'position', 'team']:
                print(f"  {k}: {v}")

    print(f"\n\nExported to: {output_path}")

    return output


if __name__ == "__main__":
    generate_weighted_tags()
