"""
Scheme & Environment Tagger - Advanced Situational Correlations

Generates 4 clusters of high-leverage tags:
1. Scheme-Specific (Formation DNA) - Shotgun/Play Action/Box Counts
2. Clutch Tags (Down & Distance) - 3rd down, chain movers
3. Environmental Tags (Turf/Dome) - Surface splits
4. Creator Tags (Independent Value) - Yards created, route winning

These isolate SCHEME and ENVIRONMENT correlations - the tie-breakers in DFS sims.
"""

import sqlite3
import pandas as pd
import numpy as np
import json
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "nfl_research.db"

def get_connection():
    return sqlite3.connect(DB_PATH)


# =============================================================================
# CLUSTER 1: SCHEME-SPECIFIC TAGS (Formation DNA)
# =============================================================================

def tag_shotgun_satellite():
    """
    SHOTGUN_SATELLITE: RBs with high shotgun snap rate
    Effect: High correlation with trailing game scripts (+0.15 to +0.30)
    """
    conn = get_connection()

    # Use shotgun_snaps which has data
    query = """
    SELECT
        g.player_id,
        g.name,
        g.position,
        g.team,
        AVG(g.shotgun_snaps) as avg_shotgun_snaps,
        AVG(g.snaps) as avg_snaps,
        AVG(g.targets) as avg_targets,
        AVG(g.fantasy_points) as avg_fp,
        COUNT(*) as games
    FROM gamelog g
    WHERE g.position = 'RB'
        AND g.season >= 2022
        AND g.shotgun_snaps IS NOT NULL
    GROUP BY g.player_id, g.name, g.position, g.team
    HAVING games >= 6 AND avg_snaps >= 10
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Calculate shotgun snap rate
    df['shotgun_rate'] = df['avg_shotgun_snaps'] / df['avg_snaps'].replace(0, np.nan)

    # Shotgun satellites: >75% shotgun usage = passing down back
    df['satellite_score'] = np.clip((df['shotgun_rate'].fillna(0) - 0.65) / 0.25, 0, 1)

    # Also boost for receiving backs (high targets relative to snaps)
    df['receiving_rate'] = df['avg_targets'] / df['avg_snaps'].replace(0, np.nan)
    df['satellite_score'] = df['satellite_score'] * (1 + np.clip(df['receiving_rate'] * 2, 0, 0.5))

    # Correlation boost with trailing scripts: +0.10 to +0.25
    df['correlation_boost_trailing'] = np.clip(df['satellite_score'] * 0.25, 0, 0.25)

    satellites = df[df['satellite_score'] > 0.3].sort_values('shotgun_rate', ascending=False)

    return {
        row['name']: {
            'tag': 'SHOTGUN_SATELLITE',
            'correlation_boost_trailing_script': round(row['correlation_boost_trailing'], 3),
            'shotgun_snap_rate': round(row['shotgun_rate'], 3) if pd.notna(row['shotgun_rate']) else None,
            'signal': f"Shotgun rate {row['shotgun_rate']:.1%}" if pd.notna(row['shotgun_rate']) else "High shotgun usage"
        }
        for _, row in satellites.iterrows()
    }


def tag_play_action_merchant():
    """
    PLAY_ACTION_MERCHANT: WRs with high PA target share
    Effect: Positive correlation with own RB1 (unusual WR/RB correlation +0.10 to +0.20)
    """
    conn = get_connection()

    query = """
    SELECT
        g.player_id,
        g.name,
        g.position,
        g.team,
        AVG(g.play_action_snaps) as avg_pa_snaps,
        AVG(g.snaps) as avg_snaps,
        AVG(g.targets) as avg_targets,
        AVG(g.fantasy_points) as avg_fp,
        COUNT(*) as games
    FROM gamelog g
    WHERE g.position = 'WR'
        AND g.season >= 2022
    GROUP BY g.player_id, g.name, g.position, g.team
    HAVING games >= 8 AND avg_snaps > 20
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Calculate play action snap rate
    df['pa_rate'] = df['avg_pa_snaps'].fillna(0) / df['avg_snaps'].replace(0, np.nan)

    # High PA rate = correlation with run game success
    df['pa_score'] = np.clip((df['pa_rate'].fillna(0) - 0.15) / 0.15, 0, 1)

    # Positive correlation with team RB1: +0.08 to +0.18
    df['correlation_boost_rb1'] = df['pa_score'] * 0.18

    pa_merchants = df[df['pa_score'] > 0.3].sort_values('pa_rate', ascending=False)

    return {
        row['name']: {
            'tag': 'PLAY_ACTION_MERCHANT',
            'correlation_boost_rb1': round(row['correlation_boost_rb1'], 3),
            'pa_snap_rate': round(row['pa_rate'], 3) if pd.notna(row['pa_rate']) else None,
            'signal': f"PA snap rate {row['pa_rate']:.1%}" if pd.notna(row['pa_rate']) else "High PA usage"
        }
        for _, row in pa_merchants.iterrows()
    }


def tag_heavy_box_bully():
    """
    HEAVY_BOX_BULLY: RBs who face 8+ men in box frequently
    Effect: Positive correlation with DST (+0.08 to +0.15) and leading scripts
    """
    conn = get_connection()

    # Use defenders_in_box which has data
    query = """
    SELECT
        g.player_id,
        g.name,
        g.position,
        g.team,
        AVG(g.defenders_in_box) as avg_box_count,
        AVG(g.defenders_in_box_run_snaps) as avg_box_run_snaps,
        AVG(g.run_snaps) as avg_run_snaps,
        AVG(g.carries) as avg_carries,
        AVG(g.yards_per_carry) as avg_ypc,
        AVG(g.fantasy_points) as avg_fp,
        COUNT(*) as games
    FROM gamelog g
    WHERE g.position = 'RB'
        AND g.season >= 2022
        AND g.defenders_in_box IS NOT NULL
    GROUP BY g.player_id, g.name, g.position, g.team
    HAVING games >= 6 AND avg_carries >= 8
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # High box count = clock killer RB (teams stack box against them)
    # Average box is ~7, so >7.5 = heavy box
    df['box_score'] = np.clip((df['avg_box_count'].fillna(7) - 7.2) / 0.8, 0, 1)

    # Also consider efficiency against heavy boxes (good YPC despite loaded box = bully)
    df['efficiency_bonus'] = np.clip((df['avg_ypc'].fillna(3.5) - 3.5) / 1.5, 0, 0.5)

    # Combined score
    df['bully_score'] = df['box_score'] * (1 + df['efficiency_bonus'])

    # Correlation with DST: +0.05 to +0.12
    df['correlation_boost_dst'] = np.clip(df['bully_score'] * 0.12, 0, 0.12)

    # Correlation with leading script: +0.10 to +0.20
    df['correlation_boost_leading'] = np.clip(df['bully_score'] * 0.20, 0, 0.20)

    bullies = df[df['bully_score'] > 0.2].sort_values('avg_box_count', ascending=False)

    return {
        row['name']: {
            'tag': 'HEAVY_BOX_BULLY',
            'correlation_boost_dst': round(row['correlation_boost_dst'], 3),
            'correlation_boost_leading_script': round(row['correlation_boost_leading'], 3),
            'avg_box_count': round(row['avg_box_count'], 2) if pd.notna(row['avg_box_count']) else None,
            'ypc_vs_box': round(row['avg_ypc'], 2) if pd.notna(row['avg_ypc']) else None,
            'signal': f"Avg box {row['avg_box_count']:.1f}, YPC {row['avg_ypc']:.1f}" if pd.notna(row['avg_box_count']) and pd.notna(row['avg_ypc']) else "Clock killer"
        }
        for _, row in bullies.iterrows()
    }


def tag_light_box_slasher():
    """
    LIGHT_BOX_SLASHER: RBs who thrive in light boxes (spread offense)
    Effect: Positive correlation with shootout game environments (+0.15 to +0.25)
    """
    conn = get_connection()

    query = """
    SELECT
        g.player_id,
        g.name,
        g.position,
        g.team,
        AVG(g.light_front_carries) as avg_light_carries,
        AVG(g.carries) as avg_carries,
        AVG(g.defenders_in_box) as avg_box_count,
        AVG(g.yards_per_carry) as avg_ypc,
        AVG(g.fantasy_points) as avg_fp,
        COUNT(*) as games
    FROM gamelog g
    WHERE g.position = 'RB'
        AND g.season >= 2022
    GROUP BY g.player_id, g.name, g.position, g.team
    HAVING games >= 8 AND avg_carries >= 5
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Light box rate
    df['light_rate'] = df['avg_light_carries'].fillna(0) / df['avg_carries'].replace(0, np.nan)

    # Low avg box count = light box slasher
    df['light_box_score'] = np.clip((7.5 - df['avg_box_count'].fillna(7)) / 1.5, 0, 1)
    df['light_carry_score'] = np.clip(df['light_rate'].fillna(0) / 0.3, 0, 1)

    # Combined score
    df['slasher_score'] = (df['light_box_score'] * 0.5 + df['light_carry_score'] * 0.5)

    # Correlation with high-scoring games: +0.10 to +0.25
    df['correlation_boost_shootout'] = df['slasher_score'] * 0.25

    slashers = df[df['slasher_score'] > 0.3].sort_values('slasher_score', ascending=False)

    return {
        row['name']: {
            'tag': 'LIGHT_BOX_SLASHER',
            'correlation_boost_shootout': round(row['correlation_boost_shootout'], 3),
            'light_box_rate': round(row['light_rate'], 3) if pd.notna(row['light_rate']) else None,
            'avg_box_count': round(row['avg_box_count'], 1) if pd.notna(row['avg_box_count']) else None,
            'signal': f"Light box {row['light_rate']:.1%}, avg box {row['avg_box_count']:.1f}" if pd.notna(row['light_rate']) else "Spread beneficiary"
        }
        for _, row in slashers.iterrows()
    }


# =============================================================================
# CLUSTER 2: CLUTCH TAGS (Down & Distance)
# =============================================================================

def tag_chain_mover():
    """
    CHAIN_MOVER: High first down conversion rate
    Effect: Positive correlation with QB volume (more plays = more opportunities)
    """
    conn = get_connection()

    # Use receiving_first_downs which has data
    query = """
    SELECT
        g.player_id,
        g.name,
        g.position,
        g.team,
        AVG(g.receiving_first_downs) as avg_first_downs,
        AVG(g.receptions) as avg_receptions,
        AVG(g.targets) as avg_targets,
        AVG(g.fantasy_points) as avg_fp,
        COUNT(*) as games
    FROM gamelog g
    WHERE g.position IN ('WR', 'TE')
        AND g.season >= 2022
        AND g.receiving_first_downs IS NOT NULL
    GROUP BY g.player_id, g.name, g.position, g.team
    HAVING games >= 6 AND avg_receptions >= 2
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # First down conversion rate (per reception)
    df['fd_rate'] = df['avg_first_downs'].fillna(0) / df['avg_receptions'].replace(0, np.nan)

    # High conversion rate = chain mover (>60% is very good)
    df['chain_score'] = np.clip((df['fd_rate'].fillna(0) - 0.45) / 0.35, 0, 1)

    # Correlation with QB pass attempts: +0.05 to +0.12
    df['correlation_boost_qb_volume'] = df['chain_score'] * 0.12

    # Floor boost (consistent value): +3% to +8%
    df['floor_boost'] = df['chain_score'] * 0.08

    chain_movers = df[df['chain_score'] > 0.2].sort_values('fd_rate', ascending=False)

    return {
        row['name']: {
            'tag': 'CHAIN_MOVER',
            'correlation_boost_qb_volume': round(row['correlation_boost_qb_volume'], 3),
            'floor_boost_pct': round(row['floor_boost'], 3),
            'first_down_rate': round(row['fd_rate'], 3) if pd.notna(row['fd_rate']) else None,
            'signal': f"1st down rate {row['fd_rate']:.1%}" if pd.notna(row['fd_rate']) else "Drive sustainer"
        }
        for _, row in chain_movers.iterrows()
    }


def tag_third_down_bailout():
    """
    THE_BAILOUT_OPTION: High target share on 3rd down
    Effect: High floor, moderate ceiling (correlation with pressure situations)
    """
    conn = get_connection()

    query = """
    SELECT
        g.player_id,
        g.name,
        g.position,
        g.team,
        AVG(g.third_down_targets) as avg_3rd_tgt,
        AVG(g.targets) as avg_targets,
        AVG(g.third_down_receiving_yards) as avg_3rd_yards,
        AVG(g.fantasy_points) as avg_fp,
        COUNT(*) as games
    FROM gamelog g
    WHERE g.position IN ('WR', 'TE', 'RB')
        AND g.season >= 2022
    GROUP BY g.player_id, g.name, g.position, g.team
    HAVING games >= 8 AND avg_targets >= 2
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # 3rd down target concentration
    df['third_down_rate'] = df['avg_3rd_tgt'].fillna(0) / df['avg_targets'].replace(0, np.nan)

    # High 3rd down rate = bailout option
    df['bailout_score'] = np.clip((df['third_down_rate'].fillna(0) - 0.20) / 0.15, 0, 1)

    # Floor boost: +5% to +10%
    df['floor_boost'] = df['bailout_score'] * 0.10

    # Slight ceiling reduction (shorter routes): -2% to -5%
    df['ceiling_penalty'] = -df['bailout_score'] * 0.05

    bailouts = df[df['bailout_score'] > 0.3].sort_values('third_down_rate', ascending=False)

    return {
        row['name']: {
            'tag': 'THE_BAILOUT_OPTION',
            'floor_boost_pct': round(row['floor_boost'], 3),
            'ceiling_penalty_pct': round(row['ceiling_penalty'], 3),
            'third_down_target_rate': round(row['third_down_rate'], 3) if pd.notna(row['third_down_rate']) else None,
            'signal': f"3rd down rate {row['third_down_rate']:.1%}" if pd.notna(row['third_down_rate']) else "Pressure safety valve"
        }
        for _, row in bailouts.iterrows()
    }


def tag_money_down_ghost():
    """
    MONEY_DOWN_GHOST: Low 3rd down target share - leaves field on crucial plays
    Effect: High variance, boom/bust profile
    """
    conn = get_connection()

    query = """
    SELECT
        g.player_id,
        g.name,
        g.position,
        g.team,
        AVG(g.third_down_targets) as avg_3rd_tgt,
        AVG(g.targets) as avg_targets,
        AVG(g.snap_share) as avg_snap_share,
        AVG(g.fantasy_points) as avg_fp,
        COUNT(*) as games
    FROM gamelog g
    WHERE g.position = 'WR'
        AND g.season >= 2022
    GROUP BY g.player_id, g.name, g.position, g.team
    HAVING games >= 8 AND avg_targets >= 4 AND avg_snap_share > 50
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # 3rd down target rate (low = ghost)
    df['third_down_rate'] = df['avg_3rd_tgt'].fillna(0) / df['avg_targets'].replace(0, np.nan)

    # Ghost score (inverse - low 3rd down rate despite volume)
    df['ghost_score'] = np.clip((0.18 - df['third_down_rate'].fillna(0)) / 0.12, 0, 1)

    # CV boost (high variance): +0.05 to +0.12
    df['cv_boost'] = df['ghost_score'] * 0.12

    ghosts = df[df['ghost_score'] > 0.4].sort_values('ghost_score', ascending=False)

    return {
        row['name']: {
            'tag': 'MONEY_DOWN_GHOST',
            'cv_boost': round(row['cv_boost'], 3),
            'third_down_target_rate': round(row['third_down_rate'], 3) if pd.notna(row['third_down_rate']) else None,
            'signal': f"3rd down rate only {row['third_down_rate']:.1%}" if pd.notna(row['third_down_rate']) else "Boom/bust profile"
        }
        for _, row in ghosts.iterrows()
    }


# =============================================================================
# CLUSTER 3: ENVIRONMENTAL TAGS
# Note: Dome/Turf data may need external join - using proxy metrics
# =============================================================================

def tag_hurry_up_weapon():
    """
    HURRY_UP_WEAPON: Players who thrive in no-huddle/2-minute drill
    Effect: Correlation with high-scoring games, shootouts
    """
    conn = get_connection()

    query = """
    SELECT
        g.player_id,
        g.name,
        g.position,
        g.team,
        AVG(g.hurry_up_snaps) as avg_hurry_snaps,
        AVG(g.snaps) as avg_snaps,
        AVG(g.fantasy_points) as avg_fp,
        COUNT(*) as games
    FROM gamelog g
    WHERE g.position IN ('WR', 'TE', 'RB')
        AND g.season >= 2022
    GROUP BY g.player_id, g.name, g.position, g.team
    HAVING games >= 8 AND avg_snaps > 20
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Hurry-up snap rate
    df['hurry_rate'] = df['avg_hurry_snaps'].fillna(0) / df['avg_snaps'].replace(0, np.nan)

    # High hurry-up rate = 2-minute drill weapon
    df['hurry_score'] = np.clip((df['hurry_rate'].fillna(0) - 0.05) / 0.10, 0, 1)

    # Correlation with high total games: +0.10 to +0.20
    df['correlation_boost_high_total'] = df['hurry_score'] * 0.20

    hurry_weapons = df[df['hurry_score'] > 0.3].sort_values('hurry_rate', ascending=False)

    return {
        row['name']: {
            'tag': 'HURRY_UP_WEAPON',
            'correlation_boost_high_total': round(row['correlation_boost_high_total'], 3),
            'hurry_up_rate': round(row['hurry_rate'], 3) if pd.notna(row['hurry_rate']) else None,
            'signal': f"Hurry-up rate {row['hurry_rate']:.1%}" if pd.notna(row['hurry_rate']) else "2-minute drill specialist"
        }
        for _, row in hurry_weapons.iterrows()
    }


# =============================================================================
# CLUSTER 4: CREATOR TAGS (Independent Value)
# =============================================================================

def tag_yards_created_god():
    """
    YARDS_CREATED_GOD: High yards created per touch
    Effect: Lower correlation with QB (QB-proof), higher floor
    """
    conn = get_connection()

    query = """
    SELECT
        g.player_id,
        g.name,
        g.position,
        g.team,
        AVG(COALESCE(g.yards_created_receiving, 0) + COALESCE(g.yards_created_rushing, 0)) as avg_yards_created,
        AVG(g.total_touches) as avg_touches,
        AVG(g.evaded_tackles) as avg_evaded,
        AVG(g.fantasy_points) as avg_fp,
        COUNT(*) as games
    FROM gamelog g
    WHERE g.position IN ('WR', 'RB', 'TE')
        AND g.season >= 2022
    GROUP BY g.player_id, g.name, g.position, g.team
    HAVING games >= 8 AND avg_touches >= 3
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Yards created per touch
    df['yc_per_touch'] = df['avg_yards_created'].fillna(0) / df['avg_touches'].replace(0, np.nan)

    # Also factor in evaded tackles
    df['evade_per_touch'] = df['avg_evaded'].fillna(0) / df['avg_touches'].replace(0, np.nan)

    # Creator score
    df['creator_score'] = np.clip(df['yc_per_touch'].fillna(0) / 3.0, 0, 1) * 0.7 + \
                          np.clip(df['evade_per_touch'].fillna(0) / 0.3, 0, 1) * 0.3

    # Lower QB correlation: -0.08 to -0.15
    df['correlation_reduction_qb'] = -df['creator_score'] * 0.15

    # Floor boost: +5% to +10%
    df['floor_boost'] = df['creator_score'] * 0.10

    creators = df[df['creator_score'] > 0.4].sort_values('yc_per_touch', ascending=False)

    return {
        row['name']: {
            'tag': 'YARDS_CREATED_GOD',
            'correlation_reduction_qb': round(row['correlation_reduction_qb'], 3),
            'floor_boost_pct': round(row['floor_boost'], 3),
            'yards_created_per_touch': round(row['yc_per_touch'], 2) if pd.notna(row['yc_per_touch']) else None,
            'signal': f"Yards created/touch {row['yc_per_touch']:.1f}" if pd.notna(row['yc_per_touch']) else "QB-proof"
        }
        for _, row in creators.iterrows()
    }


def tag_route_winner():
    """
    ROUTE_WINNER: High route win rate
    Effect: Safe floor, high correlation with pocket passers
    """
    conn = get_connection()

    query = """
    SELECT
        g.player_id,
        g.name,
        g.position,
        g.team,
        AVG(g.routes_won) as avg_routes_won,
        AVG(g.routes_run) as avg_routes,
        AVG(g.separation_at_target) as avg_separation,
        AVG(g.fantasy_points) as avg_fp,
        COUNT(*) as games
    FROM gamelog g
    WHERE g.position IN ('WR', 'TE')
        AND g.season >= 2022
    GROUP BY g.player_id, g.name, g.position, g.team
    HAVING games >= 8 AND avg_routes > 10
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Route win rate
    df['route_win_rate'] = df['avg_routes_won'].fillna(0) / df['avg_routes'].replace(0, np.nan)

    # High route win rate = consistently open
    df['winner_score'] = np.clip((df['route_win_rate'].fillna(0) - 0.40) / 0.25, 0, 1)

    # High correlation with pocket passers: +0.08 to +0.15
    df['correlation_boost_pocket_qb'] = df['winner_score'] * 0.15

    # Floor boost: +3% to +8%
    df['floor_boost'] = df['winner_score'] * 0.08

    winners = df[df['winner_score'] > 0.3].sort_values('route_win_rate', ascending=False)

    return {
        row['name']: {
            'tag': 'ROUTE_WINNER',
            'correlation_boost_pocket_qb': round(row['correlation_boost_pocket_qb'], 3),
            'floor_boost_pct': round(row['floor_boost'], 3),
            'route_win_rate': round(row['route_win_rate'], 3) if pd.notna(row['route_win_rate']) else None,
            'avg_separation': round(row['avg_separation'], 2) if pd.notna(row['avg_separation']) else None,
            'signal': f"Route win rate {row['route_win_rate']:.1%}" if pd.notna(row['route_win_rate']) else "Consistently open"
        }
        for _, row in winners.iterrows()
    }


def tag_big_hit_magnet():
    """
    BIG_HIT_MAGNET: Players who take excessive big hits
    Effect: Wider negative tail in distribution (injury/fumble risk)
    """
    conn = get_connection()

    query = """
    SELECT
        g.player_id,
        g.name,
        g.position,
        g.team,
        AVG(g.big_hits_taken) as avg_big_hits,
        AVG(g.total_touches) as avg_touches,
        AVG(g.fantasy_points) as avg_fp,
        COUNT(*) as games
    FROM gamelog g
    WHERE g.position IN ('WR', 'RB', 'TE')
        AND g.season >= 2022
    GROUP BY g.player_id, g.name, g.position, g.team
    HAVING games >= 8 AND avg_touches >= 5
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Big hits per touch
    df['hits_per_touch'] = df['avg_big_hits'].fillna(0) / df['avg_touches'].replace(0, np.nan)

    # High hit rate = injury/fumble risk
    df['magnet_score'] = np.clip(df['hits_per_touch'].fillna(0) / 0.15, 0, 1)

    # Negative tail widening (kurtosis proxy): +0.05 to +0.15
    df['negative_tail_boost'] = df['magnet_score'] * 0.15

    magnets = df[df['magnet_score'] > 0.3].sort_values('hits_per_touch', ascending=False)

    return {
        row['name']: {
            'tag': 'BIG_HIT_MAGNET',
            'negative_tail_boost': round(row['negative_tail_boost'], 3),
            'big_hits_per_touch': round(row['hits_per_touch'], 3) if pd.notna(row['hits_per_touch']) else None,
            'signal': f"Big hits/touch {row['hits_per_touch']:.2f}" if pd.notna(row['hits_per_touch']) else "High contact"
        }
        for _, row in magnets.iterrows()
    }


# =============================================================================
# MAIN: Generate all scheme/environment tags
# =============================================================================

def generate_scheme_environment_tags():
    """Generate all scheme and environment tags"""

    print("=" * 80)
    print("SCHEME & ENVIRONMENT TAGGER - Advanced Situational Correlations")
    print("=" * 80)

    all_effects = {}

    def add_effect(name, effect):
        if name not in all_effects:
            all_effects[name] = {'tags': []}
        if 'tag' in effect:
            all_effects[name]['tags'].append(effect['tag'])
            del effect['tag']
        all_effects[name].update(effect)

    # Cluster 1: Scheme-Specific
    print("\n1. Generating Scheme-Specific tags (Formation DNA)...")

    print("   - SHOTGUN_SATELLITE...")
    shotgun = tag_shotgun_satellite()
    for name, effect in shotgun.items():
        add_effect(name, effect)
    print(f"     {len(shotgun)} players tagged")

    print("   - PLAY_ACTION_MERCHANT...")
    pa = tag_play_action_merchant()
    for name, effect in pa.items():
        add_effect(name, effect)
    print(f"     {len(pa)} players tagged")

    print("   - HEAVY_BOX_BULLY...")
    heavy = tag_heavy_box_bully()
    for name, effect in heavy.items():
        add_effect(name, effect)
    print(f"     {len(heavy)} players tagged")

    print("   - LIGHT_BOX_SLASHER...")
    light = tag_light_box_slasher()
    for name, effect in light.items():
        add_effect(name, effect)
    print(f"     {len(light)} players tagged")

    # Cluster 2: Clutch Tags
    print("\n2. Generating Clutch tags (Down & Distance)...")

    print("   - CHAIN_MOVER...")
    chain = tag_chain_mover()
    for name, effect in chain.items():
        add_effect(name, effect)
    print(f"     {len(chain)} players tagged")

    print("   - THE_BAILOUT_OPTION...")
    bailout = tag_third_down_bailout()
    for name, effect in bailout.items():
        add_effect(name, effect)
    print(f"     {len(bailout)} players tagged")

    print("   - MONEY_DOWN_GHOST...")
    ghost = tag_money_down_ghost()
    for name, effect in ghost.items():
        add_effect(name, effect)
    print(f"     {len(ghost)} players tagged")

    # Cluster 3: Environmental
    print("\n3. Generating Environmental tags...")

    print("   - HURRY_UP_WEAPON...")
    hurry = tag_hurry_up_weapon()
    for name, effect in hurry.items():
        add_effect(name, effect)
    print(f"     {len(hurry)} players tagged")

    # Cluster 4: Creator Tags
    print("\n4. Generating Creator tags (Independent Value)...")

    print("   - YARDS_CREATED_GOD...")
    yc = tag_yards_created_god()
    for name, effect in yc.items():
        add_effect(name, effect)
    print(f"     {len(yc)} players tagged")

    print("   - ROUTE_WINNER...")
    rw = tag_route_winner()
    for name, effect in rw.items():
        add_effect(name, effect)
    print(f"     {len(rw)} players tagged")

    print("   - BIG_HIT_MAGNET...")
    bh = tag_big_hit_magnet()
    for name, effect in bh.items():
        add_effect(name, effect)
    print(f"     {len(bh)} players tagged")

    # Build output
    output = {
        "metadata": {
            "description": "Scheme & Environment tags for simulation pre-processing",
            "clusters": {
                "scheme_specific": "Formation-based correlations (shotgun, PA, box counts)",
                "clutch": "Down & distance situational tags",
                "environmental": "Game environment correlations",
                "creator": "Independent value (QB-proof metrics)"
            }
        },
        "effect_types": {
            "correlation_boost_trailing_script": "Add to correlation when team trailing",
            "correlation_boost_leading_script": "Add to correlation when team leading",
            "correlation_boost_rb1": "Add to correlation with team RB1 (unusual WR/RB)",
            "correlation_boost_dst": "Add to DST correlation",
            "correlation_boost_shootout": "Add to correlation in high-scoring games",
            "correlation_boost_high_total": "Add to correlation with high game totals",
            "correlation_boost_qb_volume": "Add to QB pass volume correlation",
            "correlation_boost_pocket_qb": "Add to pocket passer QB correlation",
            "correlation_reduction_qb": "Reduce QB dependency (independent scorers)",
            "floor_boost_pct": "Increase floor percentile",
            "ceiling_penalty_pct": "Reduce ceiling percentile",
            "cv_boost": "Increase coefficient of variation",
            "negative_tail_boost": "Widen left tail (injury/fumble risk)"
        },
        "player_effects": all_effects
    }

    # Export
    output_path = Path(__file__).parent.parent / "scheme_environment_tags.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 80)
    print(f"SUMMARY: {len(all_effects)} players with scheme/environment tags")
    print("=" * 80)

    # Show examples
    print("\nEXAMPLE PLAYERS:")
    print("-" * 60)

    for name in list(all_effects.keys())[:6]:
        effects = all_effects[name]
        print(f"\n{name}:")
        print(f"  Tags: {effects['tags']}")
        for k, v in effects.items():
            if k not in ['tags', 'signal']:
                print(f"  {k}: {v}")

    print(f"\n\nExported to: {output_path}")

    return output


if __name__ == "__main__":
    generate_scheme_environment_tags()
