"""
Tag Effects Quantifier - Pre-Processing Layer for Simulation Engine

This script quantifies the NUMERICAL EFFECTS of player tags to modify:
1. Correlation coefficients (same_team_correlations matrix adjustments)
2. Projection multipliers (conditional boosts based on matchup/game state)
3. Distribution parameters (CV, skew modifications)

Output: quantified_tag_effects.json - consumed by main.py before sim engine
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
# CLUSTER 1: QB RELATIONSHIP TAGS
# These modify same_team_correlations matrix
# =============================================================================

def quantify_panic_button():
    """
    THE_PANIC_BUTTON: High target share when QB under pressure/trailing
    Effect: Boost correlation with QB (floor protection)
    """
    conn = get_connection()

    # Get players with high check-down tendencies
    query = """
    SELECT
        g.player_id,
        g.name,
        g.position,
        g.team,
        AVG(g.target_share) as avg_target_share,
        AVG(g.targets) as avg_targets,
        AVG(g.receptions) as avg_receptions,
        AVG(g.fantasy_points) as avg_fp,
        AVG(CASE WHEN g.air_yards_share < 0.15 THEN g.targets ELSE 0 END) as short_targets,
        COUNT(*) as games
    FROM gamelog g
    WHERE g.position IN ('RB', 'TE')
        AND g.season >= 2022
        AND g.targets > 0
    GROUP BY g.player_id, g.name, g.position, g.team
    HAVING games >= 8 AND avg_targets >= 3
    """

    df = pd.read_sql_query(query, conn)

    # Calculate "panic button" score based on target consistency
    df['target_floor'] = df['avg_receptions'] / df['avg_targets']  # Catch rate as proxy
    df['volume_score'] = df['avg_target_share'] * 100

    # Panic buttons have high catch rate + consistent targets
    df['panic_score'] = (df['target_floor'] * 0.6 + df['volume_score'] * 0.4)

    # Quantify the correlation boost
    # Top panic buttons get +0.15 to +0.30 correlation boost with QB
    df['correlation_boost'] = np.clip((df['panic_score'] - 5) / 20, 0, 0.30)

    panic_buttons = df[df['correlation_boost'] > 0.10].sort_values('panic_score', ascending=False)

    conn.close()

    return {
        row['name']: {
            'tag': 'THE_PANIC_BUTTON',
            'correlation_boost_qb': round(row['correlation_boost'], 3),
            'floor_protection': round(row['target_floor'], 3),
            'signal': f"Catch rate {row['target_floor']:.1%}, {row['avg_targets']:.1f} targets/game"
        }
        for _, row in panic_buttons.iterrows()
    }


def quantify_deep_ball_dependent():
    """
    THE_DEEP_BALL_DEPENDENT: High aDOT, low catch rate, boom/bust
    Effect: Increase variance (CV), use LogNormal distribution
    """
    conn = get_connection()

    # Use air_yards and deep_ball_attempts for deep ball identification
    raw_query = """
    SELECT
        player_id, name, position, team, season, week,
        target_share, fantasy_points,
        deep_ball_attempts, targets, air_yards, receptions
    FROM gamelog
    WHERE position = 'WR' AND season >= 2022 AND targets > 0
    """

    df = pd.read_sql_query(raw_query, conn)

    # Calculate deep ball rate and aDOT proxy
    df['deep_rate'] = df['deep_ball_attempts'].fillna(0) / df['targets'].replace(0, np.nan)
    df['adot'] = df['air_yards'].fillna(0) / df['targets'].replace(0, np.nan)

    stats = df.groupby(['player_id', 'name', 'position', 'team']).agg({
        'target_share': 'mean',
        'deep_rate': 'mean',
        'adot': 'mean',
        'air_yards': 'mean',
        'fantasy_points': ['mean', 'std', 'count']
    }).reset_index()

    stats.columns = ['player_id', 'name', 'position', 'team',
                     'avg_target_share', 'avg_deep_rate', 'avg_adot', 'avg_air_yards',
                     'avg_fp', 'std_fp', 'games']

    stats = stats[stats['games'] >= 8].copy()
    stats['cv'] = stats['std_fp'] / stats['avg_fp'].replace(0, np.nan)

    # Deep ball dependent: high aDOT (>12) + high variance
    # Normalize aDOT around 10 (average)
    stats['adot_score'] = np.clip((stats['avg_adot'].fillna(0) - 10) / 5, 0, 1)
    stats['deep_score'] = stats['adot_score'] * stats['cv'].fillna(0.4)

    # Top deep threats get CV boost of +0.03 to +0.15
    stats['cv_boost'] = np.clip(stats['deep_score'] * 0.3, 0.03, 0.15)

    # Filter: aDOT > 11 AND CV > 0.5
    deep_dependent = stats[(stats['avg_adot'] > 11) & (stats['cv'] > 0.5)].sort_values('avg_adot', ascending=False)

    conn.close()

    return {
        row['name']: {
            'tag': 'DEEP_BALL_DEPENDENT',
            'cv_boost': round(row['cv_boost'], 3),
            'distribution_type': 'LogNormal',
            'base_cv': round(row['cv'], 3) if pd.notna(row['cv']) else 0.4,
            'adot': round(row['avg_adot'], 1),
            'signal': f"aDOT {row['avg_adot']:.1f}, CV {row['cv']:.2f}"
        }
        for _, row in deep_dependent.iterrows()
    }


def quantify_first_read():
    """
    THE_FIRST_READ: High target share, consistent volume
    Effect: High correlation with QB (if QB booms, this player booms)
    """
    conn = get_connection()

    raw_query = """
    SELECT
        player_id, name, position, team, season, week,
        target_share, targets, fantasy_points
    FROM gamelog
    WHERE position = 'WR' AND season >= 2022 AND targets > 0
    """

    df = pd.read_sql_query(raw_query, conn)

    stats = df.groupby(['player_id', 'name', 'position', 'team']).agg({
        'target_share': 'mean',
        'targets': ['mean', 'std'],
        'fantasy_points': ['mean', 'std', 'count']
    }).reset_index()

    stats.columns = ['player_id', 'name', 'position', 'team',
                     'avg_target_share', 'avg_targets', 'std_targets',
                     'avg_fp', 'std_fp', 'games']

    stats = stats[stats['games'] >= 8].copy()

    # First reads: high target share + consistent volume
    stats['target_consistency'] = 1 - (stats['std_targets'] / stats['avg_targets'].replace(0, np.nan))
    stats['first_read_score'] = stats['avg_target_share'] * stats['target_consistency']

    # Top first reads get correlation boost of +0.10 to +0.20
    stats['correlation_boost'] = np.clip((stats['first_read_score'] - 0.15) * 2, 0, 0.20)

    first_reads = stats[stats['avg_target_share'] > 0.20].sort_values('first_read_score', ascending=False)

    conn.close()

    return {
        row['name']: {
            'tag': 'THE_FIRST_READ',
            'correlation_boost_qb': round(row['correlation_boost'], 3),
            'target_share': round(row['avg_target_share'], 3),
            'consistency': round(row['target_consistency'], 3),
            'signal': f"Target share {row['avg_target_share']:.1%}, consistency {row['target_consistency']:.2f}"
        }
        for _, row in first_reads.iterrows()
    }


def quantify_vulture_victim():
    """
    THE_VULTURE_VICTIM: High yards but low red zone touches
    Effect: Negative correlation with goal-line backs
    """
    conn = get_connection()

    query = """
    SELECT
        g.player_id,
        g.name,
        g.position,
        g.team,
        SUM(COALESCE(g.receiving_yards, 0) + COALESCE(g.rushing_yards, 0)) as total_yards,
        SUM(COALESCE(g.receiving_touchdowns, 0) + COALESCE(g.rushing_touchdowns, 0)) as total_tds,
        SUM(COALESCE(g.red_zone_touches, 0) + COALESCE(g.red_zone_targets, 0)) as rz_opps,
        SUM(COALESCE(g.goal_line_carries, 0) + COALESCE(g.goal_line_targets, 0)) as gl_opps,
        COUNT(*) as games
    FROM gamelog g
    WHERE g.position IN ('RB', 'WR')
        AND g.season >= 2022
    GROUP BY g.player_id, g.name, g.position, g.team
    HAVING games >= 8 AND total_yards > 300
    """

    df = pd.read_sql_query(query, conn)

    # Filter out zeros
    df = df[df['total_yards'] > 0].copy()

    # Yards per TD - high = vulture victim (moves chains but doesn't score)
    df['yards_per_td'] = df['total_yards'] / df['total_tds'].replace(0, 0.5)

    # Goal line opportunity rate per game
    df['gl_rate'] = df['gl_opps'] / df['games']

    # Vulture victims: high yards per TD (>80) AND low goal line usage
    df['vulture_score'] = np.clip((df['yards_per_td'] - 80) / 120, 0, 1)

    # Apply goal line penalty (low GL usage = more likely vulture victim)
    df['vulture_score'] = df['vulture_score'] * (1 - np.clip(df['gl_rate'] / 2, 0, 0.7))

    # Negative correlation with goal-line backs: -0.05 to -0.12
    df['correlation_penalty_glrb'] = -df['vulture_score'] * 0.12

    vulture_victims = df[df['vulture_score'] > 0.1].sort_values('vulture_score', ascending=False).head(60)

    conn.close()

    return {
        row['name']: {
            'tag': 'VULTURE_VICTIM',
            'correlation_penalty_glrb': round(row['correlation_penalty_glrb'], 3),
            'yards_per_td': round(row['yards_per_td'], 1),
            'gl_opps_per_game': round(row['gl_rate'], 2),
            'signal': f"{row['total_yards']:.0f} yards, {row['total_tds']:.0f} TDs, {row['gl_rate']:.1f} GL opps/game"
        }
        for _, row in vulture_victims.iterrows()
    }


# =============================================================================
# CLUSTER 2: GAME SCRIPT TAGS
# These identify bring-backs and front-runners
# =============================================================================

def quantify_garbage_time_hero():
    """
    GARBAGE_TIME_HERO: High production when trailing significantly
    Effect: Positive correlation with opposing QB (game stack candidate)
    """
    conn = get_connection()

    # Use game-level data to approximate game script
    # High FP games when team likely trailing = garbage time hero
    raw_query = """
    SELECT
        g.player_id,
        g.name,
        g.position,
        g.team,
        g.season,
        g.week,
        g.fantasy_points,
        g.targets,
        g.target_share
    FROM gamelog g
    WHERE g.position IN ('WR', 'TE', 'RB')
        AND g.season >= 2022
        AND g.fantasy_points > 0
    """

    df = pd.read_sql_query(raw_query, conn)

    # Calculate per-player stats
    player_stats = df.groupby(['player_id', 'name', 'position', 'team']).agg({
        'fantasy_points': ['mean', 'std', 'max', 'count'],
        'target_share': 'mean'
    }).reset_index()

    player_stats.columns = ['player_id', 'name', 'position', 'team',
                            'avg_fp', 'std_fp', 'max_fp', 'games', 'avg_target_share']

    player_stats = player_stats[player_stats['games'] >= 8].copy()

    # Garbage time heroes have high variance and spike games
    player_stats['boom_ratio'] = player_stats['max_fp'] / player_stats['avg_fp'].replace(0, np.nan)
    player_stats['cv'] = player_stats['std_fp'] / player_stats['avg_fp'].replace(0, np.nan)

    # High boom ratio + high CV = likely garbage time beneficiary
    player_stats['garbage_score'] = player_stats['boom_ratio'] * player_stats['cv']

    # Correlation boost with opposing QB: +0.05 to +0.15
    player_stats['correlation_boost_opp_qb'] = np.clip((player_stats['garbage_score'] - 2) / 10, 0, 0.15)

    garbage_heroes = player_stats[player_stats['correlation_boost_opp_qb'] > 0.03].sort_values(
        'garbage_score', ascending=False
    )

    conn.close()

    return {
        row['name']: {
            'tag': 'GARBAGE_TIME_HERO',
            'correlation_boost_opp_qb': round(row['correlation_boost_opp_qb'], 3),
            'boom_ratio': round(row['boom_ratio'], 2),
            'bring_back_score': round(row['garbage_score'], 2),
            'signal': f"Boom ratio {row['boom_ratio']:.1f}x, CV {row['cv']:.2f}"
        }
        for _, row in garbage_heroes.iterrows()
    }


def quantify_front_runner():
    """
    THE_FRONT_RUNNER: High production when team is winning
    Effect: Positive correlation with team DST
    """
    conn = get_connection()

    query = """
    SELECT
        g.player_id,
        g.name,
        g.position,
        g.team,
        AVG(g.carries) as avg_carries,
        AVG(g.rushing_yards) as avg_rush_yards,
        AVG(g.fantasy_points) as avg_fp,
        SUM(g.carries) as total_carries,
        COUNT(*) as games
    FROM gamelog g
    WHERE g.position = 'RB'
        AND g.season >= 2022
        AND g.carries > 0
    GROUP BY g.player_id, g.name, g.position, g.team
    HAVING games >= 8 AND avg_carries >= 8
    """

    df = pd.read_sql_query(query, conn)

    # Front runners: high carry volume + yards per carry
    df['ypc'] = df['avg_rush_yards'] / df['avg_carries'].replace(0, np.nan)
    df['workload'] = df['avg_carries'] / 15  # Normalized to ~15 carries

    # Front runner score: high workload + efficiency
    df['front_runner_score'] = df['workload'] * np.clip(df['ypc'] / 4.5, 0.5, 1.5)

    # Correlation boost with DST: +0.05 to +0.15
    df['correlation_boost_dst'] = np.clip((df['front_runner_score'] - 0.8) * 0.3, 0, 0.15)

    front_runners = df[df['correlation_boost_dst'] > 0.03].sort_values('front_runner_score', ascending=False)

    conn.close()

    return {
        row['name']: {
            'tag': 'THE_FRONT_RUNNER',
            'correlation_boost_dst': round(row['correlation_boost_dst'], 3),
            'workload_score': round(row['workload'], 2),
            'ypc': round(row['ypc'], 2),
            'signal': f"{row['avg_carries']:.1f} carries/game, {row['ypc']:.1f} YPC"
        }
        for _, row in front_runners.iterrows()
    }


# =============================================================================
# CLUSTER 3: MATCHUP SPECIFIC TAGS
# These provide conditional projection boosts
# =============================================================================

def quantify_coverage_specialists():
    """
    MAN_BEATER vs ZONE_MERCHANT: Coverage-specific performance
    Effect: Conditional projection multiplier based on opponent coverage
    """
    conn = get_connection()

    # Use actual coverage-specific columns
    query = """
    SELECT
        g.player_id,
        g.name,
        g.position,
        g.team,
        AVG(g.fantasy_points) as avg_fp,
        AVG(g.target_share) as avg_target_share,
        AVG(g.routes_won_vs_man) as avg_routes_won_man,
        AVG(g.routes_won_vs_zone) as avg_routes_won_zone,
        AVG(g.yards_vs_man) as avg_yards_man,
        AVG(g.yards_vs_zone) as avg_yards_zone,
        AVG(g.targets_vs_man) as avg_targets_man,
        AVG(g.targets_vs_zone) as avg_targets_zone,
        AVG(g.receiving_yards) as avg_rec_yards,
        AVG(g.routes_run) as avg_routes,
        COUNT(*) as games
    FROM gamelog g
    WHERE g.position = 'WR'
        AND g.season >= 2022
        AND g.targets > 0
    GROUP BY g.player_id, g.name, g.position, g.team
    HAVING games >= 8
    """

    df = pd.read_sql_query(query, conn)

    # Calculate YPRR from available data
    df['yprr'] = df['avg_rec_yards'] / df['avg_routes'].replace(0, np.nan)
    df['yprr'] = df['yprr'].fillna(df['avg_fp'] / 20)

    # Calculate man vs zone efficiency differential
    df['man_efficiency'] = df['avg_yards_man'].fillna(0) / df['avg_targets_man'].replace(0, np.nan)
    df['zone_efficiency'] = df['avg_yards_zone'].fillna(0) / df['avg_targets_zone'].replace(0, np.nan)

    # Compare routes won rates
    df['man_win_rate'] = df['avg_routes_won_man'].fillna(0)
    df['zone_win_rate'] = df['avg_routes_won_zone'].fillna(0)

    # Differential scoring
    df['man_advantage'] = (df['man_efficiency'].fillna(0) - df['zone_efficiency'].fillna(0)) / 10
    df['zone_advantage'] = (df['zone_efficiency'].fillna(0) - df['man_efficiency'].fillna(0)) / 10

    results = {}

    for _, row in df.iterrows():
        if pd.notna(row['man_advantage']) and row['man_advantage'] > 0.3:
            results[row['name']] = {
                'tag': 'MAN_BEATER',
                'projection_boost_vs_man': round(0.05 + min(row['man_advantage'] * 0.15, 0.12), 3),
                'projection_penalty_vs_zone': round(-0.03, 3),
                'man_yards_per_target': round(row['man_efficiency'], 1) if pd.notna(row['man_efficiency']) else None,
                'signal': f"Man yds/tgt: {row['man_efficiency']:.1f} vs Zone: {row['zone_efficiency']:.1f}" if pd.notna(row['man_efficiency']) and pd.notna(row['zone_efficiency']) else "Coverage specialist"
            }
        elif pd.notna(row['zone_advantage']) and row['zone_advantage'] > 0.3:
            results[row['name']] = {
                'tag': 'ZONE_MERCHANT',
                'projection_boost_vs_zone': round(0.05 + min(row['zone_advantage'] * 0.12, 0.10), 3),
                'projection_penalty_vs_man': round(-0.05, 3),
                'zone_yards_per_target': round(row['zone_efficiency'], 1) if pd.notna(row['zone_efficiency']) else None,
                'signal': f"Zone yds/tgt: {row['zone_efficiency']:.1f} vs Man: {row['man_efficiency']:.1f}" if pd.notna(row['man_efficiency']) and pd.notna(row['zone_efficiency']) else "Volume-based"
            }

    conn.close()
    return results


def quantify_yac_monster():
    """
    THE_YAC_MONSTER: High yards after catch, QB-independent upside
    Effect: Lower correlation with QB accuracy, higher floor
    """
    conn = get_connection()

    # Use yards_created which captures YAC
    query = """
    SELECT
        g.player_id,
        g.name,
        g.position,
        g.team,
        AVG(g.receiving_yards) as avg_rec_yards,
        AVG(g.air_yards) as avg_air_yards,
        AVG(COALESCE(g.yards_created_receiving, g.yards_created)) as avg_yards_created,
        AVG(g.fantasy_points) as avg_fp,
        AVG(g.receptions) as avg_rec,
        AVG(g.evaded_tackles) as avg_evaded,
        COUNT(*) as games
    FROM gamelog g
    WHERE g.position IN ('WR', 'RB', 'TE')
        AND g.season >= 2022
        AND g.targets > 0
    GROUP BY g.player_id, g.name, g.position, g.team
    HAVING games >= 8 AND avg_rec >= 2
    """

    df = pd.read_sql_query(query, conn)

    # YAC = receiving yards - air yards (approximately)
    # Or use yards_created directly
    df['yac_estimate'] = df['avg_rec_yards'] - df['avg_air_yards'].fillna(0)
    df['yac_per_rec'] = df['yac_estimate'] / df['avg_rec'].replace(0, np.nan)

    # Also use yards_created and evaded tackles as YAC proxies
    df['yards_created_score'] = df['avg_yards_created'].fillna(0) / df['avg_rec'].replace(0, np.nan)
    df['evade_score'] = df['avg_evaded'].fillna(0) / df['avg_rec'].replace(0, np.nan)

    # Combined YAC score
    df['yac_score'] = np.clip(
        (df['yac_per_rec'].fillna(0) / 5) * 0.4 +
        (df['yards_created_score'].fillna(0) / 3) * 0.3 +
        (df['evade_score'].fillna(0) * 2) * 0.3,
        0, 1
    )

    # Lower QB correlation, higher floor
    df['correlation_reduction_qb'] = -df['yac_score'] * 0.10
    df['floor_boost'] = df['yac_score'] * 0.08

    yac_monsters = df[df['yac_score'] > 0.25].sort_values('yac_score', ascending=False).head(50)

    conn.close()

    return {
        row['name']: {
            'tag': 'YAC_MONSTER',
            'correlation_reduction_qb': round(row['correlation_reduction_qb'], 3),
            'floor_boost_pct': round(row['floor_boost'], 3),
            'yac_score': round(row['yac_score'], 2),
            'signal': f"YAC/rec {row['yac_per_rec']:.1f}, {row['avg_evaded']:.1f} evaded tackles/game" if pd.notna(row['yac_per_rec']) else f"Yards created {row['avg_yards_created']:.1f}"
        }
        for _, row in yac_monsters.iterrows()
    }


# =============================================================================
# CLUSTER 4: EFFICIENCY TAGS
# These help spot traps vs values
# =============================================================================

def quantify_empty_calories():
    """
    EMPTY_CALORIES: High volume, low efficiency
    Effect: Lower ceiling multiplier
    """
    conn = get_connection()

    # Calculate YPRR from routes_run and receiving_yards
    query = """
    SELECT
        g.player_id,
        g.name,
        g.position,
        g.team,
        AVG(g.snap_share) as avg_snap_share,
        AVG(g.target_share) as avg_target_share,
        AVG(g.fantasy_points) as avg_fp,
        AVG(g.receiving_yards) as avg_rec_yards,
        AVG(g.routes_run) as avg_routes,
        AVG(g.targets) as avg_targets,
        COUNT(*) as games
    FROM gamelog g
    WHERE g.position IN ('WR', 'TE')
        AND g.season >= 2022
    GROUP BY g.player_id, g.name, g.position, g.team
    HAVING games >= 8 AND avg_snap_share > 50
    """

    df = pd.read_sql_query(query, conn)

    # Calculate YPRR manually (yards per route run)
    df['yprr'] = df['avg_rec_yards'] / df['avg_routes'].replace(0, np.nan)
    df['yprr'] = df['yprr'].fillna(1.5)

    # Fantasy points per route run
    df['fpprr'] = df['avg_fp'] / df['avg_routes'].replace(0, np.nan)
    df['fpprr'] = df['fpprr'].fillna(0.3)

    # Empty calories: high snap share but LOW YPRR (below 1.8 is concerning)
    # Only tag players with significant volume (>60% snap share) but poor efficiency
    df['empty_score'] = np.clip((1.8 - df['yprr']) / 0.6, 0, 1) * np.clip((df['avg_snap_share'] - 60) / 30, 0, 1)

    # Ceiling reduction: -3% to -10%
    df['ceiling_penalty'] = -df['empty_score'] * 0.10

    # Only tag truly inefficient players
    empty_cal = df[df['empty_score'] > 0.25].sort_values('empty_score', ascending=False).head(40)

    conn.close()

    return {
        row['name']: {
            'tag': 'EMPTY_CALORIES',
            'ceiling_penalty_pct': round(row['ceiling_penalty'], 3),
            'yprr': round(row['yprr'], 2),
            'snap_share': round(row['avg_snap_share'], 1),
            'signal': f"YPRR {row['yprr']:.2f}, snap share {row['avg_snap_share']:.1f}%"
        }
        for _, row in empty_cal.iterrows()
    }


def quantify_td_regression():
    """
    TD_REGRESSION_CANDIDATE: Unsustainably high TD rate
    Effect: Negative projection adjustment
    """
    conn = get_connection()

    query = """
    SELECT
        g.player_id,
        g.name,
        g.position,
        g.team,
        SUM(g.receiving_touchdowns + COALESCE(g.rushing_touchdowns, 0)) as total_tds,
        SUM(g.receiving_yards + COALESCE(g.rushing_yards, 0)) as total_yards,
        SUM(g.targets + COALESCE(g.carries, 0)) as total_opps,
        COUNT(*) as games
    FROM gamelog g
    WHERE g.position IN ('WR', 'RB', 'TE')
        AND g.season >= 2022
    GROUP BY g.player_id, g.name, g.position, g.team
    HAVING games >= 8 AND total_opps > 50
    """

    df = pd.read_sql_query(query, conn)

    # TD rate relative to opportunities
    df['td_rate'] = df['total_tds'] / df['total_opps']
    df['yards_per_opp'] = df['total_yards'] / df['total_opps']

    # High TD rate relative to yards = regression candidate
    # Expected: ~1 TD per 100 yards. Higher = unsustainable
    df['expected_tds'] = df['total_yards'] / 100
    df['td_luck'] = df['total_tds'] / df['expected_tds'].replace(0, np.nan)
    df['td_luck'] = df['td_luck'].fillna(1)

    # Regression score: how much above expected
    df['regression_score'] = np.clip((df['td_luck'] - 1.2) * 2, 0, 1)

    # Projection fade: -3% to -10%
    df['projection_penalty'] = -df['regression_score'] * 0.10

    regression_candidates = df[df['regression_score'] > 0.2].sort_values('td_luck', ascending=False)

    conn.close()

    return {
        row['name']: {
            'tag': 'TD_REGRESSION_CANDIDATE',
            'projection_penalty_pct': round(row['projection_penalty'], 3),
            'td_luck_factor': round(row['td_luck'], 2),
            'actual_tds': int(row['total_tds']),
            'expected_tds': round(row['expected_tds'], 1),
            'signal': f"{row['total_tds']:.0f} TDs vs {row['expected_tds']:.1f} expected ({row['td_luck']:.1f}x luck)"
        }
        for _, row in regression_candidates.iterrows()
    }


def quantify_td_regression_positive():
    """
    TD_REGRESSION_BUY: Unsustainably low TD rate - positive regression candidate
    Effect: Positive projection adjustment
    """
    conn = get_connection()

    query = """
    SELECT
        g.player_id,
        g.name,
        g.position,
        g.team,
        SUM(COALESCE(g.receiving_touchdowns, 0) + COALESCE(g.rushing_touchdowns, 0)) as total_tds,
        SUM(COALESCE(g.receiving_yards, 0) + COALESCE(g.rushing_yards, 0)) as total_yards,
        SUM(COALESCE(g.targets, 0) + COALESCE(g.carries, 0)) as total_opps,
        SUM(COALESCE(g.red_zone_targets, 0) + COALESCE(g.red_zone_touches, 0)) as rz_opps,
        COUNT(*) as games
    FROM gamelog g
    WHERE g.position IN ('WR', 'RB', 'TE')
        AND g.season >= 2022
    GROUP BY g.player_id, g.name, g.position, g.team
    HAVING games >= 8 AND total_opps > 50 AND total_yards > 300
    """

    df = pd.read_sql_query(query, conn)

    # Expected TDs based on yards (1 TD per ~85-100 yards is normal)
    df['expected_tds'] = df['total_yards'] / 90

    # Also factor in RZ opportunities
    df['rz_rate'] = df['rz_opps'] / df['games']

    # Players with RZ opportunities but few TDs = unlucky
    df['td_luck'] = df['total_tds'] / df['expected_tds'].replace(0, 0.5)
    df['td_luck'] = df['td_luck'].fillna(1)

    # Buy candidates: significantly under expected (td_luck < 0.7)
    df['buy_score'] = np.clip((0.7 - df['td_luck']) * 3, 0, 1)

    # Extra boost if they have RZ opportunities (truly unlucky, not just no opportunities)
    df['buy_score'] = df['buy_score'] * (1 + np.clip(df['rz_rate'] / 3, 0, 0.5))

    # Projection boost: +2% to +6%
    df['projection_boost'] = df['buy_score'] * 0.06

    buy_candidates = df[df['buy_score'] > 0.1].sort_values('td_luck', ascending=True).head(40)

    conn.close()

    return {
        row['name']: {
            'tag': 'TD_REGRESSION_BUY',
            'projection_boost_pct': round(row['projection_boost'], 3),
            'td_unluck_factor': round(row['td_luck'], 2),
            'actual_tds': int(row['total_tds']),
            'expected_tds': round(row['expected_tds'], 1),
            'rz_opps_per_game': round(row['rz_rate'], 1),
            'signal': f"{row['total_tds']:.0f} TDs vs {row['expected_tds']:.1f} expected, {row['rz_rate']:.1f} RZ opps/game"
        }
        for _, row in buy_candidates.iterrows()
    }


# =============================================================================
# MAIN: Aggregate all quantified effects
# =============================================================================

def generate_quantified_effects():
    """Generate all quantified tag effects for simulation pre-processing"""

    print("=" * 80)
    print("TAG EFFECTS QUANTIFIER - Pre-Processing Layer for Sim Engine")
    print("=" * 80)

    all_effects = {}

    def add_effect(name, effect):
        """Add effect to player, accumulating tags in a list"""
        if name not in all_effects:
            all_effects[name] = {'tags': []}

        # Add tag to list if it has one
        if 'tag' in effect:
            all_effects[name]['tags'].append(effect['tag'])
            del effect['tag']  # Remove from effect dict, now in tags list

        # Merge other effects
        all_effects[name].update(effect)

    # Cluster 1: QB Relationship
    print("\n1. Quantifying QB Relationship tags...")

    print("   - THE_PANIC_BUTTON...")
    panic = quantify_panic_button()
    for name, effect in panic.items():
        add_effect(name, effect)
    print(f"     {len(panic)} players tagged")

    print("   - DEEP_BALL_DEPENDENT...")
    deep = quantify_deep_ball_dependent()
    for name, effect in deep.items():
        add_effect(name, effect)
    print(f"     {len(deep)} players tagged")

    print("   - THE_FIRST_READ...")
    first = quantify_first_read()
    for name, effect in first.items():
        add_effect(name, effect)
    print(f"     {len(first)} players tagged")

    print("   - VULTURE_VICTIM...")
    vulture = quantify_vulture_victim()
    for name, effect in vulture.items():
        add_effect(name, effect)
    print(f"     {len(vulture)} players tagged")

    # Cluster 2: Game Script
    print("\n2. Quantifying Game Script tags...")

    print("   - GARBAGE_TIME_HERO...")
    garbage = quantify_garbage_time_hero()
    for name, effect in garbage.items():
        add_effect(name, effect)
    print(f"     {len(garbage)} players tagged")

    print("   - THE_FRONT_RUNNER...")
    front = quantify_front_runner()
    for name, effect in front.items():
        add_effect(name, effect)
    print(f"     {len(front)} players tagged")

    # Cluster 3: Matchup Specific
    print("\n3. Quantifying Matchup Specific tags...")

    print("   - MAN_BEATER / ZONE_MERCHANT...")
    coverage = quantify_coverage_specialists()
    for name, effect in coverage.items():
        add_effect(name, effect)
    print(f"     {len(coverage)} players tagged")

    print("   - YAC_MONSTER...")
    yac = quantify_yac_monster()
    for name, effect in yac.items():
        add_effect(name, effect)
    print(f"     {len(yac)} players tagged")

    # Cluster 4: Efficiency
    print("\n4. Quantifying Efficiency tags...")

    print("   - EMPTY_CALORIES...")
    empty = quantify_empty_calories()
    for name, effect in empty.items():
        add_effect(name, effect)
    print(f"     {len(empty)} players tagged")

    print("   - TD_REGRESSION_CANDIDATE...")
    td_down = quantify_td_regression()
    for name, effect in td_down.items():
        add_effect(name, effect)
    print(f"     {len(td_down)} players tagged")

    print("   - TD_REGRESSION_BUY...")
    td_up = quantify_td_regression_positive()
    for name, effect in td_up.items():
        add_effect(name, effect)
    print(f"     {len(td_up)} players tagged")

    # Create the final output structure
    output = {
        "metadata": {
            "description": "Quantified tag effects for simulation pre-processing",
            "usage": "Apply these modifiers to projections/correlations BEFORE sim engine",
            "clusters": {
                "qb_relationship": "Modifies same_team_correlations with QB",
                "game_script": "Identifies bring-backs and front-runners",
                "matchup_specific": "Conditional projection boosts",
                "efficiency": "Ceiling/floor adjustments"
            }
        },
        "effect_types": {
            "correlation_boost_qb": "Add to base QB correlation coefficient",
            "correlation_boost_opp_qb": "Add to opposing QB correlation (game stacks)",
            "correlation_boost_dst": "Add to DST correlation (front-runner)",
            "correlation_penalty_glrb": "Subtract from goal-line RB correlation",
            "correlation_reduction_qb": "Reduce QB dependency (YAC monsters)",
            "cv_boost": "Add to coefficient of variation (volatility)",
            "projection_boost_pct": "Multiply projection by (1 + this value)",
            "projection_penalty_pct": "Multiply projection by (1 + this value) [negative]",
            "ceiling_penalty_pct": "Reduce ceiling percentile by this factor",
            "floor_boost_pct": "Increase floor percentile by this factor"
        },
        "player_effects": all_effects
    }

    # Export
    output_path = Path(__file__).parent.parent / "quantified_tag_effects.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 80)
    print(f"SUMMARY: {len(all_effects)} players with quantified effects")
    print("=" * 80)

    # Show example modifiers
    print("\nEXAMPLE PLAYER EFFECTS:")
    print("-" * 60)

    example_players = list(all_effects.keys())[:5]
    for player in example_players:
        effects = all_effects[player]
        print(f"\n{player}:")
        for key, value in effects.items():
            if key not in ['tag', 'signal']:
                print(f"  {key}: {value}")

    print(f"\n\nExported to: {output_path}")

    return output


if __name__ == "__main__":
    generate_quantified_effects()
