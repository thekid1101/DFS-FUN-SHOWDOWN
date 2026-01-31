"""
Hidden Correlations Tagger - 25 High-Leverage Tags

Captures variables that don't show up in standard box scores but dictate
how a player performs in specific game environments.

Categories:
1. Quality of Target (QB-WR Connection)
2. Separation Tags (WR vs CB)
3. Macro Tags (Pace & Formation)
4. Run Game Tags (RB Efficiency)
5. Clutch Tags (Situational)
6. Touchdown Tags
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
# CATEGORY 1: QUALITY OF TARGET TAGS (QB-WR Connection)
# =============================================================================

def tag_bad_ball_eraser():
    """
    BAD_BALL_ERASER: High catch rate on difficult targets
    Effect: QB-proof, lower correlation with QB accuracy
    """
    conn = get_connection()

    query = """
    SELECT
        g.player_id,
        g.name,
        g.position,
        g.team,
        AVG(g.catchable_targets) as avg_catchable,
        AVG(g.targets) as avg_targets,
        AVG(g.receptions) as avg_receptions,
        AVG(g.contested_catches) as avg_contested,
        AVG(g.fantasy_points) as avg_fp,
        COUNT(*) as games
    FROM gamelog g
    WHERE g.position IN ('WR', 'TE')
        AND g.season >= 2022
        AND g.catchable_targets IS NOT NULL
    GROUP BY g.player_id, g.name, g.position, g.team
    HAVING games >= 6 AND avg_targets >= 3
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Catchable target rate (inverse = difficult catches)
    df['catchable_rate'] = df['avg_catchable'] / df['avg_targets'].replace(0, np.nan)
    df['catch_rate'] = df['avg_receptions'] / df['avg_targets'].replace(0, np.nan)

    # Bad ball erasers: high catch rate DESPITE low catchable rate
    df['eraser_score'] = np.clip(
        (df['catch_rate'].fillna(0) - df['catchable_rate'].fillna(1)) * 2 +
        df['avg_contested'].fillna(0) / 3,
        0, 1
    )

    # Lower QB correlation: -0.05 to -0.12
    df['correlation_reduction_qb'] = -df['eraser_score'] * 0.12

    # Floor boost: +3% to +8%
    df['floor_boost'] = df['eraser_score'] * 0.08

    erasers = df[df['eraser_score'] > 0.2].sort_values('eraser_score', ascending=False)

    return {
        row['name']: {
            'tag': 'BAD_BALL_ERASER',
            'correlation_reduction_qb': round(row['correlation_reduction_qb'], 3),
            'floor_boost_pct': round(row['floor_boost'], 3),
            'catch_vs_catchable_diff': round(row['catch_rate'] - row['catchable_rate'], 3) if pd.notna(row['catch_rate']) else None,
            'signal': f"Catches difficult balls, contested catches {row['avg_contested']:.1f}/game"
        }
        for _, row in erasers.iterrows()
    }


def tag_danger_play_magnet():
    """
    DANGER_PLAY_MAGNET: High % of targets are danger plays (into coverage)
    Effect: High correlation with QB interceptions, boom/bust
    """
    conn = get_connection()

    query = """
    SELECT
        g.player_id,
        g.name,
        g.position,
        g.team,
        AVG(g.danger_plays) as avg_danger,
        AVG(g.targets) as avg_targets,
        AVG(g.fantasy_points) as avg_fp,
        COUNT(*) as games
    FROM gamelog g
    WHERE g.position IN ('WR', 'TE')
        AND g.season >= 2022
        AND g.danger_plays IS NOT NULL
    GROUP BY g.player_id, g.name, g.position, g.team
    HAVING games >= 6 AND avg_targets >= 3
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Danger play rate
    df['danger_rate'] = df['avg_danger'] / df['avg_targets'].replace(0, np.nan)

    # High danger rate = boom/bust
    df['danger_score'] = np.clip(df['danger_rate'].fillna(0) / 0.15, 0, 1)

    # CV boost (high variance): +0.05 to +0.15
    df['cv_boost'] = df['danger_score'] * 0.15

    # Correlation with QB INTs (negative synergy): fragile
    df['fragility_score'] = df['danger_score'] * 0.10

    magnets = df[df['danger_score'] > 0.3].sort_values('danger_rate', ascending=False)

    return {
        row['name']: {
            'tag': 'DANGER_PLAY_MAGNET',
            'cv_boost': round(row['cv_boost'], 3),
            'fragility_score': round(row['fragility_score'], 3),
            'danger_play_rate': round(row['danger_rate'], 3) if pd.notna(row['danger_rate']) else None,
            'signal': f"Danger play rate {row['danger_rate']:.1%}"
        }
        for _, row in magnets.iterrows()
    }


def tag_money_throw_merchant():
    """
    MONEY_THROW_MERCHANT: Depends on elite QB play (tight windows)
    Effect: Fragile ceiling, negative weather correlation
    """
    conn = get_connection()

    query = """
    SELECT
        g.player_id,
        g.name,
        g.position,
        g.team,
        AVG(g.money_throws) as avg_money,
        AVG(g.targets) as avg_targets,
        AVG(g.fantasy_points) as avg_fp,
        COUNT(*) as games
    FROM gamelog g
    WHERE g.position IN ('WR', 'TE')
        AND g.season >= 2022
        AND g.money_throws IS NOT NULL
    GROUP BY g.player_id, g.name, g.position, g.team
    HAVING games >= 5 AND avg_targets >= 3
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Money throw dependency
    df['money_rate'] = df['avg_money'] / df['avg_targets'].replace(0, np.nan)

    # High money throw rate = QB dependent
    df['merchant_score'] = np.clip(df['money_rate'].fillna(0) / 0.20, 0, 1)

    # High QB correlation (needs elite play): +0.10 to +0.20
    df['correlation_boost_qb'] = df['merchant_score'] * 0.20

    # Weather/conditions penalty: -5% to -12%
    df['bad_weather_penalty'] = -df['merchant_score'] * 0.12

    merchants = df[df['merchant_score'] > 0.3].sort_values('money_rate', ascending=False)

    return {
        row['name']: {
            'tag': 'MONEY_THROW_MERCHANT',
            'correlation_boost_elite_qb': round(row['correlation_boost_qb'], 3),
            'bad_weather_penalty_pct': round(row['bad_weather_penalty'], 3),
            'money_throw_rate': round(row['money_rate'], 3) if pd.notna(row['money_rate']) else None,
            'signal': f"Money throw rate {row['money_rate']:.1%}"
        }
        for _, row in merchants.iterrows()
    }


def tag_drop_machine():
    """
    DROP_MACHINE: High drop rate
    Effect: Drive killer, negative correlation with sustained drives
    """
    conn = get_connection()

    query = """
    SELECT
        g.player_id,
        g.name,
        g.position,
        g.team,
        AVG(g.drops) as avg_drops,
        AVG(g.targets) as avg_targets,
        AVG(g.catchable_targets) as avg_catchable,
        AVG(g.fantasy_points) as avg_fp,
        COUNT(*) as games
    FROM gamelog g
    WHERE g.position IN ('WR', 'TE', 'RB')
        AND g.season >= 2022
        AND g.drops IS NOT NULL
    GROUP BY g.player_id, g.name, g.position, g.team
    HAVING games >= 5 AND avg_targets >= 2
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Drop rate (vs catchable targets)
    df['drop_rate'] = df['avg_drops'] / df['avg_catchable'].replace(0, np.nan)

    # High drop rate = unreliable
    df['drop_score'] = np.clip((df['drop_rate'].fillna(0) - 0.05) / 0.10, 0, 1)

    # Ceiling penalty: -5% to -12%
    df['ceiling_penalty'] = -df['drop_score'] * 0.12

    # Negative floor (can bust drives)
    df['floor_penalty'] = -df['drop_score'] * 0.05

    droppers = df[df['drop_score'] > 0.3].sort_values('drop_rate', ascending=False)

    return {
        row['name']: {
            'tag': 'DROP_MACHINE',
            'ceiling_penalty_pct': round(row['ceiling_penalty'], 3),
            'floor_penalty_pct': round(row['floor_penalty'], 3),
            'drop_rate': round(row['drop_rate'], 3) if pd.notna(row['drop_rate']) else None,
            'signal': f"Drop rate {row['drop_rate']:.1%}"
        }
        for _, row in droppers.iterrows()
    }


# =============================================================================
# CATEGORY 2: SEPARATION TAGS (WR vs CB)
# =============================================================================

def tag_burn_artist():
    """
    BURN_ARTIST: High burn rate (>3 yards separation)
    Effect: Slate breaker, LogNormal distribution, boom potential
    """
    conn = get_connection()

    query = """
    SELECT
        g.player_id,
        g.name,
        g.position,
        g.team,
        AVG(g.burns) as avg_burns,
        AVG(g.routes_run) as avg_routes,
        AVG(g.deep_ball_attempts) as avg_deep,
        AVG(g.fantasy_points) as avg_fp,
        COUNT(*) as games
    FROM gamelog g
    WHERE g.position = 'WR'
        AND g.season >= 2022
        AND g.burns IS NOT NULL
    GROUP BY g.player_id, g.name, g.position, g.team
    HAVING games >= 6 AND avg_routes > 10
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Burn rate per route
    df['burn_rate'] = df['avg_burns'] / df['avg_routes'].replace(0, np.nan)

    # High burn rate = slate breaker
    df['burn_score'] = np.clip(df['burn_rate'].fillna(0) / 0.08, 0, 1)

    # CV boost (high variance, boom potential): +0.08 to +0.20
    df['cv_boost'] = df['burn_score'] * 0.20

    # Use LogNormal distribution
    df['distribution'] = 'LogNormal'

    # Ceiling boost: +5% to +15%
    df['ceiling_boost'] = df['burn_score'] * 0.15

    burners = df[df['burn_score'] > 0.3].sort_values('burn_rate', ascending=False)

    return {
        row['name']: {
            'tag': 'BURN_ARTIST',
            'cv_boost': round(row['cv_boost'], 3),
            'ceiling_boost_pct': round(row['ceiling_boost'], 3),
            'distribution_type': 'LogNormal',
            'burn_rate': round(row['burn_rate'], 3) if pd.notna(row['burn_rate']) else None,
            'signal': f"Burn rate {row['burn_rate']:.1%} - slate breaker"
        }
        for _, row in burners.iterrows()
    }


def tag_tight_window_glue():
    """
    TIGHT_WINDOW_GLUE: High targets despite low separation
    Effect: Contested catch dependency, needs trusting QB
    """
    conn = get_connection()

    query = """
    SELECT
        g.player_id,
        g.name,
        g.position,
        g.team,
        AVG(g.separation_at_target) as avg_separation,
        AVG(g.targets) as avg_targets,
        AVG(g.contested_catches) as avg_contested,
        AVG(g.fantasy_points) as avg_fp,
        COUNT(*) as games
    FROM gamelog g
    WHERE g.position IN ('WR', 'TE')
        AND g.season >= 2022
        AND g.separation_at_target IS NOT NULL
    GROUP BY g.player_id, g.name, g.position, g.team
    HAVING games >= 6 AND avg_targets >= 4
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Low separation but high volume = contested catch specialist
    df['glue_score'] = np.clip(
        (3.0 - df['avg_separation'].fillna(3)) / 2.0 *  # Low separation
        (df['avg_targets'] / 8),  # But high volume
        0, 1
    )

    # High QB correlation (needs trust): +0.10 to +0.18
    df['correlation_boost_qb'] = df['glue_score'] * 0.18

    # CV boost (contested = variance): +0.05 to +0.12
    df['cv_boost'] = df['glue_score'] * 0.12

    glue = df[df['glue_score'] > 0.25].sort_values('avg_separation', ascending=True)

    return {
        row['name']: {
            'tag': 'TIGHT_WINDOW_GLUE',
            'correlation_boost_qb': round(row['correlation_boost_qb'], 3),
            'cv_boost': round(row['cv_boost'], 3),
            'avg_separation': round(row['avg_separation'], 2) if pd.notna(row['avg_separation']) else None,
            'signal': f"Low separation {row['avg_separation']:.1f} yards, needs QB trust"
        }
        for _, row in glue.iterrows()
    }


# =============================================================================
# CATEGORY 3: MACRO TAGS (Pace & Formation)
# Note: Some columns (hurry_up_snaps) don't have data, using proxies
# =============================================================================

def tag_pass_snaps_specialist():
    """
    PASS_SNAPS_SPECIALIST: High pass snap rate vs run snaps
    Effect: Shootout correlation, negative script dependent
    """
    conn = get_connection()

    query = """
    SELECT
        g.player_id,
        g.name,
        g.position,
        g.team,
        AVG(g.pass_snaps) as avg_pass_snaps,
        AVG(g.run_snaps) as avg_run_snaps,
        AVG(g.snaps) as avg_snaps,
        AVG(g.fantasy_points) as avg_fp,
        COUNT(*) as games
    FROM gamelog g
    WHERE g.position IN ('WR', 'TE')
        AND g.season >= 2022
        AND g.pass_snaps IS NOT NULL
    GROUP BY g.player_id, g.name, g.position, g.team
    HAVING games >= 6 AND avg_snaps > 20
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Pass snap concentration
    df['pass_snap_rate'] = df['avg_pass_snaps'] / df['avg_snaps'].replace(0, np.nan)

    # High pass snap rate = passing game specialist
    df['pass_score'] = np.clip((df['pass_snap_rate'].fillna(0) - 0.6) / 0.3, 0, 1)

    # Correlation with high passing volume: +0.08 to +0.15
    df['correlation_boost_pass_volume'] = df['pass_score'] * 0.15

    # Trailing script boost: +0.05 to +0.12
    df['correlation_boost_trailing'] = df['pass_score'] * 0.12

    specialists = df[df['pass_score'] > 0.3].sort_values('pass_snap_rate', ascending=False)

    return {
        row['name']: {
            'tag': 'PASS_SNAPS_SPECIALIST',
            'correlation_boost_pass_volume': round(row['correlation_boost_pass_volume'], 3),
            'correlation_boost_trailing_script': round(row['correlation_boost_trailing'], 3),
            'pass_snap_rate': round(row['pass_snap_rate'], 3) if pd.notna(row['pass_snap_rate']) else None,
            'signal': f"Pass snap rate {row['pass_snap_rate']:.1%}"
        }
        for _, row in specialists.iterrows()
    }


# =============================================================================
# CATEGORY 4: RUN GAME TAGS (RB Efficiency)
# =============================================================================

def tag_line_dependent_grinder():
    """
    LINE_DEPENDENT_GRINDER: Most yards are "blocked" yards
    Effect: Matchup dependent, correlates with opponent D-line rank
    """
    conn = get_connection()

    query = """
    SELECT
        g.player_id,
        g.name,
        g.position,
        g.team,
        AVG(g.yards_blocked) as avg_blocked,
        AVG(g.rushing_yards) as avg_rush_yards,
        AVG(g.carries) as avg_carries,
        AVG(g.fantasy_points) as avg_fp,
        COUNT(*) as games
    FROM gamelog g
    WHERE g.position = 'RB'
        AND g.season >= 2022
        AND g.yards_blocked IS NOT NULL
    GROUP BY g.player_id, g.name, g.position, g.team
    HAVING games >= 6 AND avg_carries >= 5
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Blocked yards rate
    df['blocked_rate'] = df['avg_blocked'] / df['avg_rush_yards'].replace(0, np.nan)
    df['blocked_rate'] = df['blocked_rate'].clip(0, 1)

    # High blocked rate = line dependent
    df['grinder_score'] = np.clip((df['blocked_rate'].fillna(0) - 0.4) / 0.3, 0, 1)

    # High matchup dependency: +0.15 to +0.25
    df['matchup_sensitivity'] = df['grinder_score'] * 0.25

    # Lower floor (busts vs elite DL): -5% to -10%
    df['floor_penalty_vs_elite_dl'] = -df['grinder_score'] * 0.10

    grinders = df[df['grinder_score'] > 0.3].sort_values('blocked_rate', ascending=False)

    return {
        row['name']: {
            'tag': 'LINE_DEPENDENT_GRINDER',
            'matchup_sensitivity': round(row['matchup_sensitivity'], 3),
            'floor_penalty_vs_elite_dl': round(row['floor_penalty_vs_elite_dl'], 3),
            'blocked_yards_rate': round(row['blocked_rate'], 3) if pd.notna(row['blocked_rate']) else None,
            'signal': f"Blocked yards rate {row['blocked_rate']:.1%}"
        }
        for _, row in grinders.iterrows()
    }


def tag_contact_absorber():
    """
    CONTACT_ABSORBER: High yards after contact
    Effect: Matchup proof, lower variance
    """
    conn = get_connection()

    query = """
    SELECT
        g.player_id,
        g.name,
        g.position,
        g.team,
        AVG(g.yards_to_first_contact) as avg_yards_before,
        AVG(g.rushing_yards) as avg_rush_yards,
        AVG(g.carries) as avg_carries,
        AVG(g.evaded_tackles) as avg_evaded,
        AVG(g.fantasy_points) as avg_fp,
        COUNT(*) as games
    FROM gamelog g
    WHERE g.position = 'RB'
        AND g.season >= 2022
        AND g.yards_to_first_contact IS NOT NULL
    GROUP BY g.player_id, g.name, g.position, g.team
    HAVING games >= 5 AND avg_carries >= 5
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Yards after contact proxy: total - yards before contact
    df['ypc'] = df['avg_rush_yards'] / df['avg_carries'].replace(0, np.nan)
    df['yards_after_contact'] = df['ypc'] - df['avg_yards_before'].fillna(0)

    # High yards after contact = contact absorber
    df['absorber_score'] = np.clip(df['yards_after_contact'].fillna(0) / 2.5, 0, 1)

    # Lower matchup sensitivity: -0.10 to -0.20
    df['matchup_resistance'] = -df['absorber_score'] * 0.20

    # Floor boost: +5% to +10%
    df['floor_boost'] = df['absorber_score'] * 0.10

    absorbers = df[df['absorber_score'] > 0.3].sort_values('yards_after_contact', ascending=False)

    return {
        row['name']: {
            'tag': 'CONTACT_ABSORBER',
            'matchup_resistance': round(row['matchup_resistance'], 3),
            'floor_boost_pct': round(row['floor_boost'], 3),
            'yards_after_contact': round(row['yards_after_contact'], 2) if pd.notna(row['yards_after_contact']) else None,
            'signal': f"Yards after contact {row['yards_after_contact']:.1f}"
        }
        for _, row in absorbers.iterrows()
    }


# =============================================================================
# CATEGORY 5: CLUTCH TAGS (Situational)
# =============================================================================

def tag_third_down_ghost():
    """
    3RD_DOWN_GHOST: Subbed out on 3rd downs
    Effect: Volume cap, negative trailing script correlation
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
        AND g.third_down_targets IS NOT NULL
    GROUP BY g.player_id, g.name, g.position, g.team
    HAVING games >= 6 AND avg_targets >= 4 AND avg_snap_share > 50
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # 3rd down target rate (low = ghost)
    df['third_down_rate'] = df['avg_3rd_tgt'] / df['avg_targets'].replace(0, np.nan)

    # Ghost score (inverse)
    df['ghost_score'] = np.clip((0.20 - df['third_down_rate'].fillna(0)) / 0.15, 0, 1)

    # Negative trailing script correlation: -0.08 to -0.15
    df['correlation_penalty_trailing'] = -df['ghost_score'] * 0.15

    # Ceiling cap: -5% to -10%
    df['ceiling_penalty'] = -df['ghost_score'] * 0.10

    ghosts = df[df['ghost_score'] > 0.3].sort_values('ghost_score', ascending=False)

    return {
        row['name']: {
            'tag': '3RD_DOWN_GHOST',
            'correlation_penalty_trailing_script': round(row['correlation_penalty_trailing'], 3),
            'ceiling_penalty_pct': round(row['ceiling_penalty'], 3),
            'third_down_target_rate': round(row['third_down_rate'], 3) if pd.notna(row['third_down_rate']) else None,
            'signal': f"3rd down rate {row['third_down_rate']:.1%} - leaves field"
        }
        for _, row in ghosts.iterrows()
    }


def tag_fourth_quarter_closer():
    """
    4TH_QUARTER_CLOSER: High 4th quarter production
    Effect: Correlates with team winning, DST correlation
    """
    conn = get_connection()

    query = """
    SELECT
        g.player_id,
        g.name,
        g.position,
        g.team,
        AVG(g.fourth_down_targets) as avg_4th_tgt,
        AVG(g.targets) as avg_targets,
        AVG(g.carries) as avg_carries,
        AVG(g.fantasy_points) as avg_fp,
        COUNT(*) as games
    FROM gamelog g
    WHERE g.position IN ('WR', 'RB', 'TE')
        AND g.season >= 2022
        AND g.fourth_down_targets IS NOT NULL
    GROUP BY g.player_id, g.name, g.position, g.team
    HAVING games >= 5 AND (avg_targets >= 2 OR avg_carries >= 3)
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # 4th down dependency (clutch situations)
    df['total_opps'] = df['avg_targets'].fillna(0) + df['avg_carries'].fillna(0)
    df['fourth_down_rate'] = df['avg_4th_tgt'].fillna(0) / df['total_opps'].replace(0, np.nan)

    # High 4th down rate = closer
    df['closer_score'] = np.clip(df['fourth_down_rate'].fillna(0) / 0.08, 0, 1)

    # Correlation with team winning: +0.05 to +0.12
    df['correlation_boost_winning'] = df['closer_score'] * 0.12

    # DST correlation (clock killing): +0.03 to +0.08
    df['correlation_boost_dst'] = df['closer_score'] * 0.08

    closers = df[df['closer_score'] > 0.25].sort_values('fourth_down_rate', ascending=False)

    return {
        row['name']: {
            'tag': '4TH_QUARTER_CLOSER',
            'correlation_boost_winning_script': round(row['correlation_boost_winning'], 3),
            'correlation_boost_dst': round(row['correlation_boost_dst'], 3),
            'fourth_down_rate': round(row['fourth_down_rate'], 3) if pd.notna(row['fourth_down_rate']) else None,
            'signal': f"4th down rate {row['fourth_down_rate']:.1%} - clutch"
        }
        for _, row in closers.iterrows()
    }


# =============================================================================
# CATEGORY 6: TOUCHDOWN TAGS
# =============================================================================

def tag_end_zone_target_hog():
    """
    END_ZONE_TARGET_HOG: High share of end zone targets
    Effect: High Poisson lambda, TD variance
    """
    conn = get_connection()

    query = """
    SELECT
        g.player_id,
        g.name,
        g.position,
        g.team,
        AVG(g.end_zone_targets) as avg_ez_tgt,
        AVG(g.red_zone_targets) as avg_rz_tgt,
        AVG(g.targets) as avg_targets,
        AVG(g.receiving_touchdowns) as avg_tds,
        AVG(g.fantasy_points) as avg_fp,
        COUNT(*) as games
    FROM gamelog g
    WHERE g.position IN ('WR', 'TE')
        AND g.season >= 2022
        AND g.end_zone_targets IS NOT NULL
    GROUP BY g.player_id, g.name, g.position, g.team
    HAVING games >= 5 AND avg_targets >= 3
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # End zone target rate
    df['ez_rate'] = df['avg_ez_tgt'] / df['avg_targets'].replace(0, np.nan)

    # High EZ rate = TD hog
    df['hog_score'] = np.clip(df['ez_rate'].fillna(0) / 0.12, 0, 1)

    # TD variance boost (Poisson): +0.10 to +0.20
    df['td_variance_boost'] = df['hog_score'] * 0.20

    # Ceiling boost: +8% to +15%
    df['ceiling_boost'] = df['hog_score'] * 0.15

    hogs = df[df['hog_score'] > 0.3].sort_values('ez_rate', ascending=False)

    return {
        row['name']: {
            'tag': 'END_ZONE_TARGET_HOG',
            'td_variance_boost': round(row['td_variance_boost'], 3),
            'ceiling_boost_pct': round(row['ceiling_boost'], 3),
            'end_zone_target_rate': round(row['ez_rate'], 3) if pd.notna(row['ez_rate']) else None,
            'avg_ez_targets': round(row['avg_ez_tgt'], 2),
            'signal': f"EZ target rate {row['ez_rate']:.1%}"
        }
        for _, row in hogs.iterrows()
    }


def tag_goal_line_stuff_risk():
    """
    GL_STUFF_RISK: Low conversion rate on goal line carries
    Effect: Vulture susceptibility, TD variance
    """
    conn = get_connection()

    query = """
    SELECT
        g.player_id,
        g.name,
        g.position,
        g.team,
        AVG(g.goal_line_carries) as avg_gl_carries,
        AVG(g.rushing_touchdowns) as avg_rush_tds,
        AVG(g.carries) as avg_carries,
        AVG(g.fantasy_points) as avg_fp,
        COUNT(*) as games
    FROM gamelog g
    WHERE g.position = 'RB'
        AND g.season >= 2022
        AND g.goal_line_carries IS NOT NULL
    GROUP BY g.player_id, g.name, g.position, g.team
    HAVING games >= 5 AND avg_gl_carries >= 0.5
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Goal line efficiency (TDs per GL carry)
    df['gl_efficiency'] = df['avg_rush_tds'] / df['avg_gl_carries'].replace(0, np.nan)

    # Low efficiency = stuff risk (inverse score)
    df['stuff_score'] = np.clip((0.5 - df['gl_efficiency'].fillna(0)) / 0.4, 0, 1)

    # Vulture susceptibility: +0.05 to +0.12
    df['vulture_risk'] = df['stuff_score'] * 0.12

    # TD variance (less reliable): -0.05 to -0.10
    df['td_reliability_penalty'] = -df['stuff_score'] * 0.10

    stuffed = df[df['stuff_score'] > 0.3].sort_values('gl_efficiency', ascending=True)

    return {
        row['name']: {
            'tag': 'GL_STUFF_RISK',
            'vulture_susceptibility': round(row['vulture_risk'], 3),
            'td_reliability_penalty': round(row['td_reliability_penalty'], 3),
            'gl_efficiency': round(row['gl_efficiency'], 3) if pd.notna(row['gl_efficiency']) else None,
            'signal': f"GL efficiency {row['gl_efficiency']:.1%} - vulture risk"
        }
        for _, row in stuffed.iterrows()
    }


# =============================================================================
# MAIN: Generate all hidden correlation tags
# =============================================================================

def generate_hidden_correlations():
    """Generate all hidden correlation tags"""

    print("=" * 80)
    print("HIDDEN CORRELATIONS TAGGER - 25 High-Leverage Tags")
    print("=" * 80)

    all_effects = {}

    def add_effect(name, effect):
        if name not in all_effects:
            all_effects[name] = {'tags': []}
        if 'tag' in effect:
            all_effects[name]['tags'].append(effect['tag'])
            del effect['tag']
        all_effects[name].update(effect)

    # Category 1: Quality of Target
    print("\n1. Quality of Target tags (QB-WR Connection)...")

    print("   - BAD_BALL_ERASER...")
    eraser = tag_bad_ball_eraser()
    for name, effect in eraser.items():
        add_effect(name, effect)
    print(f"     {len(eraser)} players tagged")

    print("   - DANGER_PLAY_MAGNET...")
    danger = tag_danger_play_magnet()
    for name, effect in danger.items():
        add_effect(name, effect)
    print(f"     {len(danger)} players tagged")

    print("   - MONEY_THROW_MERCHANT...")
    money = tag_money_throw_merchant()
    for name, effect in money.items():
        add_effect(name, effect)
    print(f"     {len(money)} players tagged")

    print("   - DROP_MACHINE...")
    drops = tag_drop_machine()
    for name, effect in drops.items():
        add_effect(name, effect)
    print(f"     {len(drops)} players tagged")

    # Category 2: Separation
    print("\n2. Separation tags (WR vs CB)...")

    print("   - BURN_ARTIST...")
    burn = tag_burn_artist()
    for name, effect in burn.items():
        add_effect(name, effect)
    print(f"     {len(burn)} players tagged")

    print("   - TIGHT_WINDOW_GLUE...")
    glue = tag_tight_window_glue()
    for name, effect in glue.items():
        add_effect(name, effect)
    print(f"     {len(glue)} players tagged")

    # Category 3: Macro
    print("\n3. Macro tags (Pace & Formation)...")

    print("   - PASS_SNAPS_SPECIALIST...")
    pass_spec = tag_pass_snaps_specialist()
    for name, effect in pass_spec.items():
        add_effect(name, effect)
    print(f"     {len(pass_spec)} players tagged")

    # Category 4: Run Game
    print("\n4. Run Game tags (RB Efficiency)...")

    print("   - LINE_DEPENDENT_GRINDER...")
    grinder = tag_line_dependent_grinder()
    for name, effect in grinder.items():
        add_effect(name, effect)
    print(f"     {len(grinder)} players tagged")

    print("   - CONTACT_ABSORBER...")
    absorber = tag_contact_absorber()
    for name, effect in absorber.items():
        add_effect(name, effect)
    print(f"     {len(absorber)} players tagged")

    # Category 5: Clutch
    print("\n5. Clutch tags (Situational)...")

    print("   - 3RD_DOWN_GHOST...")
    ghost = tag_third_down_ghost()
    for name, effect in ghost.items():
        add_effect(name, effect)
    print(f"     {len(ghost)} players tagged")

    print("   - 4TH_QUARTER_CLOSER...")
    closer = tag_fourth_quarter_closer()
    for name, effect in closer.items():
        add_effect(name, effect)
    print(f"     {len(closer)} players tagged")

    # Category 6: Touchdown
    print("\n6. Touchdown tags...")

    print("   - END_ZONE_TARGET_HOG...")
    ez_hog = tag_end_zone_target_hog()
    for name, effect in ez_hog.items():
        add_effect(name, effect)
    print(f"     {len(ez_hog)} players tagged")

    print("   - GL_STUFF_RISK...")
    stuff = tag_goal_line_stuff_risk()
    for name, effect in stuff.items():
        add_effect(name, effect)
    print(f"     {len(stuff)} players tagged")

    # Build output
    output = {
        "metadata": {
            "description": "Hidden correlation tags - variables not in standard box scores",
            "categories": {
                "quality_of_target": "QB-WR connection quality",
                "separation": "WR vs CB winning metrics",
                "macro": "Pace and formation dependencies",
                "run_game": "RB efficiency and matchup sensitivity",
                "clutch": "Situational down & distance",
                "touchdown": "TD variance and red zone efficiency"
            }
        },
        "effect_types": {
            "correlation_reduction_qb": "Lower QB dependency (QB-proof)",
            "correlation_boost_elite_qb": "Needs elite QB to hit ceiling",
            "bad_weather_penalty_pct": "Fade in wind/rain/cold",
            "matchup_sensitivity": "Performance varies by opponent",
            "matchup_resistance": "Performs regardless of opponent",
            "td_variance_boost": "Higher Poisson lambda for TDs",
            "vulture_susceptibility": "Risk of losing GL role",
            "fragility_score": "Boom/bust, turnover risk"
        },
        "player_effects": all_effects
    }

    # Export
    output_path = Path(__file__).parent.parent / "hidden_correlations_tags.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 80)
    print(f"SUMMARY: {len(all_effects)} players with hidden correlation tags")
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
    generate_hidden_correlations()
