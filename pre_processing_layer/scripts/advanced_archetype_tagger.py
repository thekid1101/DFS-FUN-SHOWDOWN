"""
Advanced Archetype Tagger - 30+ Player Classification Tags

Comprehensive tagging system based on PlayerProfiler metrics:
1. QB DNA Tags (6 archetypes)
2. Receiver Usage Archetypes (7 archetypes)
3. RB Role Clusters (5 archetypes)
4. Game State Conditionals (4 archetypes)
5. Efficiency Traps & Gems (4 archetypes)

OUTPUT: advanced_player_tags.json - Complete player classification
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from nfl_research import query

SEASONS = (2023, 2024, 2025)
MIN_GAMES = 4


# =============================================================================
# CATEGORY 1: QB DNA TAGS
# =============================================================================

def tag_qb_dna():
    """
    QB behavioral archetypes:
    - DEEP_BALL_ADDICT: >15% deep attempts
    - RED_ZONE_VULTURE_QB: Steals RB TDs at goal line
    - PLAY_ACTION_MERCHANT: PA dependent
    - SCRAMBLE_DRILL_CREATOR: High rush share, extends plays
    - FIRST_READ_LOCK: Feeds WR1 heavily
    """

    df = query(f"""
        SELECT
            player_id, name, team,
            SUM(pass_attempts) as total_attempts,
            SUM(deep_ball_pass_attempts) as deep_attempts,
            SUM(carries_inside_5) as gl_carries,
            SUM(rushing_touchdowns) as rush_tds,
            SUM(play_action_snaps) as pa_snaps,
            SUM(snaps) as total_snaps,
            SUM(carries) as total_carries,
            SUM(rushing_yards) as rush_yards,
            SUM(passing_yards) as pass_yards,
            AVG(fantasy_points) as avg_fp,
            COUNT(*) as games
        FROM gamelog
        WHERE season IN {SEASONS}
          AND position = 'QB'
          AND pass_attempts > 10
        GROUP BY player_id, name, team
        HAVING COUNT(*) >= {MIN_GAMES}
    """)

    tags = []
    for _, row in df.iterrows():
        qb_tags = []

        # DEEP_BALL_ADDICT: >15% deep attempts
        if row['total_attempts'] > 0:
            deep_rate = (row['deep_attempts'] or 0) / row['total_attempts']
            if deep_rate > 0.15:
                qb_tags.append('DEEP_BALL_ADDICT')

        # RED_ZONE_VULTURE_QB: High GL carries for a QB
        if (row['gl_carries'] or 0) >= 5 and (row['rush_tds'] or 0) >= 3:
            qb_tags.append('RED_ZONE_VULTURE_QB')

        # PLAY_ACTION_MERCHANT: >30% PA snaps
        if row['total_snaps'] > 0:
            pa_rate = (row['pa_snaps'] or 0) / row['total_snaps']
            if pa_rate > 0.30:
                qb_tags.append('PLAY_ACTION_MERCHANT')

        # SCRAMBLE_DRILL_CREATOR: High rush share
        total_yds = (row['rush_yards'] or 0) + (row['pass_yards'] or 0)
        if total_yds > 0:
            rush_share = (row['rush_yards'] or 0) / total_yds
            if rush_share > 0.12 and (row['total_carries'] or 0) / row['games'] > 4:
                qb_tags.append('SCRAMBLE_DRILL_CREATOR')

        tags.append({
            'player_id': row['player_id'],
            'name': row['name'],
            'team': row['team'],
            'position': 'QB',
            'qb_tags': qb_tags if qb_tags else ['STANDARD_POCKET'],
            'metrics': {
                'deep_rate': round(deep_rate, 3) if row['total_attempts'] > 0 else 0,
                'rush_share': round(rush_share, 3) if total_yds > 0 else 0,
                'gl_carries': int(row['gl_carries'] or 0),
            }
        })

    return pd.DataFrame(tags)


# =============================================================================
# CATEGORY 2: RECEIVER USAGE ARCHETYPES (WR/TE)
# =============================================================================

def tag_receiver_usage():
    """
    Receiver archetypes:
    - MAN_COVERAGE_ALPHA: Wins vs Man
    - ZONE_SIT_DOWN_ARTIST: Thrives vs Zone
    - AIR_YARDS_HOG: >40% team air yards
    - CONTESTED_CATCH_SAVANT: >50% contested catch rate
    - SLOT_SAFETY_VALVE: >70% slot, low aDOT
    - YAC_CREATOR: High yards created
    - END_ZONE_OAK_TREE: Heavy end zone targets
    """

    df = query(f"""
        SELECT
            player_id, name, team, position,
            SUM(routes_won_vs_man) as won_man,
            SUM(routes_won_vs_zone) as won_zone,
            SUM(routes_vs_man) as routes_man,
            SUM(routes_vs_zone) as routes_zone,
            SUM(air_yards_share) as air_share_sum,
            SUM(contested_catches) as contested_catches,
            SUM(contested_targets) as contested_targets,
            SUM(slot_snaps) as slot_snaps,
            SUM(snaps) as total_snaps,
            SUM(yards_created) as yards_created,
            SUM(end_zone_targets) as ez_targets,
            SUM(targets) as total_targets,
            SUM(receptions) as total_receptions,
            AVG(fantasy_points) as avg_fp,
            COUNT(*) as games
        FROM gamelog
        WHERE season IN {SEASONS}
          AND position IN ('WR', 'TE')
          AND targets > 0
        GROUP BY player_id, name, team, position
        HAVING COUNT(*) >= {MIN_GAMES}
    """)

    tags = []
    for _, row in df.iterrows():
        wr_tags = []

        # MAN_COVERAGE_ALPHA vs ZONE_SIT_DOWN_ARTIST
        routes_man = row['routes_man'] or 1
        routes_zone = row['routes_zone'] or 1
        won_man = row['won_man'] or 0
        won_zone = row['won_zone'] or 0

        man_win_rate = won_man / routes_man if routes_man > 20 else 0
        zone_win_rate = won_zone / routes_zone if routes_zone > 20 else 0

        if man_win_rate > zone_win_rate * 1.15 and routes_man > 50:
            wr_tags.append('MAN_COVERAGE_ALPHA')
        elif zone_win_rate > man_win_rate * 1.15 and routes_zone > 50:
            wr_tags.append('ZONE_SIT_DOWN_ARTIST')

        # AIR_YARDS_HOG: High air yards share
        avg_air_share = (row['air_share_sum'] or 0) / row['games']
        if avg_air_share > 0.35:
            wr_tags.append('AIR_YARDS_HOG')

        # CONTESTED_CATCH_SAVANT
        contested_tgts = row['contested_targets'] or 1
        contested_catches = row['contested_catches'] or 0
        if contested_tgts >= 10:
            contested_rate = contested_catches / contested_tgts
            if contested_rate > 0.50:
                wr_tags.append('CONTESTED_CATCH_SAVANT')

        # SLOT_SAFETY_VALVE
        total_snaps = row['total_snaps'] or 1
        slot_rate = (row['slot_snaps'] or 0) / total_snaps
        if slot_rate > 0.65:
            wr_tags.append('SLOT_SAFETY_VALVE')

        # YAC_CREATOR
        receptions = row['total_receptions'] or 1
        yac_per_rec = (row['yards_created'] or 0) / receptions
        if yac_per_rec > 4.0 and row['total_receptions'] >= 20:
            wr_tags.append('YAC_CREATOR')

        # END_ZONE_OAK_TREE
        total_targets = row['total_targets'] or 1
        ez_rate = (row['ez_targets'] or 0) / total_targets
        if ez_rate > 0.08 and (row['ez_targets'] or 0) >= 5:
            wr_tags.append('END_ZONE_OAK_TREE')

        tags.append({
            'player_id': row['player_id'],
            'name': row['name'],
            'team': row['team'],
            'position': row['position'],
            'receiver_tags': wr_tags if wr_tags else ['STANDARD_RECEIVER'],
            'metrics': {
                'man_win_rate': round(man_win_rate, 3),
                'zone_win_rate': round(zone_win_rate, 3),
                'air_share': round(avg_air_share, 3),
                'slot_rate': round(slot_rate, 3),
                'ez_target_rate': round(ez_rate, 3),
            }
        })

    return pd.DataFrame(tags)


# =============================================================================
# CATEGORY 3: RB ROLE CLUSTERS
# =============================================================================

def tag_rb_roles():
    """
    RB role archetypes:
    - GREEN_ZONE_GRINDER: Goal line specialist
    - BETWEEN_20s_WORKHORSE: Volume between 20s, few RZ touches
    - PASS_GAME_SPECIALIST: >40% route participation
    - CLOSER_RB: 4th quarter specialist (high snap share late)
    - ONE_CUT_EXPLOSION: High YPC, big play dependent
    """

    df = query(f"""
        SELECT
            player_id, name, team,
            SUM(carries_inside_10) as gl_carries,
            SUM(carries) as total_carries,
            SUM(rushing_yards) as rush_yards,
            SUM(rushing_touchdowns) as rush_tds,
            SUM(routes_run) as routes_run,
            SUM(snaps) as total_snaps,
            SUM(targets) as targets,
            SUM(receptions) as receptions,
            SUM(red_zone_carries) as rz_carries,
            AVG(fantasy_points) as avg_fp,
            AVG(snap_share) as avg_snap_share,
            COUNT(*) as games
        FROM gamelog
        WHERE season IN {SEASONS}
          AND position = 'RB'
          AND (carries > 0 OR targets > 0)
        GROUP BY player_id, name, team
        HAVING COUNT(*) >= {MIN_GAMES}
    """)

    tags = []
    for _, row in df.iterrows():
        rb_tags = []

        total_carries = row['total_carries'] or 1

        # GREEN_ZONE_GRINDER: High GL carries relative to total, low snap share
        gl_rate = (row['gl_carries'] or 0) / total_carries if total_carries > 20 else 0
        if gl_rate > 0.15 and (row['avg_snap_share'] or 0) < 0.50:
            rb_tags.append('GREEN_ZONE_GRINDER')

        # BETWEEN_20s_WORKHORSE: High yards, low RZ rate
        rz_rate = (row['rz_carries'] or 0) / total_carries if total_carries > 30 else 0
        if (row['rush_yards'] or 0) > 300 and rz_rate < 0.25:
            rb_tags.append('BETWEEN_20s_WORKHORSE')

        # PASS_GAME_SPECIALIST: High route participation
        total_snaps = row['total_snaps'] or 1
        route_rate = (row['routes_run'] or 0) / total_snaps
        target_share = (row['targets'] or 0) / row['games']
        if route_rate > 0.40 and target_share > 3.0:
            rb_tags.append('PASS_GAME_SPECIALIST')

        # ONE_CUT_EXPLOSION: High YPC
        ypc = (row['rush_yards'] or 0) / total_carries if total_carries > 30 else 0
        if ypc > 5.0:
            rb_tags.append('ONE_CUT_EXPLOSION')

        # CLOSER_RB: High snap share (workhorse)
        if (row['avg_snap_share'] or 0) > 0.60 and total_carries > 100:
            rb_tags.append('CLOSER_RB')

        tags.append({
            'player_id': row['player_id'],
            'name': row['name'],
            'team': row['team'],
            'position': 'RB',
            'rb_tags': rb_tags if rb_tags else ['COMMITTEE_BACK'],
            'metrics': {
                'gl_rate': round(gl_rate, 3),
                'rz_rate': round(rz_rate, 3),
                'route_rate': round(route_rate, 3),
                'ypc': round(ypc, 2),
                'snap_share': round(row['avg_snap_share'] or 0, 3),
            }
        })

    return pd.DataFrame(tags)


# =============================================================================
# CATEGORY 4: GAME STATE CONDITIONALS
# =============================================================================

def tag_game_state():
    """
    Game state conditionals (estimated from variance patterns):
    - GARBAGE_TIME_HERO: High variance, pass-heavy
    - FRONT_RUNNER_BULLY: Low variance, rush-heavy
    - 3RD_DOWN_SECURITY: High target share (proxy)
    """

    # Get game-level data for variance analysis
    df = query(f"""
        SELECT
            player_id, name, team, position,
            fantasy_points, targets, carries, receptions
        FROM gamelog
        WHERE season IN {SEASONS}
          AND position IN ('RB', 'WR', 'TE')
          AND (targets > 0 OR carries > 0)
    """)

    # Aggregate with variance
    agg = df.groupby(['player_id', 'name', 'team', 'position']).agg({
        'fantasy_points': ['mean', 'std', 'count', 'max', 'min'],
        'targets': 'mean',
        'carries': 'mean',
        'receptions': 'mean',
    }).reset_index()

    agg.columns = ['player_id', 'name', 'team', 'position',
                   'avg_fp', 'std_fp', 'games', 'max_fp', 'min_fp',
                   'avg_targets', 'avg_carries', 'avg_receptions']

    agg = agg[agg['games'] >= MIN_GAMES]

    # Calculate CV and pass ratio
    agg['cv'] = agg['std_fp'] / agg['avg_fp'].replace(0, 1)
    agg['pass_ratio'] = agg['avg_targets'] / (agg['avg_targets'] + agg['avg_carries'] + 0.01)
    agg['boom_potential'] = agg['max_fp'] / agg['avg_fp'].replace(0, 1)

    tags = []
    median_cv = agg['cv'].median()

    for _, row in agg.iterrows():
        gs_tags = []

        # GARBAGE_TIME_HERO: High CV, pass-heavy
        if row['cv'] > median_cv * 1.2 and row['pass_ratio'] > 0.6:
            gs_tags.append('GARBAGE_TIME_HERO')

        # FRONT_RUNNER_BULLY: Low CV, rush-heavy
        if row['cv'] < median_cv * 0.8 and row['pass_ratio'] < 0.4 and row['position'] == 'RB':
            gs_tags.append('FRONT_RUNNER_BULLY')

        # 3RD_DOWN_SECURITY: High target rate (PPR floor)
        if row['avg_receptions'] > 4.0:
            gs_tags.append('3RD_DOWN_SECURITY')

        # 2_MINUTE_DRILL_WEAPON: High boom potential
        if row['boom_potential'] > 2.5:
            gs_tags.append('2_MINUTE_DRILL_WEAPON')

        tags.append({
            'player_id': row['player_id'],
            'name': row['name'],
            'team': row['team'],
            'position': row['position'],
            'game_state_tags': gs_tags if gs_tags else ['GAME_STATE_NEUTRAL'],
            'metrics': {
                'cv': round(row['cv'], 3) if pd.notna(row['cv']) else 0,
                'pass_ratio': round(row['pass_ratio'], 3),
                'boom_potential': round(row['boom_potential'], 2) if pd.notna(row['boom_potential']) else 1,
            }
        })

    return pd.DataFrame(tags)


# =============================================================================
# CATEGORY 5: EFFICIENCY TRAPS & GEMS
# =============================================================================

def tag_efficiency():
    """
    Efficiency classifications:
    - EMPTY_CALORIE_HOG: High hog rate, low FP/target
    - HYPER_EFFICIENT_GHOST: Low hog rate, elite FP/target
    - TD_REGRESSION_CANDIDATE: High TDs, low yards
    - UNLUCKY_VOLUME_BUY: High air yards, low FP
    """

    df = query(f"""
        SELECT
            player_id, name, team, position,
            SUM(hog_rate) / COUNT(*) as avg_hog_rate,
            SUM(fantasy_points) as total_fp,
            SUM(targets) as total_targets,
            SUM(receiving_touchdowns) + SUM(rushing_touchdowns) as total_tds,
            SUM(receiving_yards) + SUM(rushing_yards) as total_yards,
            SUM(air_yards) as total_air_yards,
            COUNT(*) as games
        FROM gamelog
        WHERE season IN {SEASONS}
          AND position IN ('RB', 'WR', 'TE')
          AND (targets > 0 OR carries > 0)
        GROUP BY player_id, name, team, position
        HAVING COUNT(*) >= {MIN_GAMES}
    """)

    # Calculate efficiency metrics
    df['fp_per_target'] = df['total_fp'] / df['total_targets'].replace(0, 1)
    df['yards_per_td'] = df['total_yards'] / df['total_tds'].replace(0, 1)
    df['air_yard_conversion'] = df['total_fp'] / df['total_air_yards'].replace(0, 1)

    # Calculate league benchmarks
    qualified = df[df['total_targets'] >= 30]
    median_fp_tgt = qualified['fp_per_target'].median()
    median_hog = qualified['avg_hog_rate'].median()
    top_fp_tgt = qualified['fp_per_target'].quantile(0.80)

    tags = []
    for _, row in df.iterrows():
        eff_tags = []

        hog = row['avg_hog_rate'] or 0
        fp_tgt = row['fp_per_target'] or 0

        # EMPTY_CALORIE_HOG: High hog, low efficiency
        if hog > median_hog * 1.3 and fp_tgt < median_fp_tgt * 0.85:
            eff_tags.append('EMPTY_CALORIE_HOG')

        # HYPER_EFFICIENT_GHOST: Low hog, elite efficiency
        if hog < median_hog * 0.7 and fp_tgt > top_fp_tgt:
            eff_tags.append('HYPER_EFFICIENT_GHOST')

        # TD_REGRESSION_CANDIDATE: High TDs relative to yards
        if row['total_tds'] >= 4:
            yards_per_td = row['yards_per_td'] or 999
            if yards_per_td < 80:  # Less than 80 yards per TD = unsustainable
                eff_tags.append('TD_REGRESSION_CANDIDATE')

        # UNLUCKY_VOLUME_BUY: High air yards, low conversion
        if row['total_air_yards'] > 200:
            air_conversion = row['air_yard_conversion'] or 0
            if air_conversion < 0.5:  # Less than 0.5 FP per air yard
                eff_tags.append('UNLUCKY_VOLUME_BUY')

        tags.append({
            'player_id': row['player_id'],
            'name': row['name'],
            'team': row['team'],
            'position': row['position'],
            'efficiency_tags': eff_tags if eff_tags else ['EFFICIENCY_NEUTRAL'],
            'metrics': {
                'hog_rate': round(hog, 3),
                'fp_per_target': round(fp_tgt, 2),
                'yards_per_td': round(row['yards_per_td'], 1) if pd.notna(row['yards_per_td']) and row['yards_per_td'] < 999 else None,
                'total_tds': int(row['total_tds'] or 0),
            }
        })

    return pd.DataFrame(tags)


# =============================================================================
# MAIN: COMBINE ALL TAGS
# =============================================================================

def generate_advanced_tags():
    """Generate all 30+ archetype tags."""

    print("=" * 80)
    print("ADVANCED ARCHETYPE TAGGER - 30+ Player Classifications")
    print("=" * 80)

    print("\n1. Generating QB DNA tags...")
    qb_df = tag_qb_dna()
    print(f"   {len(qb_df)} QBs tagged")

    print("\n2. Generating Receiver Usage tags...")
    wr_df = tag_receiver_usage()
    print(f"   {len(wr_df)} WR/TEs tagged")

    print("\n3. Generating RB Role tags...")
    rb_df = tag_rb_roles()
    print(f"   {len(rb_df)} RBs tagged")

    print("\n4. Generating Game State tags...")
    gs_df = tag_game_state()
    print(f"   {len(gs_df)} players tagged")

    print("\n5. Generating Efficiency tags...")
    eff_df = tag_efficiency()
    print(f"   {len(eff_df)} players tagged")

    # Merge all tags
    print("\n6. Merging all tags...")

    # Combine all dataframes
    all_tags = {}

    for df, tag_col in [(qb_df, 'qb_tags'), (wr_df, 'receiver_tags'),
                        (rb_df, 'rb_tags'), (gs_df, 'game_state_tags'),
                        (eff_df, 'efficiency_tags')]:
        for _, row in df.iterrows():
            pid = row['player_id']
            if pid not in all_tags:
                all_tags[pid] = {
                    'player_id': pid,
                    'name': row['name'],
                    'team': row['team'],
                    'position': row['position'],
                    'all_tags': [],
                    'metrics': {}
                }

            # Add tags
            tags = row.get(tag_col, [])
            if isinstance(tags, list):
                all_tags[pid]['all_tags'].extend(tags)

            # Merge metrics
            metrics = row.get('metrics', {})
            if isinstance(metrics, dict):
                all_tags[pid]['metrics'].update(metrics)

    # Remove duplicate/neutral tags
    for pid in all_tags:
        tags = all_tags[pid]['all_tags']
        # Remove neutral tags if other tags exist
        non_neutral = [t for t in tags if 'NEUTRAL' not in t and 'STANDARD' not in t and 'COMMITTEE' not in t]
        if non_neutral:
            all_tags[pid]['all_tags'] = list(set(non_neutral))
        else:
            all_tags[pid]['all_tags'] = ['UNTAGGED']

    return list(all_tags.values())


def display_tag_summary(players):
    """Display summary of tags."""

    print("\n" + "=" * 80)
    print("TAG DISTRIBUTION SUMMARY")
    print("=" * 80)

    # Count each tag
    tag_counts = {}
    for p in players:
        for tag in p['all_tags']:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    # Sort by count
    sorted_tags = sorted(tag_counts.items(), key=lambda x: -x[1])

    print(f"\n{'Tag':35s} {'Count':>6s}")
    print("-" * 45)
    for tag, count in sorted_tags[:25]:
        print(f"{tag:35s} {count:>6d}")

    # Show example players for key tags
    key_tags = ['DEEP_BALL_ADDICT', 'SCRAMBLE_DRILL_CREATOR', 'AIR_YARDS_HOG',
                'CONTESTED_CATCH_SAVANT', 'GREEN_ZONE_GRINDER', 'PASS_GAME_SPECIALIST',
                'GARBAGE_TIME_HERO', 'EMPTY_CALORIE_HOG', 'HYPER_EFFICIENT_GHOST',
                'UNLUCKY_VOLUME_BUY']

    print("\n" + "=" * 80)
    print("EXAMPLE PLAYERS BY KEY ARCHETYPE")
    print("=" * 80)

    for tag in key_tags:
        examples = [p for p in players if tag in p['all_tags']][:4]
        if examples:
            print(f"\n{tag}:")
            for p in examples:
                print(f"  - {p['name']} ({p['position']}, {p['team']})")


def export_tags(players):
    """Export tags to JSON."""

    output = {
        'metadata': {
            'seasons': list(SEASONS),
            'min_games': MIN_GAMES,
            'total_players': len(players),
            'generated': pd.Timestamp.now().strftime('%Y-%m-%d'),
        },
        'tag_definitions': {
            # QB Tags
            'DEEP_BALL_ADDICT': {
                'description': '>15% deep ball attempts',
                'correlation': 'High variance stack; needs 2 WRs to hit value',
                'sim_rule': {'variance_mult': 1.25}
            },
            'RED_ZONE_VULTURE_QB': {
                'description': 'Steals RB TDs at goal line',
                'correlation': 'Negative with own RB1',
                'sim_rule': {'rb_td_correlation': -0.15}
            },
            'SCRAMBLE_DRILL_CREATOR': {
                'description': 'High rush share, extends plays',
                'correlation': 'Positive with YAC_CREATOR WRs',
                'sim_rule': {'yac_wr_correlation': 0.15}
            },
            # Receiver Tags
            'MAN_COVERAGE_ALPHA': {
                'description': 'Wins vs Man coverage',
                'correlation': 'Boom weeks vs MAN_HEAVY defenses',
                'sim_rule': {'vs_man_heavy': {'percentile_shift': 15}}
            },
            'AIR_YARDS_HOG': {
                'description': '>35% team air yards',
                'correlation': 'Tournament winner, correlates with DEEP_BALL_ADDICT',
                'sim_rule': {'variance_mult': 1.20}
            },
            'CONTESTED_CATCH_SAVANT': {
                'description': '>50% contested catch rate',
                'correlation': 'Bad QB proof, safe floor',
                'sim_rule': {'floor_mult': 1.15}
            },
            'END_ZONE_OAK_TREE': {
                'description': 'Heavy end zone targets',
                'correlation': 'TD dependent, stack in high totals',
                'sim_rule': {'td_correlation': 0.25}
            },
            # RB Tags
            'GREEN_ZONE_GRINDER': {
                'description': 'Goal line specialist, low snap share',
                'correlation': 'TD or bust, negative with QB pass TDs',
                'sim_rule': {'td_dependency': 0.70}
            },
            'PASS_GAME_SPECIALIST': {
                'description': 'High route participation',
                'correlation': 'Negative game script god',
                'sim_rule': {'trailing_mult': 1.30}
            },
            # Efficiency Tags
            'EMPTY_CALORIE_HOG': {
                'description': 'High volume, low efficiency',
                'correlation': 'Needs massive volume to hit',
                'sim_rule': {'floor_mult': 0.85, 'volume_sensitivity': 1.5}
            },
            'HYPER_EFFICIENT_GHOST': {
                'description': 'Low volume, elite efficiency',
                'correlation': 'Volume increase = breakout',
                'sim_rule': {'upside_mult': 1.40}
            },
            'UNLUCKY_VOLUME_BUY': {
                'description': 'High air yards, low conversion',
                'correlation': 'Buy low, due for positive regression',
                'sim_rule': {'regression_mult': 1.20}
            },
        },
        'players': players,
    }

    output_path = Path(__file__).parent.parent / 'advanced_player_tags.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n\nExported to: {output_path}")


def main():
    """Generate all advanced tags."""

    players = generate_advanced_tags()
    display_tag_summary(players)
    export_tags(players)

    return players


if __name__ == '__main__':
    players = main()
