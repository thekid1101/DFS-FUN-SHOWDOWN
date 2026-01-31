"""
Player Tag Generator for Simulation Conditionals

Mines PlayerProfiler database for conditional probability tags:
1. MAN_BEATER vs ZONE_MERCHANT - Coverage scheme splits
2. FRONT_RUNNER vs GARBAGE_TIME_HERO - Game script sensitivity
3. INEFFICIENT_VOLUME vs HYPER_EFFICIENT - Volume dependency
4. RED_ZONE_DEPENDENT vs BREAKAWAY_THREAT - TD variance tags

These tags modify the base distribution parameters based on matchup/game context.

OUTPUT: player_tags.json - Merge with simulation_config_v3.json
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from nfl_research import query

SEASONS = [2023, 2024, 2025]
MIN_GAMES = 6
MIN_ROUTES = 50  # For coverage splits


def calculate_coverage_tags():
    """
    Tag 1: MAN_BEATER vs ZONE_MERCHANT

    MAN_BEATER: Target rate vs Man > 1.25x vs Zone (volatile, needs accurate QB)
    ZONE_MERCHANT: Target rate vs Zone > 1.25x vs Man (high floor, game manager friendly)
    """

    df = query(f"""
        SELECT
            player_id, name, team, position,
            SUM(routes_vs_man) as total_routes_man,
            SUM(routes_vs_zone) as total_routes_zone,
            SUM(targets_vs_man) as total_targets_man,
            SUM(targets_vs_zone) as total_targets_zone,
            SUM(yards_vs_man) as total_yards_man,
            SUM(yards_vs_zone) as total_yards_zone,
            AVG(fantasy_points) as avg_fp,
            COUNT(*) as games
        FROM gamelog
        WHERE season IN {tuple(SEASONS)}
          AND position IN ('WR', 'TE')
        GROUP BY player_id, name, team, position
        HAVING COUNT(*) >= {MIN_GAMES}
           AND SUM(routes_vs_man) >= {MIN_ROUTES}
           AND SUM(routes_vs_zone) >= {MIN_ROUTES}
    """)

    # Calculate target rate per route
    df['tpr_man'] = df['total_targets_man'] / df['total_routes_man']
    df['tpr_zone'] = df['total_targets_zone'] / df['total_routes_zone']
    df['man_zone_ratio'] = df['tpr_man'] / df['tpr_zone'].replace(0, 0.001)

    # Calculate yards per route
    df['ypr_man'] = df['total_yards_man'] / df['total_routes_man']
    df['ypr_zone'] = df['total_yards_zone'] / df['total_routes_zone']

    tags = []
    for _, row in df.iterrows():
        tag = None
        confidence = None

        if row['man_zone_ratio'] > 1.25:
            tag = 'MAN_BEATER'
            confidence = min(1.0, (row['man_zone_ratio'] - 1.0) / 0.5)
        elif row['man_zone_ratio'] < 0.80:
            tag = 'ZONE_MERCHANT'
            confidence = min(1.0, (1.0 - row['man_zone_ratio']) / 0.4)
        else:
            tag = 'COVERAGE_NEUTRAL'
            confidence = 0.5

        tags.append({
            'player_id': row['player_id'],
            'name': row['name'],
            'team': row['team'],
            'position': row['position'],
            'coverage_tag': tag,
            'coverage_confidence': round(confidence, 2),
            'tpr_man': round(row['tpr_man'], 3),
            'tpr_zone': round(row['tpr_zone'], 3),
            'man_zone_ratio': round(row['man_zone_ratio'], 2),
        })

    return pd.DataFrame(tags)


def calculate_game_script_tags():
    """
    Tag 2: FRONT_RUNNER vs GARBAGE_TIME_HERO

    Uses gamelog variance patterns to estimate script sensitivity.
    FRONT_RUNNER: Production drops >25% when trailing (RBs like Derrick Henry)
    GARBAGE_TIME_HERO: Production jumps >25% when trailing (Slot WRs, passing RBs)
    """

    # Use gamelog approach - analyze variance and pass/rush mix
    return calculate_script_tags_from_gamelog()


def calculate_script_tags_from_gamelog():
    """Fallback: Estimate script sensitivity from gamelog variance patterns."""

    # SQLite doesn't have STDEV, so calculate in pandas
    df = query(f"""
        SELECT
            player_id, name, team, position,
            fantasy_points, targets, carries
        FROM gamelog
        WHERE season IN {tuple(SEASONS)}
          AND position IN ('RB', 'WR', 'TE')
    """)

    # Aggregate with proper std calculation
    agg = df.groupby(['player_id', 'name', 'team', 'position']).agg({
        'fantasy_points': ['mean', 'std', 'count'],
        'targets': 'mean',
        'carries': 'mean'
    }).reset_index()

    agg.columns = ['player_id', 'name', 'team', 'position',
                   'avg_fp', 'std_fp', 'games', 'avg_targets', 'avg_carries']

    # Filter to min games
    df = agg[agg['games'] >= MIN_GAMES].copy()

    # Heuristic: High variance + pass-catching = likely garbage time hero
    # Low variance + rushing = likely front runner
    tags = []
    for _, row in df.iterrows():
        cv = row['std_fp'] / row['avg_fp'] if row['avg_fp'] > 0 else 0
        pass_ratio = row['avg_targets'] / (row['avg_targets'] + row['avg_carries'] + 0.01)

        if row['position'] == 'RB':
            if pass_ratio < 0.3 and cv < 0.5:
                tag = 'FRONT_RUNNER'
                confidence = 0.7
            elif pass_ratio > 0.4 and cv > 0.5:
                tag = 'GARBAGE_TIME_HERO'
                confidence = 0.7
            else:
                tag = 'SCRIPT_NEUTRAL'
                confidence = 0.5
        else:  # WR/TE
            if cv > 0.6:
                tag = 'GARBAGE_TIME_HERO'
                confidence = 0.6
            else:
                tag = 'SCRIPT_NEUTRAL'
                confidence = 0.5

        tags.append({
            'player_id': row['player_id'],
            'name': row['name'],
            'team': row['team'],
            'position': row['position'],
            'script_tag': tag,
            'script_confidence': round(confidence, 2),
            'cv': round(cv, 2),
            'pass_ratio': round(pass_ratio, 2),
        })

    return pd.DataFrame(tags)


def calculate_efficiency_tags():
    """
    Tag 3: INEFFICIENT_VOLUME vs HYPER_EFFICIENT

    INEFFICIENT_VOLUME: High target share but low FP/target (volume dependent)
    HYPER_EFFICIENT: Low target share but high FP/target (boom/bust, volume independent)
    """

    df = query(f"""
        SELECT
            player_id, name, team, position,
            AVG(target_share) as avg_target_share,
            AVG(fantasy_points) as avg_fp,
            SUM(targets) as total_targets,
            SUM(fantasy_points) as total_fp,
            COUNT(*) as games
        FROM gamelog
        WHERE season IN {tuple(SEASONS)}
          AND position IN ('WR', 'TE', 'RB')
          AND targets > 0
        GROUP BY player_id, name, team, position
        HAVING COUNT(*) >= {MIN_GAMES}
           AND SUM(targets) >= 20
    """)

    # Calculate FP per target manually
    df['avg_fp_per_target'] = df['total_fp'] / df['total_targets']

    # Calculate league averages (only for players with meaningful volume)
    qualified = df[df['total_targets'] >= 30]
    league_avg_fp_per_target = qualified['avg_fp_per_target'].median()
    top_25_threshold = qualified['avg_fp_per_target'].quantile(0.75)
    bottom_25_threshold = qualified['avg_fp_per_target'].quantile(0.25)

    # Target share thresholds
    high_share_threshold = qualified['avg_target_share'].quantile(0.75)
    low_share_threshold = qualified['avg_target_share'].quantile(0.40)

    tags = []
    for _, row in df.iterrows():
        target_share = row['avg_target_share'] if pd.notna(row['avg_target_share']) else 0
        fp_per_tgt = row['avg_fp_per_target'] if pd.notna(row['avg_fp_per_target']) else 0

        tag = None
        confidence = None

        # INEFFICIENT_VOLUME: High target share AND below-average FP/Target
        if target_share > high_share_threshold and fp_per_tgt < bottom_25_threshold:
            tag = 'INEFFICIENT_VOLUME'
            confidence = min(1.0, target_share / 0.25)
        # HYPER_EFFICIENT: Lower target share AND top-tier FP/Target
        elif target_share < low_share_threshold and fp_per_tgt > top_25_threshold:
            tag = 'HYPER_EFFICIENT'
            confidence = min(1.0, fp_per_tgt / (top_25_threshold * 1.3))
        # Also tag high volume + high efficiency as "ELITE_VOLUME"
        elif target_share > high_share_threshold and fp_per_tgt > top_25_threshold:
            tag = 'ELITE_VOLUME'
            confidence = 0.9
        else:
            tag = 'EFFICIENCY_NEUTRAL'
            confidence = 0.5

        tags.append({
            'player_id': row['player_id'],
            'name': row['name'],
            'team': row['team'],
            'position': row['position'],
            'efficiency_tag': tag,
            'efficiency_confidence': round(confidence, 2),
            'target_share': round(target_share, 3) if pd.notna(target_share) else 0,
            'fp_per_target': round(fp_per_tgt, 2) if pd.notna(fp_per_tgt) else 0,
            'league_avg_fp_tgt': round(league_avg_fp_per_target, 2),
        })

    return pd.DataFrame(tags)


def calculate_redzone_tags():
    """
    Tag 4: RED_ZONE_DEPENDENT vs BREAKAWAY_THREAT

    RED_ZONE_DEPENDENT: >40% of touches inside 20 (TD variance tied to RZ trips)
    BREAKAWAY_THREAT: >50% of TDs from outside 10 (decoupled from RZ trips)
    """

    df = query(f"""
        SELECT
            player_id, name, team, position,
            SUM(red_zone_touches) as rz_touches,
            SUM(red_zone_targets) as rz_targets,
            SUM(carries_inside_5) as carries_inside_5,
            SUM(carries_inside_10) as carries_inside_10,
            SUM(targets_inside_5) as targets_inside_5,
            SUM(targets_inside_10) as targets_inside_10,
            SUM(rushing_touchdowns) as rush_tds,
            SUM(receiving_touchdowns) as rec_tds,
            SUM(carries) as total_carries,
            SUM(targets) as total_targets,
            SUM(fantasy_points) as total_fp,
            COUNT(*) as games
        FROM gamelog
        WHERE season IN {tuple(SEASONS)}
          AND position IN ('RB', 'WR', 'TE')
        GROUP BY player_id, name, team, position
        HAVING COUNT(*) >= {MIN_GAMES}
    """)

    tags = []
    for _, row in df.iterrows():
        total_touches = (row['total_carries'] or 0) + (row['total_targets'] or 0)
        rz_touches = (row['rz_touches'] or 0)
        total_tds = (row['rush_tds'] or 0) + (row['rec_tds'] or 0)
        inside_10_opps = (row['carries_inside_10'] or 0) + (row['targets_inside_10'] or 0)

        if total_touches == 0:
            continue

        rz_touch_rate = rz_touches / total_touches

        # Estimate TDs from inside 10 vs outside
        # If high inside_10 opps relative to TDs, they're RZ dependent
        if total_tds > 0:
            inside_10_td_rate = min(1.0, inside_10_opps / (total_tds * 2))  # Heuristic
        else:
            inside_10_td_rate = 0.5

        tag = None
        confidence = None

        if rz_touch_rate > 0.40:
            tag = 'RED_ZONE_DEPENDENT'
            confidence = min(1.0, rz_touch_rate / 0.5)
        elif inside_10_td_rate < 0.35 and total_tds >= 3:
            tag = 'BREAKAWAY_THREAT'
            confidence = min(1.0, (0.5 - inside_10_td_rate) / 0.3)
        else:
            tag = 'REDZONE_NEUTRAL'
            confidence = 0.5

        tags.append({
            'player_id': row['player_id'],
            'name': row['name'],
            'team': row['team'],
            'position': row['position'],
            'redzone_tag': tag,
            'redzone_confidence': round(confidence, 2),
            'rz_touch_rate': round(rz_touch_rate, 3),
            'total_tds': int(total_tds),
            'inside_10_opps': int(inside_10_opps),
        })

    return pd.DataFrame(tags)


def generate_all_tags():
    """Generate all conditional tags and merge into single output."""

    print("=" * 80)
    print("PLAYER TAG GENERATOR - Conditional Probability Tags")
    print("=" * 80)

    print("\n1. Calculating COVERAGE tags (MAN_BEATER / ZONE_MERCHANT)...")
    coverage_df = calculate_coverage_tags()
    print(f"   Found {len(coverage_df)} players with coverage splits")

    print("\n2. Calculating GAME SCRIPT tags (FRONT_RUNNER / GARBAGE_TIME_HERO)...")
    script_df = calculate_game_script_tags()
    print(f"   Found {len(script_df)} players with script sensitivity")

    print("\n3. Calculating EFFICIENCY tags (INEFFICIENT_VOLUME / HYPER_EFFICIENT)...")
    efficiency_df = calculate_efficiency_tags()
    print(f"   Found {len(efficiency_df)} players with efficiency tags")

    print("\n4. Calculating RED ZONE tags (RED_ZONE_DEPENDENT / BREAKAWAY_THREAT)...")
    redzone_df = calculate_redzone_tags()
    print(f"   Found {len(redzone_df)} players with red zone tags")

    # Get most recent team for each player
    print("\n5. Getting most recent team assignments...")
    recent_teams = query(f"""
        SELECT player_id, name, team, position
        FROM gamelog
        WHERE season = (SELECT MAX(season) FROM gamelog)
        GROUP BY player_id
        ORDER BY MAX(week) DESC
    """)

    # Merge all tags
    print("\n6. Merging all tags...")

    # Start with recent teams as base
    merged = recent_teams.copy()

    # Merge coverage tags
    if len(coverage_df) > 0:
        cov_cols = ['player_id', 'coverage_tag', 'coverage_confidence', 'tpr_man', 'tpr_zone', 'man_zone_ratio']
        cov_cols = [c for c in cov_cols if c in coverage_df.columns]
        merged = merged.merge(coverage_df[cov_cols], on='player_id', how='left')

    # Merge script tags
    if len(script_df) > 0:
        script_cols = ['player_id', 'script_tag', 'script_confidence']
        script_cols = [c for c in script_cols if c in script_df.columns]
        merged = merged.merge(script_df[script_cols], on='player_id', how='left')

    # Merge efficiency tags
    if len(efficiency_df) > 0:
        eff_cols = ['player_id', 'efficiency_tag', 'efficiency_confidence', 'target_share', 'fp_per_target']
        eff_cols = [c for c in eff_cols if c in efficiency_df.columns]
        merged = merged.merge(efficiency_df[eff_cols], on='player_id', how='left')

    # Merge redzone tags
    if len(redzone_df) > 0:
        rz_cols = ['player_id', 'redzone_tag', 'redzone_confidence', 'rz_touch_rate', 'total_tds']
        rz_cols = [c for c in rz_cols if c in redzone_df.columns]
        merged = merged.merge(redzone_df[rz_cols], on='player_id', how='left')

    # Fill NaN tags with NEUTRAL
    for col in ['coverage_tag', 'script_tag', 'efficiency_tag', 'redzone_tag']:
        if col in merged.columns:
            neutral_name = col.replace('_tag', '').upper() + '_NEUTRAL'
            merged[col] = merged[col].fillna(neutral_name)

    # Filter out rows with missing names
    merged = merged[merged['name'].notna() & (merged['name'] != '')]

    # Filter to skill positions only
    merged = merged[merged['position'].isin(['QB', 'RB', 'WR', 'TE'])]

    return merged


def display_tag_summary(df):
    """Display summary of tag distributions."""

    print("\n" + "=" * 80)
    print("TAG DISTRIBUTION SUMMARY")
    print("=" * 80)

    for tag_col in ['coverage_tag', 'script_tag', 'efficiency_tag', 'redzone_tag']:
        if tag_col not in df.columns:
            continue
        print(f"\n{tag_col.upper()}:")
        counts = df[tag_col].value_counts()
        for tag, count in counts.items():
            pct = count / len(df) * 100
            print(f"  {tag:25s}: {count:4d} ({pct:5.1f}%)")

    # Show example players for key tags
    print("\n" + "=" * 80)
    print("EXAMPLE PLAYERS BY TAG")
    print("=" * 80)

    key_tags = [
        ('coverage_tag', 'MAN_BEATER'),
        ('coverage_tag', 'ZONE_MERCHANT'),
        ('script_tag', 'GARBAGE_TIME_HERO'),
        ('script_tag', 'FRONT_RUNNER'),
        ('efficiency_tag', 'INEFFICIENT_VOLUME'),
        ('efficiency_tag', 'HYPER_EFFICIENT'),
        ('redzone_tag', 'RED_ZONE_DEPENDENT'),
        ('redzone_tag', 'BREAKAWAY_THREAT'),
    ]

    for tag_col, tag_value in key_tags:
        if tag_col not in df.columns:
            continue
        examples = df[df[tag_col] == tag_value].head(5)
        if len(examples) > 0:
            print(f"\n{tag_value}:")
            for _, row in examples.iterrows():
                print(f"  - {row['name']} ({row.get('position', 'UNK')}, {row.get('team', 'UNK')})")


def export_tags(df):
    """Export tags to JSON and CSV."""

    output_dir = Path(__file__).parent.parent

    # JSON export (for simulation engine)
    json_output = {
        'metadata': {
            'seasons': SEASONS,
            'min_games': MIN_GAMES,
            'generated': pd.Timestamp.now().strftime('%Y-%m-%d'),
        },
        'tag_definitions': {
            'coverage_tag': {
                'MAN_BEATER': 'TPR vs Man > 1.25x vs Zone. Volatile, needs accurate QB.',
                'ZONE_MERCHANT': 'TPR vs Zone > 1.25x vs Man. High floor, game manager friendly.',
                'COVERAGE_NEUTRAL': 'Similar production vs both coverages.',
            },
            'script_tag': {
                'FRONT_RUNNER': 'Production drops >25% when trailing. Negative correlation with Opp QB.',
                'GARBAGE_TIME_HERO': 'Production jumps >25% when trailing. Positive correlation with Opp QB.',
                'SCRIPT_NEUTRAL': 'Consistent regardless of game script.',
            },
            'efficiency_tag': {
                'INEFFICIENT_VOLUME': 'High hog rate, low FP/target. High sensitivity to team play count.',
                'HYPER_EFFICIENT': 'Low hog rate, high FP/target. Boom/bust independent of volume.',
                'EFFICIENCY_NEUTRAL': 'Average efficiency relative to volume.',
            },
            'redzone_tag': {
                'RED_ZONE_DEPENDENT': '>40% touches in RZ. Correlation = 1.0 with team RZ trips.',
                'BREAKAWAY_THREAT': '>50% TDs from outside 10. Decoupled from RZ trips.',
                'REDZONE_NEUTRAL': 'Standard TD distribution.',
            },
        },
        'simulation_rules': {
            'MAN_BEATER': {'correlation_mod': {'POCKET_QB': 0.10}, 'variance_mult': 1.15},
            'ZONE_MERCHANT': {'correlation_mod': {'GAME_MANAGER_QB': 0.10}, 'variance_mult': 0.90},
            'GARBAGE_TIME_HERO': {'correlation_mod': {'OPP_QB': 0.15}, 'script_trailing_mult': 1.25},
            'FRONT_RUNNER': {'correlation_mod': {'OPP_QB': -0.10}, 'script_trailing_mult': 0.75},
            'INEFFICIENT_VOLUME': {'play_count_sensitivity': 1.5, 'floor_mult': 0.85},
            'HYPER_EFFICIENT': {'play_count_sensitivity': 0.5, 'variance_mult': 1.20},
            'RED_ZONE_DEPENDENT': {'rz_trip_correlation': 0.90, 'td_variance_mult': 1.30},
            'BREAKAWAY_THREAT': {'rz_trip_correlation': 0.30, 'breakaway_td_rate': 0.50},
        },
        'players': df.to_dict('records'),
    }

    json_path = output_dir / 'player_tags.json'
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2, default=str)

    # CSV export (for manual review)
    csv_path = output_dir / 'player_tags.csv'
    df.to_csv(csv_path, index=False)

    print(f"\n\nExported to:")
    print(f"  - {json_path}")
    print(f"  - {csv_path}")

    return json_output


def main():
    """Generate all player tags."""

    df = generate_all_tags()

    if len(df) == 0:
        print("No players found with sufficient data.")
        return

    display_tag_summary(df)
    output = export_tags(df)

    return output


if __name__ == '__main__':
    output = main()
