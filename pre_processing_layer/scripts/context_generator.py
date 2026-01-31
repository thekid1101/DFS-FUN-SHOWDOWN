"""
Context Tag Generator - Defensive & QB Profiles

Mines PBP data to classify:
1. DEFENSIVE_SCHEME: Man/Zone rates and Funnel types per team
2. QB_PRESSURE_PROFILE: Sensitivity to pass rush (COLLAPSIBLE, ICE_VEINS, CHAOS_GRENADE)
3. BETA_PIVOT: Negative correlation pairs for tournament leverage

OUTPUT: context_tags.json - Load alongside player_tags.json in simulation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from nfl_research import query

SEASONS = (2023, 2024, 2025)
MIN_PLAYS = 100


def generate_defensive_profiles():
    """
    Gem #2: Defense Funnel Tags

    PASS_FUNNEL: Top 10 Run D, Bottom 10 Pass D -> Boost Opp WR Volume
    RUN_FUNNEL: Bad Run D, Elite Secondary -> Boost Opp RB Efficiency
    MAN_HEAVY: >40% Man Coverage -> Activates MAN_BEATER WRs
    ZONE_HEAVY: >60% Zone Coverage -> Activates ZONE_MERCHANT receivers
    """

    # Get defensive stats by team
    df = query(f"""
        SELECT
            defense,
            COUNT(*) as total_plays,
            SUM(CASE WHEN pass = 1 THEN 1 ELSE 0 END) as pass_plays,
            SUM(CASE WHEN pass = 0 AND play_type NOT IN ('penalty', 'timeout') THEN 1 ELSE 0 END) as rush_plays,
            AVG(CASE WHEN pass = 1 THEN yards_gained END) as ypa_allowed,
            AVG(CASE WHEN pass = 0 THEN yards_gained END) as ypc_allowed,
            SUM(CASE WHEN coverage_type IN ('1', 'C', '2M') THEN 1 ELSE 0 END) as man_plays,
            SUM(CASE WHEN coverage_type IN ('2', '3', '4', '6') THEN 1 ELSE 0 END) as zone_plays
        FROM pbp
        WHERE season IN {SEASONS}
          AND defense IS NOT NULL
          AND play_type IS NOT NULL
        GROUP BY defense
        HAVING COUNT(*) >= {MIN_PLAYS}
    """)

    if len(df) == 0:
        print("   No defensive data found, using gamelog fallback...")
        return generate_defensive_profiles_gamelog()

    # Calculate rates
    df['man_rate'] = df['man_plays'] / (df['man_plays'] + df['zone_plays'] + 0.001)

    # Calculate league medians
    median_ypa = df['ypa_allowed'].median()
    median_ypc = df['ypc_allowed'].median()
    median_man = df['man_rate'].median()

    # Rank teams
    df['ypa_rank'] = df['ypa_allowed'].rank(ascending=False)  # Higher = worse pass D
    df['ypc_rank'] = df['ypc_allowed'].rank(ascending=False)  # Higher = worse run D
    n_teams = len(df)

    def_profiles = {}

    for _, row in df.iterrows():
        team = row['defense']
        tags = []
        metrics = {}

        # Coverage scheme tags
        if pd.notna(row['man_rate']):
            if row['man_rate'] > 0.40:
                tags.append("MAN_HEAVY")
            elif row['man_rate'] < 0.25:
                tags.append("ZONE_HEAVY")

        # Funnel tags (based on relative rankings)
        # PASS_FUNNEL: Good run D (low rank), bad pass D (high rank)
        if pd.notna(row['ypc_rank']) and pd.notna(row['ypa_rank']):
            if row['ypc_rank'] <= n_teams * 0.33 and row['ypa_rank'] >= n_teams * 0.67:
                tags.append("PASS_FUNNEL")
            # RUN_FUNNEL: Bad run D (high rank), good pass D (low rank)
            elif row['ypc_rank'] >= n_teams * 0.67 and row['ypa_rank'] <= n_teams * 0.33:
                tags.append("RUN_FUNNEL")

        # Havoc tag (for QB pressure context)
        # This would need sack/pressure data - approximate with YPA
        if pd.notna(row['ypa_allowed']) and pd.notna(median_ypa) and row['ypa_allowed'] < median_ypa * 0.85:
            tags.append("HAVOC_DEFENSE")

        metrics = {
            'ypa_allowed': round(row['ypa_allowed'], 2) if pd.notna(row['ypa_allowed']) else None,
            'ypc_allowed': round(row['ypc_allowed'], 2) if pd.notna(row['ypc_allowed']) else None,
            'man_rate': round(row['man_rate'], 2) if pd.notna(row['man_rate']) else None,
            'ypa_rank': int(row['ypa_rank']) if pd.notna(row['ypa_rank']) else None,
            'ypc_rank': int(row['ypc_rank']) if pd.notna(row['ypc_rank']) else None,
        }

        def_profiles[team] = {
            'tags': tags if tags else ['NEUTRAL'],
            'metrics': metrics
        }

    return def_profiles


def generate_defensive_profiles_gamelog():
    """Fallback: Use gamelog allowed stats if PBP lacks coverage data."""

    df = query(f"""
        SELECT
            opponent as defense,
            AVG(CASE WHEN position = 'QB' THEN fantasy_points END) as qb_fp_allowed,
            AVG(CASE WHEN position = 'RB' THEN fantasy_points END) as rb_fp_allowed,
            AVG(CASE WHEN position = 'WR' THEN fantasy_points END) as wr_fp_allowed,
            AVG(CASE WHEN position = 'TE' THEN fantasy_points END) as te_fp_allowed,
            COUNT(DISTINCT CASE WHEN position = 'QB' THEN player_id END) as qb_games
        FROM gamelog
        WHERE season IN {SEASONS}
        GROUP BY opponent
        HAVING COUNT(*) >= 50
    """)

    median_rb = df['rb_fp_allowed'].median()
    median_wr = df['wr_fp_allowed'].median()

    def_profiles = {}

    for _, row in df.iterrows():
        team = row['defense']
        tags = []

        # Simple funnel logic based on points allowed
        if row['rb_fp_allowed'] < median_rb * 0.9 and row['wr_fp_allowed'] > median_wr * 1.1:
            tags.append("PASS_FUNNEL")
        elif row['rb_fp_allowed'] > median_rb * 1.1 and row['wr_fp_allowed'] < median_wr * 0.9:
            tags.append("RUN_FUNNEL")

        def_profiles[team] = {
            'tags': tags if tags else ['NEUTRAL'],
            'metrics': {
                'rb_fp_allowed': round(row['rb_fp_allowed'], 1),
                'wr_fp_allowed': round(row['wr_fp_allowed'], 1),
            }
        }

    return def_profiles


def generate_qb_pressure_profiles():
    """
    Gem #1: QB Pressure Sensitivity Index

    COLLAPSIBLE: Efficiency drops significantly under pressure (cap at 75th percentile)
    ICE_VEINS: Maintains efficiency under pressure (ignore opponent pressure rate)
    CHAOS_GRENADE: Mobile QB - rushing increases under pressure (boost rush correlation)

    Uses gamelog sacks/hurries data since PBP pressure column is sparse.
    """

    # Use gamelog data for QB pressure profiles
    raw = query(f"""
        SELECT
            player_id, name, team, fantasy_points,
            sacks_taken, hurries, passing_yards, rushing_yards,
            carries, passing_touchdowns, interceptions
        FROM gamelog
        WHERE season IN {SEASONS}
          AND position = 'QB'
          AND pass_attempts > 10
    """)

    # Aggregate with proper std calculation
    df = raw.groupby(['player_id', 'name', 'team']).agg({
        'fantasy_points': ['mean', 'std', 'count'],
        'sacks_taken': 'mean',
        'hurries': 'mean',
        'passing_yards': 'mean',
        'rushing_yards': 'mean',
        'carries': 'mean',
        'passing_touchdowns': 'sum',
        'interceptions': 'sum',
    }).reset_index()

    df.columns = ['player_id', 'name', 'team', 'avg_fp', 'std_fp', 'games',
                  'avg_sacks', 'avg_hurries', 'avg_pass_yds', 'avg_rush_yds',
                  'avg_rush_att', 'total_pass_tds', 'total_ints']

    df = df[df['games'] >= 6]

    if len(df) == 0:
        print("   No QB data found...")
        return {}

    # Calculate pressure metrics
    df['pressure_rate'] = (df['avg_sacks'] + df['avg_hurries']) / 40  # Approximate per-dropback
    df['td_int_ratio'] = df['total_pass_tds'] / (df['total_ints'] + 0.1)
    df['rush_share'] = df['avg_rush_yds'] / (df['avg_pass_yds'] + df['avg_rush_yds'] + 0.1)
    df['cv'] = df['std_fp'] / df['avg_fp']

    # Calculate league medians
    median_pressure = df['pressure_rate'].median()
    median_cv = df['cv'].median()
    median_rush = df['rush_share'].median()

    qb_profiles = {}

    for _, row in df.iterrows():
        name = row['name']
        pressure_rate = row['pressure_rate'] if pd.notna(row['pressure_rate']) else 0
        cv = row['cv'] if pd.notna(row['cv']) else 0.5
        rush_share = row['rush_share'] if pd.notna(row['rush_share']) else 0
        td_int = row['td_int_ratio'] if pd.notna(row['td_int_ratio']) else 1

        tag = None
        confidence = 0.5

        # CHAOS_GRENADE: Mobile QB with high rush share
        if rush_share > 0.15:
            tag = "CHAOS_GRENADE"
            confidence = min(1.0, rush_share / 0.25)
        # COLLAPSIBLE: High variance + high pressure faced + poor TD/INT
        elif cv > median_cv * 1.2 and pressure_rate > median_pressure and td_int < 2.0:
            tag = "COLLAPSIBLE"
            confidence = min(1.0, cv / (median_cv * 1.5))
        # ICE_VEINS: Low variance + good TD/INT ratio despite pressure
        elif cv < median_cv * 0.85 and td_int > 3.0:
            tag = "ICE_VEINS"
            confidence = min(1.0, td_int / 5.0)
        else:
            tag = "PRESSURE_NEUTRAL"
            confidence = 0.5

        qb_profiles[name] = {
            'tag': tag,
            'team': row['team'],
            'confidence': round(confidence, 2),
            'metrics': {
                'avg_fp': round(row['avg_fp'], 1),
                'cv': round(cv, 2),
                'pressure_rate': round(pressure_rate, 3),
                'rush_share': round(rush_share, 3),
                'td_int_ratio': round(td_int, 2),
            }
        }

    return qb_profiles


def generate_beta_pivots():
    """
    Gem #3: Leverage Correlation - BETA_PIVOT pairs

    Find WR pairs on same team with negative correlation (<-0.15).
    When ALPHA hits <25th percentile, BETA should hit 75th percentile.
    Creates tournament leverage lineups.
    """

    # Get weekly fantasy points for WRs
    df = query(f"""
        SELECT
            player_id, name, team, season, week, fantasy_points
        FROM gamelog
        WHERE season IN {SEASONS}
          AND position = 'WR'
          AND fantasy_points IS NOT NULL
        ORDER BY team, season, week
    """)

    # Find same-team WR pairs
    pivots = []

    for (team, season), group in df.groupby(['team', 'season']):
        # Pivot to get players as columns
        pivot = group.pivot_table(
            index='week',
            columns='name',
            values='fantasy_points'
        )

        if len(pivot.columns) < 2:
            continue

        # Calculate correlations
        corr_matrix = pivot.corr()

        # Find negative correlations
        for i, wr1 in enumerate(corr_matrix.columns):
            for wr2 in corr_matrix.columns[i+1:]:
                corr = corr_matrix.loc[wr1, wr2]
                if pd.notna(corr) and corr < -0.15:
                    # Determine Alpha vs Beta by average FP
                    wr1_avg = pivot[wr1].mean()
                    wr2_avg = pivot[wr2].mean()

                    if wr1_avg > wr2_avg:
                        alpha, beta = wr1, wr2
                    else:
                        alpha, beta = wr2, wr1

                    pivots.append({
                        'team': team,
                        'season': season,
                        'alpha': alpha,
                        'beta': beta,
                        'correlation': round(corr, 3),
                        'alpha_avg_fp': round(max(wr1_avg, wr2_avg), 1),
                        'beta_avg_fp': round(min(wr1_avg, wr2_avg), 1),
                    })

    # Keep most recent season's pivots per team
    pivot_df = pd.DataFrame(pivots)
    if len(pivot_df) > 0:
        pivot_df = pivot_df.sort_values(['team', 'season'], ascending=[True, False])
        pivot_df = pivot_df.drop_duplicates(subset=['team', 'alpha'], keep='first')

    return pivot_df.to_dict('records') if len(pivot_df) > 0 else []


def generate_context_tags():
    """Main function to generate all context tags."""

    print("=" * 80)
    print("CONTEXT TAG GENERATOR - Defensive & QB Profiles")
    print("=" * 80)

    print("\n1. Generating Defensive Profiles (Funnel Tags)...")
    def_profiles = generate_defensive_profiles()
    print(f"   Found {len(def_profiles)} defensive profiles")

    # Count funnel types
    pass_funnels = sum(1 for p in def_profiles.values() if 'PASS_FUNNEL' in p['tags'])
    run_funnels = sum(1 for p in def_profiles.values() if 'RUN_FUNNEL' in p['tags'])
    man_heavy = sum(1 for p in def_profiles.values() if 'MAN_HEAVY' in p['tags'])
    zone_heavy = sum(1 for p in def_profiles.values() if 'ZONE_HEAVY' in p['tags'])
    print(f"   PASS_FUNNEL: {pass_funnels}, RUN_FUNNEL: {run_funnels}")
    print(f"   MAN_HEAVY: {man_heavy}, ZONE_HEAVY: {zone_heavy}")

    print("\n2. Generating QB Pressure Profiles...")
    qb_profiles = generate_qb_pressure_profiles()
    print(f"   Found {len(qb_profiles)} QB profiles")

    # Count pressure types
    collapsible = sum(1 for p in qb_profiles.values() if p['tag'] == 'COLLAPSIBLE')
    ice_veins = sum(1 for p in qb_profiles.values() if p['tag'] == 'ICE_VEINS')
    print(f"   COLLAPSIBLE: {collapsible}, ICE_VEINS: {ice_veins}")

    print("\n3. Generating Beta Pivot Pairs (Tournament Leverage)...")
    beta_pivots = generate_beta_pivots()
    print(f"   Found {len(beta_pivots)} negative correlation pairs")

    # Build output
    output = {
        'metadata': {
            'seasons': list(SEASONS),
            'min_plays': MIN_PLAYS,
            'generated': pd.Timestamp.now().strftime('%Y-%m-%d'),
        },
        'simulation_rules': {
            'defensive_tags': {
                'PASS_FUNNEL': {'wr_volume_mult': 1.15, 'rb_volume_mult': 0.90},
                'RUN_FUNNEL': {'rb_efficiency_mult': 1.10, 'wr_volume_mult': 0.95},
                'MAN_HEAVY': {'activates': 'MAN_BEATER', 'deactivates': 'ZONE_MERCHANT'},
                'ZONE_HEAVY': {'activates': 'ZONE_MERCHANT', 'deactivates': 'MAN_BEATER'},
                'HAVOC_DEFENSE': {'qb_ceiling_cap': 0.75, 'affects': 'COLLAPSIBLE'},
            },
            'qb_tags': {
                'COLLAPSIBLE': {
                    'vs_havoc': {'percentile_cap': 75, 'variance_mult': 0.80},
                    'correlation_mod': {'SATELLITE_RB': 0.15}  # Checkdowns increase
                },
                'ICE_VEINS': {
                    'vs_havoc': {'ignore': True},
                    'correlation_mod': {'DEEP_THREAT_WR': 0.10}
                },
                'CHAOS_GRENADE': {
                    'vs_havoc': {'variance_mult': 1.30, 'rush_boost': 1.20},
                    'correlation_mod': {'BELLCOW_RB': -0.10}
                },
            },
            'beta_pivot': {
                'trigger': 'alpha_below_25th_percentile',
                'action': 'beta_to_75th_percentile',
                'use_case': 'tournament_leverage'
            }
        },
        'defensive_profiles': def_profiles,
        'qb_profiles': qb_profiles,
        'beta_pivots': beta_pivots,
    }

    # Export
    output_path = Path(__file__).parent.parent / 'context_tags.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n\nExported to: {output_path}")

    # Display examples
    print("\n" + "=" * 80)
    print("EXAMPLE OUTPUTS")
    print("=" * 80)

    print("\nDefensive Profiles (Sample):")
    for team, profile in list(def_profiles.items())[:6]:
        tags = ', '.join(profile['tags'])
        print(f"  {team:4s}: {tags}")

    print("\nQB Pressure Profiles (Sample):")
    for name, profile in list(qb_profiles.items())[:8]:
        cv = profile['metrics'].get('cv', 0)
        rush = profile['metrics'].get('rush_share', 0)
        print(f"  {name:20s}: {profile['tag']:18s} (CV: {cv:.2f}, Rush: {rush:.2f})")

    print("\nBeta Pivots (Tournament Leverage):")
    for pivot in beta_pivots[:5]:
        print(f"  {pivot['team']}: {pivot['alpha']} (Alpha) vs {pivot['beta']} (Beta) | corr: {pivot['correlation']}")

    return output


if __name__ == '__main__':
    output = generate_context_tags()
