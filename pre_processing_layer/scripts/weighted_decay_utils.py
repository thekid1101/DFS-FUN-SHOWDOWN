"""
Weighted Decay Utilities - 3-Tier Weighting System

Handles recency vs. sample size tradeoff for player tagging:
- TALENT: High stability (routes_won, yards_created, contested_catches)
- ROLE: Medium stability (aDOT, slot_rate, target_share)
- LUCK: Low stability (TD rate, efficiency metrics)

Also handles coaching changes for ROLE-based tags.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Current season (adjust this each year)
CURRENT_SEASON = 2025

# Weight configurations by tag type
WEIGHTS = {
    'TALENT': {
        2025: 0.40,
        2024: 0.30,
        2023: 0.20,
        'older': 0.10  # 2020-2022 combined
    },
    'ROLE': {
        2025: 0.60,
        2024: 0.30,
        2023: 0.10,
        'older': 0.00
    },
    'LUCK': {
        2025: 0.80,
        2024: 0.20,
        2023: 0.00,
        'older': 0.00
    }
}

# Coaching changes that trigger ROLE tag resets (team -> last year of old scheme)
# 2026 offseason has tied record 10 HC changes
# Data before reset year is invalid for ROLE tags (scheme-dependent metrics)
COACHING_CHANGES_2025 = {
    # team: last_valid_year (data before this year is invalid for ROLE tags)

    # === NEW HEAD COACHES 2026 ===
    'PIT': 2025,   # Mike McCarthy replaces Tomlin after 19 years
    'BAL': 2025,   # Jesse Minter replaces Harbaugh - major scheme change
    'TEN': 2025,   # Robert Saleh replaces Callahan - defensive HC
    'MIA': 2025,   # Jeff Hafley replaces McDaniel - scheme overhaul likely
    'ATL': 2025,   # Kevin Stefanski replaces Morris - new offensive system
    'NYG': 2025,   # John Harbaugh replaces Daboll - complete reset
    'BUF': 2025,   # Joe Brady replaces McDermott - new era

    # === STILL OPEN / RECENT FIRES ===
    'ARI': 2025,   # Fired Gannon after 3-14 - HC TBD
    'LV': 2025,    # Fired Pete Carroll after 1 year - HC TBD
    'CLE': 2025,   # Stefanski left for ATL - HC TBD

    # === 2025 MID-SEASON CHANGES (carried over) ===
    'CHI': 2024,   # Fired Eberflus mid-2024, Thomas Brown interim
    'NYJ': 2024,   # Fired Saleh mid-2024 (now at TEN)
    'NO': 2024,    # Fired Dennis Allen mid-2024
    'JAX': 2024,   # Pederson out after 2024

    # === OC CHANGES (scheme impacts) ===
    'DAL': 2025,   # McCarthy gone, new HC/OC system
    'DET': 2025,   # Ben Johnson left - new OC
    'LAC': 2025,   # Minter left for BAL - new DC, possible offensive tweaks
}


def get_season_weight(season: int, tag_type: str, team: str = None) -> float:
    """
    Get the weight for a specific season based on tag type.

    Args:
        season: The NFL season year
        tag_type: 'TALENT', 'ROLE', or 'LUCK'
        team: Optional team code for coaching change check

    Returns:
        Weight between 0 and 1
    """
    weights = WEIGHTS.get(tag_type, WEIGHTS['TALENT'])

    # For ROLE tags, check for coaching changes
    if tag_type == 'ROLE' and team in COACHING_CHANGES_2025:
        reset_year = COACHING_CHANGES_2025[team]
        if season < reset_year:
            return 0.0  # Ignore pre-change data

    if season == CURRENT_SEASON:
        return weights.get(2025, 0.4)
    elif season == CURRENT_SEASON - 1:
        return weights.get(2024, 0.3)
    elif season == CURRENT_SEASON - 2:
        return weights.get(2023, 0.2)
    else:
        return weights.get('older', 0.1)


def add_season_weights(df: pd.DataFrame, tag_type: str, team_col: str = 'team') -> pd.DataFrame:
    """
    Add a 'weight' column to dataframe based on season and tag type.

    Args:
        df: DataFrame with 'season' column
        tag_type: 'TALENT', 'ROLE', or 'LUCK'
        team_col: Column name for team (used for coaching change checks)

    Returns:
        DataFrame with 'weight' column added
    """
    df = df.copy()

    if tag_type == 'ROLE' and team_col in df.columns:
        # Apply team-specific weights for ROLE tags
        df['weight'] = df.apply(
            lambda row: get_season_weight(row['season'], tag_type, row.get(team_col)),
            axis=1
        )
    else:
        # Simple season-based weights
        df['weight'] = df['season'].apply(lambda s: get_season_weight(s, tag_type))

    return df


def calculate_weighted_stat(df: pd.DataFrame, stat_col: str, tag_type: str,
                            group_cols: list = None, team_col: str = 'team',
                            min_weight: float = 0.5) -> pd.DataFrame:
    """
    Calculate weighted average of a stat across seasons.

    Args:
        df: DataFrame with game-level data
        stat_col: Column to calculate weighted average for
        tag_type: 'TALENT', 'ROLE', or 'LUCK'
        group_cols: Columns to group by (default: player_id, name, position, team)
        team_col: Column name for team
        min_weight: Minimum total weight required (filters out players with old data only)

    Returns:
        DataFrame with weighted average stats per player
    """
    if group_cols is None:
        group_cols = ['player_id', 'name', 'position', 'team']

    df = add_season_weights(df, tag_type, team_col)

    # Filter out zero-weight rows
    df = df[df['weight'] > 0].copy()

    # Calculate weighted stats
    df['weighted_stat'] = df[stat_col] * df['weight']

    result = df.groupby(group_cols).agg({
        'weighted_stat': 'sum',
        'weight': 'sum',
        stat_col: ['mean', 'std', 'count'],
        'season': 'max'  # Most recent season
    }).reset_index()

    # Flatten column names
    result.columns = [
        '_'.join(col).strip('_') if isinstance(col, tuple) else col
        for col in result.columns
    ]

    # Calculate weighted average
    result[f'weighted_{stat_col}'] = result['weighted_stat_sum'] / result['weight_sum']

    # Filter by minimum weight (ensures enough recent data)
    result = result[result['weight_sum'] >= min_weight].copy()

    return result


def calculate_weighted_multi_stats(df: pd.DataFrame, stat_cols: list, tag_type: str,
                                   group_cols: list = None, team_col: str = 'team',
                                   min_weight: float = 0.5) -> pd.DataFrame:
    """
    Calculate weighted averages for multiple stats at once.

    Args:
        df: DataFrame with game-level data
        stat_cols: List of columns to calculate weighted averages for
        tag_type: 'TALENT', 'ROLE', or 'LUCK'
        group_cols: Columns to group by
        team_col: Column name for team
        min_weight: Minimum total weight required

    Returns:
        DataFrame with weighted average stats per player
    """
    if group_cols is None:
        group_cols = ['player_id', 'name', 'position', 'team']

    df = add_season_weights(df, tag_type, team_col)

    # Filter out zero-weight rows
    df = df[df['weight'] > 0].copy()

    # Build aggregation dict
    agg_dict = {'weight': 'sum', 'season': 'max'}
    for col in stat_cols:
        df[f'{col}_weighted'] = df[col] * df['weight']
        agg_dict[f'{col}_weighted'] = 'sum'
        agg_dict[col] = 'count'

    result = df.groupby(group_cols).agg(agg_dict).reset_index()

    # Calculate weighted averages
    for col in stat_cols:
        result[f'weighted_{col}'] = result[f'{col}_weighted'] / result['weight']
        result[f'{col}_games'] = result[col]
        result.drop([f'{col}_weighted', col], axis=1, inplace=True)

    # Filter by minimum weight
    result = result[result['weight'] >= min_weight].copy()

    # Rename weight column
    result.rename(columns={'weight': 'total_weight'}, inplace=True)

    return result


def get_tag_type(tag_name: str) -> str:
    """
    Classify a tag into TALENT, ROLE, or LUCK category.

    Returns the appropriate weight tier for a given tag.
    """
    # TALENT tags - stable over time
    talent_tags = [
        'BAD_BALL_ERASER', 'BURN_ARTIST', 'CONTESTED_CATCH_SAVANT',
        'ROUTE_WINNER', 'YAC_MONSTER', 'YARDS_CREATED_GOD',
        'CONTACT_ABSORBER', 'DROP_MACHINE', 'MAN_BEATER', 'ZONE_MERCHANT',
        'TIGHT_WINDOW_GLUE', 'DANGER_PLAY_MAGNET'
    ]

    # ROLE tags - scheme/coaching dependent
    role_tags = [
        'SHOTGUN_SATELLITE', 'PLAY_ACTION_MERCHANT', 'HEAVY_BOX_BULLY',
        'LIGHT_BOX_SLASHER', 'THE_FIRST_READ', 'THE_PANIC_BUTTON',
        'PASS_SNAPS_SPECIALIST', 'CHAIN_MOVER', 'THE_BAILOUT_OPTION',
        '3RD_DOWN_GHOST', 'DEEP_BALL_DEPENDENT', 'SLOT_SAFETY_VALVE',
        'ALPHA_OUTSIDE_WR', 'SECONDARY_SLOT_WR', 'BELLCOW_EARLY_RB',
        'SATELLITE_RB', 'TIMESHARE_RB', 'ELITE_TE', 'BLOCKING_TE'
    ]

    # LUCK tags - regresses to mean quickly
    luck_tags = [
        'TD_REGRESSION_CANDIDATE', 'TD_REGRESSION_BUY', 'GARBAGE_TIME_HERO',
        'FRONT_RUNNER_BULLY', 'END_ZONE_TARGET_HOG', 'GL_STUFF_RISK',
        'VULTURE_VICTIM', 'MONEY_THROW_MERCHANT', 'EMPTY_CALORIES',
        '4TH_QUARTER_CLOSER', 'HYPER_EFFICIENT_GHOST', 'UNLUCKY_VOLUME_BUY'
    ]

    if tag_name in talent_tags:
        return 'TALENT'
    elif tag_name in role_tags:
        return 'ROLE'
    elif tag_name in luck_tags:
        return 'LUCK'
    else:
        return 'ROLE'  # Default to ROLE (medium stability)


def apply_confidence_adjustment(score: float, total_weight: float,
                                games: int, min_games: int = 8) -> float:
    """
    Adjust a tag score based on confidence (sample size + recency).

    Lower confidence = score regresses toward neutral (0.5 or 0).

    Args:
        score: Raw tag score (0 to 1)
        total_weight: Sum of season weights
        games: Number of games in sample
        min_games: Minimum games for full confidence

    Returns:
        Adjusted score
    """
    # Weight confidence (max at 1.0)
    weight_confidence = min(total_weight / 1.0, 1.0)

    # Sample size confidence
    sample_confidence = min(games / min_games, 1.0)

    # Combined confidence
    confidence = (weight_confidence * 0.6 + sample_confidence * 0.4)

    # Regress toward neutral based on confidence
    neutral = 0.0  # For most scores, neutral is 0
    adjusted_score = neutral + (score - neutral) * confidence

    return adjusted_score


# =============================================================================
# SUMMARY FUNCTIONS
# =============================================================================

def print_weight_summary():
    """Print the current weight configuration"""
    print("\n" + "=" * 60)
    print("WEIGHTED DECAY CONFIGURATION")
    print("=" * 60)
    print(f"\nCurrent Season: {CURRENT_SEASON}")

    for tier, weights in WEIGHTS.items():
        print(f"\n{tier} Tags:")
        for year, weight in weights.items():
            print(f"  {year}: {weight:.0%}")

    print(f"\nCoaching Changes (ROLE tag resets):")
    for team, year in COACHING_CHANGES_2025.items():
        print(f"  {team}: Reset to {year}+ only")


if __name__ == "__main__":
    print_weight_summary()
