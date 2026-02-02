"""
CSV loader for DraftKings Showdown projections.

Parses projections CSV and produces separate CPT/FLEX player pools.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

from ..types import ShowdownPlayer, ProjectionsData


# Column names for percentiles in the CSV
PERCENTILE_COLUMNS = {
    25: 'dk_25_percentile',
    50: 'dk_50_percentile',
    75: 'dk_75_percentile',
    85: 'dk_85_percentile',
    95: 'dk_95_percentile',
    99: 'dk_99_percentile',
}


def _player_key(name: str, team: str) -> str:
    """Build a stable key for identifying the same player across CPT/FLEX rows."""
    return f"{name}||{team}"


def _player_key_series(df: pd.DataFrame) -> pd.Series:
    """Vectorized player key series for a projections dataframe."""
    if 'Team' in df.columns:
        team = df['Team'].fillna('').astype(str)
    else:
        team = pd.Series([''] * len(df), index=df.index)
    return df['Name'].astype(str) + "||" + team


def load_projections(
    csv_path: str,
    min_projection: float = 0.0,
    salary_cap: int = 50000
) -> ProjectionsData:
    """
    Load projections from DraftKings Showdown CSV.

    The CSV contains both CPT and FLEX entries for each player.
    CPT entries have 1.5x salary compared to FLEX.

    Args:
        csv_path: Path to the projections CSV
        min_projection: Minimum median projection (dk_50_percentile) to include a player.
            Players below this threshold are removed from both CPT and FLEX pools.
        salary_cap: Salary cap for reference (default DK Showdown $50k)

    Returns:
        ProjectionsData with separate CPT and FLEX pools
    """
    df = pd.read_csv(csv_path)

    # Filter out rows without percentile data
    df = _filter_valid_players(df)

    # Filter by minimum projection threshold (applies to both CPT and FLEX rows)
    if min_projection > 0.0:
        df = _filter_by_min_projection(df, min_projection)

    # Separate CPT and FLEX entries
    cpt_rows, flex_rows = _separate_cpt_flex(df)

    # Build player objects
    cpt_players = _build_players(cpt_rows, is_cpt=True)
    flex_players = _build_players(flex_rows, is_cpt=False)

    # Link CPT to FLEX by player key (name + team)
    cpt_to_flex_map = _link_cpt_to_flex(cpt_players, flex_players)

    # Update CPT players with their FLEX links
    for cpt_idx, flex_idx in cpt_to_flex_map.items():
        cpt_players[cpt_idx].flex_player_idx = flex_idx

    # Get unique teams
    teams = list(df['Team'].dropna().unique())

    return ProjectionsData(
        cpt_players=cpt_players,
        flex_players=flex_players,
        cpt_to_flex_map=cpt_to_flex_map,
        teams=teams
    )


def _filter_valid_players(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to players with valid percentile projections."""
    # Check if primary percentile columns exist and have values
    required_cols = list(PERCENTILE_COLUMNS.values())
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            "Missing required percentile columns in CSV: "
            + ", ".join(missing_cols)
        )

    # Keep rows where at least the median projection exists
    mask = df['dk_50_percentile'].notna() & (df['dk_50_percentile'] > 0)

    filtered = df[mask].copy()

    if len(filtered) == 0:
        raise ValueError("No valid players found in CSV (all percentiles empty)")

    return filtered


def _filter_by_min_projection(df: pd.DataFrame, min_projection: float) -> pd.DataFrame:
    """
    Filter out players whose median projection is below the threshold.

    Removes all rows (CPT and FLEX) for players below the threshold,
    so both entries for the same player are dropped together.
    """
    # Find keys of players meeting the threshold (any row for that player qualifies)
    player_keys = _player_key_series(df)
    qualified_keys = set(
        player_keys[df['dk_50_percentile'] >= min_projection].unique()
    )

    filtered = df[player_keys.isin(qualified_keys)].copy()

    n_removed = len(player_keys.unique()) - len(qualified_keys)
    if n_removed > 0:
        logger.info(
            "min_projection=%.1f: removed %d players, kept %d",
            min_projection, n_removed, len(qualified_keys)
        )

    if len(filtered) == 0:
        raise ValueError(
            f"No players remain after min_projection filter ({min_projection}). "
            f"Lower the threshold or check your projections."
        )

    return filtered


def _separate_cpt_flex(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separate CPT and FLEX entries.

    In DK Showdown, CPT entries have 1.5x salary of FLEX entries.
    We identify by grouping by player key (name + team) and checking salary ratios.
    """
    # Group by player key (name + team) to find pairs
    grouped = df.groupby(_player_key_series(df))

    cpt_indices = []
    flex_indices = []

    for _, group in grouped:
        if len(group) == 1:
            # Only one entry - determine by salary threshold
            # High salary (>= typical CPT threshold) = CPT
            row = group.iloc[0]
            # If salary > 7500, likely CPT (since min FLEX is typically low)
            # Better heuristic: check if there's a pattern in the data
            # For now, use salary threshold
            if row['Salary'] >= 7500:
                cpt_indices.append(group.index[0])
            else:
                flex_indices.append(group.index[0])
        elif len(group) == 2:
            # Two entries - higher salary is CPT
            salaries = group['Salary'].values
            indices = group.index.tolist()
            if salaries[0] > salaries[1]:
                cpt_indices.append(indices[0])
                flex_indices.append(indices[1])
            else:
                cpt_indices.append(indices[1])
                flex_indices.append(indices[0])
        else:
            # More than 2 entries for same name - take highest and lowest salary
            sorted_group = group.sort_values('Salary', ascending=False)
            cpt_indices.append(sorted_group.index[0])
            flex_indices.append(sorted_group.index[-1])

    cpt_df = df.loc[cpt_indices].reset_index(drop=True)
    flex_df = df.loc[flex_indices].reset_index(drop=True)

    return cpt_df, flex_df


def _build_players(df: pd.DataFrame, is_cpt: bool) -> List[ShowdownPlayer]:
    """Build ShowdownPlayer objects from dataframe rows."""
    players = []

    for _, row in df.iterrows():
        # Extract percentiles
        percentiles = {}
        for pct, col in PERCENTILE_COLUMNS.items():
            if col in row and pd.notna(row[col]):
                percentiles[pct] = float(row[col])
            else:
                percentiles[pct] = 0.0

        # Get standard deviation
        std = float(row.get('dk_std', 0.0)) if pd.notna(row.get('dk_std')) else 0.0

        # Get ownership - prefer Adj Own, fall back to My Own
        ownership = 0.0
        if 'Adj Own' in row and pd.notna(row['Adj Own']):
            ownership = float(row['Adj Own'])
        elif 'My Own' in row and pd.notna(row['My Own']):
            ownership = float(row['My Own'])

        # Get projection - prefer dk_points, fall back to My Proj
        projection = 0.0
        if 'dk_points' in row and pd.notna(row['dk_points']):
            projection = float(row['dk_points'])
        elif 'My Proj' in row and pd.notna(row['My Proj']):
            projection = float(row['My Proj'])
        elif 50 in percentiles:
            projection = percentiles[50]

        player = ShowdownPlayer(
            id=str(row['DFS ID']),
            name=str(row['Name']),
            team=str(row['Team']) if pd.notna(row.get('Team')) else '',
            position=str(row['Pos']) if pd.notna(row.get('Pos')) else '',
            salary=int(row['Salary']),
            is_cpt=is_cpt,
            percentiles=percentiles,
            std=std,
            ownership=ownership,
            projection=projection,
            flex_player_idx=None
        )
        players.append(player)

    return players


def _link_cpt_to_flex(
    cpt_players: List[ShowdownPlayer],
    flex_players: List[ShowdownPlayer]
) -> Dict[int, int]:
    """
    Link CPT players to their corresponding FLEX entries.

    Matches by player key (name + team).

    Returns:
        Dict mapping cpt_idx -> flex_idx
    """
    # Build player key -> flex_idx lookup
    flex_by_key: Dict[str, int] = {}
    for idx, player in enumerate(flex_players):
        flex_by_key[_player_key(player.name, player.team)] = idx

    # Link CPT to FLEX
    cpt_to_flex: Dict[int, int] = {}
    for cpt_idx, cpt_player in enumerate(cpt_players):
        key = _player_key(cpt_player.name, cpt_player.team)
        if key in flex_by_key:
            cpt_to_flex[cpt_idx] = flex_by_key[key]

    return cpt_to_flex


def get_player_index_by_name(
    players: List[ShowdownPlayer],
    name: str
) -> Optional[int]:
    """Find player index by name."""
    for idx, player in enumerate(players):
        if player.name == name:
            return idx
    return None


def get_team_players(
    players: List[ShowdownPlayer],
    team: str
) -> List[int]:
    """Get indices of players on a specific team."""
    return [idx for idx, p in enumerate(players) if p.team == team]


def get_position_players(
    players: List[ShowdownPlayer],
    position: str
) -> List[int]:
    """Get indices of players at a specific position."""
    return [idx for idx, p in enumerate(players) if p.position == position]
