"""
Configuration management for DFS Showdown GTO Portfolio Builder.

Contest presets, field generation defaults, and JSON loading utilities.
"""

import json
from typing import Dict, Any
from pathlib import Path

from .types import ContestStructure, PayoutTier, FieldGenConfig


# =============================================================================
# Contest Presets
# =============================================================================

CONTEST_PRESETS: Dict[str, ContestStructure] = {
    'dk_showdown_5': ContestStructure(
        name="DraftKings Showdown $5",
        entry_fee=5.0,
        total_entries=5000,
        your_entries=150,
        payout_tiers=[
            PayoutTier(1, 1, 1000.0),
            PayoutTier(2, 2, 500.0),
            PayoutTier(3, 3, 300.0),
            PayoutTier(4, 5, 150.0),
            PayoutTier(6, 10, 75.0),
            PayoutTier(11, 20, 40.0),
            PayoutTier(21, 35, 25.0),
            PayoutTier(36, 60, 15.0),
            PayoutTier(61, 100, 12.0),
            PayoutTier(101, 175, 10.0),
            PayoutTier(176, 300, 8.0),
            PayoutTier(301, 500, 7.0),
            PayoutTier(501, 750, 6.50),
            PayoutTier(751, 1000, 6.00),
        ]
    ),

    'dk_showdown_20': ContestStructure(
        name="DraftKings Showdown $20",
        entry_fee=20.0,
        total_entries=3000,
        your_entries=150,
        payout_tiers=[
            PayoutTier(1, 1, 5000.0),
            PayoutTier(2, 2, 2500.0),
            PayoutTier(3, 3, 1500.0),
            PayoutTier(4, 5, 750.0),
            PayoutTier(6, 10, 300.0),
            PayoutTier(11, 20, 150.0),
            PayoutTier(21, 35, 80.0),
            PayoutTier(36, 60, 50.0),
            PayoutTier(61, 100, 40.0),
            PayoutTier(101, 175, 32.0),
            PayoutTier(176, 300, 28.0),
            PayoutTier(301, 500, 25.0),
            PayoutTier(501, 600, 22.0),
        ]
    ),

    'dk_showdown_milly': ContestStructure(
        name="DraftKings Showdown Milly Maker",
        entry_fee=5.0,
        total_entries=200000,
        your_entries=150,
        payout_tiers=[
            PayoutTier(1, 1, 100000.0),
            PayoutTier(2, 2, 25000.0),
            PayoutTier(3, 3, 15000.0),
            PayoutTier(4, 5, 7500.0),
            PayoutTier(6, 10, 3000.0),
            PayoutTier(11, 25, 1000.0),
            PayoutTier(26, 50, 500.0),
            PayoutTier(51, 100, 250.0),
            PayoutTier(101, 250, 100.0),
            PayoutTier(251, 500, 50.0),
            PayoutTier(501, 1000, 30.0),
            PayoutTier(1001, 2500, 20.0),
            PayoutTier(2501, 5000, 15.0),
            PayoutTier(5001, 10000, 12.0),
            PayoutTier(10001, 20000, 10.0),
            PayoutTier(20001, 35000, 8.0),
            PayoutTier(35001, 50000, 7.0),
        ]
    ),

    'dk_showdown_small': ContestStructure(
        name="DraftKings Showdown Small Field",
        entry_fee=5.0,
        total_entries=500,
        your_entries=20,
        payout_tiers=[
            PayoutTier(1, 1, 200.0),
            PayoutTier(2, 2, 100.0),
            PayoutTier(3, 5, 50.0),
            PayoutTier(6, 10, 25.0),
            PayoutTier(11, 25, 15.0),
            PayoutTier(26, 50, 10.0),
            PayoutTier(51, 100, 7.50),
        ]
    ),
}


# =============================================================================
# Default Field Generation Config
# =============================================================================

DEFAULT_FIELD_CONFIG = FieldGenConfig(
    temperature=1.0,
    salary_utilization_mean=0.98,
    salary_utilization_std=0.03,
    qb_pair_rate=0.80,
    bring_back_rate=0.65,
    dst_rate_multiplier=0.7,
    kicker_rate_multiplier=0.8,
    split_priors={
        '5-1': 0.08,
        '4-2': 0.32,
        '3-3': 0.35,
        '2-4': 0.20,
        '1-5': 0.05
    }
)


# =============================================================================
# JSON Loading Utilities
# =============================================================================

def load_contest_from_json(path: str) -> ContestStructure:
    """
    Load contest configuration from JSON file.

    Expected format:
    {
        "name": "Contest Name",
        "entry_fee": 5.0,
        "total_entries": 5000,
        "your_entries": 150,
        "payout_tiers": [
            {"start_rank": 1, "end_rank": 1, "payout": 1000.0},
            {"start_rank": 2, "end_rank": 5, "payout": 100.0},
            ...
        ]
    }
    """
    with open(path, 'r') as f:
        data = json.load(f)

    payout_tiers = [
        PayoutTier(
            start_rank=tier['start_rank'],
            end_rank=tier['end_rank'],
            payout=tier['payout']
        )
        for tier in data['payout_tiers']
    ]

    return ContestStructure(
        name=data['name'],
        entry_fee=data['entry_fee'],
        total_entries=data['total_entries'],
        your_entries=data['your_entries'],
        payout_tiers=payout_tiers
    )


def save_contest_to_json(contest: ContestStructure, path: str):
    """Save contest configuration to JSON file."""
    data = {
        'name': contest.name,
        'entry_fee': contest.entry_fee,
        'total_entries': contest.total_entries,
        'your_entries': contest.your_entries,
        'payout_tiers': [
            {
                'start_rank': tier.start_rank,
                'end_rank': tier.end_rank,
                'payout': tier.payout
            }
            for tier in contest.payout_tiers
        ]
    }

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_field_config_from_json(path: str) -> FieldGenConfig:
    """
    Load field generation config from JSON file.

    Expected format:
    {
        "temperature": 1.0,
        "qb_pair_rate": 0.80,
        "bring_back_rate": 0.65,
        "dst_rate_multiplier": 0.7,
        "kicker_rate_multiplier": 0.8,
        "split_priors": {
            "5-1": 0.08,
            "4-2": 0.32,
            "3-3": 0.35,
            "2-4": 0.20,
            "1-5": 0.05
        }
    }
    """
    with open(path, 'r') as f:
        data = json.load(f)

    return FieldGenConfig(
        temperature=data.get('temperature', 1.0),
        salary_utilization_mean=data.get('salary_utilization_mean', 0.98),
        salary_utilization_std=data.get('salary_utilization_std', 0.03),
        qb_pair_rate=data.get('qb_pair_rate', 0.80),
        bring_back_rate=data.get('bring_back_rate', 0.65),
        dst_rate_multiplier=data.get('dst_rate_multiplier', 0.7),
        kicker_rate_multiplier=data.get('kicker_rate_multiplier', 0.8),
        split_priors=data.get('split_priors', DEFAULT_FIELD_CONFIG.split_priors)
    )


def create_sample_contest_json(path: str = 'contest_sample.json'):
    """Create a sample contest JSON file for reference."""
    sample = {
        'name': 'Sample DraftKings Showdown',
        'entry_fee': 5.0,
        'total_entries': 5000,
        'your_entries': 150,
        'payout_tiers': [
            {'start_rank': 1, 'end_rank': 1, 'payout': 1000.0},
            {'start_rank': 2, 'end_rank': 5, 'payout': 100.0},
            {'start_rank': 6, 'end_rank': 20, 'payout': 50.0},
            {'start_rank': 21, 'end_rank': 50, 'payout': 25.0},
            {'start_rank': 51, 'end_rank': 100, 'payout': 15.0},
            {'start_rank': 101, 'end_rank': 250, 'payout': 10.0},
            {'start_rank': 251, 'end_rank': 500, 'payout': 7.50},
            {'start_rank': 501, 'end_rank': 1000, 'payout': 6.00},
        ]
    }

    with open(path, 'w') as f:
        json.dump(sample, f, indent=2)

    print(f"Sample contest JSON saved to {path}")
