"""
Production configuration management.

Provides locked configuration for "hit go" production usage.
A production config captures all optimizer parameters in a single
JSON file, ensuring reproducible runs without needing to remember
CLI flags.

Production config format:
{
    "version": "1.0",
    "source_profile": "balanced",
    "optimizer": { /* all pipeline kwargs */ },
    "contest_overrides": {
        "dk_showdown_milly": { "field_sharpness": 4.8 }
    }
}
"""

import json
from typing import Dict, Any, Optional
from copy import deepcopy

from .profiles import get_profile, PROFILES


PRODUCTION_CONFIG_VERSION = "1.0"


def _validate_version(config: Dict[str, Any]) -> None:
    """Validate production config version."""
    version = config.get('version', '1.0')
    if version != PRODUCTION_CONFIG_VERSION:
        raise ValueError(
            f"Unsupported production config version '{version}'. "
            f"Expected '{PRODUCTION_CONFIG_VERSION}'."
        )


def load_production_config(path: str) -> Dict[str, Any]:
    """
    Load a production config from JSON file.

    Returns the optimizer kwargs dict, ready to unpack into
    run_portfolio_optimization().

    Args:
        path: Path to production config JSON

    Returns:
        Dict of optimizer kwargs

    Raises:
        ValueError: If config version is unsupported
    """
    with open(path, 'r') as f:
        config = json.load(f)

    _validate_version(config)
    return deepcopy(config.get('optimizer', {}))


def load_production_config_with_overrides(
    path: str,
    contest_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load production config with optional contest-specific overrides applied.

    Args:
        path: Path to production config JSON
        contest_name: If provided, apply matching contest_overrides

    Returns:
        Dict of optimizer kwargs with any contest overrides merged

    Raises:
        ValueError: If config version is unsupported
    """
    with open(path, 'r') as f:
        config = json.load(f)

    _validate_version(config)
    optimizer = deepcopy(config.get('optimizer', {}))

    if contest_name and 'contest_overrides' in config:
        overrides = config['contest_overrides'].get(contest_name, {})
        optimizer.update(overrides)

    return optimizer


def build_production_config(
    profile: str,
    overrides: Optional[Dict[str, Any]] = None,
    contest_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Build a production config dict from a profile + optional overrides.

    Args:
        profile: Source profile name (aggressive, balanced, robust, defensive)
        overrides: Optional kwargs to override profile values
        contest_overrides: Optional per-contest overrides keyed by contest name

    Returns:
        Production config dict suitable for save_production_config()
    """
    optimizer = get_profile(profile)

    if overrides:
        for key, value in overrides.items():
            if value is not None:
                optimizer[key] = value

    config = {
        'version': PRODUCTION_CONFIG_VERSION,
        'source_profile': profile,
        'optimizer': optimizer,
    }

    if contest_overrides:
        config['contest_overrides'] = contest_overrides

    return config


def save_production_config(config: Dict[str, Any], path: str) -> None:
    """
    Save a production config dict to JSON file.

    Args:
        config: Production config dict (from build_production_config)
        path: Output file path
    """
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
