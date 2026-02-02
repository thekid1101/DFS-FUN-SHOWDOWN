"""
Named configuration profiles for portfolio optimization.

Four profiles mapping directly to run_portfolio_optimization() kwargs,
ranging from aggressive (max upside) to defensive (max robustness).

All profiles use greedy_marginal selection, t-copula (df=5), and
shortlist=2000. DRO and covariance gamma scale with defensiveness.
"""

import json
from typing import Dict, Any, Optional
from copy import deepcopy


# =============================================================================
# Profile Definitions
# =============================================================================

PROFILES: Dict[str, Dict[str, Any]] = {
    'aggressive': {
        'selection_method': 'greedy_marginal',
        'copula_type': 't',
        'copula_df': 5,
        'shortlist_size': 2000,
        'field_sharpness': 5.0,
        'ownership_power': 0.5,
        'dro_perturbations': 0,
        'dro_scale': 0.10,
        'dro_hhi_scale': 0.15,
        'dro_aggregation': 'mean',
        'covariance_gamma': 0.05,
    },
    'balanced': {
        'selection_method': 'greedy_marginal',
        'copula_type': 't',
        'copula_df': 5,
        'shortlist_size': 2000,
        'field_sharpness': 5.5,
        'ownership_power': 0.5,
        'dro_perturbations': 50,
        'dro_scale': 0.10,
        'dro_hhi_scale': 0.15,
        'dro_aggregation': 'mean',
        'covariance_gamma': 0.10,
    },
    'robust': {
        'selection_method': 'greedy_marginal',
        'copula_type': 't',
        'copula_df': 5,
        'shortlist_size': 2000,
        'field_sharpness': 5.5,
        'ownership_power': 0.5,
        'dro_perturbations': 50,
        'dro_scale': 0.10,
        'dro_hhi_scale': 0.15,
        'dro_aggregation': 'mean_minus_std',
        'covariance_gamma': 0.15,
    },
    'defensive': {
        'selection_method': 'greedy_marginal',
        'copula_type': 't',
        'copula_df': 5,
        'shortlist_size': 2000,
        'field_sharpness': 6.0,
        'ownership_power': 0.5,
        'dro_perturbations': 50,
        'dro_scale': 0.10,
        'dro_hhi_scale': 0.15,
        'dro_aggregation': 'cvar',
        'dro_cvar_alpha': 0.20,
        'covariance_gamma': 0.20,
    },
}

PROFILE_NAMES = list(PROFILES.keys())


def get_profile(name: str) -> Dict[str, Any]:
    """
    Get a named profile as a dict of pipeline kwargs.

    Args:
        name: Profile name (aggressive, balanced, robust, defensive)

    Returns:
        Dict of kwargs suitable for run_portfolio_optimization()

    Raises:
        ValueError: If profile name is not recognized
    """
    if name not in PROFILES:
        raise ValueError(
            f"Unknown profile '{name}'. Available: {', '.join(PROFILE_NAMES)}"
        )
    return deepcopy(PROFILES[name])


def apply_profile_overrides(
    profile: Dict[str, Any],
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge explicit overrides into a profile.

    Overrides take precedence over profile values. Only non-None
    override values are applied.

    Args:
        profile: Base profile kwargs
        overrides: Dict of override values (None values are skipped)

    Returns:
        Merged dict with overrides applied
    """
    merged = deepcopy(profile)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged


def load_profile_from_json(path: str) -> Dict[str, Any]:
    """
    Load a custom profile from a JSON file.

    The JSON should contain pipeline kwargs directly:
    {
        "selection_method": "greedy_marginal",
        "copula_type": "t",
        "copula_df": 5,
        "field_sharpness": 5.5,
        ...
    }

    Args:
        path: Path to JSON file

    Returns:
        Dict of kwargs suitable for run_portfolio_optimization()
    """
    with open(path, 'r') as f:
        return json.load(f)
