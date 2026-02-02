"""
DRO (Distributionally Robust Optimization) with two-layer perturbations.

v5 improvements over v4:
  - Layer 1: CPT ownership via Dirichlet simplex (alpha floored at 0.01)
             FLEX ownership via three-factor correlated noise model
             (global + team + role + idiosyncratic)
  - Layer 2: Condensation factor via HHI-space perturbation
             (perturb observable HHI, invert to sharpness via calibration curve)
  - K_greedy=5 representative scenarios via rank-normalized clustering
             with forced boundary scenarios
  - Union shortlist using roster overlap as fast correlation proxy
  - Aggregation: mean (default), CVaR, mean_minus_std (no lambda blending)
"""

import json
import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy.cluster.vq import kmeans2
import logging

from ..types import (
    ShowdownPlayer, LineupArrays, ContestStructure, FieldGenConfig
)
from ..field.generator import (
    generate_field, generate_field_simulated, compute_field_lineup_probabilities
)
from ..ev.approx import (
    compute_approx_lineup_evs,
    compute_approx_lineup_evs_resampled,
    compute_lineup_probabilities,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Layer 1: Ownership Perturbation
# =============================================================================

def perturb_cpt_ownership(
    cpt_players: List[ShowdownPlayer],
    concentration: float,
    rng: np.random.Generator
) -> List[ShowdownPlayer]:
    """
    Perturb CPT ownership via Dirichlet simplex sampling.

    CPT captaining is a distinct bet — perturbed as a distribution over
    the simplex rather than independently per player.

    Alpha parameters are floored at 0.01 to prevent Dirichlet crash
    on zero-ownership CPT players (e.g., K, DST rarely captained).

    Args:
        cpt_players: CPT player pool
        concentration: Controls spread of Dirichlet (higher = tighter to prior)
        rng: NumPy random generator

    Returns:
        New list of ShowdownPlayer with perturbed CPT ownership
    """
    n = len(cpt_players)
    cpt_own = np.array([max(p.ownership, 0.01) for p in cpt_players], dtype=np.float64)
    cpt_probs = cpt_own / cpt_own.sum()

    # Dirichlet alpha: probability * concentration, floored at 0.01
    alpha = np.maximum(cpt_probs * concentration, 0.01)
    perturbed_probs = rng.dirichlet(alpha)

    # Convert back to ownership scale (sum to ~100%)
    total_own = sum(p.ownership for p in cpt_players)
    perturbed_own = perturbed_probs * max(total_own, 1.0)

    perturbed = []
    for i, p in enumerate(cpt_players):
        new_own = float(np.clip(perturbed_own[i], 0.01, 99.0))
        new_p = ShowdownPlayer(
            id=p.id, name=p.name, team=p.team, position=p.position,
            salary=p.salary, is_cpt=p.is_cpt, percentiles=p.percentiles,
            std=p.std, ownership=new_own, projection=p.projection,
            flex_player_idx=p.flex_player_idx
        )
        perturbed.append(new_p)

    return perturbed


def perturb_flex_ownership(
    flex_players: List[ShowdownPlayer],
    scale: float,
    rng: np.random.Generator
) -> List[ShowdownPlayer]:
    """
    Perturb FLEX ownership via three-factor correlated noise model.

    Three-factor noise model:
    - Global factor: affects all players (macro ownership shift)
    - Team factor: same-team players move together (narrative-driven)
    - Role factor: same-role players move together across teams (positional trends)
    - Idiosyncratic: individual player noise

    Variance partition: global=20%, team=40%, role=20%, idio=20%
    (Ownership errors are mostly team-narrative driven)

    Applied in logit space to preserve positivity and create natural asymmetry.

    Args:
        flex_players: FLEX player pool
        scale: Overall noise scale
        rng: NumPy random generator

    Returns:
        New list of ShowdownPlayer with perturbed FLEX ownership
    """
    n = len(flex_players)

    # Extract team and role IDs
    team_ids = [p.team for p in flex_players]
    role_ids = [p.position for p in flex_players]

    # Generate noise components
    global_noise = rng.normal(0, scale * 0.2)
    team_noise = {t: rng.normal(0, scale * 0.4) for t in set(team_ids)}
    role_noise = {r: rng.normal(0, scale * 0.2) for r in set(role_ids)}
    idio_noise = rng.normal(0, scale * 0.2, n)

    total_noise = np.array([
        global_noise + team_noise[team_ids[i]] + role_noise[role_ids[i]] + idio_noise[i]
        for i in range(n)
    ])

    # Apply in logit space
    flex_own = np.array([max(p.ownership, 0.01) for p in flex_players], dtype=np.float64)
    flex_own_frac = np.clip(flex_own / 100.0, 0.001, 0.999)
    logits = np.log(flex_own_frac / (1 - flex_own_frac))
    perturbed_logits = logits + total_noise
    perturbed_frac = 1.0 / (1.0 + np.exp(-perturbed_logits))
    perturbed_own = perturbed_frac * 100.0

    perturbed = []
    for i, p in enumerate(flex_players):
        new_own = float(np.clip(perturbed_own[i], 0.01, 99.0))
        new_p = ShowdownPlayer(
            id=p.id, name=p.name, team=p.team, position=p.position,
            salary=p.salary, is_cpt=p.is_cpt, percentiles=p.percentiles,
            std=p.std, ownership=new_own, projection=p.projection,
            flex_player_idx=p.flex_player_idx
        )
        perturbed.append(new_p)

    return perturbed


def _perturb_ownership_legacy(
    players: List[ShowdownPlayer],
    scale: float,
    rng: np.random.Generator
) -> List[ShowdownPlayer]:
    """Legacy log-normal perturbation (kept for backwards compatibility)."""
    perturbed = []
    noise = rng.normal(0, scale, size=len(players))
    for i, p in enumerate(players):
        new_own = p.ownership * np.exp(noise[i])
        new_own = float(np.clip(new_own, 0.01, 99.0))
        new_p = ShowdownPlayer(
            id=p.id, name=p.name, team=p.team, position=p.position,
            salary=p.salary, is_cpt=p.is_cpt, percentiles=p.percentiles,
            std=p.std, ownership=new_own, projection=p.projection,
            flex_player_idx=p.flex_player_idx
        )
        perturbed.append(new_p)
    return perturbed


# =============================================================================
# Layer 2: Condensation Factor (HHI-Space Perturbation)
# =============================================================================

def load_hhi_calibration(calibration_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load sharpness-to-HHI calibration curve from JSON.

    Returns (sharpness_array, hhi_array) sorted by sharpness.
    """
    with open(calibration_path, 'r') as f:
        data = json.load(f)

    mapping = data['sharpness_to_hhi']
    sharpness_vals = sorted(float(k) for k in mapping.keys())
    hhi_vals = [mapping[str(s) if str(s) in mapping else str(float(s))]
                for s in sharpness_vals]

    return np.array(sharpness_vals), np.array(hhi_vals)


def perturb_sharpness_via_hhi(
    base_sharpness: float,
    hhi_scale: float,
    calibration_sharpness: np.ndarray,
    calibration_hhi: np.ndarray,
    rng: np.random.Generator
) -> float:
    """
    Perturb field sharpness by working in HHI space.

    Instead of perturbing sharpness directly (arbitrary units, nonlinear),
    perturb in observable HHI space and invert via calibration curve.

    Args:
        base_sharpness: Current sharpness value
        hhi_scale: Scale of log-normal perturbation in HHI space (~0.15)
        calibration_sharpness: Sharpness values from calibration
        calibration_hhi: Corresponding HHI values
        rng: NumPy random generator

    Returns:
        Perturbed sharpness value
    """
    # Interpolate base sharpness -> base HHI
    base_hhi = np.interp(base_sharpness, calibration_sharpness, calibration_hhi)

    # Perturb in HHI space (log-normal to preserve positivity)
    target_hhi = base_hhi * np.exp(rng.normal(0, hhi_scale))

    # Clip to calibrated range
    min_hhi = calibration_hhi.min()
    max_hhi = calibration_hhi.max()
    target_hhi = np.clip(target_hhi, min_hhi, max_hhi)

    # Invert: HHI -> sharpness via calibration curve
    # HHI generally increases with sharpness, so we interpolate inversely
    # Sort by HHI for proper interpolation
    sort_idx = np.argsort(calibration_hhi)
    sorted_hhi = calibration_hhi[sort_idx]
    sorted_sharpness = calibration_sharpness[sort_idx]

    perturbed_sharpness = float(np.interp(target_hhi, sorted_hhi, sorted_sharpness))

    return perturbed_sharpness


# =============================================================================
# Two-Layer Field Generation
# =============================================================================

def generate_perturbed_fields(
    cpt_players: List[ShowdownPlayer],
    flex_players: List[ShowdownPlayer],
    n_field: int,
    n_perturbations: int,
    perturbation_scale: float = 0.10,
    hhi_scale: float = 0.15,
    config: Optional[FieldGenConfig] = None,
    cpt_to_flex_map: Optional[dict] = None,
    salary_cap: int = 50000,
    seed: Optional[int] = None,
    candidate_arrays: Optional[LineupArrays] = None,
    outcomes: Optional[np.ndarray] = None,
    field_sharpness: float = 5.0,
    ownership_power: float = 0.5,
    quality_sims: int = 1000,
    calibration_path: Optional[str] = None,
    cpt_concentration: float = 50.0,
) -> List[Tuple[LineupArrays, np.ndarray]]:
    """
    Generate K perturbed opponent fields with two-layer perturbation.

    Layer 1: Ownership perturbation
      - CPT: Dirichlet simplex sampling
      - FLEX: Three-factor correlated noise (global + team + role + idio)

    Layer 2: Condensation (sharpness) perturbation
      - Perturb in HHI space, invert via calibration curve
      - Requires calibration data from validate_ownership.py

    Args:
        cpt_players: CPT player pool
        flex_players: FLEX player pool
        n_field: Number of field entries per variant
        n_perturbations: Number of perturbed fields to generate
        perturbation_scale: Ownership noise scale (default 0.10)
        hhi_scale: HHI perturbation scale (default 0.15)
        config: Field generation config
        cpt_to_flex_map: CPT->FLEX index map
        salary_cap: Salary cap
        seed: Random seed
        candidate_arrays: Candidate lineups (for simulated field)
        outcomes: Simulation outcomes (for simulated field)
        field_sharpness: Base field sharpness
        ownership_power: Ownership power
        quality_sims: Quality sims for field scoring
        calibration_path: Path to sharpness_hhi_calibration.json (for layer 2)
        cpt_concentration: Dirichlet concentration for CPT (higher = tighter)

    Returns:
        List of (field_arrays, field_counts) tuples, one per perturbation
    """
    rng = np.random.default_rng(seed)
    use_simulated = candidate_arrays is not None and outcomes is not None
    variants = []

    # Load HHI calibration if available
    cal_sharpness = None
    cal_hhi = None
    if calibration_path is not None:
        try:
            cal_sharpness, cal_hhi = load_hhi_calibration(calibration_path)
            logger.info("Loaded HHI calibration curve (%d points)", len(cal_sharpness))
        except Exception as e:
            logger.warning("Could not load HHI calibration: %s. Layer 2 disabled.", e)

    for k in range(n_perturbations):
        # Layer 1: Ownership perturbation
        perturbed_cpt = perturb_cpt_ownership(cpt_players, cpt_concentration, rng)
        perturbed_flex = perturb_flex_ownership(flex_players, perturbation_scale, rng)

        # Layer 2: Sharpness perturbation via HHI space
        perturbed_sharpness = field_sharpness
        if cal_sharpness is not None and cal_hhi is not None and hhi_scale > 0:
            perturbed_sharpness = perturb_sharpness_via_hhi(
                field_sharpness, hhi_scale, cal_sharpness, cal_hhi, rng
            )

        field_seed = rng.integers(0, 2**31)

        if use_simulated:
            field_arrays, field_counts = generate_field_simulated(
                candidate_arrays, outcomes,
                perturbed_cpt, perturbed_flex,
                n_field=n_field,
                config=config,
                field_sharpness=perturbed_sharpness,
                ownership_power=ownership_power,
                quality_sims=quality_sims,
                seed=field_seed
            )
        else:
            field_arrays, field_counts = generate_field(
                perturbed_cpt, perturbed_flex,
                n_field=n_field,
                config=config,
                cpt_to_flex_map=cpt_to_flex_map,
                salary_cap=salary_cap,
                seed=field_seed
            )
        variants.append((field_arrays, field_counts))

        if (k + 1) % 10 == 0 or k == 0:
            logger.info(
                "DRO variant %d/%d: sharpness=%.2f, %d unique, %d total",
                k + 1, n_perturbations, perturbed_sharpness,
                len(field_arrays), int(field_counts.sum())
            )

    return variants


# =============================================================================
# K_greedy=5 Representative Scenario Selection
# =============================================================================

def select_representative_scenarios(
    lineup_evs_matrix: np.ndarray,
    n_representatives: int = 5
) -> List[int]:
    """
    Select K representative field scenarios from the full set.

    Uses rank-normalized EV vectors for clustering (fixes scale dominance)
    and force-includes boundary scenarios (most pessimistic + most differentiated).

    The 5 representatives:
      1. Most pessimistic field (lowest aggregate EV)
      2. Most differentiated field (highest EV variance across lineups)
      3-5. Structural medoids from k-means clustering

    Args:
        lineup_evs_matrix: [K, n_shortlist] EV of each lineup under each field
        n_representatives: Number of representatives to select (default 5)

    Returns:
        List of field indices (into lineup_evs_matrix rows)
    """
    from scipy.stats import rankdata

    K = lineup_evs_matrix.shape[0]

    if K <= n_representatives:
        return list(range(K))

    # Rank-normalize each field's EV vector (fixes scale dominance issue #5)
    ranked = np.apply_along_axis(rankdata, 1, lineup_evs_matrix)  # [K, n_shortlist]

    # Force-include boundary scenarios (issue #6)
    forced = []

    # Most pessimistic field
    agg_evs = lineup_evs_matrix.mean(axis=1)  # [K]
    forced.append(int(np.argmin(agg_evs)))

    # Most differentiated field (highest variance across lineups)
    ev_variance = lineup_evs_matrix.var(axis=1)  # [K]
    max_var_idx = int(np.argmax(ev_variance))
    if max_var_idx not in forced:
        forced.append(max_var_idx)

    # Cluster remaining for structural medoids
    n_cluster = n_representatives - len(forced)
    remaining_mask = np.ones(K, dtype=bool)
    for idx in forced:
        remaining_mask[idx] = False

    if n_cluster > 0 and remaining_mask.sum() >= n_cluster:
        remaining_ranked = ranked[remaining_mask]
        remaining_indices = np.where(remaining_mask)[0]

        try:
            centroids, labels = kmeans2(
                remaining_ranked.astype(np.float64),
                n_cluster,
                minit='points',
                seed=42
            )

            for c in range(n_cluster):
                cluster_mask = labels == c
                if not cluster_mask.any():
                    continue
                centroid = centroids[c]
                dists = np.linalg.norm(remaining_ranked[cluster_mask] - centroid, axis=1)
                local_idx = dists.argmin()
                medoid = remaining_indices[np.where(cluster_mask)[0][local_idx]]
                forced.append(int(medoid))
        except Exception as e:
            # Fallback: evenly spaced indices
            logger.warning("K-means failed (%s), using evenly spaced fallback", e)
            remaining_list = remaining_indices.tolist()
            step = max(1, len(remaining_list) // n_cluster)
            for c in range(n_cluster):
                idx = min(c * step, len(remaining_list) - 1)
                forced.append(int(remaining_list[idx]))

    return forced[:n_representatives]


# =============================================================================
# Union Shortlist with Roster Overlap
# =============================================================================

def compute_max_roster_overlap(
    candidate_arrays: LineupArrays,
    reference_indices: List[int],
    candidate_indices: Optional[List[int]] = None
) -> np.ndarray:
    """
    Compute maximum roster overlap between each candidate and reference set.

    Roster = (cpt_idx, flex_idx[0..4]) — 6 slots total.
    Overlap = count of shared player slots. O(n^2) integer ops.

    Args:
        candidate_arrays: All candidate lineups
        reference_indices: Indices of reference lineups to compare against
        candidate_indices: Optional subset of candidate indices to compute for

    Returns:
        overlap_scores: [n_candidates or n_subset] max overlap with any reference
    """
    if candidate_indices is None:
        candidate_indices = list(range(len(candidate_arrays)))

    n = len(candidate_indices)
    overlap_scores = np.zeros(n, dtype=np.int32)

    # Build reference roster sets
    ref_rosters = []
    for ri in reference_indices:
        roster = set()
        roster.add(('cpt', int(candidate_arrays.cpt_idx[ri])))
        for fi in candidate_arrays.flex_idx[ri]:
            roster.add(('flex', int(fi)))
        ref_rosters.append(roster)

    for ci_local, ci in enumerate(candidate_indices):
        roster = set()
        roster.add(('cpt', int(candidate_arrays.cpt_idx[ci])))
        for fi in candidate_arrays.flex_idx[ci]:
            roster.add(('flex', int(fi)))

        max_overlap = 0
        for ref_roster in ref_rosters:
            overlap = len(roster & ref_roster)
            if overlap > max_overlap:
                max_overlap = overlap
        overlap_scores[ci_local] = max_overlap

    return overlap_scores


def build_union_shortlist(
    approx_evs: np.ndarray,
    robust_evs: np.ndarray,
    candidate_arrays: LineupArrays,
    target_size: int = 500
) -> List[int]:
    """
    Build union shortlist from three sources:
    - Top N by robust EV
    - Top N by nominal (approx) EV
    - Top N by low roster overlap with top-100 robust EV lineups

    Uses roster overlap as fast proxy for profit correlation (issue #7).

    Args:
        approx_evs: [n_candidates] nominal approx EVs
        robust_evs: [n_candidates] robust EVs from DRO
        candidate_arrays: All candidate lineups
        target_size: Target shortlist size

    Returns:
        List of candidate indices for shortlist
    """
    n_per_source = target_size // 3

    robust_top = np.argsort(robust_evs)[-n_per_source:].tolist()
    nominal_top = np.argsort(approx_evs)[-n_per_source:].tolist()

    # Low-overlap candidates
    top100_robust = np.argsort(robust_evs)[-100:].tolist()
    already_selected = set(robust_top + nominal_top)

    # Compute overlap for all candidates not already selected
    remaining = [i for i in range(len(candidate_arrays)) if i not in already_selected]

    if remaining:
        overlap_scores = compute_max_roster_overlap(
            candidate_arrays, top100_robust, remaining
        )
        # Sort by lowest overlap
        sorted_by_overlap = np.argsort(overlap_scores)
        low_overlap = [remaining[i] for i in sorted_by_overlap[:n_per_source]]
    else:
        low_overlap = []

    union = list(set(robust_top + nominal_top + low_overlap))
    return union


# =============================================================================
# Robust EV Computation
# =============================================================================

def compute_robust_approx_evs(
    candidate_arrays: LineupArrays,
    field_variants: List[Tuple[LineupArrays, np.ndarray]],
    outcomes: np.ndarray,
    contest: ContestStructure,
    score_bounds: Tuple[int, int],
    aggregation: str = "mean",
    cvar_alpha: float = 0.20,
    representative_k: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute robust approx EVs across perturbed fields.

    Aggregation methods (no lambda blending — each is standalone):
    - "mean": simple average across perturbations (default, safest)
    - "cvar": mean of worst alpha fraction (alpha=0.20 default)
    - "mean_minus_std": mean - 1*std

    Args:
        candidate_arrays: Candidate lineups to evaluate
        field_variants: List of (field_arrays, field_counts) from perturbed fields
        outcomes: [n_players, n_sims] simulated outcomes
        contest: Contest structure
        score_bounds: (min, max) score bounds
        aggregation: Aggregation method
        cvar_alpha: Alpha for CVaR (fraction of worst scenarios)
        representative_k: If > 0, select K representative scenarios instead of using all

    Returns:
        robust_evs: [n_candidates] robust EV per candidate
        all_evs: [n_perturbations, n_candidates] per-variant EVs
    """
    n_variants = len(field_variants)
    n_candidates = len(candidate_arrays)

    all_evs = np.zeros((n_variants, n_candidates), dtype=np.float64)

    for k, (field_arrays, field_counts) in enumerate(field_variants):
        if (k + 1) % 10 == 0 or k == 0:
            logger.info(
                "Computing approx EVs for DRO variant %d/%d...",
                k + 1, n_variants
            )
        all_evs[k] = compute_approx_lineup_evs(
            candidate_arrays, field_arrays, field_counts,
            outcomes, contest, score_bounds
        )

    # Optionally reduce to K representatives
    if representative_k > 0 and representative_k < n_variants:
        rep_indices = select_representative_scenarios(all_evs, representative_k)
        logger.info(
            "Selected %d representative scenarios: %s",
            len(rep_indices), rep_indices
        )
        all_evs_rep = all_evs[rep_indices]
    else:
        all_evs_rep = all_evs

    robust_evs = _aggregate_evs(all_evs_rep, aggregation, cvar_alpha)

    return robust_evs, all_evs


def compute_robust_approx_evs_resampled(
    candidate_arrays: LineupArrays,
    p_lineup_variants: List[np.ndarray],
    outcomes: np.ndarray,
    contest: ContestStructure,
    score_bounds: Tuple[int, int],
    field_size: int,
    aggregation: str = "mean",
    cvar_alpha: float = 0.20,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute robust approx EVs across perturbed p_lineup variants (resample mode).
    """
    n_variants = len(p_lineup_variants)
    n_candidates = len(candidate_arrays)
    rng = np.random.default_rng(seed)

    all_evs = np.zeros((n_variants, n_candidates), dtype=np.float64)

    for k, p_lineup in enumerate(p_lineup_variants):
        if (k + 1) % 10 == 0 or k == 0:
            logger.info(
                "Computing approx EVs (resampled) for DRO variant %d/%d...",
                k + 1, n_variants
            )
        variant_seed = rng.integers(0, 2**31)
        all_evs[k] = compute_approx_lineup_evs_resampled(
            candidate_arrays, p_lineup, outcomes, contest,
            score_bounds, field_size, seed=variant_seed
        )

    robust_evs = _aggregate_evs(all_evs, aggregation, cvar_alpha)
    return robust_evs, all_evs


def _aggregate_evs(
    all_evs: np.ndarray,
    aggregation: str,
    cvar_alpha: float = 0.20
) -> np.ndarray:
    """
    Aggregate per-variant EVs into robust EVs.

    No lambda blending — each method is standalone:
    - "mean": simple average
    - "cvar": mean of worst alpha fraction
    - "mean_minus_std": mean - std

    Args:
        all_evs: [n_variants, n_candidates] EV per variant
        aggregation: "mean", "cvar", or "mean_minus_std"
        cvar_alpha: Alpha for CVaR computation

    Returns:
        robust_evs: [n_candidates]
    """
    mean_evs = all_evs.mean(axis=0)

    if aggregation == "mean":
        return mean_evs

    elif aggregation == "cvar":
        n_variants = all_evs.shape[0]
        n_worst = max(1, int(np.ceil(n_variants * cvar_alpha)))
        sorted_evs = np.sort(all_evs, axis=0)
        cvar_evs = sorted_evs[:n_worst].mean(axis=0)
        return cvar_evs

    elif aggregation == "mean_minus_std":
        std_evs = all_evs.std(axis=0)
        return mean_evs - std_evs

    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")


# =============================================================================
# Legacy compatibility
# =============================================================================

def generate_perturbed_p_lineups(
    candidate_arrays: LineupArrays,
    cpt_players: List[ShowdownPlayer],
    flex_players: List[ShowdownPlayer],
    n_perturbations: int,
    perturbation_scale: float = 0.10,
    config: Optional[FieldGenConfig] = None,
    seed: Optional[int] = None,
    outcomes: Optional[np.ndarray] = None,
    field_sharpness: float = 5.0,
    ownership_power: float = 0.5,
    quality_sims: int = 1000
) -> List[np.ndarray]:
    """Generate K perturbed lineup probability vectors (for resample_per_sim mode)."""
    rng = np.random.default_rng(seed)
    use_simulated = outcomes is not None
    p_variants = []

    for k in range(n_perturbations):
        perturbed_cpt = perturb_cpt_ownership(cpt_players, 50.0, rng)
        perturbed_flex = perturb_flex_ownership(flex_players, perturbation_scale, rng)

        if use_simulated:
            p_lineup = compute_field_lineup_probabilities(
                candidate_arrays, outcomes,
                perturbed_cpt, perturbed_flex,
                config=config,
                field_sharpness=field_sharpness,
                ownership_power=ownership_power,
                quality_sims=quality_sims,
                seed=rng.integers(0, 2**31)
            )
        else:
            p_lineup = compute_lineup_probabilities(
                candidate_arrays, perturbed_cpt, perturbed_flex, config=config
            )
        p_variants.append(p_lineup)

    return p_variants
