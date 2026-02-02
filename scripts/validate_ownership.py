"""
Ownership model validation diagnostic.

Validates that the field generator produces realistic ownership distributions.
Reports player exposure, CPT vs FLEX distributions, salary utilization,
stack structure, and sharpness-to-HHI calibration curve.

No pipeline changes. Diagnostic script only. Produces calibration data
for use in Phase A1 (DRO HHI-space perturbation).

Usage:
    python -m scripts.validate_ownership projections.csv \
        --contest-preset dk_showdown_5 \
        --n-sims 10000 --seed 42
"""

import argparse
import json
import sys
import os
import numpy as np
from pathlib import Path
from collections import Counter

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from showdown_gto.types import LineupArrays, FieldGenConfig
from showdown_gto.config import CONTEST_PRESETS, load_contest_from_json
from showdown_gto.data.loader import load_projections
from showdown_gto.simulation.engine import simulate_outcomes
from showdown_gto.candidates.enumeration import enumerate_lineups
from showdown_gto.field.generator import generate_field_simulated


def compute_hhi(counts: np.ndarray) -> float:
    """Herfindahl-Hirschman Index: sum of squared shares."""
    total = counts.sum()
    if total == 0:
        return 0.0
    shares = counts / total
    return float(np.sum(shares ** 2))


def player_exposure_report(
    field_arrays: LineupArrays,
    field_counts: np.ndarray,
    cpt_players, flex_players
):
    """Report player exposure in generated field vs projected ownership."""
    total_lineups = int(field_counts.sum())

    # CPT exposure
    cpt_exposure = np.zeros(len(cpt_players), dtype=np.float64)
    for i, count in zip(field_arrays.cpt_idx, field_counts):
        cpt_exposure[i] += count
    cpt_exposure_pct = cpt_exposure / total_lineups * 100

    # FLEX exposure
    flex_exposure = np.zeros(len(flex_players), dtype=np.float64)
    for row_idx in range(len(field_arrays)):
        count = field_counts[row_idx]
        for flex_idx in field_arrays.flex_idx[row_idx]:
            flex_exposure[flex_idx] += count
    flex_exposure_pct = flex_exposure / total_lineups * 100

    print("\n" + "=" * 70)
    print("PLAYER EXPOSURE: CPT POOL")
    print("=" * 70)
    print(f"{'Player':<25} {'Pos':<5} {'Team':<5} {'Proj Own%':>9} {'Field Own%':>10} {'Ratio':>7}")
    print("-" * 70)

    cpt_sorted = sorted(range(len(cpt_players)), key=lambda i: -cpt_exposure_pct[i])
    for i in cpt_sorted:
        p = cpt_players[i]
        ratio = cpt_exposure_pct[i] / max(p.ownership, 0.01)
        print(f"{p.name:<25} {p.position:<5} {p.team:<5} {p.ownership:>8.1f}% {cpt_exposure_pct[i]:>9.1f}% {ratio:>6.2f}x")

    print("\n" + "=" * 70)
    print("PLAYER EXPOSURE: FLEX POOL")
    print("=" * 70)
    print(f"{'Player':<25} {'Pos':<5} {'Team':<5} {'Proj Own%':>9} {'Field Own%':>10} {'Ratio':>7}")
    print("-" * 70)

    flex_sorted = sorted(range(len(flex_players)), key=lambda i: -flex_exposure_pct[i])
    for i in flex_sorted:
        p = flex_players[i]
        ratio = flex_exposure_pct[i] / max(p.ownership, 0.01)
        print(f"{p.name:<25} {p.position:<5} {p.team:<5} {p.ownership:>8.1f}% {flex_exposure_pct[i]:>9.1f}% {ratio:>6.2f}x")

    return cpt_exposure_pct, flex_exposure_pct


def salary_utilization_report(field_arrays: LineupArrays, field_counts: np.ndarray, salary_cap: int):
    """Report salary utilization distribution."""
    salaries = field_arrays.salary
    weights = field_counts.astype(np.float64)
    total = weights.sum()

    weighted_mean = np.average(salaries, weights=weights)
    # Weighted percentiles via sorted expansion
    expanded_salaries = np.repeat(salaries, field_counts)
    p10 = np.percentile(expanded_salaries, 10)
    p50 = np.percentile(expanded_salaries, 50)
    p90 = np.percentile(expanded_salaries, 90)

    print("\n" + "=" * 70)
    print("SALARY UTILIZATION")
    print("=" * 70)
    print(f"  Mean:   ${weighted_mean:,.0f} ({weighted_mean/salary_cap*100:.1f}% of cap)")
    print(f"  P10:    ${p10:,.0f}")
    print(f"  Median: ${p50:,.0f}")
    print(f"  P90:    ${p90:,.0f}")
    print(f"  Cap:    ${salary_cap:,}")


def cpt_position_report(field_arrays: LineupArrays, field_counts: np.ndarray, cpt_players):
    """Report CPT selection frequency by position."""
    total = int(field_counts.sum())
    pos_counts = Counter()

    for i, count in zip(field_arrays.cpt_idx, field_counts):
        pos_counts[cpt_players[i].position] += count

    print("\n" + "=" * 70)
    print("CPT SELECTION BY POSITION")
    print("=" * 70)
    print(f"{'Position':<10} {'Count':>8} {'Share':>8}")
    print("-" * 30)
    for pos, count in sorted(pos_counts.items(), key=lambda x: -x[1]):
        print(f"{pos:<10} {count:>8} {count/total*100:>7.1f}%")


def stack_structure_report(
    field_arrays: LineupArrays, field_counts: np.ndarray,
    cpt_players, flex_players
):
    """Report same-team concentration and bring-back rates."""
    total = int(field_counts.sum())
    teams = list(set(p.team for p in flex_players))

    same_team_counts = Counter()  # number of same-team-as-CPT FLEX players
    bring_back_count = 0

    for row_idx in range(len(field_arrays)):
        count = field_counts[row_idx]
        cpt_team = cpt_players[field_arrays.cpt_idx[row_idx]].team
        opp_team = [t for t in teams if t != cpt_team]
        opp_team = opp_team[0] if opp_team else None

        flex_idxs = field_arrays.flex_idx[row_idx]
        n_same = sum(1 for fi in flex_idxs if flex_players[fi].team == cpt_team)
        n_opp = len(flex_idxs) - n_same

        same_team_counts[f"{n_same+1}-{n_opp}"] += count  # +1 for CPT

        if n_opp > 0:
            bring_back_count += count

    print("\n" + "=" * 70)
    print("STACK STRUCTURE")
    print("=" * 70)
    print(f"{'Split':<10} {'Count':>8} {'Share':>8}")
    print("-" * 30)
    for split, count in sorted(same_team_counts.items(), key=lambda x: -x[1]):
        print(f"{split:<10} {count:>8} {count/total*100:>7.1f}%")

    print(f"\nBring-back rate: {bring_back_count/total*100:.1f}%")


def sharpness_to_hhi_calibration(
    candidate_arrays: LineupArrays,
    outcomes: np.ndarray,
    cpt_players, flex_players,
    n_field: int,
    ownership_power: float = 0.5,
    quality_sims: int = 1000,
    seed: int = 42
) -> dict:
    """
    Run generate_field_simulated() across sharpness values and record HHI.

    Returns mapping of sharpness -> HHI for use in A1's condensation perturbation.
    """
    sharpness_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]

    print("\n" + "=" * 70)
    print("SHARPNESS-TO-HHI CALIBRATION CURVE")
    print("=" * 70)
    print(f"{'Sharpness':>10} {'HHI':>10} {'Unique':>8} {'Top-1%':>8} {'Eff N':>8}")
    print("-" * 50)

    calibration = {}

    for sharpness in sharpness_values:
        field_arrays, field_counts = generate_field_simulated(
            candidate_arrays, outcomes,
            cpt_players, flex_players,
            n_field=n_field,
            field_sharpness=sharpness,
            ownership_power=ownership_power,
            quality_sims=quality_sims,
            seed=seed
        )

        hhi = compute_hhi(field_counts)
        n_unique = len(field_arrays)
        total = int(field_counts.sum())
        top_1pct_count = int(max(1, total * 0.01))
        top_counts = np.sort(field_counts)[::-1][:top_1pct_count]
        top_1pct_share = float(top_counts.sum() / total * 100) if total > 0 else 0.0
        eff_n = 1.0 / hhi if hhi > 0 else float('inf')

        calibration[float(sharpness)] = float(hhi)

        print(f"{sharpness:>10.1f} {hhi:>10.6f} {n_unique:>8} {top_1pct_share:>7.1f}% {eff_n:>8.0f}")

    return calibration


def field_duplication_report(field_arrays: LineupArrays, field_counts: np.ndarray):
    """Report field lineup duplication statistics."""
    total = int(field_counts.sum())
    n_unique = len(field_arrays)
    hhi = compute_hhi(field_counts)
    max_count = int(field_counts.max()) if len(field_counts) > 0 else 0
    eff_n = 1.0 / hhi if hhi > 0 else float('inf')

    print("\n" + "=" * 70)
    print("FIELD DUPLICATION")
    print("=" * 70)
    print(f"  Unique lineups: {n_unique}")
    print(f"  Total entries:  {total}")
    print(f"  Max duplication: {max_count} ({max_count/total*100:.2f}%)")
    print(f"  HHI: {hhi:.6f}")
    print(f"  Effective N: {eff_n:.0f}")


def main():
    parser = argparse.ArgumentParser(description="Validate ownership model")
    parser.add_argument("csv_path", help="Path to projections CSV")
    parser.add_argument("--contest-preset", "-p", choices=list(CONTEST_PRESETS.keys()),
                        default="dk_showdown_5")
    parser.add_argument("--contest-file", "-c", help="Contest JSON file")
    parser.add_argument("--n-sims", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--field-sharpness", type=float, default=5.0)
    parser.add_argument("--ownership-power", type=float, default=0.5)
    parser.add_argument("--quality-sims", type=int, default=1000)
    parser.add_argument("--salary-cap", type=int, default=50000)
    parser.add_argument("--output", "-o", help="Output JSON path for calibration data")
    args = parser.parse_args()

    # Load contest
    if args.contest_file:
        contest = load_contest_from_json(args.contest_file)
    else:
        contest = CONTEST_PRESETS[args.contest_preset]

    # Load data
    print(f"Loading projections from {args.csv_path}...")
    data = load_projections(args.csv_path)
    print(f"Loaded {data.n_cpt} CPT, {data.n_flex} FLEX players")

    # Simulate outcomes
    print(f"Simulating {args.n_sims} outcomes...")
    outcomes = simulate_outcomes(data.flex_players, args.n_sims, seed=args.seed)

    # Enumerate candidates
    print("Enumerating candidates...")
    from showdown_gto.candidates.enumeration import enumerate_lineups
    candidates = enumerate_lineups(
        data.cpt_players, data.flex_players,
        salary_cap=args.salary_cap,
        cpt_to_flex_map=data.cpt_to_flex_map
    )
    candidate_arrays = LineupArrays.from_lineups(candidates)
    print(f"Enumerated {len(candidates)} candidates")

    # Generate field
    field_size = contest.field_size
    print(f"Generating {field_size} field lineups (sharpness={args.field_sharpness})...")
    field_arrays, field_counts = generate_field_simulated(
        candidate_arrays, outcomes,
        data.cpt_players, data.flex_players,
        n_field=field_size,
        field_sharpness=args.field_sharpness,
        ownership_power=args.ownership_power,
        quality_sims=args.quality_sims,
        seed=args.seed
    )

    # Reports
    player_exposure_report(field_arrays, field_counts, data.cpt_players, data.flex_players)
    salary_utilization_report(field_arrays, field_counts, args.salary_cap)
    cpt_position_report(field_arrays, field_counts, data.cpt_players)
    stack_structure_report(field_arrays, field_counts, data.cpt_players, data.flex_players)
    field_duplication_report(field_arrays, field_counts)

    # Sharpness-to-HHI calibration
    calibration = sharpness_to_hhi_calibration(
        candidate_arrays, outcomes,
        data.cpt_players, data.flex_players,
        n_field=field_size,
        ownership_power=args.ownership_power,
        quality_sims=args.quality_sims,
        seed=args.seed
    )

    # Save calibration data
    output_path = args.output or "sharpness_hhi_calibration.json"
    calibration_data = {
        "sharpness_to_hhi": calibration,
        "parameters": {
            "field_size": field_size,
            "ownership_power": args.ownership_power,
            "quality_sims": args.quality_sims,
            "n_sims": args.n_sims,
            "seed": args.seed,
            "csv_path": args.csv_path,
        }
    }
    with open(output_path, "w") as f:
        json.dump(calibration_data, f, indent=2)
    print(f"\nCalibration data saved to {output_path}")


if __name__ == "__main__":
    main()
