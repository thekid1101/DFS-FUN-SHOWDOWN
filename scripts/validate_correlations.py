"""
Correlation matrix validation diagnostic.

Validates the correlation matrix before simulations depend on it.
Checks PSD property, applies nearest-PSD projection if needed,
validates structural properties, and runs empirical Spearman correlation check.

Usage:
    python -m scripts.validate_correlations projections.csv \
        --correlation-config correlation_config_v2.json \
        --archetype-map archetype_map.json \
        --n-sims 1000 --seed 42
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from showdown_gto.data.loader import load_projections
from showdown_gto.data.correlations import (
    CorrelationMatrix, ArchetypeCorrelationConfig,
    load_archetype_mapping, _infer_archetype
)
from showdown_gto.simulation.engine import simulate_outcomes


def check_psd(matrix: np.ndarray) -> tuple:
    """Check if matrix is positive semi-definite. Returns (is_psd, min_eigenvalue, condition_number)."""
    eigenvalues = np.linalg.eigvalsh(matrix)
    min_eig = float(eigenvalues.min())
    max_eig = float(eigenvalues.max())
    cond = max_eig / max(abs(min_eig), 1e-15) if min_eig > 0 else float('inf')
    is_psd = min_eig >= -1e-10
    return is_psd, min_eig, cond


def structural_validation(
    matrix: np.ndarray, players, archetype_map
):
    """Validate structural correlation properties."""
    n = len(players)
    issues = []
    warnings = []

    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = players[i], players[j]
            corr = matrix[i, j]
            same_team = p1.team == p2.team
            arch1 = archetype_map.get(p1.name, _infer_archetype(p1))
            arch2 = archetype_map.get(p2.name, _infer_archetype(p2))

            # Same-team correlations should generally be positive
            if same_team and corr < 0 and arch1 != 'DST' and arch2 != 'DST':
                warnings.append(
                    f"  Negative same-team: {p1.name}({arch1}) <-> {p2.name}({arch2}) = {corr:.3f}"
                )

            # QB-WR/TE same team should be stronger than QB-opponent
            if same_team and 'QB' in arch1 and p2.position in ('WR', 'TE'):
                # Find if there's an opposing QB-WR pair with higher correlation
                for k in range(n):
                    if players[k].team != p1.team and players[k].position in ('WR', 'TE'):
                        opp_corr = matrix[i, k]
                        if opp_corr > corr:
                            issues.append(
                                f"  QB-pass catcher: {p1.name}-{p2.name} ({corr:.3f}) "
                                f"< QB-opp {p1.name}-{players[k].name} ({opp_corr:.3f})"
                            )
                            break

            # DST should be negative with opposing offense
            if not same_team:
                if arch1 == 'DST' and p2.position in ('QB', 'WR', 'RB', 'TE') and corr > 0:
                    warnings.append(
                        f"  DST-offense positive: {p1.name}(DST) <-> {p2.name}({arch2}) = {corr:.3f}"
                    )
                if arch2 == 'DST' and p1.position in ('QB', 'WR', 'RB', 'TE') and corr > 0:
                    warnings.append(
                        f"  DST-offense positive: {p2.name}(DST) <-> {p1.name}({arch1}) = {corr:.3f}"
                    )

    return issues, warnings


def extreme_pairs_report(matrix: np.ndarray, players):
    """Report top 10 strongest and weakest correlation pairs."""
    n = len(players)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((matrix[i, j], players[i].name, players[j].name,
                          players[i].team == players[j].team))

    pairs.sort(key=lambda x: -x[0])

    print("\n  Top 10 Strongest Pairs:")
    for corr, n1, n2, same in pairs[:10]:
        tag = "same" if same else "opp"
        print(f"    {corr:+.3f}  {n1} <-> {n2} ({tag})")

    print("\n  Top 10 Weakest Pairs:")
    for corr, n1, n2, same in pairs[-10:]:
        tag = "same" if same else "opp"
        print(f"    {corr:+.3f}  {n1} <-> {n2} ({tag})")


def empirical_spearman_check(
    matrix: np.ndarray, players, n_sims: int, seed: int
):
    """Run simulation and check Spearman rank-correlations match input."""
    outcomes = simulate_outcomes(
        players, n_sims, correlation_matrix=matrix, seed=seed
    )

    # Compute Spearman rank correlation from simulated outcomes
    n = len(players)
    spearman_matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            rho, _ = stats.spearmanr(outcomes[i], outcomes[j])
            spearman_matrix[i, j] = rho
            spearman_matrix[j, i] = rho

    # Compare input vs empirical
    diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = abs(matrix[i, j] - spearman_matrix[i, j])
            diffs.append((diff, matrix[i, j], spearman_matrix[i, j],
                          players[i].name, players[j].name))

    diffs.sort(key=lambda x: -x[0])

    print(f"\n  Empirical Spearman Check ({n_sims} sims):")
    print(f"    Mean |input - empirical|: {np.mean([d[0] for d in diffs]):.4f}")
    print(f"    Max  |input - empirical|: {diffs[0][0]:.4f}")

    print("\n  Top 10 Largest Discrepancies:")
    for diff, inp, emp, n1, n2 in diffs[:10]:
        print(f"    {diff:.4f}  input={inp:+.3f}  empirical={emp:+.3f}  {n1} <-> {n2}")


def main():
    parser = argparse.ArgumentParser(description="Validate correlation matrix")
    parser.add_argument("csv_path", help="Path to projections CSV")
    parser.add_argument("--correlation-config", required=True, help="correlation_config_v2.json")
    parser.add_argument("--archetype-map", help="Player archetype mapping JSON")
    parser.add_argument("--n-sims", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load data
    data = load_projections(args.csv_path)
    print(f"Loaded {data.n_flex} FLEX players")

    # Load archetype map
    archetype_map = {}
    if args.archetype_map:
        archetype_map = load_archetype_mapping(args.archetype_map)
        print(f"Loaded {len(archetype_map)} archetype mappings")

    # Build correlation matrix
    config = ArchetypeCorrelationConfig.from_json(args.correlation_config)
    corr_obj = CorrelationMatrix.from_archetype_config(
        data.flex_players, config, archetype_map
    )
    matrix = corr_obj.matrix

    print(f"\nCorrelation matrix: {matrix.shape[0]}x{matrix.shape[1]}")

    # PSD check
    print("\n" + "=" * 70)
    print("POSITIVE SEMI-DEFINITE CHECK")
    print("=" * 70)

    is_psd, min_eig, cond = check_psd(matrix)
    print(f"  PSD: {is_psd}")
    print(f"  Min eigenvalue: {min_eig:.6e}")
    print(f"  Condition number: {cond:.1f}")

    if not is_psd:
        print("  WARNING: Matrix is not PSD. Nearest-PSD projection will be applied by pipeline.")

    if cond > 1000:
        print(f"  WARNING: High condition number ({cond:.0f}). Auto-regularization may be needed.")

    if cond > 10000:
        print(f"  CRITICAL: Very high condition number ({cond:.0f}). Regularization required.")

    # Structural validation
    print("\n" + "=" * 70)
    print("STRUCTURAL VALIDATION")
    print("=" * 70)

    issues, warnings = structural_validation(matrix, data.flex_players, archetype_map)

    if issues:
        print(f"\n  Issues ({len(issues)}):")
        for issue in issues[:20]:
            print(issue)
    else:
        print("  No structural issues found.")

    if warnings:
        print(f"\n  Warnings ({len(warnings)}):")
        for warning in warnings[:20]:
            print(warning)

    # Extreme pairs
    print("\n" + "=" * 70)
    print("EXTREME CORRELATION PAIRS")
    print("=" * 70)
    extreme_pairs_report(matrix, data.flex_players)

    # Empirical Spearman check
    print("\n" + "=" * 70)
    print("EMPIRICAL SPEARMAN CHECK")
    print("=" * 70)
    empirical_spearman_check(matrix, data.flex_players, args.n_sims, args.seed)

    print("\nValidation complete.")


if __name__ == "__main__":
    main()
