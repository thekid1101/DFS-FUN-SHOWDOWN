"""
One-button runner for DFS Showdown GTO Portfolio Builder.

Usage:
    python run.py projections.csv archetype_map.json

    # With optional contest preset:
    python run.py projections.csv archetype_map.json --contest dk_showdown_20

    # With custom contest JSON:
    python run.py projections.csv archetype_map.json --contest-file contest.json

Everything else comes from production_config.json.
"""

import sys
import argparse
import json
import logging
from pathlib import Path

from showdown_gto.pipeline import run_portfolio_optimization
from showdown_gto.production import load_production_config
from showdown_gto.config import load_contest_from_json, CONTEST_PRESETS
from showdown_gto.data import load_projections
from showdown_gto.diagnostics import format_diagnostics

# Defaults — edit these if your file layout changes
DEFAULT_PRODUCTION_CONFIG = "production_config.json"
DEFAULT_CORRELATION_CONFIG = "correlation_config_v2.json"
DEFAULT_CONTEST_PRESET = "dk_showdown_5"
DEFAULT_N_SELECT = 150
DEFAULT_SEED = 42


def main():
    parser = argparse.ArgumentParser(
        description="One-button DFS Showdown optimizer"
    )
    parser.add_argument("csv_path", help="Path to projections CSV")
    parser.add_argument("archetype_map", help="Path to archetype map JSON")
    parser.add_argument(
        "--contest", default=DEFAULT_CONTEST_PRESET,
        choices=list(CONTEST_PRESETS.keys()),
        help=f"Contest preset (default: {DEFAULT_CONTEST_PRESET})"
    )
    parser.add_argument(
        "--contest-file", default=None,
        help="Custom contest JSON file (overrides --contest)"
    )
    parser.add_argument(
        "--n-select", type=int, default=DEFAULT_N_SELECT,
        help=f"Number of lineups (default: {DEFAULT_N_SELECT})"
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})"
    )
    parser.add_argument(
        "--config", default=DEFAULT_PRODUCTION_CONFIG,
        help=f"Production config path (default: {DEFAULT_PRODUCTION_CONFIG})"
    )
    parser.add_argument(
        "--correlation-config", default=DEFAULT_CORRELATION_CONFIG,
        help=f"Correlation config path (default: {DEFAULT_CORRELATION_CONFIG})"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output JSON path (default: results_{stem}.json)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load production config
    prod_kwargs = load_production_config(args.config)
    print(f"Config: {args.config}")
    print(f"  Profile: balanced + 10 DRO")
    print(f"  Sims: {prod_kwargs.get('n_sims', 100000)}")

    # Load contest
    if args.contest_file:
        contest = load_contest_from_json(args.contest_file)
    else:
        contest = CONTEST_PRESETS[args.contest]

    # Override your_entries to match n_select
    from showdown_gto.types import ContestStructure
    contest = ContestStructure(
        name=contest.name,
        entry_fee=contest.entry_fee,
        total_entries=contest.total_entries,
        your_entries=args.n_select,
        payout_tiers=contest.payout_tiers,
    )

    print(f"\nSlate: {args.csv_path}")
    print(f"Contest: {contest.name} (${contest.entry_fee}, {contest.total_entries} entries)")
    print(f"Your entries: {args.n_select}")
    print(f"Correlation: {args.correlation_config}")
    print(f"Archetypes: {args.archetype_map}")
    print(f"Seed: {args.seed}")
    print()

    # Run
    results = run_portfolio_optimization(
        csv_path=args.csv_path,
        contest=contest,
        n_select=args.n_select,
        correlation_config_path=args.correlation_config,
        archetype_map_path=args.archetype_map,
        seed=args.seed,
        verbose=True,
        **prod_kwargs,
    )

    if 'error' in results:
        print(f"\nERROR: {results['error']}")
        sys.exit(1)

    # Results
    diag = results['diagnostics']
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Portfolio EV:     ${diag['true_portfolio_ev']:.2f}")
    print(f"  Entry Cost:       ${diag['entry_cost']:.2f}")
    print(f"  Expected Profit:  ${diag['expected_profit']:.2f}")
    print(f"  ROI:              {diag['roi_pct']:.2f}%")
    print(f"  P(Profit):        {diag['p_profit']:.1%}")
    print(f"  Self-comp Cost:   ${diag['self_competition_cost']:.2f}")

    tm = results.get('tournament_metrics')
    if tm:
        print(f"\n  Tournament:")
        print(f"    Top 1% Rate:    {tm['top_1pct_rate']:.2%}")
        print(f"    Ceiling EV:     ${tm['ceiling_ev']:.2f}")
        print(f"    Win Rate:       {tm['win_rate']:.4%}")
        print(f"    Composite:      {tm['composite_score']:.4f}")

    # Export CSV
    stem = Path(args.csv_path).stem
    csv_out = f"portfolio_{stem}_{args.n_select}.csv"

    import csv
    proj_data = load_projections(args.csv_path)
    rows = []
    for lineup, players in zip(results['selected_lineups'], results['selected_players']):
        cpt_id = proj_data.cpt_players[lineup.cpt_player_idx].id
        flex_ids = [proj_data.flex_players[idx].id for idx in lineup.flex_player_idxs]
        cpt_name = players[0].replace(' (CPT)', '')
        flex_names = players[1:]
        rows.append({
            'CPT_ID': cpt_id,
            'FLEX1_ID': flex_ids[0], 'FLEX2_ID': flex_ids[1],
            'FLEX3_ID': flex_ids[2], 'FLEX4_ID': flex_ids[3],
            'FLEX5_ID': flex_ids[4],
            'CPT_Name': cpt_name,
            'FLEX1_Name': flex_names[0], 'FLEX2_Name': flex_names[1],
            'FLEX3_Name': flex_names[2], 'FLEX4_Name': flex_names[3],
            'FLEX5_Name': flex_names[4],
        })

    fieldnames = [
        'CPT_ID', 'FLEX1_ID', 'FLEX2_ID', 'FLEX3_ID', 'FLEX4_ID', 'FLEX5_ID',
        'CPT_Name', 'FLEX1_Name', 'FLEX2_Name', 'FLEX3_Name', 'FLEX4_Name', 'FLEX5_Name',
    ]
    with open(csv_out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Lineups: {csv_out}")

    # Export JSON
    json_out = args.output or f"results_{stem}.json"
    output_data = {
        'diagnostics': diag,
        'tournament_metrics': tm,
        'selected_players': results['selected_players'],
        'approx_evs': results['approx_evs'],
        'metadata': results['metadata'],
    }
    with open(json_out, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"  Results: {json_out}")


if __name__ == '__main__':
    main()
