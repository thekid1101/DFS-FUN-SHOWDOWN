"""
Command-line interface for DFS Showdown GTO Portfolio Builder.
"""

import click
import csv
import json
import logging
from pathlib import Path

from .types import ContestStructure, PayoutTier, FieldGenConfig
from .pipeline import run_portfolio_optimization, run_multi_contest_optimization
from .config import load_contest_from_json, CONTEST_PRESETS
from .data import load_projections


@click.command()
@click.argument('csv_path', type=click.Path(exists=True))
@click.option(
    '--contest-file', '-c',
    type=click.Path(exists=True),
    help='Contest configuration JSON file'
)
@click.option(
    '--contest-preset', '-p',
    type=click.Choice(list(CONTEST_PRESETS.keys())),
    help='Use a preset contest configuration'
)
@click.option(
    '--n-select', '-n',
    type=int,
    default=150,
    help='Number of lineups to select (default: 150)'
)
@click.option(
    '--n-sims', '-s',
    type=int,
    default=100000,
    help='Number of simulations (default: 100000)'
)
@click.option(
    '--correlation-config',
    type=click.Path(exists=True),
    help='Correlation config JSON (correlation_config_v2.json format)'
)
@click.option(
    '--archetype-map',
    type=click.Path(exists=True),
    help='Player archetype mapping JSON (player name -> archetype)'
)
@click.option(
    '--seed',
    type=int,
    help='Random seed for reproducibility'
)
@click.option(
    '--output', '-o',
    type=click.Path(),
    help='Output file for results JSON'
)
@click.option(
    '--verbose/--quiet', '-v/-q',
    default=True,
    help='Verbose output'
)
@click.option(
    '--entry-fee',
    type=float,
    help='Contest entry fee (overrides config)'
)
@click.option(
    '--field-size',
    type=int,
    help='Total contest entries (overrides config)'
)
@click.option(
    '--field-mode',
    type=click.Choice(['fixed', 'resample_per_sim']),
    default='fixed',
    help='Field mode: fixed (single field for all sims) or resample_per_sim (new field histogram each sim)'
)
@click.option(
    '--copula-type',
    type=click.Choice(['gaussian', 't']),
    default='gaussian',
    help='Copula type: gaussian (default) or t (Student-t for tail dependence)'
)
@click.option(
    '--copula-df',
    type=int,
    default=5,
    help='Degrees of freedom for t-copula (lower = heavier tails, default: 5)'
)
@click.option(
    '--selection-method',
    type=click.Choice(['top_n', 'greedy_marginal']),
    default='top_n',
    help='Selection method: top_n (fast, default) or greedy_marginal (accounts for self-competition)'
)
@click.option(
    '--shortlist-size',
    type=int,
    default=2000,
    help='Number of top candidates to consider for greedy selection (default: 2000)'
)
@click.option(
    '--greedy-sims',
    type=int,
    default=None,
    help='Max sims for greedy selection (default: min(n_sims, 10000))'
)
@click.option(
    '--effects-file',
    type=click.Path(exists=True),
    help='Path to unified_player_effects.json for pre-processing modifiers'
)
@click.option(
    '--sim-config',
    type=click.Path(exists=True),
    help='Path to simulation_config_v3.json for role distribution parameters'
)
@click.option(
    '--spread',
    type=str,
    default=None,
    help='Vegas spread: "TEAM VALUE" (e.g., "LAR -3.5"). Negative=favorite.'
)
@click.option(
    '--game-total',
    type=float,
    default=None,
    help='Vegas game total (e.g., 48.5) for conditional correlation modifiers'
)
@click.option(
    '--min-projection',
    type=float,
    default=0.0,
    help='Minimum median projection to include a player (filters low-projection players)'
)
@click.option(
    '--field-method',
    type=click.Choice(['simulated', 'ownership']),
    default='simulated',
    help='Field generation method: simulated (quality x ownership from candidates, default) or ownership (legacy)'
)
@click.option(
    '--field-sharpness',
    type=float,
    default=5.0,
    help='How projection-aware the field is: 0=pure ownership, 5.0=realistic (default), 8+=optimizer-heavy'
)
@click.option(
    '--ownership-power',
    type=float,
    default=0.5,
    help='Ownership influence on field duplication: 0=none, 0.5=balanced (default), 1.0=full'
)
@click.option(
    '--field-quality-sims',
    type=int,
    default=1000,
    help='Number of sims used for field quality scoring (default: 1000)'
)
@click.option(
    '--multi-contest',
    type=click.Path(exists=True),
    multiple=True,
    help='Additional contest JSON files for multi-contest optimization (repeatable)'
)
@click.option(
    '--dro-perturbations',
    type=int,
    default=0,
    help='Number of DRO field perturbations (0=disabled, 50=recommended)'
)
@click.option(
    '--dro-scale',
    type=float,
    default=0.10,
    help='DRO ownership perturbation scale (default: 0.10)'
)
@click.option(
    '--dro-hhi-scale',
    type=float,
    default=0.15,
    help='DRO HHI-space condensation perturbation scale (default: 0.15)'
)
@click.option(
    '--dro-aggregation',
    type=click.Choice(['mean', 'cvar', 'mean_minus_std']),
    default='mean',
    help='DRO aggregation method (default: mean)'
)
@click.option(
    '--dro-calibration',
    type=click.Path(exists=True),
    help='Path to sharpness_hhi_calibration.json for DRO layer 2'
)
@click.option(
    '--covariance-gamma',
    type=float,
    default=0.05,
    help='Profit covariance penalty for greedy: 0=disabled, 0.05=default'
)
def main(
    csv_path,
    contest_file,
    contest_preset,
    n_select,
    n_sims,
    correlation_config,
    archetype_map,
    seed,
    output,
    verbose,
    entry_fee,
    field_size,
    field_mode,
    copula_type,
    copula_df,
    selection_method,
    shortlist_size,
    greedy_sims,
    effects_file,
    sim_config,
    spread,
    game_total,
    min_projection,
    field_method,
    field_sharpness,
    ownership_power,
    field_quality_sims,
    multi_contest,
    dro_perturbations,
    dro_scale,
    dro_hhi_scale,
    dro_aggregation,
    dro_calibration,
    covariance_gamma
):
    """
    Run DFS Showdown GTO Portfolio Builder.

    CSV_PATH: Path to the projections CSV file
    """
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    # Validate critical numeric parameters
    if n_sims <= 0:
        raise click.BadParameter("n-sims must be a positive integer", param_hint="'--n-sims'")
    if n_select <= 0:
        raise click.BadParameter("n-select must be a positive integer", param_hint="'--n-select'")
    if shortlist_size <= 0:
        raise click.BadParameter("shortlist-size must be a positive integer", param_hint="'--shortlist-size'")
    if copula_type == 't' and copula_df <= 0:
        raise click.BadParameter("copula-df must be a positive integer for t-copula", param_hint="'--copula-df'")

    # Load contest configuration
    if contest_file:
        contest = load_contest_from_json(contest_file)
    elif contest_preset:
        contest = CONTEST_PRESETS[contest_preset]
    else:
        # Create default contest
        click.echo("No contest config provided. Using default settings.")
        contest = _create_default_contest(n_select, entry_fee, field_size)

    # Override contest settings if provided
    if entry_fee is not None:
        contest = ContestStructure(
            name=contest.name,
            entry_fee=entry_fee,
            total_entries=contest.total_entries,
            your_entries=n_select,
            payout_tiers=contest.payout_tiers
        )

    if field_size is not None:
        contest = ContestStructure(
            name=contest.name,
            entry_fee=contest.entry_fee,
            total_entries=field_size,
            your_entries=n_select,
            payout_tiers=contest.payout_tiers
        )

    # Ensure your_entries matches n_select
    contest = ContestStructure(
        name=contest.name,
        entry_fee=contest.entry_fee,
        total_entries=contest.total_entries,
        your_entries=n_select,
        payout_tiers=contest.payout_tiers
    )

    if contest.total_entries < contest.your_entries:
        raise click.BadParameter(
            f"n-select ({contest.your_entries}) cannot exceed total entries ({contest.total_entries}). "
            "Increase --field-size/contest total entries or reduce --n-select."
        )

    click.echo(f"Running optimization...")
    if correlation_config:
        click.echo(f"  Correlation config: {correlation_config}")
    if archetype_map:
        click.echo(f"  Archetype map: {archetype_map}")
    click.echo(f"  CSV: {csv_path}")
    click.echo(f"  Contest: {contest.name}")
    click.echo(f"  Entry fee: ${contest.entry_fee}")
    click.echo(f"  Total entries: {contest.total_entries}")
    click.echo(f"  Your entries: {n_select}")
    click.echo(f"  Simulations: {n_sims}")
    click.echo(f"  Field mode: {field_mode}")
    click.echo(f"  Field method: {field_method}")
    if field_method == 'simulated':
        click.echo(f"  Field sharpness: {field_sharpness}")
        click.echo(f"  Ownership power: {ownership_power}")
        click.echo(f"  Field quality sims: {field_quality_sims}")
    click.echo(f"  Copula: {copula_type}" + (f" (df={copula_df})" if copula_type == 't' else ''))
    sel_extra = ''
    if selection_method == 'greedy_marginal':
        sel_extra = f" (shortlist={shortlist_size})"
    click.echo(f"  Selection: {selection_method}{sel_extra}")
    if effects_file:
        click.echo(f"  Effects file: {effects_file}")
    if sim_config:
        click.echo(f"  Sim config: {sim_config}")
    if spread:
        click.echo(f"  Spread: {spread}")
    if game_total:
        click.echo(f"  Game total: {game_total}")

    # Multi-contest mode
    if multi_contest:
        if dro_perturbations > 0:
            click.echo(
                "Warning: DRO is not yet supported in multi-contest mode. "
                "DRO perturbations will be ignored.", err=True
            )
        contests = [contest]
        for mc_path in multi_contest:
            mc_contest = load_contest_from_json(mc_path)
            contests.append(mc_contest)
        click.echo(f"\n  Multi-contest mode: {len(contests)} contests")
        for i, c in enumerate(contests):
            click.echo(f"    {i+1}. {c.name} (${c.entry_fee}, {c.total_entries} entries, {c.your_entries} yours)")

        mc_results = run_multi_contest_optimization(
            csv_path=csv_path,
            contests=contests,
            n_sims=n_sims,
            correlation_config_path=correlation_config,
            archetype_map_path=archetype_map,
            seed=seed,
            verbose=verbose,
            copula_type=copula_type,
            copula_df=copula_df,
            selection_method=selection_method,
            shortlist_size=shortlist_size,
            greedy_n_sims=greedy_sims,
            effects_path=effects_file,
            sim_config_path=sim_config,
            spread_str=spread,
            game_total=game_total,
            min_projection=min_projection,
            field_method=field_method,
            field_sharpness=field_sharpness,
            ownership_power=ownership_power,
            field_quality_sims=field_quality_sims,
            covariance_gamma=covariance_gamma,
        )

        if 'error' in mc_results:
            click.echo(f"Error: {mc_results['error']}", err=True)
            return

        # Display multi-contest results
        click.echo("\n" + "=" * 60)
        click.echo("MULTI-CONTEST RESULTS")
        click.echo("=" * 60)

        for cname, cresults in mc_results['per_contest'].items():
            diag = cresults['diagnostics']
            click.echo(f"\n  {cname}:")
            click.echo(f"    EV: ${diag['true_portfolio_ev']:.2f} | "
                        f"ROI: {diag['roi_pct']:.2f}% | "
                        f"P(Profit): {diag['p_profit']:.1%} | "
                        f"Self-comp: ${diag['self_competition_cost']:.2f}")

        if mc_results.get('overlap_matrix'):
            click.echo("\n  Overlap Matrix:")
            for key, ov in mc_results['overlap_matrix'].items():
                click.echo(f"    {key}: {ov['shared_lineups']} shared lineups "
                            f"({ov['pct_of_i']:.1f}% / {ov['pct_of_j']:.1f}%)")

        # Export per-contest CSVs
        proj_data = load_projections(csv_path)
        for cname, cresults in mc_results['per_contest'].items():
            safe_name = cname.replace(' ', '_').replace('$', '').replace('/', '-')
            n_lu = len(cresults['selected_lineups'])
            csv_path_out = f"portfolio_{safe_name}_{n_lu}.csv"

            csv_rows = []
            for lineup, players in zip(cresults['selected_lineups'], cresults['selected_players']):
                cpt_idx = lineup.cpt_player_idx
                flex_idxs = lineup.flex_player_idxs
                cpt_id = proj_data.cpt_players[cpt_idx].id
                flex_ids = [proj_data.flex_players[idx].id for idx in flex_idxs]
                cpt_name = players[0].replace(' (CPT)', '')
                flex_names = players[1:]

                csv_rows.append({
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
                'CPT_Name', 'FLEX1_Name', 'FLEX2_Name', 'FLEX3_Name', 'FLEX4_Name', 'FLEX5_Name'
            ]
            with open(csv_path_out, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)
            click.echo(f"  {cname}: {csv_path_out}")

        # Save JSON summary
        summary_path = output if output and output.endswith('.json') else 'portfolio_summary.json'
        summary = {
            'per_contest': {
                name: {
                    'diagnostics': r['diagnostics'],
                    'selected_players': r['selected_players'],
                    'approx_evs': r['approx_evs'],
                }
                for name, r in mc_results['per_contest'].items()
            },
            'overlap_matrix': mc_results['overlap_matrix'],
            'metadata': mc_results['metadata'],
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        click.echo(f"\nSummary saved to {summary_path}")
        return

    # Display DRO settings
    if dro_perturbations > 0:
        click.echo(f"  DRO: {dro_perturbations} perturbations, "
                    f"scale={dro_scale}, hhi_scale={dro_hhi_scale}, "
                    f"agg={dro_aggregation}")
    if covariance_gamma > 0:
        click.echo(f"  Covariance gamma: {covariance_gamma}")

    # Run single-portfolio optimization
    results = run_portfolio_optimization(
        csv_path=csv_path,
        contest=contest,
        n_select=n_select,
        n_sims=n_sims,
        correlation_config_path=correlation_config,
        archetype_map_path=archetype_map,
        seed=seed,
        verbose=verbose,
        field_mode=field_mode,
        copula_type=copula_type,
        copula_df=copula_df,
        selection_method=selection_method,
        shortlist_size=shortlist_size,
        greedy_n_sims=greedy_sims,
        effects_path=effects_file,
        sim_config_path=sim_config,
        spread_str=spread,
        game_total=game_total,
        min_projection=min_projection,
        field_method=field_method,
        field_sharpness=field_sharpness,
        ownership_power=ownership_power,
        field_quality_sims=field_quality_sims,
        dro_perturbations=dro_perturbations,
        dro_scale=dro_scale,
        dro_hhi_scale=dro_hhi_scale,
        dro_aggregation=dro_aggregation,
        dro_calibration_path=dro_calibration,
        covariance_gamma=covariance_gamma,
    )

    if 'error' in results:
        click.echo(f"Error: {results['error']}", err=True)
        return

    # Display results
    click.echo("\n" + "=" * 60)
    click.echo("PORTFOLIO RESULTS")
    click.echo("=" * 60)

    diag = results['diagnostics']
    click.echo(f"\nPortfolio EV: ${diag['true_portfolio_ev']:.2f}")
    click.echo(f"Entry Cost: ${diag['entry_cost']:.2f}")
    click.echo(f"Expected Profit: ${diag['expected_profit']:.2f}")
    click.echo(f"ROI: {diag['roi_pct']:.2f}%")
    click.echo(f"P(Profit): {diag['p_profit']:.1%}")
    click.echo(f"Self-competition Cost: ${diag['self_competition_cost']:.2f}")

    if 'robust_ev_sum' in diag:
        click.echo(f"Robust EV Sum: ${diag['robust_ev_sum']:.2f}")
        click.echo(f"DRO Regret: ${diag['dro_regret']:.2f}")

    click.echo(f"\nSelected {len(results['selected_lineups'])} lineups")

    # Show top lineups
    click.echo("\nTop 10 Lineups by Approx EV:")
    for i, (players, approx_ev) in enumerate(
        zip(results['selected_players'][:10], results['approx_evs'][:10])
    ):
        click.echo(f"  {i+1}. ${approx_ev:.2f} - {', '.join(players)}")

    # Export CSV (always - this is the main output)
    csv_output = output if output and output.endswith('.csv') else 'portfolio_lineups.csv'
    if output and not output.endswith('.csv'):
        csv_output = output.rsplit('.', 1)[0] + '.csv'

    # Load projections to get DFS IDs
    proj_data = load_projections(csv_path)

    # Build CSV rows
    csv_rows = []
    for lineup, players in zip(results['selected_lineups'], results['selected_players']):
        cpt_idx = lineup.cpt_player_idx
        flex_idxs = lineup.flex_player_idxs

        # Get DFS IDs
        cpt_id = proj_data.cpt_players[cpt_idx].id
        flex_ids = [proj_data.flex_players[idx].id for idx in flex_idxs]

        # Get names (strip '(CPT)' from first one)
        cpt_name = players[0].replace(' (CPT)', '')
        flex_names = players[1:]

        csv_rows.append({
            'CPT_ID': cpt_id,
            'FLEX1_ID': flex_ids[0],
            'FLEX2_ID': flex_ids[1],
            'FLEX3_ID': flex_ids[2],
            'FLEX4_ID': flex_ids[3],
            'FLEX5_ID': flex_ids[4],
            'CPT_Name': cpt_name,
            'FLEX1_Name': flex_names[0],
            'FLEX2_Name': flex_names[1],
            'FLEX3_Name': flex_names[2],
            'FLEX4_Name': flex_names[3],
            'FLEX5_Name': flex_names[4],
        })

    # Write CSV
    fieldnames = ['CPT_ID', 'FLEX1_ID', 'FLEX2_ID', 'FLEX3_ID', 'FLEX4_ID', 'FLEX5_ID',
                  'CPT_Name', 'FLEX1_Name', 'FLEX2_Name', 'FLEX3_Name', 'FLEX4_Name', 'FLEX5_Name']

    with open(csv_output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    click.echo(f"\nLineups exported to {csv_output}")

    # Also save JSON if explicitly requested with .json extension
    if output and output.endswith('.json'):
        output_data = {
            'diagnostics': diag,
            'selected_players': results['selected_players'],
            'approx_evs': results['approx_evs'],
            'metadata': results['metadata'],
            'lineups': [
                {
                    'cpt_idx': lu.cpt_player_idx,
                    'flex_idxs': lu.flex_player_idxs,
                    'salary': lu.salary
                }
                for lu in results['selected_lineups']
            ]
        }

        with open(output, 'w') as f:
            json.dump(output_data, f, indent=2)
        click.echo(f"Diagnostics saved to {output}")


def _create_default_contest(
    n_select: int,
    entry_fee: float = None,
    field_size: int = None
) -> ContestStructure:
    """Create a default contest structure."""
    if entry_fee is None:
        entry_fee = 5.0
    if field_size is None:
        field_size = 5000 + n_select

    # Default payout structure (simplified)
    payout_tiers = [
        PayoutTier(1, 1, 500.0),
        PayoutTier(2, 5, 100.0),
        PayoutTier(6, 20, 50.0),
        PayoutTier(21, 50, 25.0),
        PayoutTier(51, 100, 15.0),
        PayoutTier(101, 200, 10.0),
        PayoutTier(201, 500, 7.50),
        PayoutTier(501, 1000, 6.00),
    ]

    return ContestStructure(
        name="Default Contest",
        entry_fee=entry_fee,
        total_entries=field_size,
        your_entries=n_select,
        payout_tiers=payout_tiers
    )


if __name__ == '__main__':
    main()
