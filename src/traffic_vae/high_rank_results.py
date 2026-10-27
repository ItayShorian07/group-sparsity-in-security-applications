"""Export every outcome of the separate high-rank nonlinear stress test."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .data import sha256
from .evaluation import evaluate
from .experiment import write_json


METHODS = ('VAE_Lasso', 'VAE_GroupLasso', 'OnlineDA', 'OnlineDA_rank16')
COLORS = ('#0072B2', '#D55E00', '#009E73', '#CC79A7')
LABELS = ('VAE + Lasso', 'VAE + Group Lasso', 'OnlineDA rank 5', 'OnlineDA rank 16')
MARKERS = {17: 'o', 42: 's', 2026: '^'}
SCENARIOS = ('paper_independent', 'matched_dispersed', 'destination_grouped')


def summary(frame):
    columns = [column for column in frame if column.endswith(('_f1', '_average_precision',
               '_false_positive_rate', '_precision', '_recall')) or column.startswith('relative_error')]
    result = frame.groupby(['scenario', 'gamma', 'method'])[columns].agg(['mean', 'std'])
    result.columns = ['_'.join(column) for column in result.columns]
    return result.reset_index()


def export(source, output):
    source, output = Path(source), Path(output)
    completed = json.loads((source / 'completed.json').read_text())
    settings = json.loads((source / 'settings.json').read_text())
    group = source / 'experiment_03'
    paper = source / 'experiment_02'
    config = json.loads((paper / 'config.json').read_text())
    frames = [pd.read_csv(path) for path in
              (group / 'synthetic_metrics.csv', source / 'onlineda/synthetic_metrics.csv',
               source / 'rank_control/synthetic_metrics.csv')]
    # The rank control also saves secondary threshold fields. Recompute that
    # separate evaluation for all methods below, avoiding undefined mixed columns.
    frames[-1] = frames[-1].drop(columns=[key for key in frames[-1] if key.startswith('paper_threshold_flow_')])
    frame = pd.concat(frames, ignore_index=True)
    expected = len(config['seeds']) * len(config['gammas']) * len(SCENARIOS) * len(METHODS)
    if len(frame) != expected or frame.duplicated(['seed', 'gamma', 'scenario', 'method']).any():
        raise ValueError('Incomplete or duplicate sensitivity results.')
    for (seed, gamma, scenario), selected in frame.groupby(['seed', 'gamma', 'scenario']):
        if set(selected.method) != set(METHODS):
            raise ValueError('Missing methods in a matched comparison.')
    if not np.isfinite(frame.select_dtypes(include='number').to_numpy()).all():
        raise ValueError('Non-finite exported metrics.')
    aggregated = summary(frame)
    secondary = []
    for seed in config['seeds']:
        with np.load(group / f'synthetic/seed_{seed}/attacks.npz') as values:
            new_attacks = {scenario: values[scenario].copy() for scenario in SCENARIOS[1:]}
        for index, gamma in enumerate(config['gammas']):
            with np.load(paper / f'seed_{seed}/gamma_{index:02d}/truth.npz') as values:
                attacks = {'paper_independent': values['A'].copy(), **new_attacks}
            for scenario, truth in attacks.items():
                paths = [(group / f'synthetic/seed_{seed}/gamma_{index:02d}/{scenario}/estimates.npz',
                          'lasso', 'VAE_Lasso'),
                         (group / f'synthetic/seed_{seed}/gamma_{index:02d}/{scenario}/estimates.npz',
                          'group_lasso', 'VAE_GroupLasso'),
                         (source / f'onlineda/seed_{seed}/gamma_{index:02d}/{scenario}/estimates.npz',
                          'A_hat', 'OnlineDA'),
                         (source / f'rank_control/seed_{seed}/gamma_{index:02d}/{scenario}/estimates.npz',
                          'A_hat', 'OnlineDA_rank16')]
                for path, key, method in paths:
                    with np.load(path) as values:
                        metrics = evaluate((truth != 0).ravel().astype(int), np.abs(values[key]).ravel(),
                                           config['paper_threshold'])
                    secondary.append(dict(seed=seed, gamma=gamma, scenario=scenario, method=method,
                                          **{f'flow_{key}': value for key, value in metrics.items()}))
    secondary = pd.DataFrame(secondary)
    old = pd.read_csv(Path(settings['original_source']) / 'metrics.csv')
    new = pd.read_csv(paper / 'metrics.csv')
    metric_columns = [key for key in old if key.startswith(('flow_', 'time_', 'paper_threshold_flow_', 'relative_error_'))]
    a = old[old.gamma == 0].set_index(['seed', 'method'])[metric_columns].sort_index()
    b = new[new.gamma == 0].set_index(['seed', 'method'])[metric_columns].sort_index()
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=1e-12, atol=1e-12)
    previous_export = Path('output/experiment-03/results.json')
    original_sources_unchanged = None
    if previous_export.exists():
        previous = json.loads(previous_export.read_text())
        if Path(previous['config']['synthetic_source']).resolve() == Path(settings['original_source']).resolve():
            for row in previous['synthetic_source_hashes']:
                index = config['gammas'].index(row['gamma'])
                original_case = Path(settings['original_source']) / f"seed_{row['seed']}/gamma_{index:02d}"
                if (sha256(original_case / 'truth.npz') != row['truth_sha256']
                        or sha256(original_case / 'vae_lasso/model.pt') != row['model_sha256']):
                    raise ValueError('Original data or VAE checkpoints changed since the prior export.')
            original_sources_unchanged = True
    paired = frame.pivot(index=['seed', 'gamma', 'scenario'], columns='method',
                         values=['flow_f1', 'flow_average_precision'])
    differences = pd.DataFrame(index=paired.index)
    comparison = {}
    for method in ('VAE_Lasso', 'VAE_GroupLasso'):
        for baseline in ('OnlineDA', 'OnlineDA_rank16'):
            for metric in ('flow_f1', 'flow_average_precision'):
                key = f'{method}_minus_{baseline}_{metric}'
                differences[key] = paired[metric, method] - paired[metric, baseline]
                means = differences[key].groupby(['gamma', 'scenario']).mean()
                comparison[key] = {'positive_mean_settings': int((means > 0).sum()),
                                   'settings': len(means), 'significance_tested': False}
    quality = []
    for directory in (group / 'synthetic', source / 'onlineda', source / 'rank_control'):
        for path in sorted(directory.glob('seed_*/gamma_*/*/solver.json')):
            quality.append({'case': str(path.relative_to(source)), **json.loads(path.read_text())})
    source_files = [paper / 'metrics.csv', group / 'synthetic_metrics.csv',
                    source / 'onlineda/synthetic_metrics.csv', source / 'rank_control/synthetic_metrics.csv']
    payload = {
        'settings': settings, 'completed': completed,
        'provenance': json.loads((source / 'provenance.json').read_text()),
        'training_geometry': json.loads((source / 'training_geometry.json').read_text()),
        'rank_diagnostics': pd.read_csv(paper / 'rank_diagnostics.csv').to_dict('records'),
        'original_experiment2_metrics': old.to_dict('records'),
        'high_rank_experiment2_metrics': new.to_dict('records'),
        'metrics_by_seed': frame.to_dict('records'), 'summary': aggregated.to_dict('records'),
        'secondary_fixed_threshold': config['paper_threshold'],
        'secondary_metrics_by_seed': secondary.to_dict('records'),
        'secondary_summary': summary(secondary).to_dict('records'),
        'paired_differences': differences.reset_index().to_dict('records'),
        'comparison_counts': comparison,
        'linear_control_matches_original_metrics': True,
        'original_sources_match_preexisting_export': original_sources_unchanged,
        'solver_quality': quality,
        'paper_case_solver_quality': [dict(case=str(path.relative_to(source)),
                                         **json.loads(path.read_text()))
                                     for path in sorted(paper.glob('seed_*/gamma_*/*/solver.json'))],
        'vae_source_hashes': json.loads((group / 'synthetic/source_hashes.json').read_text()),
        'online_source_hashes': json.loads((source / 'onlineda/source_hashes.json').read_text()),
        'rank_control_source_hashes': json.loads((source / 'rank_control/source_hashes.json').read_text()),
        'metric_files_sha256': {str(path): sha256(path) for path in source_files},
        'interpretation': 'Exploratory artificial stress test; report all settings, no significance claim, main project-report results unchanged',
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'results.json', payload)
    draw(frame, payload['rank_diagnostics'], output)
    print(json.dumps(comparison, indent=2))
    print(f'Exported {len(frame)} matched result rows and all secondary metrics to {output}')


def draw(frame, ranks, output):
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 230, 'pdf.fonttype': 42})
    figure, axes = plt.subplots(3, 3, figsize=(10.5, 7.8), sharex=True, sharey='row')
    titles = ('Original independent attacks', 'Matched dispersed attacks', 'Destination-grouped attacks')
    for column, (scenario, title) in enumerate(zip(SCENARIOS, titles)):
        for row, metric in enumerate(('flow_average_precision', 'flow_f1', 'group_f1')):
            axis = axes[row, column]
            for method, color, label in zip(METHODS, COLORS, LABELS):
                data = frame[(frame.scenario == scenario) & (frame.method == method)]
                for seed, marker in MARKERS.items():
                    values = data[data.seed == seed].sort_values('gamma')
                    axis.plot(values.gamma, values[metric], color=color, alpha=.17,
                              marker=marker, markersize=3, linewidth=.6)
                values = data.groupby('gamma')[metric].agg(['mean', 'std'])
                axis.errorbar(values.index, values['mean'], yerr=values['std'], color=color,
                               marker='D', markersize=3.5, capsize=2, linewidth=1.5, label=label)
            axis.set_ylim(-.02, 1.035)
            axis.set_xticks([0, .25, .5, .75, 1])
            axis.grid(alpha=.2)
            if row == 0:
                axis.set_title(title, fontsize=10)
            if row == 2:
                axis.set_xlabel('Nonlinearity gamma')
            if column == 0:
                axis.set_ylabel(('OD-flow AP', 'OD-flow F1', 'Destination-group F1')[row])
    handles = [Line2D([], [], color=color, marker='D', label=label) for color, label in zip(COLORS, LABELS)]
    handles += [Line2D([], [], color='#555555', linestyle='', marker=marker, label=f'Seed {seed}')
                for seed, marker in MARKERS.items()]
    figure.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=8)
    figure.tight_layout(rect=(0, .09, 1, 1))
    figure.savefig(output / 'method-comparison.png', bbox_inches='tight')
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(6.7, 3.6))
    rank_frame = pd.DataFrame(ranks)
    for metric, color in (('rank_95', '#0072B2'), ('rank_99', '#D55E00')):
        values = rank_frame.groupby('gamma')[metric].agg(['mean', 'std'])
        axis.errorbar(values.index, values['mean'], yerr=values['std'], marker='D',
                      capsize=3, color=color, label=metric.replace('_', ' ').upper())
    axis.axhline(5, color='#009E73', linestyle='--', label='Original OnlineDA rank 5')
    axis.axhline(16, color='#CC79A7', linestyle=':', label='Capacity-control rank 16')
    axis.set(xlabel='Nonlinearity gamma', ylabel='Effective linear rank',
             title='Normal link traffic: high-rank nonlinear stress test')
    axis.set_xticks([0, .25, .5, .75, 1])
    axis.grid(alpha=.2)
    axis.legend(fontsize=8, frameon=False, loc='lower right')
    figure.tight_layout()
    figure.savefig(output / 'normal-geometry.png', bbox_inches='tight')
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='runs/high_rank_sensitivity')
    parser.add_argument('--output', default='output/high-rank-sensitivity')
    args = parser.parse_args()
    export(args.source, args.output)


if __name__ == '__main__':
    main()
