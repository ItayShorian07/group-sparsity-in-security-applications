"""Export inspectable Experiment-3 results and paired-seed scientific figures."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .experiment import write_json
from .evaluation import evaluate


COLORS = {'VAE_Lasso': '#0072B2', 'VAE_GroupLasso': '#D55E00', 'OnlineDA': '#009E73'}
LABELS = {'VAE_Lasso': 'VAE + Lasso', 'VAE_GroupLasso': 'VAE + Group Lasso', 'OnlineDA': 'OnlineDA'}
VAE_METHODS = ('VAE_Lasso', 'VAE_GroupLasso')
MARKERS = {17: 'o', 42: 's', 2026: '^'}


def export(source, online_source=None):
    """Export matched seed-level results and figures from completed runs."""
    source = Path(source)
    config = json.loads((source / 'config.json').read_text())
    real, synthetic = pd.read_csv(source / 'real_metrics.csv'), pd.read_csv(source / 'synthetic_metrics.csv')
    online_payload = None
    if online_source is not None:
        online_source = Path(online_source)
        online_config = json.loads((online_source / 'config.json').read_text())
        if Path(online_config['comparison_source']).resolve() != source.resolve():
            raise ValueError('OnlineDA was run against a different Experiment-3 source.')
        online = pd.read_csv(online_source / 'synthetic_metrics.csv')
        keys = ['seed', 'gamma', 'scenario']
        expected = set(map(tuple, synthetic[keys].drop_duplicates().to_numpy()))
        actual = set(map(tuple, online[keys].to_numpy()))
        if (actual != expected or len(online) != len(expected)
                or set(online.method) != {'OnlineDA'} or online_config['target_fpr'] != config['target_fpr']):
            raise ValueError('Incomplete or mismatched OnlineDA comparison.')
        online_payload = {name: json.loads((online_source / f'{name}.json').read_text())
                          for name in ('config', 'completed', 'provenance', 'source_hashes')}
        from .data import sha256
        if online_payload['provenance']['comparison_metrics_sha256'] != sha256(source / 'synthetic_metrics.csv'):
            raise ValueError('The original comparison changed after OnlineDA was run.')
        synthetic = pd.concat([synthetic, online], ignore_index=True)
    methods = [method for method in COLORS if method in set(synthetic.method)]
    # Secondary operating point: the fixed 0.1 threshold already specified in
    # Experiment 2 / the paper. It is not selected from Experiment-3 test labels.
    original = Path(config['synthetic_source'])
    paper_threshold = json.loads((original / 'config.json').read_text())['paper_threshold']
    secondary = []
    for (seed, gamma, scenario), frame in synthetic.groupby(['seed', 'gamma', 'scenario']):
        index = [0., .25, .5, .75, 1.].index(gamma)
        case = source / f'synthetic/seed_{seed}/gamma_{index:02d}/{scenario}'
        if scenario == 'paper_independent':
            with np.load(original / f'seed_{seed}/gamma_{index:02d}/truth.npz') as values:
                truth = values['A']
        else:
            with np.load(source / f'synthetic/seed_{seed}/attacks.npz') as values:
                truth = values[scenario]
        with np.load(case / 'estimates.npz') as values:
            for method, key in [('VAE_Lasso', 'lasso'), ('VAE_GroupLasso', 'group_lasso')]:
                metrics = evaluate((truth != 0).ravel().astype(int), np.abs(values[key]).ravel(), paper_threshold)
                secondary.append(dict(seed=int(seed), gamma=float(gamma), scenario=scenario,
                                      method=method, **metrics))
        if online_source is not None:
            path = online_source / f'seed_{seed}/gamma_{index:02d}/{scenario}/estimates.npz'
            with np.load(path) as values:
                metrics = evaluate((truth != 0).ravel().astype(int), np.abs(values['A_hat']).ravel(), paper_threshold)
            secondary.append(dict(seed=int(seed), gamma=float(gamma), scenario=scenario,
                                  method='OnlineDA', **metrics))
    secondary = pd.DataFrame(secondary)
    secondary_summary = secondary.groupby(['scenario', 'gamma', 'method'])[
        ['f1', 'precision', 'recall', 'false_positive_rate']].agg(['mean', 'std'])
    secondary_summary.columns = ['_'.join(column) for column in secondary_summary.columns]
    directory = Path('output/experiment-03')
    directory.mkdir(parents=True, exist_ok=True)
    quality = []
    for path in sorted((source / 'synthetic').glob('seed_*/gamma_*/*/solver.json')):
        quality.append({'case': str(path.relative_to(source)), **json.loads(path.read_text())})
    if online_source is not None:
        for path in sorted(online_source.glob('seed_*/gamma_*/*/solver.json')):
            quality.append({'case': 'onlineda/' + str(path.relative_to(online_source)), **json.loads(path.read_text())})
    columns = [column for column in synthetic if column.endswith(('_f1', '_average_precision',
               '_false_positive_rate', '_precision', '_recall')) or column.startswith('relative_error')]
    synthetic_summary = synthetic.groupby(['scenario', 'gamma', 'method'])[columns].agg(['mean', 'std'])
    synthetic_summary.columns = ['_'.join(column) for column in synthetic_summary.columns]
    payload = {
        'config': config,
        'completed': json.loads((source / 'completed.json').read_text()),
        'provenance': json.loads((source / 'provenance.json').read_text()),
        'real_data_audit': json.loads((source / 'real/data_audit.json').read_text()),
        'synthetic_data_audit': json.loads((source / 'synthetic/data_audit.json').read_text()),
        'real_metrics_by_seed': real.to_dict('records'),
        'real_summary': pd.read_csv(source / 'real_summary.csv').to_dict('records'),
        'synthetic_metrics_by_seed': synthetic.to_dict('records'),
        'synthetic_summary': synthetic_summary.reset_index().to_dict('records'),
        'solver_quality': quality,
        'secondary_paper_threshold': paper_threshold,
        'secondary_paper_threshold_metrics': secondary.to_dict('records'),
        'secondary_paper_threshold_summary': secondary_summary.reset_index().to_dict('records'),
        'real_source_hashes': json.loads((source / 'real/source_hashes.json').read_text()),
        'synthetic_source_hashes': json.loads((source / 'synthetic/source_hashes.json').read_text()),
    }
    if online_payload is not None:
        payload['online_baseline'] = online_payload
    write_json(directory / 'results.json', payload)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 240, 'pdf.fonttype': 42})
    figures = Path('figures')
    figures.mkdir(exist_ok=True)
    primary = real[real.penalty_multiplier == config['real_primary_multiplier']]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
    for ax, metric, title in zip(axes, ['flow_f1', 'flow_false_positive_rate'],
                                 ['Real DDoS flows: F1 (higher is better)',
                                  'Benign Friday flows: FPR (lower is better)']):
        for seed, marker in MARKERS.items():
            rows = primary[primary.seed == seed].set_index('method').loc[list(VAE_METHODS)]
            ax.plot([0, 1], rows[metric], color='#BBBBBB', linewidth=.8, zorder=1)
            for index, method in enumerate(VAE_METHODS):
                ax.scatter(index, rows.loc[method, metric], marker=marker, s=29,
                           color=COLORS[method], edgecolor='white', linewidth=.4, zorder=3)
        for index, method in enumerate(VAE_METHODS):
            values = primary.loc[primary.method == method, metric]
            ax.errorbar(index + .05, values.mean(), yerr=values.std(ddof=1), color=COLORS[method],
                        marker='D', markersize=5, capsize=3, linewidth=1.5, zorder=4)
        ax.set_xticks([0, 1], ['VAE + Lasso', 'VAE + Group Lasso'])
        ax.set_xlim(-.4, 1.4)
        ax.set_title(title, fontsize=10)
        ax.grid(axis='y', alpha=.2)
    seed_handles = [Line2D([], [], marker=marker, linestyle='', color='#555555', label=f'Seed {seed}')
                    for seed, marker in MARKERS.items()]
    seed_handles += [Line2D([], [], marker='D', color='#555555', label='Mean +/- 1 SD')]
    fig.legend(handles=seed_handles, loc='lower center', ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, .11, 1, 1))
    for suffix in ('png', 'pdf'):
        fig.savefig(figures / f'experiment-03-real-comparison.{suffix}', bbox_inches='tight')
    plt.close(fig)

    scenarios = [('paper_independent', 'Original independent attacks'),
                 ('matched_dispersed', 'Matched dispersed attacks'),
                 ('destination_grouped', 'Destination-grouped attacks')]
    fig, axes = plt.subplots(3, 3, figsize=(10.3, 7.5), sharex=True, sharey='row')
    for column, (scenario, title) in enumerate(scenarios):
        for row, metric in enumerate(('flow_average_precision', 'flow_f1', 'group_f1')):
            ax = axes[row, column]
            selected = synthetic[synthetic.scenario == scenario]
            for method in methods:
                color = COLORS[method]
                data = selected[selected.method == method]
                for seed, marker in MARKERS.items():
                    values = data[data.seed == seed].sort_values('gamma')
                    ax.plot(values.gamma, values[metric], color=color, alpha=.22, marker=marker,
                            markersize=3, linewidth=.7)
                grouped = data.groupby('gamma')[metric].agg(['mean', 'std'])
                ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], color=color,
                            marker='D', markersize=4, linewidth=1.6, capsize=2, label=LABELS[method])
            if row == 0:
                ax.set_title(title, fontsize=10)
            elif row == 2:
                ax.set_xlabel('Nominal nonlinearity gamma')
            if column == 0:
                ax.set_ylabel(['OD-flow AP', 'OD-flow F1', 'Destination-group F1'][row])
            ax.set_xticks([0, .25, .5, .75, 1])
            ax.set_ylim(-.02, 1.035)
            ax.grid(alpha=.2)
    handles = [Line2D([], [], color=COLORS[method], marker='D', label=LABELS[method]) for method in methods]
    handles += seed_handles[:3]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, .07, 1, 1))
    for suffix in ('png', 'pdf'):
        fig.savefig(figures / f'experiment-03-synthetic-comparison.{suffix}', bbox_inches='tight')
    plt.close(fig)
    print('Exported output/experiment-03/results.json and two Experiment-3 figures.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='runs/experiment_03_full')
    parser.add_argument('--online-source', help='Completed synthetic OnlineDA extension; optional for older runs')
    args = parser.parse_args()
    export(args.source, args.online_source)


if __name__ == '__main__':
    main()
