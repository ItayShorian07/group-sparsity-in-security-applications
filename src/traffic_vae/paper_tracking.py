"""Run the paper's routing/missing-data stress settings at the linear endpoint.

This is an adapted VAE-versus-OnlineDA comparison.  It follows the alpha, pi,
sigma, beta, and 1000-sample test settings used around Fig. 4 of the reference,
while retaining this project's separate normal train/validation/calibration data.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from .experiment import write_json
from .paper_experiment import load_config, run, validate_gammas


def load_tracking_config(path):
    value = json.loads(Path(path).read_text())
    required = {'base_config', 'output_dir', 'seeds', 'test_size', 'gammas',
                'da_forgetting_factor', 'scenarios'}
    if set(value) != required:
        raise ValueError(f'Tracking fields: missing {required - set(value)}, '
                         f'unknown {set(value) - required}')
    if not isinstance(value['base_config'], str) or not value['base_config']:
        raise ValueError('base_config must be a path string.')
    if not isinstance(value['output_dir'], str) or not value['output_dir']:
        raise ValueError('output_dir must be a path string.')
    if (not isinstance(value['seeds'], list) or not value['seeds']
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in value['seeds'])
            or len(set(value['seeds'])) != len(value['seeds'])):
        raise ValueError('seeds must contain distinct uint32-compatible integers.')
    if type(value['test_size']) is not int or value['test_size'] < 4:
        raise ValueError('test_size must be an integer of at least four.')
    validate_gammas(value['gammas'])
    beta = value['da_forgetting_factor']
    if type(beta) not in (int, float) or not 0 < beta <= 1:
        raise ValueError('da_forgetting_factor must lie in (0, 1].')
    if not isinstance(value['scenarios'], list) or not value['scenarios']:
        raise ValueError('scenarios must be a nonempty list.')
    names = set()
    for scenario in value['scenarios']:
        if set(scenario) != {'name', 'routing_alpha', 'observation_probability', 'noise_std'}:
            raise ValueError('Every scenario requires name, routing_alpha, '
                             'observation_probability, and noise_std.')
        if (not isinstance(scenario['name'], str) or not scenario['name']
                or scenario['name'] in names):
            raise ValueError('Scenario names must be distinct nonempty strings.')
        names.add(scenario['name'])
        alpha, pi, sigma = (scenario[key] for key in
                            ('routing_alpha', 'observation_probability', 'noise_std'))
        if not isinstance(alpha, (int, float)) or not 0 <= alpha <= 1:
            raise ValueError('routing_alpha must lie in [0, 1].')
        if not isinstance(pi, (int, float)) or not 0 < pi <= 1:
            raise ValueError('observation_probability must lie in (0, 1].')
        if not isinstance(sigma, (int, float)) or sigma < 0:
            raise ValueError('noise_std must be nonnegative.')
    return value


def run_tracking(tracking):
    root = Path(tracking['output_dir']).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / 'config.json', tracking)
    rows = []
    for scenario in tracking['scenarios']:
        config = load_config(tracking['base_config'])
        config.update(
            output_dir=str(root / scenario['name']),
            seeds=tracking['seeds'],
            test_size=tracking['test_size'],
            gammas=tracking['gammas'],
            da_forgetting_factor=tracking['da_forgetting_factor'],
            routing_alpha=scenario['routing_alpha'],
            observation_probability=scenario['observation_probability'],
            noise_std=scenario['noise_std'],
        )
        result = run(config)
        table = pd.read_csv(result / 'metrics.csv')
        table.insert(0, 'scenario', scenario['name'])
        rows.append(table)
    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(root / 'metrics.csv', index=False)
    numeric = [column for column in combined.select_dtypes('number').columns
               if column not in {'seed'}]
    summary = combined.groupby(['scenario', 'method'])[numeric].agg(['mean', 'std'])
    summary.columns = ['_'.join(column) for column in summary.columns]
    summary.to_csv(root / 'summary.csv')
    write_json(root / 'completed.json', {
        'scenarios': len(rows),
        'fits': len(combined),
        'all_subproblems_converged': bool(combined['all_subproblems_converged'].all()),
    })
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('experiment-02-tracking.json'))
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    try:
        tracking = load_tracking_config(args.config)
        if args.validate_only:
            print(f"Configuration valid | scenarios: {len(tracking['scenarios'])} | "
                  f"routing alphas: {[item['routing_alpha'] for item in tracking['scenarios']]}")
        else:
            run_tracking(tracking)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()
