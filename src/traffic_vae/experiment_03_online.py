"""Add the unchanged Algorithm-2 OnlineDA baseline to synthetic Experiment 3.

The original two-decoder run is read-only. Each stream starts from a separate
copy of the normal-trained Experiment-2 state. No VAE is trained or evaluated.
"""
import argparse
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .anomalography import OnlineAnomalography, OnlineResult
from .data import sha256
from .experiment import write_json
from .experiment_03 import solver_summary, summarize, synthetic_metrics


def restore_tracker(path, routing, config):
    """Restore all sufficient statistics, not just the fitted basis."""
    tracker = OnlineAnomalography(
        config['da_rank'], config['da_ridge_penalty'], config['da_sparse_penalty'],
        config['lasso_max_iterations'], config['lasso_tolerance'], routing=routing,
        forgetting_factor=config['da_forgetting_factor'], output_after_update=True)
    with np.load(path) as state:
        tracker.basis, tracker.gram, tracker.cross = (
            state[key].copy() for key in ('basis', 'gram', 'cross'))
    links, rank = routing.shape[0], tracker.rank
    for value, shape in ((tracker.basis, (links, rank)),
                         (tracker.gram, (links, rank, rank)),
                         (tracker.cross, (links, rank))):
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError('Invalid saved OnlineDA state.')
    return tracker


def independent_stream(trained, observations, logger, name, chunk_size=500):
    """Chunk for logging only; updates remain continuous within this stream."""
    tracker = trained.copy()
    parts, history = [], []
    for start in range(0, len(observations), chunk_size):
        result = tracker.process(observations[start:start + chunk_size])
        parts.append(result)
        history.extend({**row, 'time_index': start + row['time_index']} for row in result.history)
        logger.info('%s | %d/%d', name, min(start + chunk_size, len(observations)), len(observations))
    return OnlineResult(np.concatenate([part.low_rank for part in parts]),
                        np.concatenate([part.sparse for part in parts]), history)


def run(source, output):
    source, output = Path(source), Path(output)
    config = json.loads((source / 'config.json').read_text())
    original = Path(config['synthetic_source'])
    paper_config = json.loads((original / 'config.json').read_text())
    original_provenance = json.loads((original / 'provenance.json').read_text())
    algorithm_path = Path(__file__).with_name('anomalography.py')
    if sha256(algorithm_path) != original_provenance['source_sha256']['anomalography.py']:
        raise ValueError('OnlineDA implementation differs from the Experiment-2 baseline.')
    expected_sources = {(row['seed'], row['gamma']): row for row in
                        json.loads((source / 'synthetic/source_hashes.json').read_text())}
    output.mkdir(parents=True, exist_ok=False)
    online_config = {
        'comparison_source': str(source), 'synthetic_source': str(original),
        'seeds': config['seeds'], 'synthetic_gamma_indices': config['synthetic_gamma_indices'],
        'target_fpr': config['target_fpr'],
        'parameters': {key: paper_config[key] for key in paper_config if
                       key.startswith(('da_', 'lasso_'))},
        'scenario_start': 'Independent copies of the same normal-trained state; causal updates within each stream',
        'calibration': 'Independent normal stream, replayed and checked against Experiment 2',
        'paper_independent': 'Full replay checked against the original cached estimates',
        'new_vae_training': False, 'test_label_tuning': False,
    }
    write_json(output / 'config.json', online_config)
    write_json(output / 'provenance.json', {
        'python': sys.version, 'numpy': np.__version__,
        'source_sha256': {name: sha256(Path(__file__).with_name(name)) for name in
                          ('anomalography.py', 'experiment_03.py', 'experiment_03_online.py', 'evaluation.py')},
        'comparison_config_sha256': sha256(source / 'config.json'),
        'comparison_metrics_sha256': sha256(source / 'synthetic_metrics.csv'),
        'paper_config_sha256': sha256(original / 'config.json'),
    })
    logger = logging.getLogger('experiment_03_online')
    logger.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(output / 'run.log')):
        handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s', '%H:%M:%S'))
        logger.addHandler(handler)
    started = time.perf_counter()
    records, provenance = [], []
    with threadpool_limits(limits=config['threads']):
        for seed in config['seeds']:
            attacks_path = source / f'synthetic/seed_{seed}/attacks.npz'
            with np.load(attacks_path) as saved:
                flows = saved['flows'].copy()
                new_attacks = {key: saved[key].copy() for key in ('matched_dispersed', 'destination_grouped')}
            groups = [np.flatnonzero(flows[:, 1] == node) for node in np.unique(flows[:, 1])]
            for index in config['synthetic_gamma_indices']:
                case_source = original / f'seed_{seed}/gamma_{index:02d}'
                truth_path = case_source / 'truth.npz'
                with np.load(truth_path) as saved:
                    truth = {key: saved[key].copy() for key in saved.files}
                gamma, routing = float(truth['gamma']), truth['R']
                expected = expected_sources[(seed, gamma)]
                if sha256(truth_path) != expected['truth_sha256']:
                    raise ValueError('Truth arrays changed since the two-decoder Experiment-3 run.')
                if not all(truth[f'mask_{split}'].all() and not truth[f'routing_ids_{split}'].any()
                           for split in ('calibration', 'test')):
                    raise ValueError('Expected the same fully observed static-routing data.')
                state_path = case_source / 'onlineda/training_state.npz'
                trained = restore_tracker(state_path, routing, paper_config)
                case_dir = output / f'seed_{seed}/gamma_{index:02d}'
                case_dir.mkdir(parents=True)
                prefix = f'seed {seed} gamma {gamma:.2f}'
                calibration = independent_stream(trained, truth['D_calibration'], logger, prefix + ' calibration')
                with np.load(case_source / 'onlineda/estimates.npz') as cached:
                    np.testing.assert_allclose(calibration.sparse, cached['calibration_A_hat'], rtol=1e-10, atol=1e-12)
                    expected_original = cached['A_hat'].copy()
                np.savez_compressed(case_dir / 'calibration.npz', A_hat=calibration.sparse)
                pd.DataFrame(calibration.history).to_csv(case_dir / 'calibration_solver_history.csv', index=False)
                for scenario, attacks in {'paper_independent': truth['A'], **new_attacks}.items():
                    scenario_dir = case_dir / scenario
                    scenario_dir.mkdir()
                    observations = truth['D_test'] if scenario == 'paper_independent' else (
                        truth['L_test'] + truth['N_test'] + attacks @ routing.T)
                    result = independent_stream(trained, observations, logger, prefix + ' ' + scenario)
                    if scenario == 'paper_independent':
                        np.testing.assert_allclose(result.sparse, expected_original, rtol=1e-10, atol=1e-12)
                    metrics = synthetic_metrics(result.sparse, calibration.sparse, attacks, groups,
                                                config['target_fpr'], routing)
                    quality = solver_summary(result.history)
                    records.append(dict(seed=seed, gamma=gamma, scenario=scenario, method='OnlineDA',
                                        **metrics, **{f'solver_{key}': value for key, value in quality.items()}))
                    np.savez_compressed(scenario_dir / 'estimates.npz', A_hat=result.sparse, L_hat=result.low_rank)
                    pd.DataFrame(result.history).to_csv(scenario_dir / 'solver_history.csv', index=False)
                    write_json(scenario_dir / 'solver.json', {'onlineda': quality,
                               'onlineda_calibration': solver_summary(calibration.history)})
                    logger.info('RESULT %s %s | flow F1 %.4f | AP %.4f | group F1 %.4f',
                                prefix, scenario, metrics['flow_f1'], metrics['flow_average_precision'], metrics['group_f1'])
                provenance.append(dict(seed=seed, gamma=gamma, truth_sha256=sha256(truth_path),
                                       training_state_sha256=sha256(state_path), attacks_sha256=sha256(attacks_path),
                                       original_estimates_sha256=sha256(case_source / 'onlineda/estimates.npz'),
                                       calibration_replay_matches=True, original_test_replay_matches=True))
    frame = pd.DataFrame(records)
    frame.to_csv(output / 'synthetic_metrics.csv', index=False)
    summarize(frame, ['scenario', 'gamma', 'method'], output / 'synthetic_summary.csv')
    write_json(output / 'source_hashes.json', provenance)
    write_json(output / 'completed.json', {'records': len(frame), 'original_test_replays': len(provenance),
               'calibration_replays': len(provenance), 'elapsed_seconds': time.perf_counter() - started})
    logger.info('Completed OnlineDA extension: %s', output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='runs/experiment_03_full')
    parser.add_argument('--output', default='runs/experiment_03_onlineda')
    args = parser.parse_args()
    run(args.source, args.output)


if __name__ == '__main__':
    main()
