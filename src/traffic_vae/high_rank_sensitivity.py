"""Exploratory high-linear-rank, two-latent-dimensional nonlinear stress test.

One declared generator, no search for favorable test outcomes. Original
Experiments 1--3 and their report remain unchanged. All new models and arrays
are retained under a separate output directory.
"""
import argparse
import json
import logging
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import ndtr
import torch
from threadpoolctl import threadpool_limits

from .anomalography import OnlineAnomalography
from .data import sha256
from .evaluation import evaluate
from .experiment import write_json
from .experiment_03 import run as run_group, solver_summary, summarize, synthetic_metrics
from .experiment_03_online import independent_stream, run as run_online
from .paper_experiment import load_config, run_case, summarize as paper_summary
from .paper_synthetic import generate
from .synthetic import spectrum


def fourier_features(latent, modes):
    """A smooth two-dimensional manifold, not independent high-dimensional noise.

Gaussian coordinates map to uniform angles. Distinct non-opposite integer
frequencies give orthogonal sine/cosine pairs in population.
"""
    latent, modes = np.asarray(latent), np.asarray(modes)
    if latent.ndim != 2 or latent.shape[1] != 2 or modes.ndim != 2 or modes.shape[1] != 2:
        raise ValueError('This stress generator requires exactly two latent coordinates.')
    if not np.isfinite(latent).all() or not np.isfinite(modes).all() or not np.equal(modes, np.round(modes)).all():
        raise ValueError('Finite latent values and integer Fourier modes required.')
    unique = set()
    for row in modes:
        key = tuple(int(value) for value in row)
        opposite = tuple(-value for value in key)
        if key == (0, 0) or key in unique or opposite in unique:
            raise ValueError('Zero, duplicate, and opposite modes are not allowed.')
        unique.add(key)
    angles = 2 * np.pi * ndtr(latent)
    phases = angles @ modes.T
    return np.sqrt(2.) * np.column_stack([np.sin(phases), np.cos(phases)])


def replace_nonlinear(base, seed, settings):
    """Equal-power routed modes using topology only; scale/center use normal train.

OD loadings are a minimum-norm routing preimage of random orthonormal link
directions. This deliberately constructed mathematical stress test is not a
claim of realistic independently sampled OD traffic.
"""
    modes = np.asarray(settings['fourier_modes'], dtype=int)
    width = 2 * len(modes)
    routing = base.network['R']
    left, singular, right = np.linalg.svd(routing, full_matrices=False)
    rank = int(np.sum(singular > singular[0] * max(routing.shape) * np.finfo(float).eps))
    if width > rank:
        raise ValueError('Too many Fourier features for the routing column space.')
    rng = np.random.default_rng(np.random.SeedSequence([seed, settings['generator_seed_namespace']]))
    coordinates, _ = np.linalg.qr(rng.normal(size=(rank, width)))
    directions = left[:, :rank] @ coordinates
    loadings = (right[:rank].T / singular[:rank]) @ coordinates
    np.testing.assert_allclose(routing @ loadings, directions, atol=1e-12, rtol=1e-12)
    features = {split: fourier_features(values, modes) for split, values in base.latent.items()}
    center = features['train'].mean(axis=0)
    values = {split: (feature - center) @ loadings.T for split, feature in features.items()}
    target = float(np.sqrt(np.mean(base.route(base.linear['train'], 'train')**2)))
    raw = float(np.sqrt(np.mean(base.route(values['train'], 'train')**2)))
    scale = target / raw
    base.nonlinear = {split: value * scale for split, value in values.items()}
    base.nonlinear_scale = scale
    base.maps.update(H_modes=modes, H_od_loadings=loadings,
                     H_link_directions=directions, H_training_feature_center=center)
    diagnostic = spectrum(base.route(base.nonlinear['train'], 'train'))
    squared = np.square(diagnostic['singular_values'])
    diagnostic['top5_energy_fraction'] = float(np.sum(squared[:5]) / np.sum(squared))
    diagnostic['intrinsic_latent_dimension'] = 2
    diagnostic['fourier_feature_count'] = width
    diagnostic['training_routed_rms'] = target
    if (diagnostic['rank_95'] < settings['minimum_training_rank_95']
            or diagnostic['top5_energy_fraction'] > settings['maximum_training_top5_energy']):
        raise ValueError('Declared training-only high-rank requirement not satisfied; do not tune using test outcomes.')
    return diagnostic


def save_generator(base, path):
    np.savez_compressed(path, **base.network, **base.maps, nonlinear_scale=base.nonlinear_scale,
                        **{f'W_{key}': value for key, value in base.latent.items()},
                        **{f'R_bank_{key}': value for key, value in base.routing_banks.items()},
                        **{f'R_ids_{key}': value for key, value in base.routing_ids.items()},
                        **{f'mask_{key}': value for key, value in base.masks.items()})


def rank_control(config, group_source, paper_source, output, rank, logger):
    """Change only OnlineDA's rank, not the data, penalties, or threshold rule."""
    output.mkdir()
    write_json(output / 'config.json', dict(rank=rank, paper_source=str(paper_source),
                                           comparison_source=str(group_source),
                                           parameter_selection='Fixed before test outcomes'))
    records, source_hashes = [], []
    for seed in config['seeds']:
        with np.load(group_source / f'synthetic/seed_{seed}/attacks.npz') as saved:
            flows = saved['flows'].copy()
            attacks_new = {name: saved[name].copy() for name in ('matched_dispersed', 'destination_grouped')}
        groups = [np.flatnonzero(flows[:, 1] == node) for node in np.unique(flows[:, 1])]
        for index, gamma in enumerate(config['gammas']):
            source = paper_source / f'seed_{seed}/gamma_{index:02d}'
            with np.load(source / 'truth.npz') as saved:
                truth = {name: saved[name].copy() for name in saved.files}
            routing = truth['R']
            directory = output / f'seed_{seed}/gamma_{index:02d}'
            directory.mkdir(parents=True)
            tracker = OnlineAnomalography(rank, config['da_ridge_penalty'], config['da_sparse_penalty'],
                config['lasso_max_iterations'], config['lasso_tolerance'], routing=routing,
                forgetting_factor=config['da_forgetting_factor'], output_after_update=True)
            warmup = config['da_warmup_size']
            tracker.initialize(truth['D_train'][:warmup])
            # Static full observations, exactly the same training stream as rank 5.
            training_history = []
            for start in range(warmup, len(truth['D_train']), 1000):
                result = tracker.process(truth['D_train'][start:start + 1000])
                training_history.extend({**row, 'time_index': start + row['time_index']} for row in result.history)
                logger.info('rank %d seed %d gamma %.2f train | %d/%d', rank, seed, gamma,
                            min(start + 1000, len(truth['D_train'])), len(truth['D_train']))
            np.savez_compressed(directory / 'training_state.npz', basis=tracker.basis,
                                gram=tracker.gram, cross=tracker.cross)
            calibration = independent_stream(tracker, truth['D_calibration'], logger,
                                             f'rank {rank} seed {seed} gamma {gamma} calibration')
            np.savez_compressed(directory / 'calibration.npz', A_hat=calibration.sparse)
            pd.DataFrame(training_history).to_csv(directory / 'train_solver_history.csv', index=False)
            pd.DataFrame(calibration.history).to_csv(directory / 'calibration_solver_history.csv', index=False)
            for scenario, attacks in {'paper_independent': truth['A'], **attacks_new}.items():
                scenario_dir = directory / scenario
                scenario_dir.mkdir()
                observations = truth['D_test'] if scenario == 'paper_independent' else (
                    truth['L_test'] + truth['N_test'] + attacks @ routing.T)
                result = independent_stream(tracker, observations, logger,
                                             f'rank {rank} seed {seed} gamma {gamma} {scenario}')
                metrics = synthetic_metrics(result.sparse, calibration.sparse, attacks, groups,
                                            config['target_fpr'], routing)
                fixed = evaluate((attacks != 0).ravel().astype(int), np.abs(result.sparse).ravel(),
                                 config['paper_threshold'])
                quality = solver_summary(result.history)
                records.append(dict(seed=seed, gamma=gamma, scenario=scenario, method=f'OnlineDA_rank{rank}',
                                    **metrics, **{f'solver_{key}': value for key, value in quality.items()},
                                    **{f'paper_threshold_flow_{key}': value for key, value in fixed.items()}))
                np.savez_compressed(scenario_dir / 'estimates.npz', A_hat=result.sparse, L_hat=result.low_rank)
                pd.DataFrame(result.history).to_csv(scenario_dir / 'solver_history.csv', index=False)
                write_json(scenario_dir / 'solver.json', {'test': quality,
                           'calibration': solver_summary(calibration.history), 'train': solver_summary(training_history)})
                logger.info('RESULT rank %d seed %d gamma %.2f %s | F1 %.4f | AP %.4f', rank, seed, gamma,
                            scenario, metrics['flow_f1'], metrics['flow_average_precision'])
            source_hashes.append(dict(seed=seed, gamma=gamma, truth_sha256=sha256(source / 'truth.npz'),
                                     trained_state_sha256=sha256(directory / 'training_state.npz')))
            frame = pd.DataFrame(records)
            frame.to_csv(output / 'synthetic_metrics.csv', index=False)
    summarize(pd.DataFrame(records), ['scenario', 'gamma', 'method'], output / 'synthetic_summary.csv')
    write_json(output / 'source_hashes.json', source_hashes)
    write_json(output / 'completed.json', {'records': len(records)})


def run(settings_path):
    settings = json.loads(Path(settings_path).read_text())
    config = load_config(settings['paper_config'])
    if config['linear_rank'] != 2 or config['routing_alpha'] != 0 or config['observation_probability'] != 1:
        raise ValueError('This stress test requires the existing static, full-observation two-latent protocol.')
    if settings['test_label_parameter_selection'] or settings['replace_original_results']:
        raise ValueError('Do not select a generator from test labels or overwrite the original experiment.')
    root = Path(settings['output_dir'])
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / 'settings.json', settings)
    write_json(root / 'provenance.json', {
        'python': sys.version, 'platform': platform.platform(),
        'source_sha256': {path.name: sha256(path) for path in Path(__file__).parent.glob('*.py')},
        'design': 'One Fourier-manifold generator declared before evaluating new test metrics',
        'model_selection': 'Original training/validation early stopping only; no test tuning',
        'original_report_changed': False,
    })
    logger = logging.getLogger('high_rank_sensitivity')
    logger.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(root / 'run.log')):
        handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s', '%H:%M:%S'))
        logger.addHandler(handler)
    started = time.perf_counter()
    paper_dir = root / 'experiment_02'
    paper_dir.mkdir()
    config['output_dir'] = str(paper_dir.resolve())
    write_json(paper_dir / 'config.json', config)
    write_json(paper_dir / 'provenance.json', {
        'source_sha256': {path.name: sha256(path) for path in Path(__file__).parent.glob('*.py')},
        'generator': 'Two-dimensional equal-power 16-feature Fourier manifold; topology-based routed loading',
        'exploratory_stress_test': True, 'test_label_tuning': False,
        'online_adaptation': 'Same normal-prefix SVD warm start and causal Algorithm-2 updates',
    })
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    metrics, ranks, audits = [], [], []
    with threadpool_limits(limits=config['threads']):
        for seed in config['seeds']:
            base = generate(config, seed)
            audit = replace_nonlinear(base, seed, settings)
            audits.append(dict(seed=seed, **audit))
            logger.info('TRAIN-ONLY GEOMETRY seed %d | rank95 %d | top5 energy %.4f',
                        seed, audit['rank_95'], audit['top5_energy_fraction'])
            directory = paper_dir / f'seed_{seed}'
            directory.mkdir()
            save_generator(base, directory / 'generator.npz')
            with np.load(Path(settings['original_source']) / f'seed_{seed}/gamma_00/truth.npz') as original:
                _, _, linear = base.observed(0)
                for split in ('train', 'validation', 'calibration', 'test'):
                    np.testing.assert_array_equal(linear[split], original[f'D_{split}'])
                np.testing.assert_array_equal(base.attacks, original['A'])
                np.testing.assert_array_equal(base.network['R'], original['R'])
            for index, gamma in enumerate(config['gammas']):
                rows, rank_row = run_case(config, base, seed, gamma, directory / f'gamma_{index:02d}', logger)
                metrics.extend(rows)
                ranks.append(rank_row)
                pd.DataFrame(metrics).to_csv(paper_dir / 'metrics.csv', index=False)
                pd.DataFrame(ranks).to_csv(paper_dir / 'rank_diagnostics.csv', index=False)
            write_json(root / 'training_geometry.json', audits)
        paper_summary(pd.DataFrame(metrics), paper_dir)
        write_json(paper_dir / 'completed.json', {'runs': len(metrics), 'new_vae_models': len(metrics) // 2})
    group_config = json.loads(Path(settings['group_config']).read_text())
    group_config.update(synthetic_source=str(paper_dir), output_dir=str(root / 'experiment_03'))
    group_path = root / 'group_config.json'
    write_json(group_path, group_config)
    run_group(group_path, ['synthetic'])
    run_online(root / 'experiment_03', root / 'onlineda')
    with threadpool_limits(limits=config['threads']):
        rank_control(config, root / 'experiment_03', paper_dir, root / 'rank_control',
                     settings['online_rank_control'], logger)
    write_json(root / 'completed.json', {'elapsed_seconds': time.perf_counter() - started,
               'new_vae_models': 15, 'gamma_settings': config['gammas'], 'seeds': config['seeds'],
               'original_report_changed': False})
    logger.info('COMPLETED high-rank sensitivity: %s', root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='high-rank-sensitivity.json')
    args = parser.parse_args()
    run(args.config)


if __name__ == '__main__':
    main()
