"""Matched VAE+Lasso / VAE+Group-Lasso experiments on real and routed data.

The real-data arm is identity-design score denoising, not routed OD recovery.
The synthetic arm freezes the saved Experiment-2 VAEs and changes only decoding.
All settings are declared in experiment-03.json before looking at test metrics.
"""
import argparse
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from .anomalography import solve_lasso
from .data import sha256
from .evaluation import calibrate, evaluate
from .experiment import write_json
from .group_sparse import shrink_scores, solve_group_lasso_batch
from .model import VAE
from .paper_experiment import reconstruct


def metadata_groups(frame, row_indices, window_seconds):
    selected = frame.iloc[np.asarray(row_indices, dtype=int)]
    stamps = pd.to_datetime(selected['Timestamp'], format='mixed', dayfirst=True, errors='raise')
    if stamps.isna().any() or selected['Destination IP'].isna().any():
        raise ValueError('Missing group metadata.')
    keys = pd.MultiIndex.from_arrays([selected['Destination IP'].astype(str),
                                     stamps.dt.floor(f'{window_seconds}s')])
    return pd.factorize(keys, sort=False)[0]


def group_scores(scores, ids):
    sizes = np.bincount(ids)
    return np.sqrt(np.bincount(ids, weights=np.asarray(scores)**2) / sizes)


def real_metrics(scores, calibration, labels, ids, calibration_ids, target_fpr):
    result = {f'flow_{key}': value for key, value in evaluate(
        labels, scores, calibrate(calibration, target_fpr)).items()}
    test_group = group_scores(scores, ids)
    normal_group = group_scores(calibration, calibration_ids)
    group_labels = (np.bincount(ids, weights=labels) > 0).astype(int)
    result.update({f'group_{key}': value for key, value in evaluate(
        group_labels, test_group, calibrate(normal_group, target_fpr)).items()})
    return result


def real_arm(config, output, logger):
    source = Path(config['real_source'])
    manifest = pd.read_csv(source / 'split_manifest.csv')
    metadata = {}
    for name, path in [('normal', config['normal_csv']), ('test', config['test_csv'])]:
        frame = pd.read_csv(path, usecols=lambda col: col.strip() in ('Destination IP', 'Timestamp', 'Label'))
        frame.columns = frame.columns.str.strip()
        metadata[name] = frame
    selected = {name: manifest.loc[manifest.split == name, 'source_row'].to_numpy()
                for name in ('validation', 'test')}
    ids = metadata_groups(metadata['test'], selected['test'], config['real_window_seconds'])
    normal_ids = metadata_groups(metadata['normal'], selected['validation'], config['real_window_seconds'])
    labels = (metadata['test'].iloc[selected['test']]['Label'].str.strip().str.upper() == 'DDOS').to_numpy(dtype=int)
    if metadata['normal'].iloc[selected['validation']]['Label'].str.strip().str.upper().ne('BENIGN').any():
        raise ValueError('Normal calibration contains attacks.')
    directory = output / 'real'
    directory.mkdir()
    np.savez_compressed(directory / 'groups.npz', test_ids=ids, calibration_ids=normal_ids,
                        test_rows=selected['test'], calibration_rows=selected['validation'], labels=labels)
    sizes = np.bincount(ids)
    positives = np.bincount(ids, weights=labels)
    audit = {
        'test_flows': len(ids), 'calibration_flows': len(normal_ids),
        'test_groups': len(sizes), 'calibration_groups': len(np.unique(normal_ids)),
        'singleton_group_fraction': float(np.mean(sizes == 1)),
        'test_group_size_median': float(np.median(sizes)), 'test_group_size_max': int(sizes.max()),
        'mixed_label_groups': int(np.sum((positives > 0) & (positives < sizes))),
        'attacked_groups': int(np.sum(positives > 0)),
        'grouping': 'Destination IP and fixed 60-second wall-clock window; no label-based grouping',
        'normal_csv_sha256': sha256(Path(config['normal_csv'])),
        'test_csv_sha256': sha256(Path(config['test_csv'])),
        'limitations': [
            'Identity-design denoising of scalar anomaly scores, not routed attack-amplitude recovery',
            'Groups are evaluated after the complete window; this is not a causal online detector',
            'Reused Monday validation was also used for VAE early stopping in Experiment 1',
            'One fixed lab DDoS capture; not evidence for arbitrary real-world attacks',
        ],
    }
    source_provenance = json.loads((source / 'provenance.json').read_text())
    source_hashes = {Path(path).name: digest for path, digest in source_provenance['input_sha256'].items()}
    for key in ('normal', 'test'):
        if source_hashes[Path(config[f'{key}_csv']).name] != audit[f'{key}_csv_sha256']:
            raise ValueError('CSV files differ from those used to produce the cached VAE scores.')
    write_json(directory / 'data_audit.json', audit)
    records, provenance = [], []
    for seed in config['seeds']:
        normal_path = source / f'seed_{seed}/calibration_scores.csv'
        test_path = source / f'seed_{seed}/test_predictions.csv'
        normal = pd.read_csv(normal_path)
        test = pd.read_csv(test_path)
        np.testing.assert_array_equal(normal.source_row, selected['validation'])
        np.testing.assert_array_equal(test.source_row, selected['test'])
        np.testing.assert_array_equal(test.label, labels)
        normal_scores, scores = normal.score.to_numpy(), test.score.to_numpy()
        base_penalty = float(np.median(normal_scores))
        if base_penalty <= 0:
            raise ValueError('Normal-score median must be positive.')
        seed_dir = directory / f'seed_{seed}'
        seed_dir.mkdir()
        row = real_metrics(scores, normal_scores, labels, ids, normal_ids, config['target_fpr'])
        records.append(dict(seed=seed, method='VAE', penalty_multiplier=0., penalty=0., **row))
        arrays = {'vae': scores}
        for multiplier in config['real_penalty_multipliers']:
            penalty = multiplier * base_penalty
            variants = {
                'VAE_Lasso': (np.maximum(scores - penalty, 0), np.maximum(normal_scores - penalty, 0)),
                'VAE_GroupLasso': (shrink_scores(scores, ids, penalty),
                                   shrink_scores(normal_scores, normal_ids, penalty)),
            }
            for method, (estimate, calibration) in variants.items():
                row = real_metrics(estimate, calibration, labels, ids, normal_ids, config['target_fpr'])
                records.append(dict(seed=seed, method=method, penalty_multiplier=multiplier,
                                    penalty=penalty, **row))
                if multiplier == config['real_primary_multiplier']:
                    arrays[method] = estimate
                logger.info('Real seed %d | %s | multiplier %.1f | flow F1 %.4f | AP %.4f',
                            seed, method, multiplier, row['flow_f1'], row['flow_average_precision'])
        np.savez_compressed(seed_dir / 'scores.npz', **arrays)
        provenance.append({'seed': seed, 'scores_sha256': sha256(test_path),
                           'normal_scores_sha256': sha256(normal_path),
                           'model_sha256': sha256(source / f'seed_{seed}/model.pt')})
    write_json(directory / 'source_hashes.json', provenance)
    return pd.DataFrame(records)


def matched_attacks(flows, size, attacked_times, seed):
    """Equal OD support/energy and attacked times; only support grouping differs.

    Each active time has n_nodes-1 positive unit attacks. Destination grouping is
    topological, not inferred from labels. Routed energy may differ and is saved.
    """
    nodes = np.unique(flows[:, 1])
    groups = [np.flatnonzero(flows[:, 1] == node) for node in nodes]
    width = len(groups[0])
    if any(len(group) != width for group in groups) or not 0 < attacked_times < size:
        raise ValueError('Expected equal OD destination groups and valid attacked-time count.')
    rng = np.random.default_rng(np.random.SeedSequence([seed, 3003]))
    times = np.sort(rng.choice(size, attacked_times, replace=False))
    dispersed, grouped = np.zeros((size, len(flows))), np.zeros((size, len(flows)))
    for index in times:
        dispersed[index, rng.choice(len(flows), width, replace=False)] = 1.
        grouped[index, groups[int(rng.integers(len(groups)))]] = 1.
    return {'matched_dispersed': dispersed, 'destination_grouped': grouped}, groups


def synthetic_metrics(estimate, calibration, truth, groups, target_fpr, routing):
    result = {}
    scores = {
        'flow': (np.abs(estimate).ravel(), np.abs(calibration).ravel(), (truth != 0).ravel()),
        'group': (np.stack([np.linalg.norm(estimate[:, group], axis=1) for group in groups], axis=1).ravel(),
                  np.stack([np.linalg.norm(calibration[:, group], axis=1) for group in groups], axis=1).ravel(),
                  np.stack([np.any(truth[:, group] != 0, axis=1) for group in groups], axis=1).ravel()),
        'time': (np.linalg.norm(estimate, axis=1), np.linalg.norm(calibration, axis=1), np.any(truth != 0, axis=1)),
    }
    for level, (score, normal_score, labels) in scores.items():
        threshold = calibrate(normal_score, target_fpr)
        result.update({f'{level}_{key}': value for key, value in evaluate(
            labels.astype(int), score, threshold).items()})
        result[f'calibration_{level}_false_positive_rate'] = float(np.mean(normal_score > threshold))
    result['relative_error_A'] = float(np.linalg.norm(estimate - truth) / np.linalg.norm(truth))
    result['relative_error_routed_A'] = float(np.linalg.norm((estimate - truth) @ routing.T)
                                              / np.linalg.norm(truth @ routing.T))
    return result


def decode_group(residuals, routing, groups, config, logger, name):
    estimates, history = [], []
    for start in range(0, len(residuals), 512):
        estimated, diagnostics = solve_group_lasso_batch(
            routing, residuals[start:start + 512], groups, config['synthetic_penalty'],
            config['group_max_iterations'], config['group_kkt_tolerance'], config['batch_size'])
        estimates.append(estimated)
        history.extend(diagnostics)
        logger.info('%s | Group Lasso %d/%d | max KKT %.2g', name, min(start + 512, len(residuals)),
                    len(residuals), max(row['relative_kkt_residual'] for row in diagnostics))
    return np.concatenate(estimates), history


def decode_lasso(residuals, routing, config, logger, name):
    estimates, history = [], []
    for index, residual in enumerate(residuals):
        estimate, diagnostics = solve_lasso(routing, residual, config['synthetic_penalty'], 10000, 1e-7)
        estimates.append(estimate)
        history.append(diagnostics)
        if (index + 1) % 1000 == 0:
            logger.info('%s | Lasso %d/%d', name, index + 1, len(residuals))
    return np.asarray(estimates), history


def solver_summary(history):
    return {'subproblems': len(history), 'unconverged': sum(not row['converged'] for row in history),
            'max_relative_kkt_residual': max(float(row['relative_kkt_residual']) for row in history)}


def synthetic_arm(config, output, logger):
    source = Path(config['synthetic_source'])
    source_config = json.loads((source / 'config.json').read_text())
    directory = output / 'synthetic'
    directory.mkdir()
    records, audits, provenance = [], [], []
    for seed in config['seeds']:
        seed_dir = directory / f'seed_{seed}'
        seed_dir.mkdir()
        generator_path = source / f'seed_{seed}/generator.npz'
        with np.load(generator_path) as generator:
            flows = generator['flows'].copy()
        first_path = source / f'seed_{seed}/gamma_00/truth.npz'
        with np.load(first_path) as first:
            size = len(first['A'])
        cases, groups = matched_attacks(flows, size, config['matched_attacked_times'], seed)
        np.savez_compressed(seed_dir / 'attacks.npz', flows=flows, **cases)
        for gamma_index in config['synthetic_gamma_indices']:
            case_source = source / f'seed_{seed}/gamma_{gamma_index:02d}'
            case_dir = seed_dir / f'gamma_{gamma_index:02d}'
            case_dir.mkdir()
            logger.info('Synthetic seed %d | gamma index %d | loading frozen VAE', seed, gamma_index)
            checkpoint_path = case_source / 'vae_lasso/model.pt'
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            model = VAE(checkpoint['input_dim'], checkpoint['hidden_dims'], checkpoint['latent_dim'])
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            with np.load(case_source / 'truth.npz') as truth_file:
                truth = {key: truth_file[key].copy() for key in truth_file.files}
            if not all(truth[f'mask_{split}'].all() and not truth[f'routing_ids_{split}'].any()
                       for split in ('calibration', 'test')):
                raise ValueError('Experiment 3 requires the static, fully observed Experiment-2 data.')
            routing, gamma = truth['R'], float(truth['gamma'])
            mean = checkpoint['mean'].numpy()
            scale = float(checkpoint['scale'])

            def residual(observed):
                values = ((observed - mean) / scale).astype(np.float32)
                return observed - (reconstruct(model, values, 256) * scale + mean)

            normal_residual = residual(truth['D_calibration'])
            calibration_group, calibration_history = decode_group(
                normal_residual, routing, groups, config, logger, 'normal calibration')
            with np.load(case_source / 'vae_lasso/estimates.npz') as saved:
                calibration_lasso = saved['calibration_A_hat'].copy()
                original_lasso = saved['A_hat'].copy()
                expected_low = saved['L_hat'].copy()
            np.savez_compressed(case_dir / 'calibration.npz', lasso=calibration_lasso,
                                group_lasso=calibration_group)
            original_residual = residual(truth['D_test'])
            np.testing.assert_allclose(truth['D_test'] - original_residual, expected_low, rtol=1e-5, atol=1e-6)
            case_truths = {'paper_independent': truth['A'], **cases}
            for scenario, attacks in case_truths.items():
                scenario_dir = case_dir / scenario
                scenario_dir.mkdir()
                observations = truth['D_test'] if scenario == 'paper_independent' else (
                    truth['L_test'] + truth['N_test'] + attacks @ routing.T)
                current_residual = original_residual if scenario == 'paper_independent' else residual(observations)
                if scenario == 'paper_independent':
                    estimate_lasso = original_lasso
                    lasso_history = pd.read_csv(case_source / 'vae_lasso/test_solver_history.csv').to_dict('records')
                else:
                    estimate_lasso, lasso_history = decode_lasso(current_residual, routing, config, logger, scenario)
                estimate_group, group_history = decode_group(current_residual, routing, groups, config, logger, scenario)
                audit = dict(seed=seed, gamma=gamma, scenario=scenario,
                             attack_time_count=int(np.any(attacks != 0, axis=1).sum()),
                             attack_cell_count=int(np.count_nonzero(attacks)),
                             flow_attack_rms=float(np.sqrt(np.mean(attacks**2))),
                             routed_attack_rms=float(np.sqrt(np.mean((attacks @ routing.T)**2))))
                audits.append(audit)
                for method, estimate, calibration, history in (
                        ('VAE_Lasso', estimate_lasso, calibration_lasso, lasso_history),
                        ('VAE_GroupLasso', estimate_group, calibration_group, group_history)):
                    metrics = synthetic_metrics(estimate, calibration, attacks, groups, config['target_fpr'], routing)
                    quality = solver_summary(history)
                    records.append(dict(seed=seed, gamma=gamma, scenario=scenario, method=method,
                                        **metrics, **{f'solver_{key}': value for key, value in quality.items()}))
                    logger.info('RESULT seed %d gamma %.2f %s %s | group F1 %.4f | flow F1 %.4f',
                                seed, gamma, scenario, method, metrics['group_f1'], metrics['flow_f1'])
                np.savez_compressed(scenario_dir / 'estimates.npz', lasso=estimate_lasso,
                                    group_lasso=estimate_group)
                write_json(scenario_dir / 'solver.json', {'lasso': solver_summary(lasso_history),
                           'group_lasso': solver_summary(group_history),
                           'group_calibration': solver_summary(calibration_history)})
                pd.DataFrame(group_history).to_csv(scenario_dir / 'group_solver_history.csv', index=False)
            provenance.append(dict(seed=seed, gamma=gamma, model_sha256=sha256(checkpoint_path),
                                   truth_sha256=sha256(case_source / 'truth.npz'),
                                   generator_sha256=sha256(generator_path)))
    write_json(directory / 'data_audit.json', audits)
    write_json(directory / 'source_hashes.json', provenance)
    return pd.DataFrame(records)


def summarize(frame, keys, path):
    columns = [column for column in frame if column.endswith(('_f1', '_average_precision',
               '_false_positive_rate', '_precision', '_recall')) or column.startswith('relative_error')]
    summary = frame.groupby(keys)[columns].agg(['mean', 'std'])
    summary.columns = ['_'.join(column) for column in summary.columns]
    summary.to_csv(path)
    return summary.reset_index()


def run(config_path, arms):
    config = json.loads(Path(config_path).read_text())
    output = Path(config['output_dir'])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'config.json', config)
    write_json(output / 'provenance.json', {
        'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
        'source_sha256': {path.name: sha256(path) for path in Path(__file__).parent.glob('*.py')},
        'model_training': 'No new VAE training. Frozen checkpoints and unchanged source splits.',
        'test_label_tuning': False,
    })
    logger = logging.getLogger('experiment_03')
    logger.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(output / 'run.log')):
        handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s', '%H:%M:%S'))
        logger.addHandler(handler)
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    with threadpool_limits(limits=config['threads']):
        if 'real' in arms:
            real = real_arm(config, output, logger)
            real.to_csv(output / 'real_metrics.csv', index=False)
            summarize(real, ['method', 'penalty_multiplier'], output / 'real_summary.csv')
        if 'synthetic' in arms:
            synthetic = synthetic_arm(config, output, logger)
            synthetic.to_csv(output / 'synthetic_metrics.csv', index=False)
            summarize(synthetic, ['scenario', 'gamma', 'method'], output / 'synthetic_summary.csv')
    write_json(output / 'completed.json', {'arms': arms, 'elapsed_seconds': time.perf_counter() - started})
    logger.info('Completed Experiment 3: %s', output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='experiment-03.json')
    parser.add_argument('--arms', nargs='+', choices=['real', 'synthetic'], default=['real', 'synthetic'])
    args = parser.parse_args()
    run(args.config, args.arms)


if __name__ == '__main__':
    main()
