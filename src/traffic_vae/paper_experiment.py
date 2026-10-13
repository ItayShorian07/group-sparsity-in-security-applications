"""Experiment 02 using the paper's linear generator and our nonlinear extension."""
import importlib.metadata
import logging
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from .anomalography import OnlineAnomalography, OnlineResult, solve_lasso
from .data import sha256
from .evaluation import calibrate, evaluate
from .experiment import write_json
from .model import VAE, train
from .paper_synthetic import generate
from .synthetic import spectrum


def load_config(path):
    import json

    config = json.loads(Path(path).read_text())
    integers = ('n_nodes', 'n_links', 'linear_rank', 'topology_max_attempts', 'train_size',
                'validation_size', 'calibration_size', 'test_size', 'latent_dim', 'epochs',
                'batch_size', 'patience', 'threads', 'da_rank', 'da_warmup_size', 'lasso_max_iterations')
    positive = ('connection_radius', 'learning_rate', 'da_ridge_penalty', 'da_sparse_penalty',
                'vae_sparse_penalty', 'lasso_tolerance', 'paper_threshold')
    fractions = ('attack_probability', 'target_fpr')
    required = set(integers + positive + fractions) | {
        'data_model', 'output_dir', 'seeds', 'gammas', 'hidden_dims', 'noise_std', 'vae_kl_weight',
        'routing_alpha', 'observation_probability', 'da_forgetting_factor'}
    if set(config) != required:
        raise ValueError(f'Config fields: missing {required - set(config)}, unknown {set(config) - required}')
    if config['data_model'] != 'paper_routed':
        raise ValueError('data_model must be paper_routed.')
    for key in integers:
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer.')
    for key in positive + fractions + ('noise_std', 'vae_kl_weight', 'routing_alpha',
                                        'observation_probability', 'da_forgetting_factor'):
        value = config[key]
        if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
            raise ValueError(f'{key} must be finite and nonnegative.')
    for key in positive:
        if config[key] == 0:
            raise ValueError(f'{key} must be positive.')
    for key in fractions:
        if not 0 < config[key] < 1:
            raise ValueError(f'{key} must lie in (0, 1).')
    if not 0 <= config['routing_alpha'] <= 1:
        raise ValueError('routing_alpha must lie in [0, 1].')
    if not 0 < config['observation_probability'] <= 1:
        raise ValueError('observation_probability must lie in (0, 1].')
    if not 0 < config['da_forgetting_factor'] <= 1:
        raise ValueError('da_forgetting_factor must lie in (0, 1].')
    nodes, links = config['n_nodes'], config['n_links']
    if nodes < 2 or links % 2 or not 2 * (nodes - 1) <= links <= nodes * (nodes - 1):
        raise ValueError('A connected bidirectional graph needs an even, feasible link count.')
    if config['linear_rank'] > min(links, nodes * (nodes - 1)):
        raise ValueError('linear_rank exceeds the observed dimension.')
    if not config['da_rank'] <= config['da_warmup_size'] < config['train_size']:
        raise ValueError('Require da_rank <= da_warmup_size < train_size.')
    if config['da_rank'] > links or config['latent_dim'] > links:
        raise ValueError('Model dimensions must not exceed n_links.')
    if min(config[f'{split}_size'] for split in ('train', 'validation', 'calibration', 'test')) < 4:
        raise ValueError('Every split requires at least four time points.')
    for key in ('seeds', 'hidden_dims'):
        values = config[key]
        if (not isinstance(values, list) or not values
                or any(type(x) is not int or x < (0 if key == 'seeds' else 1) for x in values)):
            raise ValueError(f'{key} must contain valid integers.')
    if len(set(config['seeds'])) != len(config['seeds']) or max(config['seeds']) >= 2**32:
        raise ValueError('Seeds must be distinct and less than 2**32.')
    validate_gammas(config['gammas'])
    if not isinstance(config['output_dir'], str) or not config['output_dir'].strip():
        raise ValueError('output_dir must be a nonempty path.')
    config['output_dir'] = str(Path(config['output_dir']).expanduser().resolve())
    return config


def validate_gammas(gammas):
    if (not isinstance(gammas, list) or not gammas
            or any(type(value) not in (int, float) or not np.isfinite(value)
                   or not 0 <= value <= 1 for value in gammas)
            or len(set(gammas)) != len(gammas)):
        raise ValueError('gammas must be distinct finite numbers in [0, 1].')
    if 0 not in gammas:
        raise ValueError('Include gamma=0 to retain the paper-based linear control.')


@torch.no_grad()
def reconstruct(model, values, batch_size):
    model.eval()
    result = np.concatenate([
        model(torch.from_numpy(values[start:start + batch_size]), sample=False)[0].numpy()
        for start in range(0, len(values), batch_size)
    ]).astype(np.float64)
    if not np.isfinite(result).all():
        raise FloatingPointError('Non-finite VAE reconstruction.')
    return result


def decode_flows(residuals, base, config, logger, split):
    """Infer OD anomalies from link residuals using an elementwise L1 penalty."""
    anomalies, history = [], []
    for index, residual in enumerate(residuals):
        routing = base.routing_banks[split][base.routing_ids[split][index]]
        mask = base.masks[split][index]
        anomaly, diagnostics = solve_lasso(routing[mask], residual[mask], config['vae_sparse_penalty'],
                                           config['lasso_max_iterations'], config['lasso_tolerance'])
        anomalies.append(anomaly)
        history.append({'time_index': index, 'observed_links': int(mask.sum()), **diagnostics})
        if (index + 1) % 1000 == 0:
            logger.info('VAE flow decoding | %s | %d/%d', split, index + 1, len(residuals))
    return np.asarray(anomalies), history


def process_online(tracker, values, base, logger, split, offset=0):
    """Process one continuous stream in chunks used only for progress reporting."""
    results, history = [], []
    for start in range(0, len(values), 1000):
        stop = start + 1000
        route_start, route_stop = offset + start, offset + stop
        result = tracker.process(
            values[start:stop],
            routings=base.expanded_routing(split, route_start, route_stop),
            masks=base.masks[split][route_start:route_stop],
        )
        history.extend({**row, 'time_index': offset + start + row['time_index']}
                       for row in result.history)
        results.append(result)
        logger.info('OnlineDA | %s | %d/%d', split, min(start + 1000, len(values)), len(values))
    return OnlineResult(np.concatenate([result.low_rank for result in results]),
                        np.concatenate([result.sparse for result in results]), history)


def detection_metrics(estimated, calibration, truth, target_fpr, fixed_threshold):
    metrics = {}
    scores = {
        'time': (np.linalg.norm(estimated, axis=1), np.linalg.norm(calibration, axis=1),
                 np.any(truth != 0, axis=1)),
        'flow': (np.abs(estimated).ravel(), np.abs(calibration).ravel(), (truth != 0).ravel()),
    }
    for level, (score, calibration_score, labels) in scores.items():
        values = evaluate(labels.astype(int), score, calibrate(calibration_score, target_fpr))
        metrics.update({f'{level}_{key}': value for key, value in values.items()})
    fixed = evaluate((truth != 0).ravel().astype(int), np.abs(estimated).ravel(), fixed_threshold)
    metrics.update({f'paper_threshold_flow_{key}': value for key, value in fixed.items()})
    return metrics


def save_solver_history(directory, histories):
    diagnostics = {}
    for split, history in histories.items():
        pd.DataFrame(history).to_csv(directory / f'{split}_solver_history.csv', index=False)
        diagnostics[split] = {
            'all_subproblems_converged': all(row['converged'] for row in history),
            'unconverged_count': sum(not row['converged'] for row in history),
            'max_relative_kkt_residual': max(row['relative_kkt_residual'] for row in history),
        }
    write_json(directory / 'solver.json', diagnostics)
    return all(item['all_subproblems_converged'] for item in diagnostics.values())


def summarize(table, output):
    measures = ['precision', 'recall', 'f1', 'false_positive_rate', 'average_precision', 'roc_auc']
    columns = [f'{level}_{metric}' for level in ('time', 'flow', 'paper_threshold_flow')
               for metric in measures]
    columns += ['relative_error_L', 'relative_error_S', 'relative_error_routed_S']
    summary = table.groupby(['gamma', 'method'])[columns].agg(['mean', 'std'])
    summary.columns = ['_'.join(column) for column in summary.columns]
    summary.to_csv(output / 'summary.csv')
    summary.loc[[0.0]].to_csv(output / 'linear_comparison.csv')
    paired = table.pivot(index=['seed', 'gamma'], columns='method', values=columns)
    differences = pd.DataFrame({
        key: paired[(key, 'VAE_Lasso')] - paired[(key, 'OnlineDA')]
        for key in columns
    })
    differences.to_csv(output / 'paired_vae_minus_online_da.csv')


def run_case(config, base, seed, gamma, directory, logger):
    directory.mkdir()
    flows, clean, observed = base.observed(gamma)
    routing = base.network['R']
    train_mask = base.masks['train']
    observed_counts = train_mask.sum(axis=0)
    if not np.all(observed_counts):
        raise ValueError('Every link must be observed at least once in training.')
    mean = np.sum(observed['train'] * train_mask, axis=0) / observed_counts
    scale = float(np.sqrt(np.sum(((observed['train'] - mean) * train_mask)**2)
                             / train_mask.sum()))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Training observations have invalid scale.')
    arrays = {}
    for name, values in observed.items():
        transformed = ((values - mean) / scale).astype(np.float32)
        transformed[~base.masks[name]] = 0
        arrays[name] = transformed
    routed_attacks = base.route(base.attacks, 'test')
    np.savez_compressed(directory / 'truth.npz', gamma=gamma, routing_alpha=config['routing_alpha'],
                        observation_probability=config['observation_probability'], mean=mean, scale=scale,
                        R=routing, A=base.attacks, S=routed_attacks,
                        **{f'mask_{key}': value for key, value in base.masks.items()},
                        **{f'routing_ids_{key}': value for key, value in base.routing_ids.items()},
                        **{f'Z_{key}': value for key, value in flows.items()},
                        **{f'L_{key}': value for key, value in clean.items()},
                        **{f'D_{key}': value for key, value in observed.items()},
                        **{f'N_{key}': value for key, value in base.noise.items()})
    spectral = spectrum(clean['test'])
    write_json(directory / 'spectrum.json', spectral)
    rank_row = {
        'seed': seed, 'gamma': gamma,
        'rank_95': spectral['rank_95'], 'rank_99': spectral['rank_99'],
        'clean_rms': float(np.sqrt(np.mean(clean['test']**2))),
        'noise_rms': float(np.sqrt(np.mean(base.noise['test']**2))),
        'flow_attack_rms': float(np.sqrt(np.mean(base.attacks**2))),
        'routed_attack_rms': float(np.sqrt(np.mean(routed_attacks**2))),
        'routing_states_test': len(base.routing_banks['test']),
        'observed_fraction_test': float(base.masks['test'].mean()),
    }
    rows = []
    for method in ('VAE_Lasso', 'OnlineDA'):
        started = time.perf_counter()
        method_dir = directory / method.lower()
        method_dir.mkdir()
        logger.info('Seed %d | gamma %.2f | routing alpha %.3f | %s',
                    seed, gamma, config['routing_alpha'], method)
        predictions, histories = {}, {}
        if method == 'VAE_Lasso':
            torch.manual_seed(seed)
            model = VAE(config['n_links'], config['hidden_dims'], config['latent_dim'])
            train_config = config | {'seed': seed, 'beta': config['vae_kl_weight']}
            history = train(model, arrays['train'], arrays['validation'], train_config, logger)
            pd.DataFrame(history).to_csv(method_dir / 'training_history.csv', index=False)
            torch.save({'state_dict': model.state_dict(), 'input_dim': config['n_links'],
                        'hidden_dims': config['hidden_dims'], 'latent_dim': config['latent_dim'],
                        'mean': torch.from_numpy(mean), 'scale': scale}, method_dir / 'model.pt')
            for split in ('calibration', 'test'):
                low = reconstruct(model, arrays[split], config['batch_size']) * scale + mean
                sparse, histories[split] = decode_flows(observed[split] - low, base,
                                                        config, logger, split)
                predictions[split] = (low, sparse)
        else:
            tracker = OnlineAnomalography(config['da_rank'], config['da_ridge_penalty'],
                                          config['da_sparse_penalty'], config['lasso_max_iterations'],
                                          config['lasso_tolerance'], routing=routing,
                                          forgetting_factor=config['da_forgetting_factor'],
                                          output_after_update=True)
            warmup = config['da_warmup_size']
            tracker.initialize(observed['train'][:warmup])
            training_result = process_online(tracker, observed['train'][warmup:], base,
                                             logger, 'train', offset=warmup)
            histories['train_after_warmup'] = training_result.history
            np.savez(method_dir / 'training_state.npz', basis=tracker.basis,
                     gram=tracker.gram, cross=tracker.cross)
            for split in ('calibration', 'test'):
                result = process_online(tracker.copy(), observed[split], base, logger, split)
                predictions[split] = (result.low_rank, result.sparse)
                histories[split] = result.history
        solver_ok = save_solver_history(method_dir, histories)
        if not solver_ok:
            logger.warning('%s has unconverged Lasso subproblems; inspect solver.json', method)
        low, sparse = predictions['test']
        values = detection_metrics(sparse, predictions['calibration'][1], base.attacks,
                                   config['target_fpr'], config['paper_threshold'])
        values.update(
            seed=seed, gamma=gamma, routing_alpha=config['routing_alpha'], method=method,
            all_subproblems_converged=solver_ok,
            elapsed_seconds=time.perf_counter() - started,
            relative_error_L=float(np.linalg.norm(low - clean['test']) / np.linalg.norm(clean['test'])),
            relative_error_S=float(np.linalg.norm(sparse - base.attacks) / np.linalg.norm(base.attacks)),
            relative_error_routed_S=float(np.linalg.norm(base.route(sparse - base.attacks, 'test'))
                                          / np.linalg.norm(routed_attacks)),
        )
        np.savez_compressed(method_dir / 'estimates.npz', L_hat=low, A_hat=sparse,
                            S_hat=base.route(sparse, 'test'),
                            calibration_A_hat=predictions['calibration'][1])
        write_json(method_dir / 'metrics.json', values)
        rows.append(values)
        logger.info('%s | gamma %.2f | time recall %.4f | flow recall %.4f | '
                    'flow F1 %.4f | flow AP %.4f | flow FPR %.4f',
                    method, gamma, values['time_recall'], values['flow_recall'],
                    values['flow_f1'], values['flow_average_precision'],
                    values['flow_false_positive_rate'])
    return rows, rank_row


def run(config):
    output = Path(config['output_dir'])
    output.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger(f'paper_synthetic.{output}')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(output / 'run.log')]
    for handler in handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', '%H:%M:%S'))
        logger.addHandler(handler)
    try:
        write_json(output / 'config.json', config)
        write_json(output / 'provenance.json', {
            'python': sys.version,
            'platform': platform.platform(),
            'device': 'cpu',
            'versions': {key: importlib.metadata.version(key) for key in
                         ('numpy', 'pandas', 'torch', 'scikit-learn', 'threadpoolctl')},
            'source_sha256': {path.name: sha256(path)
                              for path in sorted(Path(__file__).parent.glob('*.py'))},
            'reference': 'https://arxiv.org/abs/1208.4043',
            'data_source': 'New samples from the Section VI-A linear model; original author arrays unavailable',
            'nonlinear_source': 'Our extension; gamma denotes nonlinearity',
            'routing_alpha': 'Paper notation: per-step probability of a routing edge removal/addition event',
            'observations': 'D = Z(gamma) R_t.T + A R_t.T + N; Omega_t controls observed links',
            'protocol': 'Normal-only training; independent model-selection, calibration, and test splits',
            'online_adaptation': 'Algorithm-2 updates with normal-prefix SVD warm start; post-update nominal estimate',
            'vae_method': 'Normal-trained VAE link reconstruction followed by Lasso over the known routing matrix',
            'limitations': [
                'New graph conditioned on connectivity and link count; deterministic BFS tie-breaking',
                'Shared normal training is additional information relative to the original paper stream',
                'OnlineDA adapts during test; VAE weights are frozen',
                'Different representation capacities; no claim of isolated architecture advantage',
                'Interpolation can change signal energy; inspect rank_diagnostics.csv',
                'No test-based parameter selection; settings are not independently optimized',
            ],
        })
        torch.set_num_threads(config['threads'])
        torch.use_deterministic_algorithms(True)
        metrics_rows, rank_rows = [], []
        with threadpool_limits(limits=config['threads']):
            for seed in config['seeds']:
                logger.info('Seed %d | sampling a paper-model topology and paired data', seed)
                base = generate(config, seed)
                seed_dir = output / f'seed_{seed}'
                seed_dir.mkdir()
                np.savez_compressed(
                    seed_dir / 'generator.npz', **base.network, **base.maps,
                    nonlinear_scale=base.nonlinear_scale,
                    **{f'W_{key}': value for key, value in base.latent.items()},
                    **{f'R_bank_{key}': value for key, value in base.routing_banks.items()},
                    **{f'R_ids_{key}': value for key, value in base.routing_ids.items()},
                    **{f'mask_{key}': value for key, value in base.masks.items()},
                )
                logger.info('Topology accepted after %d draws | %d links | %d flows',
                            base.network['topology_attempts'], *base.network['R'].shape)
                for index, gamma in enumerate(config['gammas']):
                    rows, rank_row = run_case(config, base, seed, gamma,
                                              seed_dir / f'gamma_{index:02d}', logger)
                    metrics_rows.extend(rows)
                    rank_rows.append(rank_row)
                    pd.DataFrame(metrics_rows).to_csv(output / 'metrics.csv', index=False)
                    pd.DataFrame(rank_rows).to_csv(output / 'rank_diagnostics.csv', index=False)
        table = pd.DataFrame(metrics_rows)
        summarize(table, output)
        write_json(output / 'completed.json', {
            'runs': len(table),
            'includes_linear_control': True,
            'all_subproblems_converged': bool(table['all_subproblems_converged'].all()),
        })
        logger.info('Completed experiment 02 | gamma=0 linear comparison: %s',
                    output / 'linear_comparison.csv')
        return output
    except Exception:
        logger.exception('Experiment failed; partial artifacts retained')
        raise
    finally:
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)
