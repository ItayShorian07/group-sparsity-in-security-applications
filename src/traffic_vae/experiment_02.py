"""Experiment 02: fully synthetic nonlinearity sweep comparing VAE with RPCA."""
import argparse
import importlib.metadata
import importlib.util
import json
import logging
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from .data import sha256
from .evaluation import calibrate, evaluate
from .experiment import write_json
from .model import VAE, train
from .rpca import solve
from .synthetic import generate, spectrum


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    required = {
        'output_dir', 'seeds', 'alphas', 'n_features', 'group_size', 'train_size',
        'validation_size', 'test_size', 'period', 'noise_std', 'attack_amplitude',
        'attack_duration', 'attacked_time_fraction', 'hidden_dims', 'latent_dim',
        'epochs', 'batch_size', 'learning_rate', 'beta', 'patience', 'target_fpr',
        'rpca_sparse_penalty', 'rpca_max_iterations', 'rpca_tolerance', 'threads',
    }
    if set(config) != required:
        raise ValueError(f'Config fields: missing {required - set(config)}, unknown {set(config) - required}')
    for key in ('n_features', 'group_size', 'train_size', 'validation_size', 'test_size',
                'period', 'attack_duration', 'latent_dim', 'epochs', 'batch_size',
                'patience', 'rpca_max_iterations', 'threads'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer.')
    for key in ('noise_std', 'attack_amplitude', 'attacked_time_fraction', 'learning_rate',
                'beta', 'target_fpr', 'rpca_sparse_penalty', 'rpca_tolerance'):
        value = config[key]
        if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
            raise ValueError(f'{key} must be finite and nonnegative.')
    for key in ('attack_amplitude', 'learning_rate', 'rpca_sparse_penalty', 'rpca_tolerance'):
        if config[key] == 0:
            raise ValueError(f'{key} must be positive.')
    for key in ('target_fpr', 'attacked_time_fraction'):
        if not 0 < config[key] < 1:
            raise ValueError(f'{key} must lie strictly between zero and one.')
    for key in ('seeds', 'hidden_dims'):
        if not isinstance(config[key], list) or not config[key]:
            raise ValueError(f'{key} must be a nonempty list.')
        if any(type(x) is not int or x < (0 if key == 'seeds' else 1) for x in config[key]):
            raise ValueError(f'Invalid {key}.')
    if len(set(config['seeds'])) != len(config['seeds']) or max(config['seeds']) >= 2**32:
        raise ValueError('Seeds must be unique and smaller than 2**32.')
    if not isinstance(config['alphas'], list) or not config['alphas']:
        raise ValueError('alphas must be a nonempty list.')
    if any(type(a) not in (int, float) or not np.isfinite(a) or not 0 <= a <= 1
           for a in config['alphas']) or len(set(config['alphas'])) != len(config['alphas']):
        raise ValueError('alphas must be distinct finite numbers in [0, 1].')
    if config['n_features'] % config['group_size'] or config['n_features'] < 3:
        raise ValueError('n_features must be at least three and divisible by group_size.')
    if min(config['train_size'], config['validation_size'], config['test_size']) < 4:
        raise ValueError('Each split must have at least four time points.')
    # Equal matrix shapes keep the batch RPCA penalty and calibration context matched.
    if config['validation_size'] != config['test_size']:
        raise ValueError('validation_size must equal test_size for batch RPCA calibration.')
    if not isinstance(config['output_dir'], str) or not config['output_dir'].strip():
        raise ValueError('output_dir must be a nonempty path.')
    config['output_dir'] = str(Path(config['output_dir']).expanduser().resolve())
    slots = config['test_size'] // config['attack_duration']
    count = max(1, int(round(config['attacked_time_fraction'] * config['test_size'] / config['attack_duration'])))
    if not 1 <= count < slots:
        raise ValueError('Attack settings must leave attacked and unattacked time blocks.')
    return config


@torch.no_grad()
def reconstruct(model: VAE, values: np.ndarray, batch_size: int) -> np.ndarray:
    model.eval()
    result = []
    for start in range(0, len(values), batch_size):
        x = torch.from_numpy(values[start:start + batch_size])
        result.append(model(x, sample=False)[0].numpy())
    result = np.concatenate(result).astype(np.float64)
    if not np.isfinite(result).all():
        raise FloatingPointError('Non-finite VAE reconstruction.')
    return result


def detection_metrics(sparse, calibration_sparse, truth, group_size, target_fpr):
    scores = {
        'time': (np.linalg.norm(sparse, axis=1), np.linalg.norm(calibration_sparse, axis=1),
                 np.any(truth != 0, axis=1)),
        'group': (np.linalg.norm(sparse.reshape(len(sparse), -1, group_size), axis=2).ravel(),
                  np.linalg.norm(calibration_sparse.reshape(len(calibration_sparse), -1, group_size), axis=2).ravel(),
                  np.any(truth.reshape(len(truth), -1, group_size) != 0, axis=2).ravel()),
        'element': (np.abs(sparse).ravel(), np.abs(calibration_sparse).ravel(), (truth != 0).ravel()),
    }
    metrics = {}
    for level, (test_score, calibration_score, labels) in scores.items():
        threshold = calibrate(calibration_score, target_fpr)
        values = evaluate(labels.astype(int), test_score, threshold)
        metrics.update({f'{level}_{key}': value for key, value in values.items()})
    return metrics


def plot_results(table: pd.DataFrame, ranks: pd.DataFrame, output: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    metrics = [('time_f1', 'Time detection F1'), ('time_false_positive_rate', 'Time false-positive rate'),
               ('time_average_precision', 'Time average precision'),
               ('relative_error_L', 'Relative clean-signal error'),
               ('relative_error_S', 'Relative attack error')]
    for ax, (metric, title) in zip(axes.flat, metrics):
        for method, rows in table.groupby('method'):
            grouped = rows.groupby('alpha')[metric].agg(['mean', 'std']).sort_index()
            ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'].fillna(0),
                        marker='o', capsize=3, label=method)
        ax.set(xlabel='Nonlinearity alpha', title=title)
        ax.grid(alpha=.25)
        ax.legend()
    ax = axes.flat[-1]
    for metric in ('rank_95', 'rank_99'):
        grouped = ranks.groupby('alpha')[metric].agg(['mean', 'std']).sort_index()
        ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'].fillna(0),
                    marker='o', capsize=3, label=metric)
    ax.set(xlabel='Nonlinearity alpha', title='Clean signal energy rank')
    ax.grid(alpha=.25)
    ax.legend()
    fig.suptitle('Synthetic VAE vs batch RPCA | bars: SD across paired seeds')
    fig.tight_layout()
    fig.savefig(output / 'comparison.png', dpi=180)
    fig.savefig(output / 'comparison.pdf')
    plt.close(fig)


def run(config: dict) -> Path:
    output = Path(config['output_dir'])
    output.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger(f'synthetic.{output}')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(output / 'run.log')]
    for handler in handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', '%H:%M:%S'))
        logger.addHandler(handler)
    try:
        write_json(output / 'config.json', config)
        write_json(output / 'provenance.json', {
            'python': sys.version, 'platform': platform.platform(), 'device': 'cpu',
            'versions': {key: importlib.metadata.version(key) for key in
                         ('numpy', 'pandas', 'torch', 'scikit-learn', 'threadpoolctl')},
            'source_sha256': {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
            'protocol': 'VAE normal-only training; RPCA separate transductive validation/test decompositions',
            'calibration_source': 'normal validation, shared with VAE early stopping',
        })
        torch.set_num_threads(config['threads'])
        torch.use_deterministic_algorithms(True)
        metrics_rows, rank_rows = [], []
        with threadpool_limits(limits=config['threads']):
            for seed in config['seeds']:
                base = generate(config, seed)
                seed_dir = output / f'seed_{seed}'
                seed_dir.mkdir()
                np.savez_compressed(seed_dir / 'generator.npz', **base.maps, **base.normalization,
                                    **{f'z_{name}': values for name, values in base.latent.items()})
                for alpha_index, alpha in enumerate(config['alphas']):
                    logger.info('Seed %d | alpha %.4f | generating paired data', seed, alpha)
                    directory = seed_dir / f'alpha_{alpha_index:02d}'
                    directory.mkdir()
                    clean, observed = base.observed(alpha)
                    # An affine training-only transform preserves D = L + S + N.
                    mean = observed['train'].mean(axis=0)
                    scale = observed['train'].std(axis=0)
                    scale = np.maximum(scale, 1e-8)
                    arrays = {name: ((values - mean) / scale).astype(np.float32)
                              for name, values in observed.items()}
                    np.savez_compressed(directory / 'truth.npz', alpha=alpha, mean=mean, scale=scale,
                                        S=base.attacks,
                                        **{f'L_{k}': v for k, v in clean.items()},
                                        **{f'D_{k}': v for k, v in observed.items()},
                                        **{f'N_{k}': v for k, v in base.noise.items()})
                    diagnostics = spectrum(clean['test'])
                    write_json(directory / 'spectrum.json', diagnostics)
                    rank_rows.append({'seed': seed, 'alpha': alpha,
                                      'rank_95': diagnostics['rank_95'], 'rank_99': diagnostics['rank_99'],
                                      'clean_rms': float(np.sqrt(np.mean(clean['test']**2))),
                                      'noise_rms': float(np.sqrt(np.mean(base.noise['test']**2))),
                                      'attack_rms': float(np.sqrt(np.mean(base.attacks**2)))})
                    for method in ('VAE', 'RPCA'):
                        started = time.perf_counter()
                        method_dir = directory / method.lower()
                        method_dir.mkdir()
                        solver_ok = True
                        if method == 'VAE':
                            torch.manual_seed(seed)
                            model = VAE(config['n_features'], config['hidden_dims'], config['latent_dim'])
                            history = train(model, arrays['train'], arrays['validation'], config | {'seed': seed}, logger)
                            pd.DataFrame(history).to_csv(method_dir / 'history.csv', index=False)
                            validation_low = reconstruct(model, arrays['validation'], config['batch_size'])
                            low = reconstruct(model, arrays['test'], config['batch_size'])
                            sparse = arrays['test'].astype(float) - low
                            calibration_sparse = arrays['validation'].astype(float) - validation_low
                            torch.save({'state_dict': model.state_dict(), 'input_dim': config['n_features'],
                                        'hidden_dims': config['hidden_dims'], 'latent_dim': config['latent_dim']},
                                       method_dir / 'model.pt')
                        else:
                            logger.info('Seed %d | alpha %.4f | batch RPCA validation and test', seed, alpha)
                            results = {}
                            for split in ('validation', 'test'):
                                result = solve(arrays[split], config['rpca_sparse_penalty'],
                                               config['rpca_max_iterations'], config['rpca_tolerance'])
                                results[split] = result
                                pd.DataFrame(result.history).to_csv(method_dir / f'{split}_history.csv', index=False)
                                if not result.converged:
                                    logger.warning('RPCA %s reached iteration limit; convergence not established', split)
                            result = results['test']
                            low, sparse = result.low_rank, result.sparse
                            calibration_sparse = results['validation'].sparse
                            solver_ok = all(item.converged for item in results.values())
                            write_json(method_dir / 'solver.json', {
                                split: {'converged': r.converged, 'iterations': len(r.history),
                                        'relative_gradient_mapping': r.history[-1]['relative_gradient_mapping'],
                                        'nuclear_penalty': r.nuclear_penalty, 'sparse_penalty': r.sparse_penalty}
                                for split, r in results.items()})
                        # Report recovery in the original generator units.
                        estimated_clean = low * scale + mean
                        estimated_attack = sparse * scale
                        values = detection_metrics(sparse, calibration_sparse, base.attacks,
                                                   config['group_size'], config['target_fpr'])
                        values.update(seed=seed, alpha=alpha, method=method,
                                      rpca_converged=solver_ok if method == 'RPCA' else None,
                                      elapsed_seconds=time.perf_counter() - started,
                                      relative_error_L=float(np.linalg.norm(estimated_clean - clean['test']) / np.linalg.norm(clean['test'])),
                                      relative_error_S=float(np.linalg.norm(estimated_attack - base.attacks) / np.linalg.norm(base.attacks)))
                        write_json(method_dir / 'metrics.json', values)
                        np.savez_compressed(method_dir / 'estimates.npz', L_hat=estimated_clean,
                                            S_hat=estimated_attack, calibration_sparse=calibration_sparse)
                        metrics_rows.append(values)
                        pd.DataFrame(metrics_rows).to_csv(output / 'metrics.csv', index=False)
                        logger.info('%s | alpha %.2f | F1 %.4f | AP %.4f | FPR %.4f | L error %.4f',
                                    method, alpha, values['time_f1'], values['time_average_precision'],
                                    values['time_false_positive_rate'], values['relative_error_L'])
        table = pd.DataFrame(metrics_rows)
        ranks = pd.DataFrame(rank_rows)
        ranks.to_csv(output / 'rank_diagnostics.csv', index=False)
        numeric = ['time_f1', 'time_false_positive_rate', 'time_average_precision',
                   'group_f1', 'group_average_precision', 'relative_error_L', 'relative_error_S']
        summary = table.groupby(['alpha', 'method'])[numeric].agg(['mean', 'std'])
        summary.columns = ['_'.join(column) for column in summary.columns]
        summary.to_csv(output / 'summary.csv')
        # Paired differences use the same generator seed at each alpha.
        paired = table.pivot(index=['seed', 'alpha'], columns='method', values=numeric)
        differences = pd.DataFrame({key: paired[(key, 'VAE')] - paired[(key, 'RPCA')] for key in numeric})
        differences.to_csv(output / 'paired_vae_minus_rpca.csv')
        plots_available = importlib.util.find_spec('matplotlib') is not None
        if plots_available:
            plot_results(table, ranks, output)
        else:
            logger.info('Optional Matplotlib is absent; plot-ready CSV results were saved')
        write_json(output / 'completed.json', {'runs': len(table), 'plots_created': plots_available,
                   'all_rpca_converged': bool(table.loc[table.method == 'RPCA', 'rpca_converged'].all())})
        logger.info('Completed experiment 02 | results: %s', output)
        return output
    except Exception:
        logger.exception('Experiment failed; partial artifacts retained')
        raise
    finally:
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('experiment_02.json'))
    args = parser.parse_args()
    try:
        run(load_config(args.config))
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()
