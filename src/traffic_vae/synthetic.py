"""Paired synthetic traffic: fixed latent factors, mixing maps, noise, and attacks."""
from dataclasses import dataclass
import numpy as np


@dataclass
class SyntheticBase:
    linear: dict[str, np.ndarray]
    nonlinear: dict[str, np.ndarray]
    noise: dict[str, np.ndarray]
    attacks: np.ndarray
    latent: dict[str, np.ndarray]
    maps: dict[str, np.ndarray]
    normalization: dict[str, np.ndarray]

    def observed(self, alpha: float):
        clean = {name: (1 - alpha) * self.linear[name] + alpha * self.nonlinear[name]
                 for name in self.linear}
        observed = {name: clean[name] + self.noise[name] for name in clean}
        observed['test'] = observed['test'] + self.attacks
        return clean, observed


def latent_sequence(rng: np.random.Generator, size: int, period: int) -> np.ndarray:
    phase = rng.uniform(0, 2 * np.pi)
    angle = 2 * np.pi * np.arange(size) / period + phase
    load = np.empty(size)
    load[0] = rng.normal()
    for index in range(1, size):
        load[index] = 0.9 * load[index - 1] + np.sqrt(1 - 0.9**2) * rng.normal()
    return np.column_stack([np.sin(angle), np.cos(angle), load])


def generate(config: dict, seed: int) -> SyntheticBase:
    rng = np.random.default_rng(seed)
    features = config['n_features']
    maps = {name: rng.normal(size=(3, features)) / np.sqrt(3) for name in ('A', 'B', 'C')}
    sizes = {name: config[f'{name}_size'] for name in ('train', 'validation', 'test')}
    latent = {name: latent_sequence(rng, size, config['period']) for name, size in sizes.items()}
    linear = {name: z @ maps['A'] for name, z in latent.items()}
    nonlinear = {name: np.sin(2 * z @ maps['B']) + 0.3 * (z @ maps['C'])**2
                 for name, z in latent.items()}
    normalization = {}
    # Match component energy using training data only; never use test statistics.
    for label, component in [('linear', linear), ('nonlinear', nonlinear)]:
        center = component['train'].mean(axis=0)
        rms = np.sqrt(np.mean((component['train'] - center)**2))
        normalization[f'{label}_center'] = center
        normalization[f'{label}_rms'] = np.asarray(rms)
        for name in component:
            component[name] = (component[name] - center) / rms
    noise = {name: rng.normal(0, config['noise_std'], size=(size, features))
             for name, size in sizes.items()}
    attacks = np.zeros_like(linear['test'])
    duration = config['attack_duration']
    slots = sizes['test'] // duration
    count = max(1, int(round(config['attacked_time_fraction'] * sizes['test'] / duration)))
    if not 1 <= count < slots:
        raise ValueError('Attack settings must leave both attacked and unattacked time blocks.')
    # Nonoverlapping time blocks make the actual support unambiguous.
    for slot in rng.choice(slots, size=count, replace=False):
        group = int(rng.integers(features // config['group_size']))
        first = group * config['group_size']
        attacks[slot * duration:(slot + 1) * duration,
                first:first + config['group_size']] = config['attack_amplitude']
    return SyntheticBase(linear, nonlinear, noise, attacks, latent, maps, normalization)


def spectrum(values: np.ndarray) -> dict:
    singular = np.linalg.svd(values, compute_uv=False)
    energy = np.cumsum(singular**2) / np.sum(singular**2)
    return {'singular_values': singular.tolist(),
            'rank_95': int(np.searchsorted(energy, .95) + 1),
            'rank_99': int(np.searchsorted(energy, .99) + 1)}
