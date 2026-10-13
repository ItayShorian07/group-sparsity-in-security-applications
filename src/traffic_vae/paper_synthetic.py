"""Section VI-A linear traffic generator with a paired nonlinear extension.

Reference: https://arxiv.org/abs/1208.4043. Arrays are time-by-channel;
the routing matrix is links-by-flows. These are new draws, not author data.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np


def routing_from_adjacency(adjacency, edges, flows):
    """Build minimum-hop routing on an active graph using fixed physical-link rows."""
    nodes = len(adjacency)
    edge_index = {tuple(edge): index for index, edge in enumerate(edges)}
    routing = np.zeros((len(edges), len(flows)))
    for source in range(nodes):
        parent = {source: source}
        queue = deque([source])
        while queue:
            current = queue.popleft()
            for neighbor in np.flatnonzero(adjacency[current]):
                if neighbor not in parent:
                    parent[neighbor] = current
                    queue.append(neighbor)
        if len(parent) != nodes:
            return None
        for column, (origin, target) in enumerate(flows):
            if origin != source:
                continue
            current = target
            while current != source:
                previous = parent[current]
                routing[edge_index[(previous, current)], column] = 1
                current = previous
    return routing


def geometric_routing(rng, nodes=15, links=52, radius=.35, max_attempts=100000):
    """Sample a connected geometric graph with the paper's directed link count.

    Rejection conditions on topology only, never on detection performance.
    Sorted BFS neighbors give a reproducible tie rule for minimum-hop paths.
    """
    for attempt in range(1, max_attempts + 1):
        positions = rng.uniform(size=(nodes, 2))
        distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)
        adjacency = (distances < radius) & ~np.eye(nodes, dtype=bool)
        if adjacency.sum() != links:
            continue
        edges = np.argwhere(adjacency)
        flows = np.asarray([(source, target) for source in range(nodes) for target in range(nodes)
                            if source != target])
        routing = routing_from_adjacency(adjacency, edges, flows)
        if routing is not None:
            return {'R': routing, 'positions': positions, 'edges': edges, 'adjacency': adjacency,
                    'flows': np.asarray(flows), 'topology_attempts': np.asarray(attempt)}
    raise ValueError('No connected topology with the requested link count; increase topology_max_attempts.')


def dynamic_routing_plan(network, size, alpha, rng):
    """Follow the paper's remove/add-edge model and retain compact routing change points."""
    physical = network['adjacency'].copy()
    effective = physical.copy()
    undirected_physical = [tuple(edge) for edge in np.argwhere(np.triu(physical, 1))]
    bank = [network['R']]
    ids = np.zeros(size, dtype=np.int32)
    for time_index in range(1, size):
        changed = False
        if rng.random() < alpha:
            active = [edge for edge in undirected_physical if effective[edge]]
            for choice in rng.permutation(len(active)):
                first, second = active[int(choice)]
                candidate = effective.copy()
                candidate[first, second] = candidate[second, first] = False
                if routing_from_adjacency(candidate, network['edges'], network['flows']) is not None:
                    effective = candidate
                    changed = True
                    break
        if rng.random() < alpha:
            inactive = [edge for edge in undirected_physical if not effective[edge]]
            if inactive:
                first, second = inactive[int(rng.integers(len(inactive)))]
                effective[first, second] = effective[second, first] = True
                changed = True
        if changed:
            routing = routing_from_adjacency(effective, network['edges'], network['flows'])
            if routing is None:
                raise RuntimeError('Dynamic routing update disconnected the graph.')
            bank.append(routing)
        ids[time_index] = len(bank) - 1
    return np.asarray(bank), ids


@dataclass
class PaperSyntheticBase:
    network: dict[str, np.ndarray]
    maps: dict[str, np.ndarray]
    latent: dict[str, np.ndarray]
    linear: dict[str, np.ndarray]
    nonlinear: dict[str, np.ndarray]
    noise: dict[str, np.ndarray]
    attacks: np.ndarray
    nonlinear_scale: float
    routing_banks: dict[str, np.ndarray]
    routing_ids: dict[str, np.ndarray]
    masks: dict[str, np.ndarray]

    def route(self, values, split):
        """Route time-by-flow values using the split's compact R_t plan."""
        output = np.empty((len(values), self.network['R'].shape[0]))
        for route_id, routing in enumerate(self.routing_banks[split]):
            selected = self.routing_ids[split] == route_id
            output[selected] = values[selected] @ routing.T
        return output

    def expanded_routing(self, split, start=None, stop=None):
        selected_ids = self.routing_ids[split][slice(start, stop)]
        return self.routing_banks[split][selected_ids]

    def observed(self, gamma):
        """Return nominal OD flows, nominal link traffic, and observed link traffic."""
        if not np.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError('gamma must lie in [0, 1].')
        flows = {name: (1 - gamma) * self.linear[name] + gamma * self.nonlinear[name]
                 for name in self.linear}
        clean = {name: self.route(values, name) for name, values in flows.items()}
        observed = {name: values + self.noise[name] for name, values in clean.items()}
        observed['test'] = observed['test'] + self.route(self.attacks, 'test')
        return flows, clean, observed


def generate(config, seed):
    # Topology rejection cannot change the latent/noise/attack random streams.
    streams = [np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(7)]
    topology_rng, map_rng, latent_rng, noise_rng, attack_rng, routing_rng, mask_rng = streams
    network = geometric_routing(topology_rng, config['n_nodes'], config['n_links'],
                                config['connection_radius'], config['topology_max_attempts'])
    flow_count = config['n_nodes'] * (config['n_nodes'] - 1)
    rank = config['linear_rank']
    maps = {'U': map_rng.normal(size=(flow_count, rank)) / np.sqrt(flow_count),
            'B': map_rng.normal(size=(rank, flow_count)) / np.sqrt(rank),
            'C': map_rng.normal(size=(rank, flow_count)) / np.sqrt(rank)}
    sizes = {name: config[f'{name}_size'] for name in ('train', 'validation', 'calibration', 'test')}
    latent = {name: latent_rng.normal(size=(size, rank)) for name, size in sizes.items()}
    # The linear endpoint is left in the paper's units: no centering or rescaling.
    linear = {name: w @ maps['U'].T for name, w in latent.items()}
    nonlinear = {name: np.sin(2 * w @ maps['B'])
                 + .3 * ((w @ maps['C'])**2 - np.sum(maps['C']**2, axis=0))
                 for name, w in latent.items()}
    # Match endpoint link energy using training data only; do not change U or W.
    linear_rms = np.sqrt(np.mean((linear['train'] @ network['R'].T)**2))
    nonlinear_rms = np.sqrt(np.mean((nonlinear['train'] @ network['R'].T)**2))
    scale = float(linear_rms / nonlinear_rms)
    nonlinear = {name: values * scale for name, values in nonlinear.items()}
    noise = {name: noise_rng.normal(0, config['noise_std'], size=(size, config['n_links']))
             for name, size in sizes.items()}
    support = attack_rng.random((sizes['test'], flow_count)) < config['attack_probability']
    attacks = support * attack_rng.choice([-1., 1.], size=support.shape)
    if not support.any() or not np.any(~support.any(axis=1)):
        raise ValueError('The sampled test set needs attacks and attack-free times; use a larger test_size.')
    routing_banks, routing_ids, masks = {}, {}, {}
    for name, size in sizes.items():
        routing_banks[name], routing_ids[name] = dynamic_routing_plan(
            network, size, config['routing_alpha'], routing_rng
        )
        masks[name] = mask_rng.random((size, config['n_links'])) < config['observation_probability']
        empty = ~masks[name].any(axis=1)
        if empty.any():
            rows = np.flatnonzero(empty)
            masks[name][rows, mask_rng.integers(config['n_links'], size=len(rows))] = True
    return PaperSyntheticBase(network, maps, latent, linear, nonlinear, noise, attacks, scale,
                              routing_banks, routing_ids, masks)
