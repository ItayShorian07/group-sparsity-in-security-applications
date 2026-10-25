"""Group-sparse proximal operators and a checked, fixed-design decoder.

The routing decoder solves a convex problem only after the VAE is frozen. It
does not differentiate through the VAE or claim unique OD-flow recovery.
"""
import numpy as np


def partition(groups, width):
    arrays = [np.asarray(group, dtype=int) for group in groups]
    if not arrays or any(group.ndim != 1 or not len(group) for group in arrays):
        raise ValueError('Groups must be nonempty one-dimensional index arrays.')
    order = np.concatenate(arrays)
    if not np.array_equal(np.sort(order), np.arange(width)):
        raise ValueError('Groups must partition all coordinates without overlap.')
    sizes = np.array([len(group) for group in arrays])
    return order, sizes, np.r_[0, np.cumsum(sizes)[:-1]]


def shrink_scores(scores, group_ids, penalty):
    """Exact identity-design Group Lasso, with sqrt(group size) weights.

    This acts on scalar VAE anomaly scores, not physical traffic amplitudes.
    Setting every group to a singleton gives ordinary soft thresholding.
    """
    scores = np.asarray(scores, dtype=float)
    ids = np.asarray(group_ids, dtype=int)
    if (scores.ndim != 1 or ids.shape != scores.shape or not len(scores)
            or not np.isfinite(scores).all() or (ids < 0).any()
            or not np.isfinite(penalty) or penalty < 0):
        raise ValueError('Invalid scores, groups, or penalty.')
    sizes = np.bincount(ids)
    norms = np.sqrt(np.bincount(ids, weights=scores**2))
    factors = np.maximum(1 - penalty * np.sqrt(sizes) / np.maximum(norms, 1e-300), 0)
    return scores * factors[ids]


def solve_group_lasso_batch(design, targets, groups, penalty, max_iterations=10000,
                            tolerance=1e-5, batch_size=256):
    """Solve .5||r-Ra||^2 + penalty sum_g sqrt(|g|)||a_g||_2.

    FISTA with adaptive restart; every solution receives a normalized group-KKT
    residual. A batch never declares convergence based on objective change alone.
    """
    design = np.asarray(design, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if (design.ndim != 2 or targets.ndim != 2 or targets.shape[1] != design.shape[0]
            or not min(design.shape) or not np.isfinite(design).all()
            or not np.isfinite(targets).all() or penalty <= 0
            or not np.isfinite(penalty) or tolerance <= 0
            or max_iterations < 1 or batch_size < 1):
        raise ValueError('Invalid fixed-design group-decoder inputs.')
    order, sizes, starts = partition(groups, design.shape[1])
    routing = np.ascontiguousarray(design[:, order])
    coordinate_group = np.repeat(np.arange(len(sizes)), sizes)
    weights = penalty * np.sqrt(sizes)
    lipschitz = float(np.linalg.norm(routing, 2)**2)
    if lipschitz == 0:
        raise ValueError('The design must contain nonzero columns.')
    estimates, diagnostics = [], []
    for begin in range(0, len(targets), batch_size):
        target = targets[begin:begin + batch_size]
        a = np.zeros((len(target), routing.shape[1]))
        y, momentum = a.copy(), np.ones((len(target), 1))
        gradient_zero = -target @ routing
        normalizer = np.maximum(1., np.sqrt(np.add.reduceat(gradient_zero**2, starts, axis=1)).max(axis=1))
        for iteration in range(1, max_iterations + 1):
            candidate = y - ((y @ routing.T - target) @ routing) / lipschitz
            norms = np.sqrt(np.add.reduceat(candidate**2, starts, axis=1))
            factors = np.maximum(1 - weights / (lipschitz * np.maximum(norms, 1e-300)), 0)
            candidate *= factors[:, coordinate_group]
            next_momentum = (1 + np.sqrt(1 + 4 * momentum**2)) / 2
            extrapolated = candidate + ((momentum - 1) / next_momentum) * (candidate - a)
            restart = np.sum((y - candidate) * (candidate - a), axis=1) > 0
            extrapolated[restart] = candidate[restart]
            next_momentum[restart] = 1
            a, y, momentum = candidate, extrapolated, next_momentum
            if iteration % 25 == 0 or iteration == max_iterations:
                gradient = (a @ routing.T - target) @ routing
                norms = np.sqrt(np.add.reduceat(a**2, starts, axis=1))
                gradient_norms = np.sqrt(np.add.reduceat(gradient**2, starts, axis=1))
                stationarity = gradient + a * (weights / np.maximum(norms, 1e-300))[:, coordinate_group]
                active_violation = np.sqrt(np.add.reduceat(stationarity**2, starts, axis=1))
                violation = np.where(norms > 0, active_violation,
                                     np.maximum(gradient_norms - weights, 0))
                relative = violation.max(axis=1) / normalizer
                if np.all(relative <= tolerance):
                    break
        if not np.isfinite(a).all() or not np.isfinite(relative).all():
            raise FloatingPointError('Non-finite group-decoder solution.')
        restored = np.empty_like(a)
        restored[:, order] = a
        estimates.append(restored)
        diagnostics.extend({'iterations': iteration, 'relative_kkt_residual': float(error),
                            'converged': bool(error <= tolerance)} for error in relative)
    return np.concatenate(estimates), diagnostics
