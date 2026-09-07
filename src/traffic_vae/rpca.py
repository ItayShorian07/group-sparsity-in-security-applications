"""Penalized stable RPCA solved by proximal gradient with a fixed-point check."""
from dataclasses import dataclass
import numpy as np


@dataclass
class RPCAResult:
    low_rank: np.ndarray
    sparse: np.ndarray
    history: list[dict]
    converged: bool
    nuclear_penalty: float
    sparse_penalty: float


def soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0)


def solve(values: np.ndarray, sparse_penalty: float = 0.1,
          max_iterations: int = 1000, tolerance: float = 1e-5) -> RPCAResult:
    """Minimize 0.5||D-L-S||_F^2 + lambda_*||L||_* + lambda_1||S||_1.

    The joint smooth gradient has Lipschitz constant 2, hence step size 1/2.
    lambda_1/lambda_* = 1/sqrt(max(shape)); the absolute scale is configurable.
    Convergence refers to the relative proximal-gradient mapping, not recovery.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError('RPCA requires a nonempty finite matrix.')
    if (not np.isfinite(sparse_penalty) or not np.isfinite(tolerance)
            or sparse_penalty <= 0 or type(max_iterations) is not int
            or max_iterations < 1 or tolerance <= 0):
        raise ValueError('RPCA penalties, iteration limit, and tolerance must be positive.')
    nuclear_penalty = sparse_penalty * np.sqrt(max(values.shape))
    low_rank = np.zeros_like(values)
    sparse = np.zeros_like(values)
    normalizer = max(1.0, np.linalg.norm(values))
    history = []
    converged = False
    for iteration in range(1, max_iterations + 1):
        residual = low_rank + sparse - values
        u, singular, vt = np.linalg.svd(low_rank - 0.5 * residual, full_matrices=False)
        shrunk = np.maximum(singular - 0.5 * nuclear_penalty, 0)
        next_low_rank = (u * shrunk) @ vt
        next_sparse = soft_threshold(sparse - 0.5 * residual, 0.5 * sparse_penalty)
        mapping = 2 * np.sqrt(np.sum((next_low_rank - low_rank)**2)
                              + np.sum((next_sparse - sparse)**2)) / normalizer
        low_rank, sparse = next_low_rank, next_sparse
        objective = (0.5 * np.sum((values - low_rank - sparse)**2)
                     + nuclear_penalty * shrunk.sum() + sparse_penalty * np.abs(sparse).sum())
        history.append({'iteration': iteration, 'objective': float(objective),
                        'relative_gradient_mapping': float(mapping),
                        'rank': int(np.count_nonzero(shrunk))})
        if not np.isfinite(objective):
            raise FloatingPointError('Non-finite RPCA objective.')
        if mapping <= tolerance:
            converged = True
            break
    return RPCAResult(low_rank, sparse, history, converged, nuclear_penalty, sparse_penalty)
