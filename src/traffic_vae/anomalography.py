"""Algorithm-2 updates for full observations and identity routing, with warm start.

Reference: Mardani, Mateos, Giannakis, arXiv:1208.4043, equations (11)-(14).
The synthetic protocol uses R=I, Omega=I, and forgetting factor one.
A normal-only SVD prefix initializes full-rank sufficient statistics; this is an
explicit adaptation, not the paper's random P / zero-statistics initialization.
"""
from dataclasses import dataclass
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso


@dataclass
class OnlineResult:
    low_rank: np.ndarray
    sparse: np.ndarray
    history: list[dict]


class OnlineAnomalography:
    def __init__(self, rank: int, ridge: float, sparse_penalty: float,
                 max_iterations: int = 3000, tolerance: float = 1e-7):
        if (type(rank) is not int or rank < 1 or type(max_iterations) is not int
                or max_iterations < 1 or not np.isfinite([ridge, sparse_penalty, tolerance]).all()
                or min(ridge, sparse_penalty, tolerance) <= 0):
            raise ValueError('Invalid online anomalography parameters.')
        self.rank = rank
        self.ridge = ridge
        self.sparse_penalty = sparse_penalty
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.basis = None
        self.gram = None
        self.cross = None

    def initialize(self, normal_prefix: np.ndarray):
        values = np.asarray(normal_prefix, dtype=np.float64)
        if (values.ndim != 2 or min(values.shape) < self.rank
                or not np.isfinite(values).all()):
            raise ValueError('Warm-up matrix must be finite with both dimensions >= rank.')
        u, singular, _ = np.linalg.svd(values, full_matrices=False)
        if singular[self.rank - 1] <= 1e-10:
            raise ValueError('Warm-up matrix does not span the requested rank.')
        coordinates = u[:, :self.rank] * np.sqrt(singular[:self.rank])
        self.gram = coordinates.T @ coordinates
        self.cross = values.T @ coordinates
        self.basis = np.linalg.solve(self.gram + self.ridge * np.eye(self.rank), self.cross.T).T

    def copy(self):
        clone = OnlineAnomalography(self.rank, self.ridge, self.sparse_penalty,
                                    self.max_iterations, self.tolerance)
        clone.basis = self.basis.copy()
        clone.gram = self.gram.copy()
        clone.cross = self.cross.copy()
        return clone

    def coefficients(self, observation):
        """Solve the joint ridge-coordinate/L1-anomaly subproblem at fixed P."""
        y = np.asarray(observation, dtype=np.float64)
        if self.basis is None or y.shape != (len(self.basis),) or not np.isfinite(y).all():
            raise ValueError('Initialize the tracker and supply a finite feature vector.')
        projection = np.linalg.solve(self.ridge * np.eye(self.rank) + self.basis.T @ self.basis,
                                     self.basis.T)
        # F.T @ F = I - P @ (lambda I + P.T @ P)^(-1) @ P.T.
        gradient_at_zero = y - self.basis @ (projection @ y)
        iterations, warning_seen = 0, False
        if np.max(np.abs(gradient_at_zero)) <= self.sparse_penalty:
            anomaly = np.zeros_like(y)
        else:
            design = np.vstack([np.eye(len(y)) - self.basis @ projection,
                                np.sqrt(self.ridge) * projection])
            target = design @ y
            # sklearn averages the squared loss; divide alpha by the row count.
            lasso = Lasso(alpha=self.sparse_penalty / len(target), fit_intercept=False,
                          max_iter=self.max_iterations, tol=self.tolerance, selection='cyclic')
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                lasso.fit(design, target)
            warning_seen = any(issubclass(item.category, ConvergenceWarning) for item in caught)
            anomaly = lasso.coef_
            iterations = int(lasso.n_iter_)
        coordinates = projection @ (y - anomaly)
        nominal = self.basis @ coordinates
        gradient = nominal + anomaly - y
        active = anomaly != 0
        violation = np.where(active, np.abs(gradient + self.sparse_penalty * np.sign(anomaly)),
                             np.maximum(np.abs(gradient) - self.sparse_penalty, 0))
        relative_kkt = float(np.max(violation) / max(1., np.linalg.norm(y, ord=np.inf)))
        converged = not warning_seen and relative_kkt <= max(1e-5, 100 * self.tolerance)
        return nominal, anomaly, coordinates, {'lasso_iterations': iterations,
                                               'relative_kkt_residual': relative_kkt,
                                               'converged': converged}

    def process(self, values: np.ndarray, update: bool = True) -> OnlineResult:
        """Score each observation using the current basis, then update causally."""
        values = np.asarray(values, dtype=np.float64)
        low_rank, sparse = np.empty_like(values), np.empty_like(values)
        history = []
        for index, y in enumerate(values):
            nominal, anomaly, q, diagnostics = self.coefficients(y)
            low_rank[index], sparse[index] = nominal, anomaly
            history.append({'time_index': index, **diagnostics})
            if update:
                self.gram += np.outer(q, q)
                self.cross += np.outer(y - anomaly, q)
                self.basis = np.linalg.solve(self.gram + self.ridge * np.eye(self.rank), self.cross.T).T
        return OnlineResult(low_rank, sparse, history)
